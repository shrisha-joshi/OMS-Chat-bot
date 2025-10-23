"""
Feedback API routes for collecting user feedback and system improvement.
This module handles user feedback collection, rating systems,
and data for future fine-tuning operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    response: str
    rating: str = Field(..., pattern="^(thumbs_up|thumbs_down|helpful|not_helpful|1|2|3|4|5)$")
    correction: Optional[str] = None
    feedback_text: Optional[str] = None
    response_time: Optional[float] = None
    sources_helpful: Optional[bool] = None
    category: Optional[str] = None

class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: str
    message: str

class FeedbackStats(BaseModel):
    total_feedback: int
    positive_feedback: int
    negative_feedback: int
    average_rating: float
    improvement_suggestions: int

class FeedbackSummary(BaseModel):
    feedback_id: str
    session_id: str
    query: str
    rating: str
    timestamp: str
    has_correction: bool
    category: Optional[str]

@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    Submit user feedback for a chat interaction.
    
    Args:
        feedback: Feedback data including rating and optional correction
        mongo_client: MongoDB client
        redis_client: Redis client
    
    Returns:
        Feedback submission confirmation
    """
    try:
        # Validate feedback data
        if not feedback.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not feedback.response.strip():
            raise HTTPException(status_code=400, detail="Response cannot be empty")
        
        # Store feedback in database
        success = await mongo_client.store_feedback(
            session_id=feedback.session_id,
            query=feedback.query,
            response=feedback.response,
            rating=feedback.rating,
            correction=feedback.correction
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
        
        # Store additional feedback metadata
        feedback_metadata = {
            "feedback_text": feedback.feedback_text,
            "response_time": feedback.response_time,
            "sources_helpful": feedback.sources_helpful,
            "category": feedback.category,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache feedback for analytics
        feedback_key = f"feedback:{feedback.session_id}:{int(datetime.utcnow().timestamp())}"
        await redis_client.set_cache(feedback_key, feedback_metadata, expiry_seconds=86400)
        
        # Update feedback counters
        await _update_feedback_counters(feedback.rating, redis_client)
        
        logger.info(f"Feedback submitted for session {feedback.session_id}: {feedback.rating}")
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_key,
            message="Feedback submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    days: int = Query(30, ge=1, le=365),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Get feedback statistics for the specified time period.
    
    Args:
        days: Number of days to include in statistics
        mongo_client: MongoDB client
    
    Returns:
        Feedback statistics summary
    """
    try:
        from datetime import timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query feedback from database
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_feedback": {"$sum": 1},
                    "ratings": {"$push": "$rating"},
                    "corrections": {
                        "$sum": {
                            "$cond": [
                                {"$ne": ["$correction", None]},
                                1,
                                0
                            ]
                        }
                    }
                }
            }
        ]
        
        cursor = mongo_client.database.feedback.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if not results:
            return FeedbackStats(
                total_feedback=0,
                positive_feedback=0,
                negative_feedback=0,
                average_rating=0.0,
                improvement_suggestions=0
            )
        
        data = results[0]
        ratings = data.get("ratings", [])
        
        # Calculate statistics
        positive_ratings = ["thumbs_up", "helpful", "4", "5"]
        negative_ratings = ["thumbs_down", "not_helpful", "1", "2"]
        
        positive_count = sum(1 for r in ratings if r in positive_ratings)
        negative_count = sum(1 for r in ratings if r in negative_ratings)
        
        # Calculate average rating (convert to numeric)
        numeric_ratings = []
        for rating in ratings:
            if rating == "thumbs_up" or rating == "helpful":
                numeric_ratings.append(5)
            elif rating == "thumbs_down" or rating == "not_helpful":
                numeric_ratings.append(1)
            elif rating.isdigit():
                numeric_ratings.append(int(rating))
        
        average_rating = sum(numeric_ratings) / len(numeric_ratings) if numeric_ratings else 0
        
        return FeedbackStats(
            total_feedback=data.get("total_feedback", 0),
            positive_feedback=positive_count,
            negative_feedback=negative_count,
            average_rating=round(average_rating, 2),
            improvement_suggestions=data.get("corrections", 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback statistics")

@router.get("/recent", response_model=List[FeedbackSummary])
async def get_recent_feedback(
    limit: int = Query(20, ge=1, le=100),
    rating_filter: Optional[str] = Query(None),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Get recent feedback submissions.
    
    Args:
        limit: Maximum number of feedback entries to return
        rating_filter: Optional filter by rating type
        mongo_client: MongoDB client
    
    Returns:
        List of recent feedback summaries
    """
    try:
        # Build query
        query = {}
        if rating_filter:
            query["rating"] = rating_filter
        
        # Get recent feedback
        cursor = mongo_client.database.feedback.find(query).sort("created_at", -1).limit(limit)
        feedback_list = []
        
        async for feedback in cursor:
            feedback_list.append(FeedbackSummary(
                feedback_id=str(feedback["_id"]),
                session_id=feedback.get("session_id", ""),
                query=feedback.get("query", "")[:100] + "..." if len(feedback.get("query", "")) > 100 else feedback.get("query", ""),
                rating=feedback.get("rating", ""),
                timestamp=feedback.get("created_at", datetime.utcnow()).isoformat(),
                has_correction=bool(feedback.get("correction")),
                category=feedback.get("category")
            ))
        
        return feedback_list
        
    except Exception as e:
        logger.error(f"Failed to get recent feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent feedback")

@router.get("/corrections")
async def get_feedback_corrections(
    limit: int = Query(50, ge=1, le=200),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Get feedback entries with corrections for fine-tuning.
    
    Args:
        limit: Maximum number of corrections to return
        mongo_client: MongoDB client
    
    Returns:
        List of feedback entries with user corrections
    """
    try:
        # Query feedback with corrections
        query = {"correction": {"$ne": None}}
        cursor = mongo_client.database.feedback.find(query).sort("created_at", -1).limit(limit)
        
        corrections = []
        async for feedback in cursor:
            corrections.append({
                "feedback_id": str(feedback["_id"]),
                "session_id": feedback.get("session_id", ""),
                "query": feedback.get("query", ""),
                "original_response": feedback.get("response", ""),
                "user_correction": feedback.get("correction", ""),
                "rating": feedback.get("rating", ""),
                "timestamp": feedback.get("created_at", datetime.utcnow()).isoformat()
            })
        
        return {
            "corrections": corrections,
            "count": len(corrections),
            "message": f"Retrieved {len(corrections)} feedback corrections"
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback corrections: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback corrections")

@router.get("/export")
async def export_feedback_for_training(
    format: str = Query("jsonl", pattern="^(jsonl|csv)$"),
    days: int = Query(30, ge=1, le=365),
    min_rating: Optional[str] = Query(None),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Export feedback data for model training and fine-tuning.
    
    Args:
        format: Export format (jsonl or csv)
        days: Number of days of data to export
        min_rating: Minimum rating to include
        mongo_client: MongoDB client
    
    Returns:
        Exported feedback data
    """
    try:
        from datetime import timedelta
        import json
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query = {
            "created_at": {"$gte": start_date, "$lte": end_date}
        }
        
        if min_rating:
            query["rating"] = {"$in": [min_rating, "thumbs_up", "helpful", "4", "5"]}
        
        # Get feedback data
        cursor = mongo_client.database.feedback.find(query).sort("created_at", -1)
        
        export_data = []
        async for feedback in cursor:
            training_example = {
                "query": feedback.get("query", ""),
                "response": feedback.get("correction") or feedback.get("response", ""),
                "rating": feedback.get("rating", ""),
                "timestamp": feedback.get("created_at", datetime.utcnow()).isoformat()
            }
            
            if format == "jsonl":
                export_data.append(json.dumps(training_example))
            else:
                export_data.append(training_example)
        
        return {
            "format": format,
            "data": export_data,
            "count": len(export_data),
            "exported_at": datetime.utcnow().isoformat(),
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to export feedback data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export feedback data")

@router.delete("/session/{session_id}")
async def delete_session_feedback(
    session_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Delete all feedback for a specific session."""
    try:
        result = await mongo_client.database.feedback.delete_many({"session_id": session_id})
        
        return {
            "success": True,
            "session_id": session_id,
            "deleted_count": result.deleted_count,
            "message": f"Deleted {result.deleted_count} feedback entries"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete session feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session feedback")

@router.get("/analytics/trends")
async def get_feedback_trends(
    days: int = Query(30, ge=7, le=365),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Get feedback trends over time for analytics."""
    try:
        from datetime import timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Aggregate feedback by day
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at"
                        }
                    },
                    "total_feedback": {"$sum": 1},
                    "positive_feedback": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$rating", ["thumbs_up", "helpful", "4", "5"]]},
                                1,
                                0
                            ]
                        }
                    },
                    "negative_feedback": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$rating", ["thumbs_down", "not_helpful", "1", "2"]]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        cursor = mongo_client.database.feedback.aggregate(pipeline)
        trends = []
        
        async for day_data in cursor:
            trends.append({
                "date": day_data["_id"],
                "total_feedback": day_data["total_feedback"],
                "positive_feedback": day_data["positive_feedback"],
                "negative_feedback": day_data["negative_feedback"],
                "satisfaction_rate": (
                    day_data["positive_feedback"] / day_data["total_feedback"] * 100
                    if day_data["total_feedback"] > 0 else 0
                )
            })
        
        return {
            "trends": trends,
            "period": f"{days} days",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback trends")

# Utility functions

async def _update_feedback_counters(rating: str, redis_client: RedisClient):
    """Update feedback counters in Redis for real-time analytics."""
    try:
        # Update overall counter
        await redis_client.increment_counter("feedback:total", 86400)
        
        # Update rating-specific counters
        positive_ratings = ["thumbs_up", "helpful", "4", "5"]
        if rating in positive_ratings:
            await redis_client.increment_counter("feedback:positive", 86400)
        else:
            await redis_client.increment_counter("feedback:negative", 86400)
        
        # Update daily counter
        today = datetime.utcnow().strftime("%Y-%m-%d")
        await redis_client.increment_counter(f"feedback:daily:{today}", 86400)
        
    except Exception as e:
        logger.error(f"Failed to update feedback counters: {e}")

@router.get("/health")
async def feedback_health_check():
    """Health check for feedback service."""
    try:
        return {
            "status": "healthy",
            "service": "feedback",
            "features": {
                "rating_collection": True,
                "correction_tracking": True,
                "export_formats": ["jsonl", "csv"],
                "analytics": True
            }
        }
    except Exception as e:
        logger.error(f"Feedback health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "feedback",
            "error": str(e)
        }