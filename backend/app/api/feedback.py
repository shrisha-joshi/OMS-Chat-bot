"""
Feedback API routes for collecting user feedback and system improvement.
This module handles user feedback collection, rating systems,
and data for future fine-tuning operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import asyncio
import logging
from bson import ObjectId

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# MongoDB query constants
RATING_FIELD = "$rating"
COND_OPERATOR = "$cond"

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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Cache feedback for analytics
        feedback_key = f"feedback:{feedback.session_id}:{int(datetime.now(timezone.utc).timestamp())}"
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
        end_date = datetime.now(timezone.utc)
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
                    "ratings": {"$push": RATING_FIELD},
                    "corrections": {
                        "$sum": {
                            COND_OPERATOR: [
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
                timestamp=feedback.get("created_at", datetime.now(timezone.utc)).isoformat(),
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
                "timestamp": feedback.get("created_at", datetime.now(timezone.utc)).isoformat()
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
        end_date = datetime.now(timezone.utc)
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
                "timestamp": feedback.get("created_at", datetime.now(timezone.utc)).isoformat()
            }
            
            if format == "jsonl":
                export_data.append(json.dumps(training_example))
            else:
                export_data.append(training_example)
        
        return {
            "format": format,
            "data": export_data,
            "count": len(export_data),
            "exported_at": datetime.now(timezone.utc).isoformat(),
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
        end_date = datetime.now(timezone.utc)
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
                            COND_OPERATOR: [
                                {"$in": [RATING_FIELD, ["thumbs_up", "helpful", "4", "5"]]},
                                1,
                                0
                            ]
                        }
                    },
                    "negative_feedback": {
                        "$sum": {
                            COND_OPERATOR: [
                                {"$in": [RATING_FIELD, ["thumbs_down", "not_helpful", "1", "2"]]},
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

# ============================================================================
# Phase 5: ADVANCED FEEDBACK SYSTEM FOR CONTINUOUS IMPROVEMENT
# ============================================================================

@router.post("/submit-advanced")
async def submit_advanced_feedback(
    session_id: str,
    query: str,
    response: str,
    rating: int,
    feedback_type: str = "general",
    comment: Optional[str] = None,
    source_documents: Optional[List[str]] = None,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Advanced feedback submission for ML-driven improvements.
    
    Args:
        session_id: Session ID
        query: Original query
        response: System response
        rating: Rating 1-5
        feedback_type: Type of feedback (helpful, incorrect, incomplete, misleading)
        comment: Optional detailed comment
        source_documents: Optional list of source document IDs
        mongo_client: MongoDB client
        
    Returns:
        Feedback submission confirmation
    """
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Store advanced feedback
        feedback_doc = {
            "session_id": session_id,
            "query": query,
            "response": response[:500],  # Store truncated
            "rating": rating,
            "feedback_type": feedback_type,
            "comment": comment,
            "source_documents": source_documents or [],
            "created_at": datetime.now(timezone.utc),
            "processed": False,
            "training_data_generated": False,
            "phase": "phase_5_advanced"
        }
        
        result = await mongo_client.database.feedback_submissions.insert_one(feedback_doc)
        feedback_id = str(result.inserted_id)
        
        logger.info(f"✅ Advanced feedback submitted: {feedback_id} (rating: {rating}/5, type: {feedback_type})")
        
        # Log low ratings for priority processing
        if rating <= 2:
            logger.warning(f"⚠️  Critical feedback: {feedback_type} - {comment or 'No comment'}")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "rating": rating,
            "feedback_type": feedback_type,
            "message": "Advanced feedback recorded for system improvement"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit advanced feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.get("/analytics-advanced")
async def get_advanced_analytics(  # noqa: python:S3776
    time_period: str = "7d",
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Get comprehensive feedback analytics for system improvement.
    
    Args:
        time_period: "1d", "7d", "30d", or "all"
        mongo_client: MongoDB client
        
    Returns:
        Detailed analytics and insights
    """
    try:
        from datetime import timedelta
        
        # Time range calculation
        time_ranges = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "all": None
        }
        
        time_delta = time_ranges.get(time_period, timedelta(days=7))
        query = {}
        
        if time_delta:
            cutoff_date = datetime.now(timezone.utc) - time_delta
            query["created_at"] = {"$gte": cutoff_date}
        
        # Get all feedback
        feedback_entries = await mongo_client.database.feedback_submissions.find(query).to_list(None)
        total_feedback = len(feedback_entries)
        
        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "message": "No feedback data available for period",
                "period": time_period
            }
        
        # Calculate comprehensive metrics
        ratings = [f.get("rating", 3) for f in feedback_entries if "rating" in f]
        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
        
        # Rating distribution
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            if 1 <= rating <= 5:
                rating_dist[rating] += 1
        
        # Feedback type distribution
        feedback_types = {}
        for entry in feedback_entries:
            ftype = entry.get("feedback_type", "general")
            feedback_types[ftype] = feedback_types.get(ftype, 0) + 1
        
        # Issue analysis for low ratings
        issues = {}
        low_rating_entries = [f for f in feedback_entries if f.get("rating", 5) <= 2]
        
        for entry in low_rating_entries:
            issue = entry.get("feedback_type", "other")
            issues[issue] = issues.get(issue, 0) + 1
        
        top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate insights
        insights = []
        
        if avg_rating < 2.5:
            insights.append({
                "severity": "critical",
                "message": "System quality critically low",
                "recommendation": "Immediate model retraining required"
            })
        elif avg_rating < 3.0:
            insights.append({
                "severity": "warning",
                "message": "System quality below acceptable threshold",
                "recommendation": "Focus on accuracy improvements"
            })
        
        low_rating_pct = (len(low_rating_entries) / total_feedback * 100) if total_feedback > 0 else 0
        if low_rating_pct > 20:
            insights.append({
                "severity": "warning",
                "message": f"{low_rating_pct:.1f}% of responses rated poorly",
                "recommendation": "Investigate and improve retrieval and generation"
            })
        
        # Quality metrics
        quality_metrics = {
            "accuracy": (rating_dist[5] / total_feedback * 100) if total_feedback > 0 else 0,
            "acceptability": ((rating_dist[4] + rating_dist[5]) / total_feedback * 100) if total_feedback > 0 else 0,
            "poor_quality": ((rating_dist[1] + rating_dist[2]) / total_feedback * 100) if total_feedback > 0 else 0
        }
        
        logger.info(f"✅ Analytics generated: {total_feedback} entries, avg {avg_rating:.2f}/5")
        
        return {
            "period": time_period,
            "total_feedback": total_feedback,
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_dist,
            "feedback_type_distribution": feedback_types,
            "top_issues": [{"issue": issue, "count": count} for issue, count in top_issues],
            "quality_metrics": quality_metrics,
            "insights": insights,
            "actionable_recommendations": [
                "Review low-rated responses for patterns",
                "Analyze feedback comments for improvement areas",
                "Consider retraining models with collected data",
                "Improve document retrieval quality",
                "Enhance response validation"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get advanced analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@router.post("/generate-training-data")
async def generate_training_data_from_feedback(
    min_rating: int = 4,
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    redis_client: RedisClient = Depends(get_redis_client)
) -> Dict[str, Any]:
    """
    Generate training data from high-quality feedback for model improvement.
    
    Args:
        min_rating: Minimum rating for positive examples (default: 4)
        mongo_client: MongoDB client
        redis_client: Redis client
        
    Returns:
        Training data generation status
    """
    try:
        # Get high-quality feedback
        high_quality = await mongo_client.database.feedback_submissions.find({
            "rating": {"$gte": min_rating},
            "training_data_generated": False
        }).to_list(None)
        
        # Get low-quality feedback for negative examples
        low_quality = await mongo_client.database.feedback_submissions.find({
            "rating": {"$lt": 3},
            "training_data_generated": False
        }).to_list(None)
        
        training_data = []
        
        # Positive examples
        for entry in high_quality:
            training_data.append({
                "query": entry.get("query"),
                "response": entry.get("response"),
                "label": "positive",
                "rating": entry.get("rating"),
                "feedback_type": entry.get("feedback_type"),
                "source_id": str(entry.get("_id"))
            })
        
        # Negative examples
        for entry in low_quality:
            training_data.append({
                "query": entry.get("query"),
                "response": entry.get("response"),
                "label": "negative",
                "rating": entry.get("rating"),
                "feedback_type": entry.get("feedback_type"),
                "source_id": str(entry.get("_id"))
            })
        
        # Store in Redis
        if training_data:
            await redis_client.set_json(
                "training_data:phase5:generated",
                {
                    "entries": training_data,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_count": len(training_data),
                    "positive_count": len(high_quality),
                    "negative_count": len(low_quality),
                    "ready_for_retraining": True
                },
                expire_seconds=604800  # 7 days
            )
            
            # Mark as processed
            processed_ids = [entry["source_id"] for entry in training_data]
            await mongo_client.database.feedback_submissions.update_many(
                {"_id": {"$in": [ObjectId(id) for id in processed_ids]}},
                {"$set": {"training_data_generated": True}}
            )
            
            logger.info(f"✅ Training data generated: {len(training_data)} samples "
                       f"({len(high_quality)} positive, {len(low_quality)} negative)")
        
        return {
            "success": True,
            "training_samples": len(training_data),
            "positive_examples": len(high_quality),
            "negative_examples": len(low_quality),
            "status": "ready_for_retraining" if training_data else "no_new_data",
            "message": "Training data ready for model improvement"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate training data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate training data")


@router.get("/system-insights")
async def get_system_insights_from_feedback(
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Get high-level system insights from all feedback.
    
    Args:
        mongo_client: MongoDB client
        
    Returns:
        System-level insights and recommendations
    """
    try:
        total_entries = await mongo_client.database.feedback_submissions.count_documents({})
        
        if total_entries == 0:
            return {
                "total_feedback": 0,
                "status": "no_data",
                "message": "No feedback collected yet"
            }
        
        all_feedback = await mongo_client.database.feedback_submissions.find({}).to_list(None)
        ratings = [f.get("rating", 3) for f in all_feedback if "rating" in f]
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
        
        # Rating distribution
        excellent = sum(1 for r in ratings if r >= 4)
        acceptable = sum(1 for r in ratings if r == 3)
        poor = sum(1 for r in ratings if r <= 2)
        
        # Issues breakdown
        issues_map = {}
        for entry in all_feedback:
            if entry.get("rating", 5) <= 2:
                issue = entry.get("feedback_type", "other")
                issues_map[issue] = issues_map.get(issue, 0) + 1
        
        # System health status
        if avg_rating >= 4.0:
            health_status = "excellent"
            priority = "monitor"
        elif avg_rating >= 3.5:
            health_status = "good"
            priority = "maintain"
        elif avg_rating >= 3.0:
            health_status = "acceptable"
            priority = "improve"
        else:
            health_status = "poor"
            priority = "critical"
        
        # Generate actionable insights
        recommendations = []
        
        if health_status == "poor":
            recommendations.append("🚨 CRITICAL: System requires immediate attention")
            recommendations.append("   Action: Review and retrain models with collected feedback data")
        
        if (poor / total_entries * 100) > 25:
            recommendations.append(f"⚠️  {(poor / total_entries * 100):.1f}% of responses rated poorly")
            recommendations.append("   Action: Investigate retrieval and generation quality")
        
        if issues_map.get("incorrect", 0) > total_entries * 0.2:
            recommendations.append(f"⚠️  High incorrect response rate ({issues_map.get('incorrect', 0)} cases)")
            recommendations.append("   Action: Verify source documents and improve accuracy")
        
        if issues_map.get("incomplete", 0) > total_entries * 0.3:
            recommendations.append(f"⚠️  Many incomplete responses ({issues_map.get('incomplete', 0)} cases)")
            recommendations.append("   Action: Improve context retrieval and response length")
        
        logger.info(f"✅ System insights: {total_entries} feedback entries, health={health_status}, priority={priority}")
        
        return {
            "feedback_summary": {
                "total_feedback": total_entries,
                "average_rating": round(avg_rating, 2),
                "rating_breakdown": {
                    "excellent_4_5": excellent,
                    "acceptable_3": acceptable,
                    "poor_1_2": poor
                }
            },
            "system_health": {
                "status": health_status,
                "priority": priority,
                "confidence": round(min(excellent / total_entries, 1.0), 2) if total_entries > 0 else 0
            },
            "issues_detected": dict(sorted(issues_map.items(), key=lambda x: x[1], reverse=True)[:5]),
            "recommendations": recommendations,
            "next_actions": {
                "immediate": "Review feedback for patterns",
                "short_term": "Generate and apply training data",
                "medium_term": "Retrain models with feedback",
                "long_term": "Implement continuous improvement loop"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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