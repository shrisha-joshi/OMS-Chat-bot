"""
Response validation service for ensuring chatbot responses are based on documents.
Validates that responses use source materials and aren't generic.
"""

import asyncio
import logging
import re
from typing import Dict, Any, Tuple, List

from ..config import settings

logger = logging.getLogger(__name__)


class ResponseValidationService:
    """Service for validating that responses use document sources."""
    
    # Generic phrases that indicate non-document-based responses
    GENERIC_PHRASES = [
        r"(?i)I don't have access to",
        r"(?i)I cannot find information",
        r"(?i)I'm not able to",
        r"(?i)I don't have any information",
        r"(?i)I apologize, but I",
        r"(?i)I don't know",
        r"(?i)I am not sure",
        r"(?i)I cannot say",
        r"(?i)based on my training data",
        r"(?i)in general",
        r"(?i)typically",
        r"(?i)usually",
        r"(?i)generally speaking",
    ]
    
    # Citation patterns
    CITATION_PATTERNS = [
        r"\[[\d\-\,\s]+\]",  # [1], [1-3], [1,2,3]
        r"\(\d+\)",  # (1)
        r"(?:according to|from|in)\s+(?:\[.+?\]|the document|the file)",
        r"based on the (?:provided|uploaded|given) (?:document|material|file|content)",
    ]
    
    def __init__(self):
        self.generic_regex = [re.compile(p) for p in self.GENERIC_PHRASES]
        self.citation_regex = [re.compile(p) for p in self.CITATION_PATTERNS]
    
    def validate_response(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        query: str = ""
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that response is based on documents, not generic.
        
        Args:
            response: The generated response text
            sources: List of source documents used
            query: Original user query
        
        Returns:
            Tuple of (is_valid, validation_details)
        """
        if not settings.validate_document_usage:
            logger.debug("Document usage validation disabled")
            return True, {"validation_enabled": False}
        
        try:
            validation_details = {
                "validation_enabled": True,
                "is_valid": True,
                "has_citations": False,
                "has_generic_phrases": False,
                "citation_count": 0,
                "generic_phrase_count": 0,
                "source_count": len(sources),
                "warnings": [],
                "issues": []
            }
            
            # Check 1: Presence of citations
            citations = self._find_citations(response)
            validation_details["has_citations"] = len(citations) > 0
            validation_details["citation_count"] = len(citations)
            
            logger.debug(f"Found {len(citations)} citations in response")
            
            # Check 2: Generic phrases
            generic_matches = self._find_generic_phrases(response)
            validation_details["has_generic_phrases"] = len(generic_matches) > 0
            validation_details["generic_phrase_count"] = len(generic_matches)
            
            if generic_matches:
                logger.warning(f"Found generic phrases in response: {generic_matches}")
                validation_details["issues"].append(
                    f"Response contains generic phrases: {', '.join(generic_matches[:3])}"
                )
            
            # Check 3: Citation requirements
            if settings.require_citations and not validation_details["has_citations"]:
                validation_details["issues"].append(
                    "Response requires citations but none found"
                )
                logger.warning("Response validation failed: no citations found")
            
            # Check 4: Minimum citation count
            if settings.min_citation_count > 0:
                if validation_details["citation_count"] < settings.min_citation_count:
                    validation_details["warnings"].append(
                        f"Response has {validation_details['citation_count']} citations, "
                        f"minimum required is {settings.min_citation_count}"
                    )
                    logger.debug(f"Citation count below minimum: {validation_details['citation_count']}/{settings.min_citation_count}")
            
            # Check 5: Document reference in response
            has_document_reference = self._has_document_reference(response)
            if not has_document_reference and len(sources) > 0:
                validation_details["warnings"].append(
                    "Response doesn't explicitly reference document sources"
                )
            
            # Check 6: Generic response detection
            if settings.reject_generic_responses:
                if self._is_generic_response(response, sources):
                    validation_details["issues"].append(
                        "Response appears to be generic despite available documents"
                    )
                    logger.warning("Generic response detected despite documents available")
            
            # Determine overall validity
            if validation_details["issues"]:
                validation_details["is_valid"] = False
                logger.warning(f"Response validation failed with issues: {validation_details['issues']}")
            else:
                logger.info("✅ Response validation passed")
            
            return validation_details["is_valid"], validation_details
        
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return True, {"validation_enabled": True, "error": str(e)}
    
    def _find_citations(self, text: str) -> List[str]:
        """Extract citation markers from text."""
        citations = []
        try:
            for pattern in self.citation_regex:
                matches = pattern.findall(text)
                citations.extend(matches)
        except Exception as e:
            logger.debug(f"Error finding citations: {e}")
        return list(set(citations))  # Remove duplicates
    
    def _find_generic_phrases(self, text: str) -> List[str]:
        """Find generic phrases that indicate non-document-based response."""
        phrases = []
        try:
            for pattern in self.generic_regex:
                if pattern.search(text):
                    phrases.append(pattern.pattern)
        except Exception as e:
            logger.debug(f"Error finding generic phrases: {e}")
        return phrases
    
    def _has_document_reference(self, text: str) -> bool:
        """Check if response references documents explicitly."""
        document_references = [
            r"(?i)based on the (?:document|material|content)",
            r"(?i)according to the (?:provided|uploaded) (?:document|file)",
            r"(?i)the document (?:states|shows|indicates|mentions)",
            r"(?i)in (?:this|the) (?:document|material|file)",
            r"(?i)\[[\d\-\,\s]+\]",  # Citation format
        ]
        
        try:
            for pattern in document_references:
                if re.search(pattern, text):
                    return True
        except Exception as e:
            logger.debug(f"Error checking document reference: {e}")
        
        return False
    
    def _is_generic_response(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect if response is generic despite available documents.
        """
        try:
            # If no sources, response might be expected to be generic
            if not sources:
                return False
            
            # Check for generic indicators
            generic_indicators = 0
            
            # Long strings of generic phrases
            if sum(1 for p in self.generic_regex if p.search(response)) > 2:
                generic_indicators += 1
            
            # Response is too short (less than 100 chars) for document-based answer
            if len(response.strip()) < 100:
                generic_indicators += 1
            
            # No citations despite documents available
            if not self._find_citations(response) and len(sources) > 0:
                generic_indicators += 1
            
            # Response doesn't reference specific document content
            if not self._has_specific_details(response, sources):
                generic_indicators += 1
            
            return generic_indicators >= 2
        
        except Exception as e:
            logger.debug(f"Error detecting generic response: {e}")
            return False
    
    def _has_specific_details(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> bool:
        """Check if response contains specific details (not generic)."""
        try:
            # Look for numbers, dates, proper nouns, or specific terms
            specific_patterns = [
                r"\d+(?:\%|dollars?|euros?|pounds?|years?|months?|days?)?",  # Numbers with units
                r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",  # Dates
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",  # Proper nouns (multiple caps)
                r"(?:January|February|March|April|May|June|July|August|September|October|November|December)",
            ]
            
            for pattern in specific_patterns:
                if re.search(pattern, response):
                    return True
            
            # Check if response mentions source-specific content
            for source in sources:
                filename = source.get("filename", "").lower()
                if filename.split('/')[-1][:10] in response.lower():
                    return True
            
            return False
        
        except Exception as e:
            logger.debug(f"Error checking for specific details: {e}")
            return True  # Assume specific if we can't determine
    
    def get_validation_score(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a validation score (0-1) for the response.
        Higher score = better document usage.
        
        Args:
            response: Generated response text
            sources: Source documents used
        
        Returns:
            Score between 0 and 1
        """
        try:
            score = 1.0
            
            # Deduct for generic phrases
            generic_count = len(self._find_generic_phrases(response))
            score -= generic_count * 0.1
            
            # Add bonus for citations
            citation_count = len(self._find_citations(response))
            score += min(citation_count * 0.1, 0.3)
            
            # Add bonus for document references
            if self._has_document_reference(response):
                score += 0.2
            
            # Deduct if response is too short
            if len(response.strip()) < 50:
                score -= 0.2
            
            # Add bonus for specific details
            if self._has_specific_details(response, sources):
                score += 0.15
            
            # Ensure score is in valid range
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.warning(f"Error calculating validation score: {e}")
            return 0.5


# Global validation service instance
response_validation_service = ResponseValidationService()


def get_response_validation_service() -> ResponseValidationService:
    """Dependency injection for response validation service."""
    return response_validation_service
