"""
Prompt Engineering Service for RAG-enhanced responses.
This module manages system prompts, prompt templates, and response generation strategies
to maximize LLM accuracy and consistency.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptService:
    """Service for managing prompts and response generation."""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize prompt templates for different scenarios."""
        
        # Search/QA Template
        self.templates["search"] = {
            "system": """You are a helpful AI assistant specializing in document analysis and question answering.
You have access to retrieved documents and context.

INSTRUCTIONS:
1. Answer based ONLY on the provided context/documents
2. If information is not in the context, clearly state: "This information is not available in the provided documents"
3. Cite specific sources when making claims
4. Format citations as [Source: filename.pdf] or [Source: document_name]
5. Be concise and accurate
6. If unsure, express uncertainty clearly
7. Break complex answers into numbered points

RETRIEVED CONTEXT:
{context}

SOURCES:
{sources}""",
            "examples": [
                {
                    "query": "What is the main topic?",
                    "response": "Based on the provided documents, the main topic is [specific topic]. This is evident from [reference to source]."
                }
            ]
        }
        
        # Summarization Template
        self.templates["summarize"] = {
            "system": """You are an expert summarization assistant.
Your task is to create clear, concise summaries of provided content.

INSTRUCTIONS:
1. Identify key points and main ideas
2. Remove redundant information
3. Maintain logical flow
4. Use clear, simple language
5. Preserve critical details
6. Format as bullet points or numbered list
7. Total summary should be 20-30% of original length

SOURCE MATERIAL:
{context}""",
            "examples": [
                {
                    "query": "Summarize this document",
                    "response": "Key Points:\n• [Point 1]\n• [Point 2]\n• [Point 3]"
                }
            ]
        }
        
        # Analysis Template
        self.templates["analyze"] = {
            "system": """You are an analytical AI assistant specialized in document analysis.

INSTRUCTIONS:
1. Examine the provided context systematically
2. Identify patterns, trends, and relationships
3. Highlight key findings and insights
4. Provide data-backed conclusions
5. Acknowledge limitations and uncertainties
6. Suggest implications or next steps
7. Cite sources for each claim

ANALYTICAL APPROACH:
- Examine data from multiple angles
- Consider context and background
- Compare and contrast information
- Draw logical conclusions

CONTEXT:
{context}""",
            "examples": [
                {
                    "query": "Analyze the key factors",
                    "response": "Analysis reveals several key factors:\n1. [Factor] - Evidence: [Source]\n2. [Factor] - Evidence: [Source]"
                }
            ]
        }
        
        # Comparison Template
        self.templates["compare"] = {
            "system": """You are an expert at comparative analysis.
Compare the provided items, concepts, or documents systematically.

INSTRUCTIONS:
1. Identify similarities and differences
2. Organize comparison in a structured format
3. Use tables or lists where appropriate
4. Highlight key distinctions
5. Cite sources for each comparison
6. Note any ambiguities or missing information

COMPARISON STRUCTURE:
- Similarities
- Differences
- Notable distinctions

CONTENT TO COMPARE:
{context}""",
            "examples": [
                {
                    "query": "Compare A and B",
                    "response": "Similarities:\n• Both are [similarity]\n\nDifferences:\n• A: [difference]\n• B: [difference]"
                }
            ]
        }
        
        # Entity Extraction Template
        self.templates["extract"] = {
            "system": """You are an expert at extracting structured information from text.

INSTRUCTIONS:
1. Extract entities of requested type
2. Return as JSON array
3. Include confidence scores where applicable
4. Note any ambiguous entities
5. Preserve exact quotes when needed

FORMAT:
{{"entities": [{{entity: "name", type: "type", source: "location"}}]}}

TEXT TO ANALYZE:
{context}""",
            "examples": [
                {
                    "query": "Extract all person names",
                    "response": '{"entities": [{"entity": "John Smith", "type": "PERSON", "source": "paragraph 3"}]}'
                }
            ]
        }
        
        # Validation Template
        self.templates["validate"] = {
            "system": """You are a fact-checking assistant.
Validate whether statements are supported by provided documents.

INSTRUCTIONS:
1. Check if each statement is supported by the provided context
2. Provide confidence level (high/medium/low)
3. Cite specific sources
4. Note any conflicting information
5. Flag unsupported claims

VALIDATION CRITERIA:
- Directly mentioned in source
- Logically inferable from source
- Not supported by source
- Contradicted by source

STATEMENT TO VALIDATE:
{statement}

SUPPORTING DOCUMENTS:
{context}""",
            "examples": [
                {
                    "query": "Is this true?",
                    "response": "✓ SUPPORTED (High confidence): This is directly stated in [source].\nEvidence: [quote]"
                }
            ]
        }
    
    def get_system_prompt(
        self,
        task_type: str = "search",
        context: str = "",
        sources: str = "",
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Get a system prompt for the given task type.
        
        Args:
            task_type: Type of task (search, summarize, analyze, compare, extract, validate)
            context: Context/documents to include
            sources: Source citations
            additional_instructions: Additional custom instructions
            
        Returns:
            Formatted system prompt
        """
        try:
            template = self.templates.get(task_type, self.templates["search"])
            system_prompt = template["system"]
            
            # Replace placeholders
            system_prompt = system_prompt.replace("{context}", context)
            system_prompt = system_prompt.replace("{sources}", sources)
            
            # Add additional instructions if provided
            if additional_instructions:
                system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{additional_instructions}"
            
            return system_prompt
            
        except Exception as e:
            logger.error(f"Error generating system prompt: {e}")
            # Return default safe prompt
            return "You are a helpful AI assistant. Answer based on provided context."
    
    def enhance_query(self, query: str, task_type: str = "search") -> str:
        """
        Enhance user query with task-specific instructions.
        
        Args:
            query: Original user query
            task_type: Type of task
            
        Returns:
            Enhanced query
        """
        enhancements = {
            "search": "Please answer the following question based on the provided documents:\n",
            "summarize": "Please provide a concise summary of the following content, highlighting key points:\n",
            "analyze": "Please analyze the following content and provide insights:\n",
            "compare": "Please compare the following items and highlight key differences:\n",
            "extract": "Please extract structured information from the following text:\n",
            "validate": "Please validate whether the following statement is supported by the documents:\n"
        }
        
        enhancement = enhancements.get(task_type, "")
        return enhancement + query
    
    def format_context_with_citations(
        self,
        chunks: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Format context chunks with proper citations.
        
        Args:
            chunks: List of context chunks with metadata
            
        Returns:
            Tuple of (formatted_context, citations)
        """
        try:
            context_parts = []
            citations = []
            
            for i, chunk in enumerate(chunks, 1):
                # Extract chunk content and metadata
                text = chunk.get("text", "")
                source = chunk.get("source", "Unknown")
                doc_name = chunk.get("filename", "Unknown")
                page = chunk.get("page", "")
                chunk_id = chunk.get("chunk_id", "")
                
                # Build citation
                citation = f"[{i}] {doc_name}"
                if page:
                    citation += f" (Page {page})"
                if chunk_id:
                    citation += f" - Section {chunk_id}"
                
                citations.append(citation)
                context_parts.append(f"[{i}] {text}")
            
            context_str = "\n\n".join(context_parts)
            citations_str = "\n".join(citations)
            
            return context_str, citations_str
            
        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            return "", ""
    
    def create_few_shot_examples(
        self,
        task_type: str,
        num_examples: int = 2
    ) -> str:
        """
        Create few-shot examples for improved performance.
        
        Args:
            task_type: Type of task
            num_examples: Number of examples to include
            
        Returns:
            Formatted few-shot prompt
        """
        try:
            template = self.templates.get(task_type, self.templates["search"])
            examples = template.get("examples", [])
            
            if not examples:
                return ""
            
            few_shot = "\nFEW-SHOT EXAMPLES:\n"
            for i, example in enumerate(examples[:num_examples], 1):
                few_shot += f"\nExample {i}:\n"
                few_shot += f"Q: {example['query']}\n"
                few_shot += f"A: {example['response']}\n"
            
            return few_shot
            
        except Exception as e:
            logger.error(f"Error creating few-shot examples: {e}")
            return ""
    
    def build_complete_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        task_type: str = "search",
        include_few_shot: bool = True,
        additional_instructions: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Build complete system and user prompts.
        
        Args:
            query: User query
            context: Context chunks
            task_type: Type of task
            include_few_shot: Whether to include examples
            additional_instructions: Additional instructions
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            # Format context with citations
            formatted_context, citations = self.format_context_with_citations(context)
            
            # Build system prompt
            system_prompt = self.get_system_prompt(
                task_type=task_type,
                context=formatted_context,
                sources=citations,
                additional_instructions=additional_instructions
            )
            
            # Enhance user query
            user_prompt = self.enhance_query(query, task_type)
            
            # Add few-shot examples
            if include_few_shot:
                few_shot = self.create_few_shot_examples(task_type)
                system_prompt += few_shot
            
            return system_prompt, user_prompt
            
        except Exception as e:
            logger.error(f"Error building complete prompt: {e}")
            return "You are a helpful AI assistant.", query
    
    def detect_task_type(self, query: str) -> str:
        """
        Detect the task type from the user query.
        
        Args:
            query: User query
            
        Returns:
            Detected task type
        """
        query_lower = query.lower()
        
        # Define keywords for each task type
        keywords = {
            "summarize": ["summarize", "summary", "brief", "overview", "tldr", "condense"],
            "analyze": ["analyze", "analysis", "analyze", "examine", "breakdown", "deep dive"],
            "compare": ["compare", "difference", "contrast", "similar", "same as", "versus"],
            "extract": ["extract", "list", "find", "get", "what are", "identify", "find all"],
            "validate": ["is this true", "verify", "confirm", "check", "validate", "correct"],
            "search": []  # Default
        }
        
        for task_type, kw_list in keywords.items():
            if any(kw in query_lower for kw in kw_list):
                return task_type
        
        return "search"  # Default to search


# Global instance
prompt_service = PromptService()
