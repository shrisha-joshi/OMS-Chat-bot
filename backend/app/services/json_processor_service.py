"""
JSON file processing and parsing service.
Extracts structured data from JSON files and stores in MongoDB for retrieval.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class JSONProcessorService:
    """Service for processing and parsing JSON files."""
    
    def __init__(self):
        self.supported_schemas = [
            'flat',           # Flat key-value pairs
            'nested',         # Nested objects
            'array',          # Array of objects
            'table',          # Table-like structure
            'hierarchical'    # Hierarchical structure
        ]
    
    async def process_json_file(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a JSON file and extract structured data.
        
        Args:
            content: Raw file content
            filename: Original filename
            
        Returns:
            Dictionary with extracted data and metadata
        """
        try:
            # Parse JSON
            json_data = json.loads(content.decode('utf-8'))
            logger.info(f"Parsed JSON file: {filename}")
            
            # Detect schema type
            schema_type = await self._detect_schema(json_data)
            logger.info(f"Detected schema type: {schema_type}")
            
            # Extract structured data
            extracted_data = await self._extract_data(json_data, schema_type)
            
            # Generate embeddings text
            embeddings_text = await self._generate_embeddings_text(extracted_data)
            
            # Create document record
            result = {
                'filename': filename,
                'schema_type': schema_type,
                'data': extracted_data,
                'embeddings_text': embeddings_text,
                'statistics': {
                    'total_records': len(extracted_data.get('records', [])),
                    'fields': list(extracted_data.get('fields', {}).keys()),
                    'field_count': len(extracted_data.get('fields', {}))
                },
                'processed_at': datetime.utcnow().isoformat()
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            raise
    
    async def _detect_schema(self, data: Any) -> str:
        """Detect the schema type of JSON data."""
        if isinstance(data, dict):
            # Check if it's a table-like structure
            if all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
                return 'flat'
            
            # Check if it has nested objects
            if any(isinstance(v, dict) for v in data.values()):
                return 'nested'
            
            # Check if it has arrays
            if any(isinstance(v, list) for v in data.values()):
                return 'hierarchical'
        
        elif isinstance(data, list):
            if len(data) > 0:
                # Check if array contains objects
                if all(isinstance(item, dict) for item in data):
                    return 'array'
        
        return 'flat'
    
    async def _extract_data(self, data: Any, schema_type: str) -> Dict[str, Any]:
        """Extract structured data based on schema type."""
        records = []
        fields = {}
        
        if schema_type == 'flat':
            records = [data]
            fields = {k: type(v).__name__ for k, v in data.items()}
        
        elif schema_type == 'nested':
            records = self._flatten_nested(data)
            fields = self._get_fields_from_nested(data)
        
        elif schema_type == 'array':
            records = data
            if len(data) > 0:
                fields = {k: type(v).__name__ for k, v in data[0].items() if isinstance(data[0], dict)}
        
        elif schema_type == 'hierarchical':
            records = self._flatten_hierarchical(data)
            fields = self._get_hierarchical_fields(data)
        
        return {
            'records': records,
            'fields': fields,
            'record_count': len(records)
        }
    
    def _flatten_nested(self, data: Dict, parent_key: str = '', sep: str = '.') -> List[Dict]:
        """Flatten nested dictionary structure."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_nested(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return [dict(items)] if items else []
    
    def _get_fields_from_nested(self, data: Dict) -> Dict[str, str]:
        """Extract field types from nested structure."""
        fields = {}
        
        def extract_fields(obj: Any, prefix: str = ''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        extract_fields(v, full_key)
                    else:
                        fields[full_key] = type(v).__name__
        
        extract_fields(data)
        return fields
    
    def _flatten_hierarchical(self, data: Dict) -> List[Dict]:
        """Flatten hierarchical structure."""
        records = []
        
        def flatten(obj: Any, parent_data: Dict = None):
            if parent_data is None:
                parent_data = {}
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, list):
                        for item in v:
                            child_data = parent_data.copy()
                            child_data[k + '_item'] = item
                            flatten(item, child_data)
                    elif isinstance(v, dict):
                        child_data = parent_data.copy()
                        child_data.update({k: str(v)})
                        flatten(v, child_data)
                    else:
                        parent_data[k] = v
            
            if parent_data:
                records.append(parent_data.copy())
        
        flatten(data)
        return records
    
    def _get_hierarchical_fields(self, data: Dict) -> Dict[str, str]:
        """Extract field types from hierarchical structure."""
        fields = {}
        
        def extract_fields(obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        fields[k] = type(v).__name__
                    elif isinstance(v, (dict, list)):
                        if isinstance(v, list) and len(v) > 0:
                            extract_fields(v[0])
                        else:
                            extract_fields(v)
        
        extract_fields(data)
        return fields
    
    async def _generate_embeddings_text(self, extracted_data: Dict[str, Any]) -> str:
        """Generate text content for embeddings from structured data."""
        text_parts = []
        
        # Add field names
        if 'fields' in extracted_data:
            text_parts.append("Fields: " + ", ".join(extracted_data['fields'].keys()))
        
        # Add sample records
        if 'records' in extracted_data and len(extracted_data['records']) > 0:
            sample_records = extracted_data['records'][:3]  # First 3 records
            
            for idx, record in enumerate(sample_records):
                text_parts.append(f"Record {idx + 1}: " + self._record_to_text(record))
        
        # Add statistics
        if 'record_count' in extracted_data:
            text_parts.append(f"Total records: {extracted_data['record_count']}")
        
        return "\n".join(text_parts)
    
    def _record_to_text(self, record: Dict) -> str:
        """Convert a record dictionary to readable text."""
        if not isinstance(record, dict):
            return str(record)
        
        parts = []
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{key}: {json.dumps(value)}")
            else:
                parts.append(f"{key}: {value}")
        
        return "; ".join(parts)
    
    async def create_qa_pairs(self, extracted_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from structured data for better retrieval.
        
        Args:
            extracted_data: Extracted data from JSON
            
        Returns:
            List of question-answer pairs
        """
        qa_pairs = []
        records = extracted_data.get('records', [])
        fields = extracted_data.get('fields', {})
        
        # Generate field-based questions
        for field in fields.keys():
            qa_pairs.append({
                'question': f"What is the '{field}' field?",
                'answer': f"The field '{field}' contains {fields[field]} type data.",
                'type': 'metadata'
            })
        
        # Generate record-based questions
        for idx, record in enumerate(records[:5]):  # First 5 records
            if isinstance(record, dict):
                question = f"What is record {idx + 1}?"
                answer = self._record_to_text(record)
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'data'
                })
        
        # Generate summary questions
        qa_pairs.append({
            'question': f"How many records are in this JSON file?",
            'answer': f"This JSON file contains {len(records)} records.",
            'type': 'summary'
        })
        
        qa_pairs.append({
            'question': f"What are the fields in this JSON file?",
            'answer': f"The fields are: {', '.join(fields.keys())}",
            'type': 'summary'
        })
        
        return qa_pairs


# Singleton instance
_json_processor = None


async def get_json_processor() -> JSONProcessorService:
    """Get or create JSON processor instance."""
    global _json_processor
    if _json_processor is None:
        _json_processor = JSONProcessorService()
    return _json_processor
