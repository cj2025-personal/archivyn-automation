"""
Data Cleaning Service Integration
Integrates the data cleaning pipeline with the existing API services
"""
from typing import List, Dict, Optional
import os
import sys

# Add parent directory to path to import the pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_cleaning_pipeline import DataCleaningPipeline


class DataCleaningService:
    """Service wrapper for the data cleaning pipeline"""
    
    def __init__(
        self, 
        target_words_per_chunk: int = 325, 
        min_words_per_chunk: int = 250, 
        max_words_per_chunk: int = 400,
        use_llm_cleaning: bool = False,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None
    ):
        """
        Initialize the service
        
        Args:
            target_words_per_chunk: Target words per chunk (default: 325)
            min_words_per_chunk: Minimum words per chunk (default: 250)
            max_words_per_chunk: Maximum words per chunk (default: 400)
            use_llm_cleaning: Enable LLM-based chunk cleaning (default: False)
            llm_provider: LLM provider ("openai" or "ollama", default: "openai")
            llm_model: LLM model name (default: "gpt-4o-mini" for OpenAI, "llama3" for Ollama)
            llm_api_key: API key for LLM provider (optional, uses env vars if not provided)
            ollama_host: Ollama host URL (optional, uses env var if not provided)
        """
        self.pipeline = DataCleaningPipeline(
            target_words_per_chunk=target_words_per_chunk,
            min_words_per_chunk=min_words_per_chunk,
            max_words_per_chunk=max_words_per_chunk,
            use_llm_cleaning=use_llm_cleaning,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            ollama_host=ollama_host
        )
    
    def clean_and_chunk_text(self, text: str, profile_url: str = "", 
                            section_header: str = "") -> List[Dict]:
        """
        Clean and chunk text using the pipeline
        
        Args:
            text: Raw text to process
            profile_url: Source profile URL
            section_header: Optional section header
            
        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []
        
        return self.pipeline.process_text(
            text=text,
            profile_url=profile_url,
            section_header=section_header
        )
    
    def clean_and_chunk_from_json(self, json_file: str, output_file: str = "chunks.json") -> List[Dict]:
        """
        Process JSON file and create chunks
        
        Args:
            json_file: Path to input JSON file
            output_file: Path to output chunks JSON file
            
        Returns:
            List of chunk dictionaries
        """
        return self.pipeline.process_json_file(json_file, output_file)
    
    def normalize_text(self, text: str) -> str:
        """Normalize raw text"""
        return self.pipeline.normalize_text(text)
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text"""
        return self.pipeline.split_into_sections(text)
    
    def clean_section(self, text: str) -> str:
        """Clean a section of text"""
        return self.pipeline.clean_section(text)


# Singleton instance
_cleaning_service_instance = None

def get_data_cleaning_service(
    target_words_per_chunk: int = 325, 
    min_words_per_chunk: int = 250, 
    max_words_per_chunk: int = 400,
    use_llm_cleaning: bool = False,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    ollama_host: Optional[str] = None
) -> DataCleaningService:
    """Get or create data cleaning service instance"""
    global _cleaning_service_instance
    
    if _cleaning_service_instance is None:
        _cleaning_service_instance = DataCleaningService(
            target_words_per_chunk=target_words_per_chunk,
            min_words_per_chunk=min_words_per_chunk,
            max_words_per_chunk=max_words_per_chunk,
            use_llm_cleaning=use_llm_cleaning,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            ollama_host=ollama_host
        )
    
    return _cleaning_service_instance

