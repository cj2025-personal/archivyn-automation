import os
from typing import List, Optional, Dict
import numpy as np


class EmbeddingsService:

    
    def __init__(self, model_name: str = None):
        """
        Initialize embeddings service
        
        Args:
            model_name: Model to use ("sentence-transformers", "ollama", "openai")
                        If None, reads from config
        """
        # Try to get from config if not provided
        config_model_name = None
        if model_name is None:
            try:
                from config.pinecone_config import EMBEDDING_MODEL, EMBEDDING_MODEL_NAME
                model_name = EMBEDDING_MODEL
                config_model_name = EMBEDDING_MODEL_NAME
            except ImportError:
                model_name = "sentence-transformers"
                config_model_name = None
        
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.config_model_name = config_model_name
        
        if model_name == "sentence-transformers":
            self._init_sentence_transformers()
        elif model_name == "ollama":
            self._init_ollama()
        elif model_name == "openai":
            # Use model name from config if available
            openai_model = config_model_name or 'text-embedding-3-small'
            self._init_openai(model=openai_model)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _init_sentence_transformers(self):
        """Initialize sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
            print("[Embeddings] Initialized sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
    
    def _init_ollama(self):
        """Initialize Ollama embeddings"""
        try:
            import ollama
            self.model = ollama
            # Ollama embedding dimensions vary by model
            # nomic-embed-text: 768, mxbai-embed-large: 1024
            self.dimension = 768  # Default for nomic-embed-text
            print("[Embeddings] Initialized Ollama embeddings")
        except ImportError:
            raise ImportError("ollama not installed. Install with: pip install ollama")
    
    def _init_openai(self, model: str = "text-embedding-3-large"):
        """Initialize OpenAI embeddings"""
        try:
            from openai import OpenAI
            import httpx
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            # Create a custom httpx client without proxy settings to avoid conflicts
            try:
                # Try with custom httpx client that explicitly excludes proxies
                http_client = httpx.Client(
                    timeout=60.0,
                    # Don't pass proxies parameter
                )
                self.model = OpenAI(api_key=api_key, http_client=http_client)
            except Exception as e:
                # Fallback: try without custom http_client
                if 'proxies' in str(e).lower() or 'unexpected keyword' in str(e).lower():
                    print(f"[Embeddings] Warning: Trying OpenAI initialization without custom http_client...")
                    # Try with minimal initialization
                    self.model = OpenAI(api_key=api_key)
                else:
                    raise
            
            self.openai_model = model
            
            # Set dimension based on model
            if model == "text-embedding-3-large":
                self.dimension = 3072  # Can be reduced to 256, 1024, or 3072
            elif model == "text-embedding-3-small":
                self.dimension = 1536  # Can be reduced to 512 or 1536
            elif model == "text-embedding-ada-002":
                self.dimension = 1536
            else:
                self.dimension = 1536  # Default
            
            print(f"[Embeddings] Initialized OpenAI embeddings with model: {model} (dimension: {self.dimension})")
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")
    
    def embed_text(self, text: str, max_tokens: int = 8000) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            max_tokens: Maximum tokens allowed (default 8000, leaving buffer for 8192 limit)
        
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.dimension
        
        # Use tiktoken to accurately count tokens for OpenAI models
        if self.model_name == "openai":
            try:
                import tiktoken
                # Get encoding for the model (text-embedding-3-small uses cl100k_base)
                model_to_use = getattr(self, 'openai_model', 'text-embedding-3-small')
                try:
                    encoding = tiktoken.encoding_for_model(model_to_use)
                except KeyError:
                    # Fallback to cl100k_base if model not found
                    encoding = tiktoken.get_encoding("cl100k_base")
                
                # Count actual tokens
                tokens = encoding.encode(text)
                token_count = len(tokens)
                
                if token_count > max_tokens:
                    print(f"[Embeddings] ⚠️ Text too long ({token_count} tokens), truncating to {max_tokens} tokens (GPT limit: 8192)")
                    # Truncate tokens directly (respects GPT token limits exactly)
                    # Leave a small buffer to account for any edge cases
                    safe_max = max_tokens - 50  # 50 token buffer for safety
                    truncated_tokens = tokens[:safe_max]
                    # Decode back to text
                    text = encoding.decode(truncated_tokens)
                    
                    # Verify final token count
                    final_tokens = encoding.encode(text)
                    final_count = len(final_tokens)
                    if final_count > max_tokens:
                        # If still too long (due to decode edge cases), truncate more aggressively
                        print(f"[Embeddings] ⚠️ After decode still {final_count} tokens, truncating further")
                        truncated_tokens = tokens[:max_tokens - 200]  # Leave larger buffer
                        text = encoding.decode(truncated_tokens)
                        final_tokens = encoding.encode(text)
                        final_count = len(final_tokens)
                    
                    print(f"[Embeddings] ✅ Truncated to {final_count} tokens ({len(text)} chars) - within GPT limit of 8192")
                else:
                    print(f"[Embeddings] Token count: {token_count} (within limit of {max_tokens})")
                    
            except ImportError:
                print(f"[Embeddings] ⚠️ tiktoken not installed, using character-based estimation")
                # Fallback to character-based estimation if tiktoken not available
                # Rough estimate: 1 token ≈ 4 characters for English text
                max_chars = int(max_tokens * 3.5)
                original_length = len(text)
                
                if original_length > max_chars:
                    print(f"[Embeddings] ⚠️ Text too long ({original_length} chars, est. {original_length//4} tokens), truncating")
                    truncated = text[:max_chars]
                    last_period = truncated.rfind('.')
                    last_newline = truncated.rfind('\n')
                    cutoff = max(last_period, last_newline)
                    if cutoff > max_chars * 0.8:
                        text = truncated[:cutoff + 1] + "... [truncated]"
                    else:
                        text = truncated + "... [truncated]"
                    print(f"[Embeddings] Truncated to {len(text)} chars")
            except Exception as e:
                print(f"[Embeddings] ⚠️ Error counting tokens: {str(e)}, using character-based estimation")
                # Fallback on any error
                max_chars = int(max_tokens * 3.5)
                if len(text) > max_chars:
                    text = text[:max_chars] + "... [truncated]"
        
        try:
            if self.model_name == "sentence-transformers":
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            
            elif self.model_name == "ollama":
                # Use nomic-embed-text model
                response = self.model.embeddings(
                    model="nomic-embed-text",
                    prompt=text
                )
                return response["embedding"]
            
            elif self.model_name == "openai":
                model_to_use = getattr(self, 'openai_model', 'text-embedding-3-small')
                try:
                    response = self.model.embeddings.create(
                        model=model_to_use,
                        input=text,
                        dimensions=self.dimension if model_to_use.startswith('text-embedding-3') else None
                    )
                    embedding = response.data[0].embedding
                    # Validate embedding is not all zeros
                    if all(v == 0.0 for v in embedding):
                        error_msg = "OpenAI returned zero vector - this indicates an API error"
                        print(f"[Embeddings] ❌ ERROR: {error_msg}")
                        raise ValueError(error_msg)
                    # Check if embedding has reasonable values
                    if len(embedding) != self.dimension:
                        error_msg = f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                        print(f"[Embeddings] ❌ ERROR: {error_msg}")
                        raise ValueError(error_msg)
                    return embedding
                except Exception as e:
                    print(f"[Embeddings] OpenAI API error: {str(e)}")
                    raise
            
        except Exception as e:
            print(f"[Embeddings] ❌ ERROR generating embedding: {str(e)}")
            import traceback
            traceback.print_exc()
            # Don't return zero vector - raise error instead
            raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if self.model_name == "sentence-transformers":
                # Sentence transformers handles batching efficiently
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                return embeddings.tolist()
            
            elif self.model_name == "ollama":
                # Process in batches
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = []
                    for text in batch:
                        embedding = self.embed_text(text)
                        batch_embeddings.append(embedding)
                    all_embeddings.extend(batch_embeddings)
                return all_embeddings
            
            elif self.model_name == "openai":
                # OpenAI handles batching, but we need to truncate each text to respect token limits
                model_to_use = getattr(self, 'openai_model', 'text-embedding-3-small')
                max_tokens_per_text = 8000  # Leave buffer for 8192 limit
                
                # Truncate each text if needed using tiktoken
                try:
                    import tiktoken
                    try:
                        encoding = tiktoken.encoding_for_model(model_to_use)
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    
                    truncated_texts = []
                    for i, text in enumerate(texts):
                        if not text or not text.strip():
                            truncated_texts.append("")
                            continue
                        
                        tokens = encoding.encode(text)
                        token_count = len(tokens)
                        
                        if token_count > max_tokens_per_text:
                            print(f"[Embeddings] ⚠️ Text {i+1}/{len(texts)} too long ({token_count} tokens), truncating")
                            safe_max = max_tokens_per_text - 50
                            truncated_tokens = tokens[:safe_max]
                            text = encoding.decode(truncated_tokens)
                            # Verify
                            final_tokens = encoding.encode(text)
                            if len(final_tokens) > max_tokens_per_text:
                                truncated_tokens = tokens[:max_tokens_per_text - 200]
                                text = encoding.decode(truncated_tokens)
                            print(f"[Embeddings] ✅ Text {i+1} truncated to {len(encoding.encode(text))} tokens")
                        
                        truncated_texts.append(text)
                    
                    texts = truncated_texts
                except ImportError:
                    print(f"[Embeddings] ⚠️ tiktoken not installed for batch, using character-based estimation")
                    # Fallback: truncate by characters
                    max_chars = int(max_tokens_per_text * 3.5)
                    texts = [text[:max_chars] if len(text) > max_chars else text for text in texts]
                except Exception as e:
                    print(f"[Embeddings] ⚠️ Error truncating batch texts: {str(e)}")
                    # Continue with original texts, API will handle or error
                
                try:
                    response = self.model.embeddings.create(
                        model=model_to_use,
                        input=texts,
                        dimensions=self.dimension if model_to_use.startswith('text-embedding-3') else None
                    )
                    embeddings = [item.embedding for item in response.data]
                    # Validate embeddings are not all zeros
                    for i, emb in enumerate(embeddings):
                        if all(v == 0.0 for v in emb):
                            print(f"[Embeddings] WARNING: OpenAI returned zero vector for text {i}, this might indicate an error")
                    return embeddings
                except Exception as e:
                    print(f"[Embeddings] OpenAI API error: {str(e)}")
                    raise
        
        except Exception as e:
            print(f"[Embeddings] ❌ ERROR generating batch embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            # Don't return zero vectors - raise error instead
            raise ValueError(f"Failed to generate batch embeddings: {str(e)}")
    
    def embed_profile(self, profile_data: Dict) -> List[float]:
        """
        Generate embedding for profile data
        
        Args:
            profile_data: Profile data dictionary
        
        Returns:
            Embedding vector
        """
        # Combine relevant profile fields into a single text
        text_parts = []
        
        if profile_data.get("name"):
            text_parts.append(f"Name: {profile_data['name']}")
        if profile_data.get("university"):
            text_parts.append(f"University: {profile_data['university']}")
        if profile_data.get("department"):
            text_parts.append(f"Department: {profile_data['department']}")
        if profile_data.get("position"):
            text_parts.append(f"Position: {profile_data['position']}")
        if profile_data.get("email"):
            text_parts.append(f"Email: {profile_data['email']}")
        if profile_data.get("full_text"):
            text_parts.append(profile_data['full_text'])
        
        combined_text = " ".join(text_parts)
        return self.embed_text(combined_text)
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension


# Singleton instances
_embeddings_instances = {}

def get_embeddings_service(model_name: str = None) -> EmbeddingsService:
    """Get or create EmbeddingsService instance"""
    # If no model_name provided, use None to read from config
    if model_name is None:
        try:
            from config.pinecone_config import EMBEDDING_MODEL
            model_name = EMBEDDING_MODEL
        except ImportError:
            model_name = "sentence-transformers"
    
    if model_name not in _embeddings_instances:
        _embeddings_instances[model_name] = EmbeddingsService(model_name=model_name)
    return _embeddings_instances[model_name]

