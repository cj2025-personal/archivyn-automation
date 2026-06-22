"""
Pinecone Vector Database Service
Handles vector storage and retrieval for faculty profile data
"""
import os
from typing import List, Dict, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
import hashlib
import json


class VectorDBService:
    """
    Service for managing vector embeddings in Pinecone
    """
    
    def __init__(self, api_key: Optional[str] = None, index_name: str = "faculty-profiles", dimension: int = 384):
        """
        Initialize Pinecone connection
        
        Args:
            api_key: Pinecone API key (defaults to environment variable or config)
            index_name: Name of the Pinecone index
            dimension: Vector dimension (must match embedding model)
        """
        # Try to get from config first, then environment, then parameter
        try:
            from config.pinecone_config import PINECONE_API_KEY as CONFIG_KEY
            self.api_key = api_key or os.getenv("PINECONE_API_KEY") or CONFIG_KEY
        except ImportError:
            self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable or configure in config/pinecone_config.py")
        
        self.index_name = index_name
        self.dimension = dimension
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure the index exists, create if it doesn't"""
        try:
            # Try to get config values
            try:
                from config.pinecone_config import (
                    INDEX_METRIC, SERVERLESS_CLOUD, SERVERLESS_REGION
                )
                metric = INDEX_METRIC
                cloud = SERVERLESS_CLOUD
                region = SERVERLESS_REGION
            except ImportError:
                # Use defaults
                metric = "cosine"
                cloud = "aws"
                region = "us-east-1"
            
            # Check if index exists
            if self.index_name in self.pc.list_indexes().names():
                self.index = self.pc.Index(self.index_name)
                print(f"[VectorDB] Connected to existing index: {self.index_name}")
            else:
                # Create new index
                print(f"[VectorDB] Creating new index: {self.index_name}")
                print(f"[VectorDB] Dimension: {self.dimension}, Metric: {metric}, Region: {region}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(5)
                self.index = self.pc.Index(self.index_name)
                print(f"[VectorDB] Index created successfully: {self.index_name}")
        except PineconeException as e:
            print(f"[VectorDB] Error managing index: {str(e)}")
            raise
    
    def _generate_id(self, profile_url: str, content_type: str, chunk_index: int = 0) -> str:
        """
        Generate a unique ID for a vector
        
        Args:
            profile_url: Profile URL
            content_type: Type of content (profile, document, webpage)
            chunk_index: Index of chunk if content is split
        
        Returns:
            Unique vector ID
        """
        content = f"{profile_url}:{content_type}:{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def upsert_profile(
        self,
        profile_url: str,
        profile_data: Dict,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store or update a profile vector
        
        Args:
            profile_url: Profile URL
            profile_data: Profile data dictionary
            embedding: Vector embedding (list of floats)
            metadata: Additional metadata to store
        
        Returns:
            True if successful
        """
        try:
            vector_id = self._generate_id(profile_url, "profile")
            
            # Prepare metadata
            vector_metadata = {
                "profile_url": profile_url,
                "content_type": "profile",
                "name": profile_data.get("name", ""),
                "university": profile_data.get("university", ""),
                "department": profile_data.get("department", ""),
                "email": profile_data.get("email", ""),
                "profile_data": json.dumps(profile_data),
            }
            
            if metadata:
                vector_metadata.update(metadata)
            
            # Validate embedding dimension
            if len(embedding) != self.dimension:
                error_msg = f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                print(f"[VectorDB] ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Upsert vector
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                }]
            )
            
            print(f"[VectorDB] ✅ Upserted profile vector: {profile_url[:60]}... (ID: {vector_id})")
            return True
            
        except Exception as e:
            import traceback
            print(f"[VectorDB] ❌ ERROR upserting profile: {str(e)}")
            print(f"[VectorDB] Full traceback:")
            traceback.print_exc()
            return False
    
    def upsert_professor(
        self,
        professor_name: str,
        combined_content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store professor content as vectors (with chunking if content is too large)
        Each chunk is stored as a separate vector with professor_name as identifier
        
        Args:
            professor_name: Professor's name (used as unique identifier)
            combined_content: All content combined (profile + documents + webpages)
            embedding: Vector embedding for the combined content (used for first chunk)
            metadata: Additional metadata (profile_url, university, department, etc.)
        
        Returns:
            True if successful
        """
        try:
            from api.services.embeddings import get_embeddings_service
            
            # Generate normalized name for IDs
            import re
            normalized_name = re.sub(r'[^\w\s-]', '', professor_name.lower().strip())
            normalized_name = re.sub(r'\s+', '_', normalized_name)
            
            # Pinecone metadata limit is 40KB per record
            # Reserve ~5KB for other metadata fields, so we can store ~35KB of content per chunk
            MAX_CONTENT_PER_CHUNK = 35000  # ~35KB per chunk
            
            # Check if we need to chunk
            if len(combined_content) <= MAX_CONTENT_PER_CHUNK:
                # Content fits in one record - store as single vector
                vector_id = f"professor_{normalized_name}"
                
                vector_metadata = {
                    "professor_name": professor_name,
                    "content_type": "professor_aggregated",
                    "content": combined_content,  # Full content
                    "content_preview": combined_content[:500],  # Quick preview
                    "content_length": len(combined_content),
                    "chunk_index": 0,  # Single chunk
                    "total_chunks": 1,
                }
                
                if metadata:
                    vector_metadata.update(metadata)
                
                # Validate embedding
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}")
                if all(v == 0.0 for v in embedding):
                    raise ValueError(f"Embedding is all zeros for {professor_name}")
                
                self.index.upsert(vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                }])
                
                print(f"[VectorDB] ✅ Stored professor in single vector: {professor_name} ({len(combined_content)} chars)")
                return True
            else:
                # Content is too large - chunk it into multiple records
                print(f"[VectorDB] Content too large ({len(combined_content)} chars), chunking into multiple records...")
                
                # Chunk the content (no overlap needed - we want clean separation)
                chunks = self._chunk_text(combined_content, MAX_CONTENT_PER_CHUNK, overlap=0)
                total_chunks = len(chunks)
                
                print(f"[VectorDB] Split into {total_chunks} chunks for {professor_name}")
                
                # Get embeddings service for generating chunk embeddings
                embeddings_service = get_embeddings_service()
                
                vectors_to_upsert = []
                for idx, chunk in enumerate(chunks):
                    # Generate unique ID for each chunk
                    vector_id = f"professor_{normalized_name}_chunk_{idx}"
                    
                    # Generate embedding for this chunk
                    try:
                        chunk_embedding = embeddings_service.embed_text(chunk, max_tokens=8000)
                        print(f"[VectorDB] Generated embedding for chunk {idx+1}/{total_chunks}")
                    except Exception as emb_error:
                        print(f"[VectorDB] ⚠️ Error generating embedding for chunk {idx+1}, using main embedding: {str(emb_error)}")
                        chunk_embedding = embedding  # Fallback to main embedding
                    
                    # Validate chunk embedding
                    if len(chunk_embedding) != self.dimension:
                        print(f"[VectorDB] ⚠️ Chunk embedding dimension mismatch, using main embedding")
                        chunk_embedding = embedding
                    if all(v == 0.0 for v in chunk_embedding):
                        print(f"[VectorDB] ⚠️ Chunk embedding is all zeros, using main embedding")
                        chunk_embedding = embedding
                    
                    # Prepare metadata for this chunk
                    chunk_metadata = {
                        "professor_name": professor_name,
                        "content_type": "professor_aggregated",
                        "content": chunk,  # Full chunk content
                        "content_preview": chunk[:500] if idx == 0 else chunk[:200],  # Preview
                        "content_length": len(combined_content),  # Original total length
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                    }
                    
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": chunk_embedding,
                        "metadata": chunk_metadata
                    })
                
                # Delete old chunks for this professor first (if updating)
                # Query for existing chunks
                try:
                    import random
                    existing_filter = {"professor_name": professor_name, "content_type": "professor_aggregated"}
                    random_vector = [random.uniform(-0.01, 0.01) for _ in range(self.dimension)]
                    existing_query = self.index.query(
                        vector=random_vector,
                        top_k=1000,
                        include_metadata=True,
                        filter=existing_filter
                    )
                    existing_ids = [match.id for match in existing_query.matches if match.id.startswith(f"professor_{normalized_name}")]
                    if existing_ids:
                        print(f"[VectorDB] Deleting {len(existing_ids)} existing chunks for {professor_name}...")
                        self.index.delete(ids=existing_ids)
                except Exception as del_error:
                    print(f"[VectorDB] ⚠️ Could not delete existing chunks (may not exist): {str(del_error)}")
                
                # Upsert all chunks in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                print(f"[VectorDB] ✅ Stored professor in {total_chunks} chunks: {professor_name} ({len(combined_content)} chars total)")
                return True
            
        except Exception as e:
            import traceback
            print(f"[VectorDB] ❌ ERROR upserting professor: {str(e)}")
            print(f"[VectorDB] Full traceback:")
            traceback.print_exc()
            return False
    
    def upsert_document(
        self,
        profile_url: str,
        document_url: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> int:
        """
        Store document content as vectors (with chunking)
        
        Args:
            profile_url: Associated profile URL
            document_url: Document URL
            content: Document text content
            embedding: Vector embedding for full document (or first chunk)
            metadata: Additional metadata
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            Number of chunks stored
        """
        try:
            # Split content into chunks
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            
            vectors_to_upsert = []
            for idx, chunk in enumerate(chunks):
                vector_id = self._generate_id(document_url, "document", idx)
                
                chunk_metadata = {
                    "profile_url": profile_url,
                    "document_url": document_url,
                    "content_type": "document",
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk[:500],  # Store preview
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                # For now, use the provided embedding (in production, generate per chunk)
                # TODO: Generate embeddings for each chunk separately
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,  # Should be chunk-specific embedding
                    "metadata": chunk_metadata
                })
            
            # Validate embedding dimension
            if len(embedding) != self.dimension:
                error_msg = f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                print(f"[VectorDB] ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"[VectorDB] ✅ Upserted {len(chunks)} document chunks: {document_url[:60]}...")
            return len(chunks)
            
        except Exception as e:
            import traceback
            print(f"[VectorDB] ❌ ERROR upserting document: {str(e)}")
            print(f"[VectorDB] Full traceback:")
            traceback.print_exc()
            return 0
    
    def upsert_webpage(
        self,
        profile_url: str,
        webpage_url: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store webpage content as vector
        
        Args:
            profile_url: Associated profile URL
            webpage_url: Webpage URL
            content: Webpage text content
            embedding: Vector embedding
            metadata: Additional metadata
        
        Returns:
            True if successful
        """
        try:
            vector_id = self._generate_id(webpage_url, "webpage")
            
            vector_metadata = {
                "profile_url": profile_url,
                "webpage_url": webpage_url,
                "content_type": "webpage",
                "content_preview": content[:500],
            }
            
            if metadata:
                vector_metadata.update(metadata)
            
            # Validate embedding dimension
            if len(embedding) != self.dimension:
                error_msg = f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                print(f"[VectorDB] ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                }]
            )
            
            print(f"[VectorDB] ✅ Upserted webpage vector: {webpage_url[:60]}... (ID: {vector_id})")
            return True
            
        except Exception as e:
            import traceback
            print(f"[VectorDB] ❌ ERROR upserting webpage: {str(e)}")
            print(f"[VectorDB] Full traceback:")
            traceback.print_exc()
            return False
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"university": "UT Dallas"})
            include_metadata: Whether to include metadata in results
        
        Returns:
            List of search results with scores and metadata
        """
        try:
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            results = []
            for match in query_response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else None
                })
            
            return results
            
        except Exception as e:
            print(f"[VectorDB] Error searching: {str(e)}")
            return []
    
    def delete_by_profile(self, profile_url: str) -> bool:
        """
        Delete all vectors associated with a profile
        
        Args:
            profile_url: Profile URL
        
        Returns:
            True if successful
        """
        try:
            # Search for all vectors with this profile_url
            # Note: Pinecone doesn't support delete by metadata directly
            # You'd need to query first, then delete by IDs
            # This is a simplified version - in production, maintain an index of IDs
            
            # For now, we'll need to query and delete
            # This is inefficient for large datasets - consider maintaining a separate index
            print(f"[VectorDB] Delete by profile not fully implemented. Consider maintaining ID index.")
            return True
            
        except Exception as e:
            print(f"[VectorDB] Error deleting profile vectors: {str(e)}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """
        Split text into chunks, trying to break at sentence boundaries
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks (default 0 for professor content)
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at sentence boundary if not at the end
            if end < len(text):
                # Look for sentence endings
                last_period = chunk.rfind('. ')
                last_exclamation = chunk.rfind('! ')
                last_question = chunk.rfind('? ')
                last_newline = chunk.rfind('\n\n')  # Paragraph break
                
                # Find the best break point
                break_points = [p for p in [last_period, last_exclamation, last_question, last_newline] if p > 0]
                if break_points:
                    break_point = max(break_points)
                    # Only use if we're past 70% of chunk size (to avoid tiny chunks)
                    if break_point > chunk_size * 0.7:
                        chunk = chunk[:break_point + 1].strip()
                        end = start + len(chunk)
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            start = end - overlap if overlap > 0 else end
        
        return chunks
    
    def _get_professor_chunks(self, professor_name: str) -> List[Dict]:
        """
        Retrieve all chunks for a professor and combine them
        
        Args:
            professor_name: Professor's name
        
        Returns:
            List of chunk records sorted by chunk_index
        """
        try:
            import random
            # Use random vector to query
            random_vector = [random.uniform(-0.01, 0.01) for _ in range(self.dimension)]
            
            # Filter by professor name
            filter_dict = {
                "professor_name": professor_name,
                "content_type": "professor_aggregated"
            }
            
            # Query for all chunks (up to 100 chunks per professor should be enough)
            query_response = self.index.query(
                vector=random_vector,
                top_k=100,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Collect and sort chunks
            chunks = []
            for match in query_response.matches:
                metadata = match.metadata or {}
                chunk_index = metadata.get('chunk_index', 0)
                chunks.append({
                    'chunk_index': chunk_index,
                    'content': metadata.get('content', ''),
                    'metadata': metadata,
                    'id': match.id
                })
            
            # Sort by chunk_index
            chunks.sort(key=lambda x: x['chunk_index'])
            
            return chunks
            
        except Exception as e:
            print(f"[VectorDB] Error retrieving chunks for {professor_name}: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get index statistics
        
        Returns:
            Dictionary with index stats
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else None
            }
        except Exception as e:
            print(f"[VectorDB] Error getting stats: {str(e)}")
            return {}
    
    def get_all_professors(self, limit: int = 1000) -> List[Dict]:
        """
        Get all professor records from the index
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of professor records with metadata
        """
        try:
            import random
            # Use a small random vector instead of zero vector (zero vectors are rejected)
            # This is a workaround since Pinecone doesn't have a direct "list all" API
            random_vector = [random.uniform(-0.01, 0.01) for _ in range(self.dimension)]
            
            # Query with filter for professor_aggregated content type
            filter_dict = {'content_type': 'professor_aggregated'}
            
            # Query with high top_k to get as many records as possible
            query_response = self.index.query(
                vector=random_vector,
                top_k=min(limit, 10000),  # Pinecone max is usually 10000
                include_metadata=True,
                filter=filter_dict
            )
            
            # Group matches by professor name to aggregate chunks
            professor_dict = {}
            for match in query_response.matches:
                metadata = match.metadata or {}
                if metadata.get('content_type') == 'professor_aggregated':
                    prof_name = metadata.get('professor_name', '')
                    if prof_name not in professor_dict:
                        professor_dict[prof_name] = []
                    professor_dict[prof_name].append({
                        'match': match,
                        'metadata': metadata,
                        'chunk_index': metadata.get('chunk_index', 0)
                    })
            
            # Aggregate chunks for each professor
            results = []
            for prof_name, chunks_list in professor_dict.items():
                # Sort chunks by chunk_index
                chunks_list.sort(key=lambda x: x['chunk_index'])
                
                # Combine all chunk contents
                full_content_parts = []
                content_length = 0
                first_chunk_metadata = chunks_list[0]['metadata']
                
                for chunk_data in chunks_list:
                    chunk_content = chunk_data['metadata'].get('content', '')
                    if chunk_content:
                        full_content_parts.append(chunk_content)
                    # Get original content length from first chunk
                    if content_length == 0:
                        content_length = chunk_data['metadata'].get('content_length', len(chunk_content))
                
                # Combine all chunks
                full_content = ' '.join(full_content_parts)
                total_chunks = len(chunks_list)
                
                # Use first chunk's metadata for other fields
                results.append({
                    "id": chunks_list[0]['match'].id,  # Use first chunk's ID
                    "professor_name": prof_name,
                    "university": first_chunk_metadata.get('university', ''),
                    "department": first_chunk_metadata.get('department', ''),
                    "email": first_chunk_metadata.get('email', ''),
                    "position": first_chunk_metadata.get('position', ''),
                    "profile_url": first_chunk_metadata.get('profile_url', ''),
                    "content": full_content,  # Combined content from all chunks
                    "content_preview": first_chunk_metadata.get('content_preview', ''),
                    "content_length": content_length or len(full_content),
                    "content_truncated": False,  # No truncation when chunked
                    "total_chunks": total_chunks,
                    "score": chunks_list[0]['match'].score,
                    "metadata": first_chunk_metadata
                })
            
            print(f"[VectorDB] Retrieved {len(results)} professor records (aggregated from chunks)")
            return results
            
        except Exception as e:
            print(f"[VectorDB] Error getting all professors: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_professors(
        self,
        query_text: str = "",
        university: str = "",
        department: str = "",
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for professors by text query or filters
        
        Args:
            query_text: Text query to search for (will generate embedding)
            university: Filter by university
            department: Filter by department
            limit: Maximum number of results
        
        Returns:
            List of matching professor records
        """
        try:
            from api.services.embeddings import get_embeddings_service
            
            # Build filter
            filter_dict = {}
            if university:
                filter_dict['university'] = university
            if department:
                filter_dict['department'] = department
            filter_dict['content_type'] = 'professor_aggregated'
            
            if query_text:
                # Generate embedding for text query
                embeddings_service = get_embeddings_service()
                query_embedding = embeddings_service.embed_text(query_text)
                
                # Search with embedding
                query_response = self.index.query(
                    vector=query_embedding,
                    top_k=limit,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
            else:
                # No text query, use small random vector with filters
                import random
                random_vector = [random.uniform(-0.01, 0.01) for _ in range(self.dimension)]
                query_response = self.index.query(
                    vector=random_vector,
                    top_k=limit,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
            
            # Group matches by professor name to aggregate chunks
            professor_dict = {}
            for match in query_response.matches:
                metadata = match.metadata or {}
                if metadata.get('content_type') == 'professor_aggregated':
                    prof_name = metadata.get('professor_name', '')
                    if prof_name not in professor_dict:
                        professor_dict[prof_name] = []
                    professor_dict[prof_name].append({
                        'match': match,
                        'metadata': metadata,
                        'chunk_index': metadata.get('chunk_index', 0)
                    })
            
            # Aggregate chunks for each professor
            results = []
            for prof_name, chunks_list in professor_dict.items():
                # Sort chunks by chunk_index
                chunks_list.sort(key=lambda x: x['chunk_index'])
                
                # Combine all chunk contents
                full_content_parts = []
                content_length = 0
                first_chunk_metadata = chunks_list[0]['metadata']
                
                for chunk_data in chunks_list:
                    chunk_content = chunk_data['metadata'].get('content', '')
                    if chunk_content:
                        full_content_parts.append(chunk_content)
                    # Get original content length from first chunk
                    if content_length == 0:
                        content_length = chunk_data['metadata'].get('content_length', len(chunk_content))
                
                # Combine all chunks
                full_content = ' '.join(full_content_parts)
                total_chunks = len(chunks_list)
                
                # Use first chunk's metadata for other fields
                results.append({
                    "id": chunks_list[0]['match'].id,  # Use first chunk's ID
                    "professor_name": prof_name,
                    "university": first_chunk_metadata.get('university', ''),
                    "department": first_chunk_metadata.get('department', ''),
                    "email": first_chunk_metadata.get('email', ''),
                    "position": first_chunk_metadata.get('position', ''),
                    "profile_url": first_chunk_metadata.get('profile_url', ''),
                    "content": full_content,  # Combined content from all chunks
                    "content_preview": first_chunk_metadata.get('content_preview', ''),
                    "content_length": content_length or len(full_content),
                    "content_truncated": False,  # No truncation when chunked
                    "total_chunks": total_chunks,
                    "score": chunks_list[0]['match'].score,
                    "metadata": first_chunk_metadata
                })
            
            print(f"[VectorDB] Search returned {len(results)} professor records (aggregated from chunks)")
            return results
            
        except Exception as e:
            print(f"[VectorDB] Error searching professors: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def delete_all(self, namespace: str = "") -> bool:
        """
        Delete all vectors from the index
        
        Args:
            namespace: Namespace to delete from (empty string for default namespace)
        
        Returns:
            True if successful
        """
        try:
            print(f"[VectorDB] Deleting all vectors from index '{self.index_name}' (namespace: '{namespace or 'default'}')...")
            
            # Get stats before deletion
            stats_before = self.get_stats()
            total_before = stats_before.get('total_vectors', 0)
            
            if total_before == 0:
                print(f"[VectorDB] Index is already empty. Nothing to delete.")
                return True
            
            # Delete all vectors
            self.index.delete(delete_all=True, namespace=namespace)
            
            print(f"[VectorDB] ✅ Successfully deleted {total_before} vector(s) from index '{self.index_name}'")
            return True
            
        except Exception as e:
            print(f"[VectorDB] ❌ Error deleting all vectors: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# Singleton instance
_vector_db_instance = None

def get_vector_db(api_key: Optional[str] = None, index_name: str = None, dimension: int = None) -> VectorDBService:
    """Get or create VectorDBService instance"""
    global _vector_db_instance
    
    # Try to get from config if not provided
    if index_name is None or dimension is None:
        try:
            from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
            index_name = index_name or INDEX_NAME
            dimension = dimension or INDEX_DIMENSION
        except ImportError:
            index_name = index_name or "faculty-profiles"
            dimension = dimension or 384
    
    # Create new instance if index_name or dimension changed
    if _vector_db_instance is None or _vector_db_instance.index_name != index_name or _vector_db_instance.dimension != dimension:
        _vector_db_instance = VectorDBService(api_key=api_key, index_name=index_name, dimension=dimension)
    
    return _vector_db_instance

