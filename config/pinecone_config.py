"""
Pinecone Configuration
Store your Pinecone API key and index settings here
"""
import os

# Pinecone API Key
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    "pcsk_6ubN2X_CDEH5YcJ6YCcJKT4uHatg8WrBDdYCRNXdQ5kbVQmLTycRrCATikXHUL3AnTZegA"
)

# Index Configuration
INDEX_NAME = "ngo-profiles" 
# Dimension options:
# - 384: all-MiniLM-L6-v2 (sentence-transformers) - FREE
# - 1536: text-embedding-3-small (OpenAI) - RECOMMENDED
# - 3072: text-embedding-3-large (OpenAI) - BEST QUALITY
INDEX_DIMENSION = 1536  # Matches your Pinecone index "ngo-profiles" (text-embedding-3-small)
INDEX_METRIC = "cosine"  # ALWAYS use cosine for text embeddings

# Serverless Configuration
SERVERLESS_CLOUD = "aws"  # Options: aws, gcp, azure
SERVERLESS_REGION = "us-east-1"  # Change to your preferred region

# Embedding Model Configuration
# Options: "sentence-transformers" (free), "ollama" (free), "openai" (paid)
# Recommended: "openai" with text-embedding-3-small for best balance
# NOTE: If using OpenAI, you MUST set OPENAI_API_KEY environment variable
EMBEDDING_MODEL = "openai"  # Using OpenAI embeddings (requires OPENAI_API_KEY in .env)
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # Matches your Pinecone index

# Document Chunking Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

