"""
Helper script to create .env file
Run this to set up your environment variables
"""
import os
from pathlib import Path

def create_env_file():
    """Create .env file with template"""
    env_path = Path(__file__).parent / '.env'
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled. Existing .env file preserved.")
            return
    
    print("\n" + "=" * 60)
    print("Environment Variables Setup")
    print("=" * 60)
    
    # Get OpenAI API key
    print("\n1. OpenAI API Key (required for OpenAI embeddings)")
    print("   Get your key from: https://platform.openai.com/api-keys")
    openai_key = input("   Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    # Get Pinecone API key (optional, already in config)
    print("\n2. Pinecone API Key (optional - already in config/pinecone_config.py)")
    pinecone_key = input("   Enter your Pinecone API key (or press Enter to skip): ").strip()
    
    # Create .env file
    env_content = []
    env_content.append("# OpenAI API Key (required for OpenAI embeddings)")
    env_content.append("# Get your key from: https://platform.openai.com/api-keys")
    if openai_key:
        env_content.append(f"OPENAI_API_KEY={openai_key}")
    else:
        env_content.append("OPENAI_API_KEY=your-openai-api-key-here")
    
    env_content.append("")
    env_content.append("# Pinecone API Key (optional - already in config/pinecone_config.py)")
    if pinecone_key:
        env_content.append(f"PINECONE_API_KEY={pinecone_key}")
    else:
        env_content.append("# PINECONE_API_KEY=your-pinecone-api-key-here")
    
    env_content.append("")
    env_content.append("# Optional: Ollama Configuration (if using Ollama embeddings)")
    env_content.append("# OLLAMA_BASE_URL=http://localhost:11434")
    
    with open(env_path, 'w') as f:
        f.write('\n'.join(env_content))
    
    print(f"\n✅ Created .env file at: {env_path}")
    print("\n📝 Next steps:")
    print("   1. Restart your server for changes to take effect")
    print("   2. Run: python check_openai_key.py to verify")
    print("   3. The .env file is already in .gitignore (won't be committed)")

if __name__ == "__main__":
    create_env_file()

