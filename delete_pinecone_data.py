"""
Script to delete all data from Pinecone vector database
"""
import os
from dotenv import load_dotenv
from api.services.vector_db import get_vector_db
from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION

# Load environment variables
load_dotenv()


def delete_all_data(confirm: bool = True):
    """
    Delete all vectors from Pinecone index
    
    Args:
        confirm: If True, prompt for confirmation. If False, proceed without confirmation.
    """
    print("="*60)
    print("Pinecone Data Deletion Script")
    print("="*60)
    print(f"Index: {INDEX_NAME}")
    print("="*60)
    
    try:
        # Initialize vector DB service
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        
        # Get stats before deletion
        stats = vector_db.index.describe_index_stats()
        total_vectors = stats.total_vector_count
        print(f"\n[Current State] Total vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            print("\n[Result] Index is already empty. Nothing to delete.")
            return True
        
        # Confirm deletion if requested
        if confirm:
            print(f"\n[Warning] This will delete ALL {total_vectors} vectors from the index '{INDEX_NAME}'")
            response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
            
            if response != 'yes':
                print("[Cancelled] Deletion cancelled by user.")
                return False
        else:
            print(f"\n[Warning] Deleting ALL {total_vectors} vectors from the index '{INDEX_NAME}' (non-interactive mode)")
        
        # Delete all vectors
        print("\n[Deleting] Deleting all vectors...")
        success = vector_db.delete_all(namespace="")
        
        if success:
            # Verify deletion
            stats_after = vector_db.index.describe_index_stats()
            total_after = stats_after.total_vector_count
            print(f"\n[Verification] Vectors remaining: {total_after}")
            
            if total_after == 0:
                print("\n[Result] ✅ Successfully deleted all vectors from Pinecone index!")
                return True
            else:
                print(f"\n[Warning] ⚠️ Some vectors may still remain: {total_after}")
                return False
        else:
            print("\n[Result] ❌ Failed to delete vectors")
            return False
            
    except Exception as e:
        print(f"\n[Error] ❌ Error during deletion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    # Check for --yes flag to skip confirmation
    skip_confirm = '--yes' in sys.argv or '-y' in sys.argv
    delete_all_data(confirm=not skip_confirm)

