import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
    exit(1)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def cleanup_database():
    """
    Clears all rows from articles_master.
    Due to ON DELETE CASCADE, this will also clear:
    - article_embeddings
    - daily_digest
    """
    print("Starting database cleanup...")
    
    try:
        # We use a filter like .neq("id", 0) because Supabase requires a filter for deletes
        # articles_master uses bigserial primary key, so id 0 is safe/unlikely
        response = supabase.table("articles_master").delete().neq("id", 0).execute()
        
        # Check if response has data or error (depending on library version)
        # The supabase-py library usually returns data in .data
        print(f"Successfully cleared articles_master (and cascaded tables).")
        
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

if __name__ == "__main__":
    cleanup_database()
