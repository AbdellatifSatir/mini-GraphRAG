import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def setup_database():
    if not all([URI, USER, PASSWORD]):
        print("❌ Error: Missing Neo4j credentials in .env file.")
        return

    print(f"Connecting to: {URI}...")
    
    # Use neo4j+s for encrypted connections (Aura default)
    # If SSL issues persist, we might try neo4j+ssc (Self-signed certificate)
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        with driver.session() as session:
            print("--- Initializing Neo4j Schema ---")
            
            # 1. Verify Connection
            session.run("RETURN 1")
            print("✅ Connection Verified.")

            # 2. Create Uniqueness Constraint for Entities
            # This ensures we don't have duplicate nodes when we import the KG.
            print("Applying Uniqueness Constraint on Entity(id)...")
            session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            
            # 3. Create Index for Entity types
            print("Applying Index on Entity(type)...")
            session.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            
            print("--- Schema Setup Complete! ---")
            
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print("\n💡 Troubleshooting Tip: Ensure your URI starts with 'neo4j+s://' for Aura.")
    finally:
        driver.close()

if __name__ == "__main__":
    setup_database()
