import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def test_connection():
    if not all([URI, USER, PASSWORD]):
        print("❌ Error: Missing Neo4j credentials in .env file.")
        return

    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful!' AS message")
            message = result.single()["message"]
            print(f"✅ {message}")
        driver.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
