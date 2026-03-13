from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def verify():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        print("--- Node Counts ---")
        res = session.run("MATCH (c:Community) RETURN c.level as level, count(c) as count")
        for r in res:
            print(f"Level {r['level']}: {r['count']} nodes")
        
        print("\n--- Relationship Counts ---")
        res = session.run("MATCH ()-[r:BELONGS_TO|CHILD_OF]->() RETURN type(r) as type, count(r) as count")
        for r in res:
            print(f"Relationship {r['type']}: {r['count']} instances")

    driver.close()

if __name__ == "__main__":
    verify()
