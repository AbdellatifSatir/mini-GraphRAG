from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def check_all():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        print("--- All Node Labels ---")
        res = session.run("CALL db.labels()")
        for r in res:
            label = r[0]
            count_res = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            print(f"- {label}: {count_res.single()['count']} nodes")
        
        print("\n--- All Relationship Types ---")
        res = session.run("CALL db.relationshipTypes()")
        for r in res:
            rel_type = r[0]
            count_res = session.run(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count")
            print(f"- {rel_type}: {count_res.single()['count']} instances")

    driver.close()

if __name__ == "__main__":
    check_all()
