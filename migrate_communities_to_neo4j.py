import json
import os
from neo4j import GraphDatabase
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
    COMMUNITY_SUMMARIES_FILE
)

def migrate_communities():
    if not os.path.exists(COMMUNITY_SUMMARIES_FILE):
        print(f"❌ Error: {COMMUNITY_SUMMARIES_FILE} not found.")
        return

    with open(COMMUNITY_SUMMARIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("--- Migrating Level 0 Communities ---")
        for comm_id, info in data.get("level_0", {}).items():
            summary = info.get("summary", "")
            nodes = info.get("nodes", [])
            
            # 1. Create Level 0 Community Node
            session.run(
                "MERGE (c:Community {id: $id, level: 0}) SET c.summary = $summary",
                id=comm_id, summary=summary
            )
            
            # 2. Link Entities to Community
            for node_id in nodes:
                session.run(
                    """
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (c:Community {id: $comm_id, level: 0})
                    MERGE (e)-[:BELONGS_TO]->(c)
                    """,
                    entity_id=node_id, comm_id=comm_id
                )
            print(f"✅ Community {comm_id} (Level 0) migrated with {len(nodes)} entities.")

        print("\n--- Migrating Level 1 Communities ---")
        for meta_id, info in data.get("level_1", {}).items():
            summary = info.get("summary", "")
            children = info.get("children", [])
            
            # 1. Create Level 1 Community Node (prefixed id to avoid conflict)
            meta_node_id = f"meta-{meta_id}"
            session.run(
                "MERGE (c:Community {id: $id, level: 1}) SET c.summary = $summary",
                id=meta_node_id, summary=summary
            )
            
            # 2. Link Level 0 Communities to Level 1 parent
            for child_id in children:
                session.run(
                    """
                    MATCH (child:Community {id: $child_id, level: 0})
                    MATCH (parent:Community {id: $parent_id, level: 1})
                    MERGE (child)-[:CHILD_OF]->(parent)
                    """,
                    child_id=child_id, parent_id=meta_node_id
                )
            print(f"✅ Meta-Community {meta_node_id} (Level 1) migrated with {len(children)} children.")

    driver.close()
    print("\n🎉 Migration to Neo4j complete!")

if __name__ == "__main__":
    migrate_communities()
