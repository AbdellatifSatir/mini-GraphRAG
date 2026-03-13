1. Testing "Local" Retrieval (Neo4j + Vector Search)
  These questions test the system's ability to map fuzzy user terms to exact graph nodes using FAISS and then traverse the graph using Neo4j Cypher queries.     

   * "Who founded the technology firm based in Berlin?"
       * Goal: Tests if the system can map "technology firm in Berlin" to Quantum Dynamics and then retrieve the founded_by relationship to Sarah Chen.
   * "What are the research areas of the institution that Quantum Dynamics collaborates with?"
       * Goal: Tests 2-hop traversal. It must go from Quantum Dynamics $\rightarrow$ University of Berlin $\rightarrow$ its attributes/relationships.
   * "When was Sarah Chen's company established?"
       * Goal: Tests mapping a person (Sarah Chen) to their organization and retrieving a specific attribute (date).


  2. Testing "Global" Retrieval (Hierarchical Summaries)
  These test the Routing mechanism and the use of community_summaries.json.

   * "What is the primary theme of this text?"
       * Goal: Tests if the routing chooses GLOBAL and if it pulls the Level 1 summary (the bird's eye view).
   * "Can you summarize all the entities mentioned and their core roles?"
       * Goal: Tests if the system provides the Level 0 "Detailed Themes" from each community.


  3. Testing Semantic "Fuzzy" Resolution
   * "Tell me about the Berlin-based Quantum center."
       * Goal: Tests if FAISS can resolve "Quantum center" to Quantum Dynamics even though the words don't match exactly.
