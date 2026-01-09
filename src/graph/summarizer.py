"""
Community Summarizer

Generates summaries for knowledge graph communities using an LLM.
Implements Equation 3 from the SEMRAG paper.
"""

import networkx as nx
from typing import List, Dict, Tuple
import json


class CommunitySummarizer:
    """
    Generates community summaries using LLM.
    
    Implements Equation 3:
    S(C_i) = LLM_summarize(⋃_{v∈V_i} s(v) ∪ ⋃_{(v_j,v_k)∈E_i} s(v_j, v_k))
    
    Where:
    - C_i is a community
    - V_i is the set of nodes (entities) in the community
    - E_i is the set of edges (relationships) in the community
    - s(v) is the summary of a node
    - s(v_j, v_k) is the summary of an edge
    """
    
    def __init__(self, llm_client):
        """
        Initialize community summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm = llm_client
    
    def summarize_community(
        self,
        community_id: int,
        subgraph: nx.Graph,
        entity_to_chunks: Dict[str, List[int]],
        chunks: List[Dict]
    ) -> Dict:
        """
        Generate a summary for a community.
        
        Args:
            community_id: Community identifier
            subgraph: Subgraph for this community
            entity_to_chunks: Mapping of entities to chunks
            chunks: Original chunk data
            
        Returns:
            Community summary dictionary
        """
        # Collect all nodes (entities) in community
        nodes = list(subgraph.nodes())
        
        # Collect all edges (relationships) in community
        edges = list(subgraph.edges(data=True))
        
        # Build context for LLM
        context = self._build_community_context(nodes, edges, entity_to_chunks, chunks)
        
        # Generate summary using LLM
        summary = self._generate_summary(context, community_id)
        
        return {
            "community_id": community_id,
            "num_entities": len(nodes),
            "num_relationships": len(edges),
            "entities": nodes,
            "summary": summary,
            "context": context
        }
    
    def _build_community_context(
        self,
        nodes: List[str],
        edges: List[Tuple],
        entity_to_chunks: Dict[str, List[int]],
        chunks: List[Dict]
    ) -> str:
        """
        Build context string from community graph structure.
        
        Args:
            nodes: List of entity names
            edges: List of edges with data
            entity_to_chunks: Entity to chunk mapping
            chunks: Original chunks
            
        Returns:
            Context string for LLM
        """
        context_parts = []
        
        # Add entities section
        context_parts.append("ENTITIES IN THIS COMMUNITY:")
        for node in nodes[:20]:  # Limit to prevent context overflow
            context_parts.append(f"- {node}")
        
        if len(nodes) > 20:
            context_parts.append(f"... and {len(nodes) - 20} more entities")
        
        # Add relationships section
        context_parts.append("\nRELATIONSHIPS:")
        for u, v, data in edges[:20]:  # Limit relationships
            relation = data.get("relation", "related_to")
            context_parts.append(f"- {u} --[{relation}]--> {v}")
        
        if len(edges) > 20:
            context_parts.append(f"... and {len(edges) - 20} more relationships")
        
        # Add sample text context from chunks
        context_parts.append("\nRELEVANT TEXT EXCERPTS:")
        chunk_ids = set()
        for node in nodes[:10]:  # Sample from first 10 entities
            if node in entity_to_chunks:
                chunk_ids.update(entity_to_chunks[node][:2])  # Max 2 chunks per entity
        
        for chunk_id in list(chunk_ids)[:5]:  # Max 5 chunks total
            chunk = chunks[chunk_id]
            text_preview = chunk["text"][:200]  # First 200 chars
            context_parts.append(f"- {text_preview}...")
        
        return "\n".join(context_parts)
    
    def _generate_summary(self, context: str, community_id: int) -> str:
        """
        Generate summary using LLM.
        
        Args:
            context: Context string
            community_id: Community ID
            
        Returns:
            Generated summary
        """
        prompt = f"""You are analyzing a knowledge graph community about Dr. B.R. Ambedkar's works.

Below is information about Community {community_id}, including entities, relationships, and relevant text excerpts.

{context}

Please provide a concise summary (2-3 sentences) describing:
1. The main theme or topic of this community
2. Key entities and their relationships
3. The significance of this community in understanding Ambedkar's works

Summary:"""
        
        try:
            summary = self.llm.generate(prompt, max_tokens=200)
            return summary.strip()
        except Exception as e:
            print(f"  Warning: Failed to generate summary for community {community_id}: {e}")
            # Fallback summary
            return f"Community {community_id} contains entities and relationships related to Ambedkar's works."
    
    def summarize_all_communities(
        self,
        communities: Dict[str, int],
        graph_builder,
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Generate summaries for all communities.
        
        Args:
            communities: Dictionary mapping entity to community ID
            graph_builder: KnowledgeGraphBuilder instance
            chunks: Original chunks
            
        Returns:
            List of community summaries
        """
        print("\nGenerating community summaries...")
        
        # Get unique community IDs
        community_ids = sorted(set(communities.values()))
        summaries = []
        
        for comm_id in community_ids:
            # Get subgraph for this community
            subgraph = graph_builder.get_community_subgraph(comm_id, communities)
            
            # Generate summary
            summary = self.summarize_community(
                comm_id,
                subgraph,
                graph_builder.entity_to_chunks,
                chunks
            )
            summaries.append(summary)
            
            print(f"  Community {comm_id}: {summary['num_entities']} entities, {summary['num_relationships']} relationships")
        
        return summaries


def demo():
    """Demo community summarization."""
    # Mock LLM client
    class MockLLM:
        def generate(self, prompt, max_tokens=100):
            return "This community focuses on Ambedkar's early life and education, including his birthplace and academic achievements."
    
    # Sample data
    import networkx as nx
    
    subgraph = nx.Graph()
    subgraph.add_nodes_from(["Dr. B.R. Ambedkar", "Mhow", "India"])
    subgraph.add_edge("Dr. B.R. Ambedkar", "Mhow", relation="born_in")
    
    entity_to_chunks = {
        "Dr. B.R. Ambedkar": [0],
        "Mhow": [0],
        "India": [0]
    }
    
    chunks = [
        {"chunk_id": 0, "text": "Dr. B.R. Ambedkar was born in Mhow, India in 1891."}
    ]
    
    summarizer = CommunitySummarizer(MockLLM())
    summary = summarizer.summarize_community(0, subgraph, entity_to_chunks, chunks)
    
    print("\n" + "="*60)
    print("COMMUNITY SUMMARY:")
    print("="*60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    demo()
