"""
Knowledge Graph Builder

Constructs a knowledge graph from entities and relationships,
then applies community detection as described in Section 3.2.2 of the SEMRAG paper.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple
import numpy as np
from collections import defaultdict


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph with:
    - Nodes = entities
    - Edges = relationships between entities
    
    Also performs community detection (Leiden/Louvain).
    """
    
    def __init__(self):
        """Initialize knowledge graph builder."""
        self.graph = nx.Graph()
        self.entity_to_chunks = {}  # Entity -> list of chunk IDs
        self.chunk_to_entities = defaultdict(list)  # Chunk ID -> list of entities
        self.relationships = []
    
    def add_entity(self, entity_text: str, entity_data: Dict):
        """
        Add an entity node to the graph.
        
        Args:
            entity_text: Entity name
            entity_data: Entity metadata
        """
        if not self.graph.has_node(entity_text):
            self.graph.add_node(
                entity_text,
                label=entity_data.get("label", "ENTITY"),
                type="entity"
            )
    
    def add_relationship(self, subject: str, relation: str, obj: str, metadata: Dict = None):
        """
        Add a relationship edge to the graph.
        
        Args:
            subject: Subject entity
            relation: Relationship type
            obj: Object entity
            metadata: Additional edge metadata
        """
        # Ensure both entities exist
        if not self.graph.has_node(subject):
            self.add_entity(subject, {"label": "ENTITY"})
        if not self.graph.has_node(obj):
            self.add_entity(obj, {"label": "ENTITY"})
        
        # Add edge
        if not self.graph.has_edge(subject, obj):
            self.graph.add_edge(
                subject,
                obj,
                relation=relation,
                metadata=metadata or {}
            )
        
        self.relationships.append({
            "subject": subject,
            "relation": relation,
            "object": obj
        })
    
    def link_entity_to_chunk(self, entity_text: str, chunk_id: int):
        """
        Link an entity to a chunk.
        
        Args:
            entity_text: Entity name
            chunk_id: Chunk identifier
        """
        if entity_text not in self.entity_to_chunks:
            self.entity_to_chunks[entity_text] = []
        
        if chunk_id not in self.entity_to_chunks[entity_text]:
            self.entity_to_chunks[entity_text].append(chunk_id)
        
        self.chunk_to_entities[chunk_id].append(entity_text)
    
    def build_from_extractions(
        self,
        extraction_results: List[Dict],
        chunks: List[Dict]
    ):
        """
        Build knowledge graph from entity extraction results.
        
        Args:
            extraction_results: List of extraction results from EntityExtractor
            chunks: Original chunks
        """
        print("\nBuilding knowledge graph...")
        
        # Add entities and link to chunks
        for result in extraction_results:
            chunk_id = result["chunk_id"]
            
            for entity in result["entities"]:
                entity_text = entity["text"]
                self.add_entity(entity_text, entity)
                self.link_entity_to_chunk(entity_text, chunk_id)
        
        # Add relationships
        for result in extraction_results:
            for rel in result["relationships"]:
                self.add_relationship(
                    rel["subject"],
                    rel["relation"],
                    rel["object"],
                    {"sentence": rel.get("sentence", "")}
                )
        
        print(f"  Nodes (entities): {self.graph.number_of_nodes()}")
        print(f"  Edges (relationships): {self.graph.number_of_edges()}")
        print(f"  Connected components: {nx.number_connected_components(self.graph)}")
    
    def detect_communities(self, algorithm: str = "louvain", resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in the knowledge graph.
        
        Args:
            algorithm: Community detection algorithm ('louvain' or 'leiden')
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary mapping entity to community ID
        """
        print(f"\nDetecting communities using {algorithm} algorithm...")
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        if algorithm == "louvain":
            communities = self._louvain_communities(resolution)
        elif algorithm == "leiden":
            communities = self._leiden_communities(resolution)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"  Found {len(set(communities.values()))} communities")
        
        return communities
    
    def _louvain_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Apply Louvain community detection.
        
        Args:
            resolution: Resolution parameter
            
        Returns:
            Dictionary mapping node to community ID
        """
        try:
            import community as community_louvain
            
            # Louvain algorithm
            partition = community_louvain.best_partition(
                self.graph,
                resolution=resolution
            )
            return partition
        
        except ImportError:
            print("  Warning: python-louvain not available, using simple connected components")
            return self._connected_components()
    
    def _leiden_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Apply Leiden community detection.
        
        Args:
            resolution: Resolution parameter
            
        Returns:
            Dictionary mapping node to community ID
        """
        try:
            import igraph as ig
            import leidenalg
            
            # Convert networkx graph to igraph
            edges = list(self.graph.edges())
            nodes = list(self.graph.nodes())
            
            g = ig.Graph()
            g.add_vertices(len(nodes))
            
            # Map node names to indices
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
            g.add_edges(edge_list)
            
            # Run Leiden
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution
            )
            
            # Map back to node names
            communities = {nodes[i]: partition.membership[i] for i in range(len(nodes))}
            return communities
        
        except ImportError:
            print("  Warning: leidenalg not available, using Louvain instead")
            return self._louvain_communities(resolution)
    
    def _connected_components(self) -> Dict[str, int]:
        """
        Fallback: Use connected components as communities.
        
        Returns:
            Dictionary mapping node to community ID
        """
        communities = {}
        for i, component in enumerate(nx.connected_components(self.graph)):
            for node in component:
                communities[node] = i
        return communities
    
    def get_community_members(self, communities: Dict[str, int]) -> Dict[int, List[str]]:
        """
        Group entities by community.
        
        Args:
            communities: Dictionary mapping entity to community ID
            
        Returns:
            Dictionary mapping community ID to list of entities
        """
        community_members = defaultdict(list)
        for entity, comm_id in communities.items():
            community_members[comm_id].append(entity)
        return dict(community_members)
    
    def get_community_subgraph(self, community_id: int, communities: Dict[str, int]) -> nx.Graph:
        """
        Extract subgraph for a specific community.
        
        Args:
            community_id: Community identifier
            communities: Dictionary mapping entity to community ID
            
        Returns:
            Subgraph containing only entities in the community
        """
        # Get entities in this community
        entities = [e for e, c in communities.items() if c == community_id]
        
        # Extract subgraph
        subgraph = self.graph.subgraph(entities).copy()
        return subgraph
    
    def get_statistics(self) -> Dict:
        """
        Get graph statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_connected_components": nx.number_connected_components(self.graph),
            "density": nx.density(self.graph),
            "num_entities_with_chunks": len(self.entity_to_chunks),
        }
        
        if self.graph.number_of_nodes() > 0:
            stats["avg_degree"] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        
        return stats


def demo():
    """Demo knowledge graph construction."""
    # Sample extraction results
    extraction_results = [
        {
            "chunk_id": 0,
            "entities": [
                {"text": "Dr. B.R. Ambedkar", "label": "PERSON"},
                {"text": "Mhow", "label": "GPE"},
                {"text": "India", "label": "GPE"}
            ],
            "relationships": [
                {"subject": "Dr. B.R. Ambedkar", "relation": "born", "object": "Mhow"}
            ]
        },
        {
            "chunk_id": 1,
            "entities": [
                {"text": "Ambedkar", "label": "PERSON"},
                {"text": "Indian Constitution", "label": "WORK_OF_ART"}
            ],
            "relationships": [
                {"subject": "Ambedkar", "relation": "draft", "object": "Indian Constitution"}
            ]
        }
    ]
    
    chunks = [{"chunk_id": 0, "text": "..."}, {"chunk_id": 1, "text": "..."}]
    
    builder = KnowledgeGraphBuilder()
    builder.build_from_extractions(extraction_results, chunks)
    
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH:")
    print("="*60)
    stats = builder.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    communities = builder.detect_communities()
    community_members = builder.get_community_members(communities)
    print(f"\nCommunities: {len(community_members)}")
    for comm_id, members in list(community_members.items())[:3]:
        print(f"  Community {comm_id}: {members[:5]}")


if __name__ == "__main__":
    demo()
