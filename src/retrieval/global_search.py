"""
Global Graph RAG Search

Implements Equation 5 from the SEMRAG paper:
D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))), score(p_j, Q))
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class GlobalGraphRAG:
    """
    Global Graph RAG Search implementation.
    
    Retrieves information through community-level search:
    1. Find top-K relevant community summaries
    2. Extract chunks from those communities
    3. Score and rank all points/sub-pieces
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_communities: int = 3,
        top_k_points: int = 10
    ):
        """
        Initialize Global Graph RAG.
        
        Args:
            embedding_model: Model for embeddings
            top_k_communities: Number of top communities to retrieve
            top_k_points: Number of final points to return
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.top_k_communities = top_k_communities
        self.top_k_points = top_k_points
        
        # Storage
        self.community_summaries = []  # List of community summary dicts
        self.community_embeddings = {}  # community_id -> embedding
        self.chunks = []  # chunk data
        self.chunk_embeddings = {}  # chunk_id -> embedding
        self.communities = {}  # entity -> community_id
    
    def index_community_summaries(self, summaries: List[Dict]):
        """
        Generate and store embeddings for community summaries.
        
        Args:
            summaries: List of community summary dictionaries
        """
        print("Indexing community summaries...")
        self.community_summaries = summaries
        
        summary_texts = [s["summary"] for s in summaries]
        
        if len(summary_texts) == 0:
            return
        
        embeddings = self.embedding_model.encode(
            summary_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        for summary, embedding in zip(summaries, embeddings):
            comm_id = summary["community_id"]
            self.community_embeddings[comm_id] = embedding
        
        print(f"  Indexed {len(self.community_embeddings)} community summaries")
    
    def index_chunks(self, chunks: List[Dict]):
        """
        Store chunk data and embeddings.
        
        Args:
            chunks: List of chunk dictionaries
        """
        print("Indexing chunks for global search...")
        self.chunks = chunks
        
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        if len(chunk_texts) == 0:
            return
        
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            self.chunk_embeddings[chunk_id] = embedding
        
        print(f"  Indexed {len(self.chunk_embeddings)} chunks")
    
    def set_communities(self, communities: Dict[str, int], entity_to_chunks: Dict[str, List[int]]):
        """
        Set community membership data.
        
        Args:
            communities: Dictionary mapping entity to community ID
            entity_to_chunks: Dictionary mapping entity to chunk IDs
        """
        self.communities = communities
        self.entity_to_chunks = entity_to_chunks
        print(f"Set community data for {len(communities)} entities")
    
    def search(self, query: str) -> List[Dict]:
        """
        Perform Global Graph RAG search.
        
        Implements Equation 5:
        D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))))
        
        Args:
            query: User query
            
        Returns:
            List of retrieved results
        """
        # Step 1: Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Step 2: Find top-K most relevant community reports
        top_communities = self._find_top_communities(query_embedding)
        
        if len(top_communities) == 0:
            print("  Warning: No communities found")
            return []
        
        # Step 3: Get chunks from those communities
        community_chunks = self._get_chunks_from_communities(top_communities)
        
        # Step 4: Score each chunk/point against query
        scored_points = self._score_points(query_embedding, community_chunks)
        
        # Step 5: Return top-K points
        results = scored_points[:self.top_k_points]
        
        return results
    
    def _find_top_communities(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """
        Find top-K most relevant community summaries.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of (community_id, similarity_score) tuples
        """
        community_scores = []
        
        for comm_id, comm_emb in self.community_embeddings.items():
            # Calculate similarity
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                comm_emb.reshape(1, -1)
            )[0][0]
            
            community_scores.append((comm_id, sim))
        
        # Sort by similarity and take top-K
        community_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = community_scores[:self.top_k_communities]
        
        print(f"  Selected top {len(top_k)} communities")
        
        return top_k
    
    def _get_chunks_from_communities(
        self,
        top_communities: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, float]]:
        """
        Get all chunks associated with top communities.
        
        Args:
            top_communities: List of (community_id, score) tuples
            
        Returns:
            List of (chunk_id, community_id, community_score) tuples
        """
        community_chunks = []
        seen_chunks = set()
        
        for comm_id, comm_score in top_communities:
            # Find entities in this community
            community_entities = [
                entity for entity, cid in self.communities.items()
                if cid == comm_id
            ]
            
            # Get chunks linked to these entities
            for entity in community_entities:
                if entity not in self.entity_to_chunks:
                    continue
                
                for chunk_id in self.entity_to_chunks[entity]:
                    if chunk_id not in seen_chunks:
                        community_chunks.append((chunk_id, comm_id, comm_score))
                        seen_chunks.add(chunk_id)
        
        print(f"  Retrieved {len(community_chunks)} chunks from communities")
        
        return community_chunks
    
    def _score_points(
        self,
        query_embedding: np.ndarray,
        community_chunks: List[Tuple[int, int, float]]
    ) -> List[Dict]:
        """
        Score each chunk/point against the query.
        
        Args:
            query_embedding: Query embedding vector
            community_chunks: List of (chunk_id, community_id, community_score) tuples
            
        Returns:
            Sorted list of result dictionaries
        """
        scored_points = []
        
        for chunk_id, comm_id, comm_score in community_chunks:
            # Get chunk embedding
            if chunk_id not in self.chunk_embeddings:
                continue
            
            chunk_emb = self.chunk_embeddings[chunk_id]
            
            # Calculate chunk-query similarity
            chunk_score = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_emb.reshape(1, -1)
            )[0][0]
            
            # Combine community score and chunk score
            # Weighted average: 0.3 * community + 0.7 * chunk
            final_score = 0.3 * comm_score + 0.7 * chunk_score
            
            result = {
                "chunk_id": chunk_id,
                "chunk_text": self.chunks[chunk_id]["text"],
                "score": float(final_score),
                "chunk_score": float(chunk_score),
                "community_id": comm_id,
                "community_score": float(comm_score),
                "source": "global_rag"
            }
            scored_points.append(result)
        
        # Sort by final score
        scored_points.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_points


def demo():
    """Demo global graph RAG search."""
    # Sample data
    chunks = [
        {"chunk_id": 0, "text": "Dr. B.R. Ambedkar was a social reformer and jurist."},
        {"chunk_id": 1, "text": "Ambedkar drafted the Indian Constitution."},
        {"chunk_id": 2, "text": "He fought against caste discrimination in India."}
    ]
    
    summaries = [
        {
            "community_id": 0,
            "summary": "This community focuses on Ambedkar's constitutional work and legal contributions.",
            "entities": ["Ambedkar", "Indian Constitution"]
        },
        {
            "community_id": 1,
            "summary": "This community discusses social reform and anti-discrimination movements.",
            "entities": ["Ambedkar", "caste discrimination"]
        }
    ]
    
    communities = {
        "Ambedkar": 0,
        "Indian Constitution": 0,
        "caste discrimination": 1
    }
    
    entity_to_chunks = {
        "Ambedkar": [0, 1, 2],
        "Indian Constitution": [1],
        "caste discrimination": [2]
    }
    
    # Initialize and index
    global_rag = GlobalGraphRAG(top_k_communities=2, top_k_points=3)
    global_rag.index_community_summaries(summaries)
    global_rag.index_chunks(chunks)
    global_rag.set_communities(communities, entity_to_chunks)
    
    # Search
    query = "Tell me about constitutional reforms"
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    results = global_rag.search(query)
    
    print("\nRESULTS:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.3f} (Community {result['community_id']})")
        print(f"   Text: {result['chunk_text'][:80]}...")


if __name__ == "__main__":
    demo()
