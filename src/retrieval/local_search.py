"""
Local Graph RAG Search

Implements Equation 4 from the SEMRAG paper:
D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class LocalGraphRAG:
    """
    Local Graph RAG Search implementation.
    
    Retrieves relevant entities and their associated chunks based on:
    1. Entity similarity to query (τ_e threshold)
    2. Chunk-entity relevance (τ_d threshold)
    3. Top-k ranking
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau_e: float = 0.6,
        tau_d: float = 0.5,
        top_k: int = 5
    ):
        """
        Initialize Local Graph RAG.
        
        Args:
            embedding_model: Model for embeddings
            tau_e: Entity similarity threshold (τ_e)
            tau_d: Chunk-entity distance threshold (τ_d)
            top_k: Number of results to return
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tau_e = tau_e
        self.tau_d = tau_d
        self.top_k = top_k
        
        # Storage
        self.entity_embeddings = {}  # entity -> embedding
        self.chunk_embeddings = {}   # chunk_id -> embedding
        self.entity_to_chunks = {}   # entity -> [chunk_ids]
        self.chunks = []             # chunk data
    
    def index_entities(self, entities: List[str]):
        """
        Generate and store embeddings for entities.
        
        Args:
            entities: List of entity texts
        """
        print("Indexing entities...")
        unique_entities = list(set(entities))
        
        if len(unique_entities) == 0:
            return
        
        embeddings = self.embedding_model.encode(
            unique_entities,
            show_progress_bar=True,
            batch_size=32
        )
        
        for entity, embedding in zip(unique_entities, embeddings):
            self.entity_embeddings[entity] = embedding
        
        print(f"  Indexed {len(self.entity_embeddings)} entities")
    
    def index_chunks(self, chunks: List[Dict]):
        """
        Generate and store embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        print("Indexing chunks...")
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
    
    def set_entity_chunk_links(self, entity_to_chunks: Dict[str, List[int]]):
        """
        Set the entity-to-chunk mapping.
        
        Args:
            entity_to_chunks: Dictionary mapping entity to chunk IDs
        """
        self.entity_to_chunks = entity_to_chunks
        print(f"Linked {len(entity_to_chunks)} entities to chunks")
    
    def search(
        self,
        query: str,
        history: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform Local Graph RAG search.
        
        Implements Equation 4:
        D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
        
        Args:
            query: User query
            history: Optional conversation history
            
        Returns:
            List of retrieved results with entities and chunks
        """
        # Step 1: Combine query and history
        combined_query = query
        if history:
            combined_query = f"{history} {query}"
        
        # Step 2: Embed query
        query_embedding = self.embedding_model.encode([combined_query])[0]
        
        # Step 3: Find relevant entities (sim(v, Q+H) > τ_e)
        relevant_entities = self._find_relevant_entities(query_embedding)
        
        if len(relevant_entities) == 0:
            print("  Warning: No entities matched the query threshold")
            return []
        
        # Step 4: Find relevant chunks linked to entities (sim(g, v) > τ_d)
        relevant_chunks = self._find_relevant_chunks(
            query_embedding,
            relevant_entities
        )
        
        # Step 5: Rank and return top-k
        results = self._rank_results(relevant_entities, relevant_chunks)
        
        return results[:self.top_k]
    
    def _find_relevant_entities(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """
        Find entities with similarity > τ_e.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        relevant = []
        
        for entity, entity_emb in self.entity_embeddings.items():
            # Calculate similarity
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity_emb.reshape(1, -1)
            )[0][0]
            
            # Filter by threshold
            if sim > self.tau_e:
                relevant.append((entity, sim))
        
        # Sort by similarity
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Found {len(relevant)} relevant entities (threshold: {self.tau_e})")
        
        return relevant
    
    def _find_relevant_chunks(
        self,
        query_embedding: np.ndarray,
        relevant_entities: List[Tuple[str, float]]
    ) -> List[Tuple[int, str, float]]:
        """
        Find chunks linked to relevant entities with similarity > τ_d.
        
        Args:
            query_embedding: Query embedding vector
            relevant_entities: List of (entity, score) tuples
            
        Returns:
            List of (chunk_id, entity, similarity_score) tuples
        """
        relevant_chunks = []
        seen_chunks = set()
        
        for entity, entity_score in relevant_entities:
            # Get chunks linked to this entity
            if entity not in self.entity_to_chunks:
                continue
            
            chunk_ids = self.entity_to_chunks[entity]
            
            for chunk_id in chunk_ids:
                if chunk_id in seen_chunks:
                    continue
                
                # Get chunk embedding
                if chunk_id not in self.chunk_embeddings:
                    continue
                
                chunk_emb = self.chunk_embeddings[chunk_id]
                
                # Calculate chunk-query similarity
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    chunk_emb.reshape(1, -1)
                )[0][0]
                
                # Filter by threshold τ_d
                if sim > self.tau_d:
                    relevant_chunks.append((chunk_id, entity, sim))
                    seen_chunks.add(chunk_id)
        
        print(f"  Found {len(relevant_chunks)} relevant chunks (threshold: {self.tau_d})")
        
        return relevant_chunks
    
    def _rank_results(
        self,
        entities: List[Tuple[str, float]],
        chunks: List[Tuple[int, str, float]]
    ) -> List[Dict]:
        """
        Rank and format results.
        
        Args:
            entities: List of (entity, score) tuples
            chunks: List of (chunk_id, entity, score) tuples
            
        Returns:
            Ranked list of result dictionaries
        """
        # Combine entity scores and chunk scores
        chunk_scores = {}
        chunk_entities = {}
        
        for chunk_id, entity, score in chunks:
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = []
                chunk_entities[chunk_id] = []
            
            chunk_scores[chunk_id].append(score)
            chunk_entities[chunk_id].append(entity)
        
        # Average scores per chunk
        results = []
        for chunk_id in chunk_scores:
            avg_score = np.mean(chunk_scores[chunk_id])
            
            result = {
                "chunk_id": chunk_id,
                "chunk_text": self.chunks[chunk_id]["text"],
                "score": float(avg_score),
                "entities": chunk_entities[chunk_id],
                "source": "local_rag"
            }
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results


def demo():
    """Demo local graph RAG search."""
    # Sample data
    chunks = [
        {"chunk_id": 0, "text": "Dr. B.R. Ambedkar was a social reformer and jurist."},
        {"chunk_id": 1, "text": "Ambedkar drafted the Indian Constitution."},
        {"chunk_id": 2, "text": "He fought against caste discrimination."}
    ]
    
    entities = ["Dr. B.R. Ambedkar", "Ambedkar", "Indian Constitution", "caste discrimination"]
    
    entity_to_chunks = {
        "Dr. B.R. Ambedkar": [0],
        "Ambedkar": [1, 2],
        "Indian Constitution": [1],
        "caste discrimination": [2]
    }
    
    # Initialize and index
    local_rag = LocalGraphRAG(top_k=3)
    local_rag.index_entities(entities)
    local_rag.index_chunks(chunks)
    local_rag.set_entity_chunk_links(entity_to_chunks)
    
    # Search
    query = "What did Ambedkar do for the Constitution?"
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    results = local_rag.search(query)
    
    print("\nRESULTS:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.3f}")
        print(f"   Entities: {result['entities']}")
        print(f"   Text: {result['chunk_text'][:80]}...")


if __name__ == "__main__":
    demo()
