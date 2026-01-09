"""
Semantic Chunking Implementation (Algorithm 1 from SEMRAG Paper)

This module implements semantic chunking via LLM embedding and cosine similarity
as described in Section 3.2.2 of the SEMRAG paper.
"""

import re
import nltk
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SemanticChunker:
    """
    Implements Algorithm 1: Semantic Chunking via LLM Embedding and Cosine Similarity
    
    Process:
    1. Split document into sentences
    2. Buffer merge sentences for context
    3. Embed merged sentences
    4. Calculate cosine distance between adjacent embeddings
    5. Group sentences when distance < threshold
    6. Split oversized chunks with overlap
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        buffer_size: int = 5,
        cosine_threshold: float = 0.5,
        max_tokens: int = 1024,
        overlap_tokens: int = 128
    ):
        """
        Initialize the Semantic Chunker.
        
        Args:
            embedding_model: Model for sentence embeddings
            buffer_size: Number of sentences to merge (b in paper)
            cosine_threshold: θ - threshold for semantic coherence
            max_tokens: Tmax - maximum chunk size in tokens
            overlap_tokens: Overlap size for sub-chunks
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.buffer_size = buffer_size
        self.cosine_threshold = cosine_threshold
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split document into sentences (Line 2 of Algorithm 1).
        
        Args:
            text: Input document text
            
        Returns:
            List of sentences
        """
        # Use NLTK for sentence tokenization
        sentences = nltk.sent_tokenize(text)
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def buffer_merge(self, sentences: List[str]) -> List[str]:
        """
        Merge sentences with buffer for contextual continuity (Line 3 of Algorithm 1).
        
        Args:
            sentences: List of individual sentences
            
        Returns:
            List of merged sentences with buffer context
        """
        if len(sentences) == 0:
            return []
        
        merged = []
        for i in range(len(sentences)):
            # Calculate window boundaries
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            
            # Merge sentences in buffer window
            buffer_text = " ".join(sentences[start:end])
            merged.append(buffer_text)
        
        return merged
    
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for sentences (Line 4 of Algorithm 1).
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            Numpy array of embeddings (shape: [n_sentences, embedding_dim])
        """
        embeddings = self.embedding_model.encode(
            sentences,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def calculate_cosine_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Calculate cosine distance between adjacent embeddings (Lines 5-6 of Algorithm 1).
        
        Cosine distance = 1 - cosine_similarity
        
        Args:
            embeddings: Array of sentence embeddings
            
        Returns:
            List of cosine distances between adjacent embeddings
        """
        distances = []
        for i in range(len(embeddings) - 1):
            # Compute cosine similarity
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            
            # Convert to cosine distance
            distance = 1 - sim
            distances.append(distance)
        
        return distances
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (simple whitespace-based).
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        # Simple approximation: split by whitespace
        return len(text.split())
    
    def create_chunks(
        self,
        sentences: List[str],
        distances: List[float]
    ) -> List[str]:
        """
        Group sentences into chunks based on semantic coherence (Lines 7-12 of Algorithm 1).
        
        Args:
            sentences: Original sentences
            distances: Cosine distances between adjacent sentences
            
        Returns:
            List of semantically coherent chunks
        """
        chunks = []
        current_chunk = []
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            
            # Check if we should split here
            if i < len(distances):
                if distances[i] >= self.cosine_threshold:
                    # High distance -> semantic boundary
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def split_with_overlap(self, chunk: str) -> List[str]:
        """
        Split oversized chunks with overlap (Lines 13-15 of Algorithm 1).
        
        Implements Equation 2 from the paper:
        g = ⋃[j=1 to m] g_j, where g_j ∩ g_{j+1} ≠ ∅
        
        Args:
            chunk: Oversized chunk to split
            
        Returns:
            List of sub-chunks with overlap
        """
        words = chunk.split()
        sub_chunks = []
        
        # Calculate chunk size in words (approximation)
        max_words = self.max_tokens
        overlap_words = self.overlap_tokens
        
        i = 0
        while i < len(words):
            # Take max_words from current position
            end = min(i + max_words, len(words))
            sub_chunk = " ".join(words[i:end])
            sub_chunks.append(sub_chunk)
            
            # Move forward with overlap
            i += max_words - overlap_words
            
            if i >= len(words):
                break
        
        return sub_chunks
    
    def process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Process chunks to enforce token limits (Lines 13-15 of Algorithm 1).
        
        Args:
            chunks: Initial chunks
            
        Returns:
            Processed chunks respecting token limits
        """
        processed_chunks = []
        
        for chunk in chunks:
            token_count = self.count_tokens(chunk)
            
            if token_count > self.max_tokens:
                # Split into overlapping sub-chunks
                sub_chunks = self.split_with_overlap(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def chunk_document(self, text: str) -> List[Dict]:
        """
        Complete semantic chunking pipeline (Algorithm 1).
        
        Args:
            text: Input document text
            
        Returns:
            List of chunk dictionaries with metadata
        """
        print("Step 1: Splitting into sentences...")
        sentences = self.split_into_sentences(text)
        print(f"  Found {len(sentences)} sentences")
        
        if len(sentences) == 0:
            return []
        
        print("Step 2: Buffer merging...")
        merged_sentences = self.buffer_merge(sentences)
        print(f"  Created {len(merged_sentences)} merged contexts")
        
        print("Step 3: Generating embeddings...")
        embeddings = self.embed_sentences(merged_sentences)
        print(f"  Generated embeddings with shape {embeddings.shape}")
        
        print("Step 4: Calculating cosine distances...")
        distances = self.calculate_cosine_distances(embeddings)
        print(f"  Calculated {len(distances)} distances")
        
        print("Step 5: Creating semantic chunks...")
        chunks = self.create_chunks(sentences, distances)
        print(f"  Created {len(chunks)} initial chunks")
        
        print("Step 6: Processing chunks (token limits)...")
        processed_chunks = self.process_chunks(chunks)
        print(f"  Final chunk count: {len(processed_chunks)}")
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(processed_chunks):
            chunk_obj = {
                "chunk_id": i,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text),
                "embedding": None  # Will be populated later for retrieval
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects


def demo():
    """Demo the semantic chunker."""
    sample_text = """
    Dr. B.R. Ambedkar was a visionary leader. He fought for social justice.
    His contributions to the Indian Constitution are immense. He believed in equality.
    Ambedkar was also a scholar of Buddhism. He converted to Buddhism in 1956.
    His writings continue to inspire millions. Education was his primary tool for empowerment.
    """
    
    chunker = SemanticChunker(buffer_size=2, cosine_threshold=0.6)
    chunks = chunker.chunk_document(sample_text)
    
    print("\n" + "="*60)
    print("SEMANTIC CHUNKS:")
    print("="*60)
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']} ({chunk['token_count']} tokens):")
        print(f"  {chunk['text'][:100]}...")


if __name__ == "__main__":
    demo()
