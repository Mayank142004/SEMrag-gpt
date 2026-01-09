"""
Unit tests for Semantic Chunker
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking.semantic_chunker import SemanticChunker


def test_sentence_splitting():
    """Test sentence splitting."""
    chunker = SemanticChunker()
    
    text = "Dr. Ambedkar was a reformer. He drafted the Constitution. He fought for equality."
    sentences = chunker.split_into_sentences(text)
    
    assert len(sentences) == 3
    print("✓ Sentence splitting works")


def test_buffer_merging():
    """Test buffer merging."""
    chunker = SemanticChunker(buffer_size=1)
    
    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    merged = chunker.buffer_merge(sentences)
    
    assert len(merged) == 3
    assert "First" in merged[0] and "Second" in merged[0]  # Buffer includes neighbors
    print("✓ Buffer merging works")


def test_token_counting():
    """Test token counting."""
    chunker = SemanticChunker()
    
    text = "This is a test sentence with several words."
    count = chunker.count_tokens(text)
    
    assert count == 8
    print("✓ Token counting works")


def test_chunk_document():
    """Test full chunking pipeline."""
    chunker = SemanticChunker(
        buffer_size=2,
        cosine_threshold=0.6,
        max_tokens=50
    )
    
    text = """
    Dr. B.R. Ambedkar was born in 1891. He was a social reformer.
    Ambedkar studied at Columbia University. He earned multiple degrees.
    Later he drafted the Indian Constitution. This was his major contribution.
    """
    
    chunks = chunker.chunk_document(text)
    
    assert len(chunks) > 0
    assert all('chunk_id' in c for c in chunks)
    assert all('text' in c for c in chunks)
    print(f"✓ Created {len(chunks)} chunks")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Semantic Chunker")
    print("="*60)
    
    test_sentence_splitting()
    test_buffer_merging()
    test_token_counting()
    test_chunk_document()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
