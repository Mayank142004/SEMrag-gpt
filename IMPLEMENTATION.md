# SEMRAG Implementation Guide

## Overview

This document explains the implementation details of the SEMRAG system for answering questions about Dr. B.R. Ambedkar's works.

## Implementation of SEMRAG Paper Components

### 1. Semantic Chunking (Algorithm 1)

**File:** `src/chunking/semantic_chunker.py`

**Implementation Details:**

The semantic chunker implements Algorithm 1 exactly as described in the paper:

```python
def chunk_document(text):
    # Line 2: Split into sentences
    sentences = split_into_sentences(text)
    
    # Line 3: Buffer merge (context window)
    merged_sentences = buffer_merge(sentences, buffer_size=b)
    
    # Line 4: Generate embeddings
    embeddings = embed_sentences(merged_sentences)
    
    # Lines 5-6: Calculate cosine distances
    distances = calculate_cosine_distances(embeddings)
    
    # Lines 7-12: Create chunks based on threshold
    chunks = create_chunks(sentences, distances, threshold=θ)
    
    # Lines 13-15: Split oversized chunks with overlap
    final_chunks = process_chunks(chunks, max_tokens=1024, overlap=128)
    
    return final_chunks
```

**Key Equations:**

**Equation 1: Cosine Distance**
```
d(ci, ci+k) = 1 - (d(ci) · d(ci+k)) / (||d(ci)||2 * ||d(ci+k)||2)
```

Where:
- d(c) is the embedding of sentence/sentence group
- · represents dot product
- || || is L2 norm

**Equation 2: Chunk Splitting with Overlap**
```
g = ⋃[j=1 to m] gj
where gj ∩ gj+1 ≠ ∅, |gj| ≤ 1024, |gj ∩ gj+1| = 128
```

### 2. Knowledge Graph Construction

**Files:** 
- `src/graph/entity_extractor.py`
- `src/graph/graph_builder.py`

**Process:**

1. **Entity Extraction** (NER):
   - Uses spaCy's `en_core_web_sm` model
   - Extracts: PERSON, ORG, GPE, DATE, WORK_OF_ART, etc.

2. **Relationship Extraction**:
   - Uses dependency parsing
   - Identifies subject-verb-object triples
   - Links entities through verbs

3. **Graph Structure**:
   - Nodes: Entities
   - Edges: Relationships
   - Attributes: Entity types, relationship types

4. **Community Detection**:
   - Leiden algorithm (preferred) or Louvain
   - Groups semantically related entities
   - Creates hierarchical structure

### 3. Community Summarization (Equation 3)

**File:** `src/graph/summarizer.py`

**Equation 3 Implementation:**
```
S(Ci) = LLM_summarize(⋃_{v∈Vi} s(v) ∪ ⋃_{(vj,vk)∈Ei} s(vj, vk))
```

Where:
- Ci is a community
- Vi is the set of nodes in the community
- Ei is the set of edges in the community
- s(v) is the summary of a node
- s(vj, vk) is the summary of an edge

**Implementation:**
```python
def summarize_community(community_id, subgraph, entity_to_chunks, chunks):
    # Collect nodes and edges
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges(data=True))
    
    # Build context
    context = build_context(nodes, edges, entity_to_chunks, chunks)
    
    # Generate summary with LLM
    summary = llm.generate(
        f"Summarize this community:\n{context}",
        max_tokens=200
    )
    
    return summary
```

### 4. Local Graph RAG (Equation 4)

**File:** `src/retrieval/local_search.py`

**Equation 4 Implementation:**
```
Dretrieved = Topk({v ∈ V, g ∈ G | 
              sim(v, Q+H) > τe ∧ sim(g, v) > τd})
```

Where:
- V is the set of entities
- G is the set of chunks
- Q is the query
- H is optional history
- τe is entity similarity threshold
- τd is chunk-entity distance threshold

**Algorithm:**
```python
def local_search(query, history=None):
    # Step 1: Combine query and history
    combined_query = query + (history or "")
    query_embedding = embed(combined_query)
    
    # Step 2: Find relevant entities (sim > τe)
    relevant_entities = []
    for entity, entity_emb in entity_embeddings.items():
        sim = cosine_similarity(query_embedding, entity_emb)
        if sim > tau_e:
            relevant_entities.append((entity, sim))
    
    # Step 3: Find chunks linked to entities (sim > τd)
    relevant_chunks = []
    for entity, _ in relevant_entities:
        for chunk_id in entity_to_chunks[entity]:
            chunk_emb = chunk_embeddings[chunk_id]
            sim = cosine_similarity(query_embedding, chunk_emb)
            if sim > tau_d:
                relevant_chunks.append((chunk_id, entity, sim))
    
    # Step 4: Rank and return top-k
    ranked = sort_by_score(relevant_chunks)
    return ranked[:top_k]
```

### 5. Global Graph RAG (Equation 5)

**File:** `src/retrieval/global_search.py`

**Equation 5 Implementation:**
```
Dretrieved = Topk(⋃_{r ∈ RTop-K(Q)} ⋃_{ci ∈ Cr} 
              (⋃_{pj ∈ ci} (pj, score(pj, Q))), score(pj, Q))
```

Where:
- RTop-K(Q) are the top-K community reports for query Q
- Cr are the chunks in report r
- pj are the points (sub-pieces) in chunk ci

**Algorithm:**
```python
def global_search(query):
    # Step 1: Embed query
    query_embedding = embed(query)
    
    # Step 2: Find top-K community summaries
    community_scores = []
    for comm_id, comm_emb in community_embeddings.items():
        sim = cosine_similarity(query_embedding, comm_emb)
        community_scores.append((comm_id, sim))
    
    top_communities = sort_by_score(community_scores)[:top_k_communities]
    
    # Step 3: Get chunks from these communities
    community_chunks = []
    for comm_id, comm_score in top_communities:
        # Find entities in this community
        entities = [e for e, c in communities.items() if c == comm_id]
        
        # Get their chunks
        for entity in entities:
            for chunk_id in entity_to_chunks[entity]:
                community_chunks.append((chunk_id, comm_id, comm_score))
    
    # Step 4: Score each chunk against query
    scored_points = []
    for chunk_id, comm_id, comm_score in community_chunks:
        chunk_emb = chunk_embeddings[chunk_id]
        chunk_score = cosine_similarity(query_embedding, chunk_emb)
        
        # Combine scores (weighted average)
        final_score = 0.3 * comm_score + 0.7 * chunk_score
        scored_points.append((chunk_id, final_score))
    
    # Step 5: Return top-K points
    ranked = sort_by_score(scored_points)
    return ranked[:top_k_points]
```

### 6. LLM Integration

**Files:**
- `src/llm/llm_client.py`
- `src/llm/answer_generator.py`

**LLM Usage:**

1. **Community Summarization**:
   - Input: Entities + relationships + context
   - Output: 2-3 sentence summary
   - Temperature: 0.1 (deterministic)

2. **Answer Generation**:
   - Input: Query + retrieved context (local + global)
   - Output: Grounded answer with citations
   - Temperature: 0.1 (factual)

**Prompt Template:**
```
System: You are an AI assistant specialized in Dr. B.R. Ambedkar's works.
Answer based ONLY on the provided context.

Context:
[Retrieved chunks from local and global search]

Question: [User query]

Answer:
```

## Configuration Parameters

### Chunking Parameters

- **buffer_size (b)**: 5
  - Number of sentences to merge for context
  - Trade-off: Larger = more context, slower processing
  - Paper recommendation: 2-5 for news articles, 5-10 for dense text

- **cosine_threshold (θ)**: 0.5
  - Semantic similarity threshold for grouping
  - Lower = larger chunks, higher = smaller chunks
  - Paper recommendation: 0.4-0.6

- **max_chunk_tokens**: 1024
  - Maximum chunk size before splitting
  - Limited by LLM context window

- **overlap_tokens**: 128
  - Overlap between sub-chunks
  - Preserves continuity across splits

### Retrieval Parameters

- **tau_e (τe)**: 0.6
  - Entity similarity threshold for local search
  - Higher = more selective, fewer results

- **tau_d (τd)**: 0.5
  - Chunk-entity distance threshold
  - Filters chunks by relevance to entities

- **top_k_communities**: 3
  - Number of communities for global search
  - Paper recommendation: 3-5

- **top_k_points**: 10
  - Final number of points to retrieve
  - Balance between coverage and context size

### LLM Parameters

- **temperature**: 0.1
  - Low for factual, deterministic answers
  - Higher (0.7-0.9) for creative generation

- **max_tokens**: 1000
  - Maximum length of generated answer
  - Adjust based on query complexity

## Performance Optimization

### 1. Embedding Caching
```python
# Cache embeddings to avoid recomputation
self.embedding_cache = {}

def embed_with_cache(text):
    if text not in self.embedding_cache:
        self.embedding_cache[text] = self.model.encode(text)
    return self.embedding_cache[text]
```

### 2. Batch Processing
```python
# Process multiple texts in batches
embeddings = self.model.encode(
    texts,
    batch_size=32,  # Adjust based on GPU memory
    show_progress_bar=True
)
```

### 3. Index Persistence
```python
# Save built index for reuse
pipeline.save_index("data/processed")

# Load instead of rebuilding
pipeline.load_index("data/processed")
```

## Testing

Run component tests:
```bash
python tests/test_chunking.py
python tests/test_retrieval.py
python tests/test_integration.py
```

## Deployment Checklist

- [ ] Ollama installed and running
- [ ] LLM model pulled (llama3:8b)
- [ ] spaCy model downloaded (en_core_web_sm)
- [ ] PDF file placed in data/
- [ ] Dependencies installed
- [ ] Config tuned for your data
- [ ] Index built and saved
- [ ] Demo script tested

## Live Demo Preparation

1. **Pre-build the index**:
   ```bash
   python demo.py --pdf data/Ambedkar_book.pdf
   ```

2. **Test with sample questions**:
   ```bash
   python demo.py --load --interactive
   ```

3. **Prepare 5 diverse questions**:
   - Factual: "When was Ambedkar born?"
   - Analytical: "What was Ambedkar's philosophy on caste?"
   - Comparative: "How did Ambedkar view Buddhism vs Hinduism?"
   - Synthesis: "What were Ambedkar's major contributions?"
   - Specific: "What did Ambedkar write about education?"

4. **Monitor system resources**:
   - Check memory usage: `top` or `htop`
   - Ensure Ollama is responsive
   - Have backup (MockLLM) ready

## Troubleshooting

### Issue: Slow chunking
**Solution:** Reduce buffer_size or use smaller embedding model

### Issue: Poor retrieval quality
**Solution:** Tune tau_e and tau_d thresholds

### Issue: Generic answers
**Solution:** Increase top_k for more context

### Issue: Out of memory
**Solution:** Reduce batch_size in embedding, lower max_chunk_tokens

### Issue: Ollama connection error
**Solution:** Check `ollama serve` is running, verify model is pulled

## References

- SEMRAG Paper: "Semantic Knowledge-Augmented RAG for Improved Question-Answering"
- Sentence Transformers: https://www.sbert.net
- spaCy: https://spacy.io
- Ollama: https://ollama.ai
- NetworkX: https://networkx.org
