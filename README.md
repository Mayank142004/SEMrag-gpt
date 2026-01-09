# AmbedkarGPT - SEMRAG System

A production-grade implementation of **SEMRAG (Semantic Knowledge-Augmented RAG)** for answering questions about Dr. B.R. Ambedkar's works.

This system implements the complete architecture described in the research paper "SEMRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering".

## 🎯 Features

- ✅ **Semantic Chunking** (Algorithm 1): Preserves semantic coherence using cosine similarity
- ✅ **Knowledge Graph Construction**: Entities and relationships with community detection
- ✅ **Local Graph RAG** (Equation 4): Entity-based retrieval
- ✅ **Global Graph RAG** (Equation 5): Community-level retrieval
- ✅ **LLM Integration**: Local LLM support via Ollama (Llama3/Mistral)
- ✅ **Citation Support**: Grounded answers with source attribution
- ✅ **Modular Architecture**: Clean, testable, production-ready code

## 📁 Project Structure

```
ambedkargpt/
├── data/
│   ├── Ambedkar_book.pdf         # Input: 94-page PDF
│   └── processed/                 # Generated: Chunks, graph, embeddings
│       ├── chunks.json
│       └── knowledge_graph.pkl
├── src/
│   ├── chunking/
│   │   └── semantic_chunker.py   # Algorithm 1: Semantic chunking
│   ├── graph/
│   │   ├── entity_extractor.py   # NER and relation extraction
│   │   ├── graph_builder.py      # Knowledge graph construction
│   │   └── summarizer.py         # Community summarization (Eq. 3)
│   ├── retrieval/
│   │   ├── local_search.py       # Local Graph RAG (Eq. 4)
│   │   └── global_search.py      # Global Graph RAG (Eq. 5)
│   ├── llm/
│   │   ├── llm_client.py         # Ollama integration
│   │   └── answer_generator.py   # Answer generation
│   └── pipeline/
│       └── ambedkargpt.py        # Main SEMRAG pipeline
├── config.yaml                    # Hyperparameters
├── requirements.txt               # Dependencies
├── demo.py                        # Demo script
└── README.md                      # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

- **Python 3.9+**
- **Ollama** (for local LLM)
- **8GB+ RAM** (16GB recommended)
- **10GB+ disk space**

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Install and Configure Ollama

```bash
# Install Ollama (https://ollama.ai)
# On Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

# On Windows: Download from https://ollama.com/download

# Pull the LLM model (choose one):
ollama pull llama3:8b      # Recommended
ollama pull mistral:7b     # Alternative
ollama pull gemma2:9b      # Alternative

# Verify installation
ollama list
```

### 4. Place the Data

Put `Ambedkar_book.pdf` in the `data/` directory:

```bash
cp /path/to/Ambedkar_book.pdf data/
```

## 📖 Usage

### Quick Start (Demo Mode)

```bash
# Build index and run demo
python demo.py --pdf data/Ambedkar_book.pdf

# Or load pre-built index
python demo.py --load
```

### Interactive Mode

```bash
# Interactive Q&A session
python demo.py --pdf data/Ambedkar_book.pdf --interactive

# Or with pre-built index
python demo.py --load --interactive
```

### Python API

```python
from src.pipeline.ambedkargpt import SEMRAGPipeline

# Initialize pipeline
pipeline = SEMRAGPipeline(config_path="config.yaml")

# Build index from PDF
text = pipeline.load_pdf("data/Ambedkar_book.pdf")
pipeline.build_index(text)

# Save index for reuse
pipeline.save_index()

# Query the system
result = pipeline.query("What was Ambedkar's role in drafting the Constitution?")
print(result['answer'])

# Show formatted response with citations
print(pipeline.answer_generator.format_response(result))
```

## ⚙️ Configuration

Edit `config.yaml` to tune hyperparameters:

```yaml
chunking:
  buffer_size: 5              # Context window for semantic chunking
  cosine_threshold: 0.5       # θ - semantic coherence threshold
  max_chunk_tokens: 1024      # Maximum chunk size
  overlap_tokens: 128         # Overlap for sub-chunks

retrieval:
  local:
    tau_e: 0.6               # τe - entity similarity threshold
    tau_d: 0.5               # τd - chunk-entity threshold
    top_k: 5                 # Number of results
  global:
    top_k_communities: 3     # Top communities to retrieve
    top_k_points: 10         # Top points to return

llm:
  model_name: "llama3:8b"    # Ollama model
  temperature: 0.1           # Generation temperature
  max_tokens: 1000           # Max tokens to generate
```

## 🧪 Testing Components

Each module can be tested independently:

```bash
# Test semantic chunker
python src/chunking/semantic_chunker.py

# Test entity extraction
python src/graph/entity_extractor.py

# Test knowledge graph
python src/graph/graph_builder.py

# Test local search
python src/retrieval/local_search.py

# Test global search
python src/retrieval/global_search.py

# Test LLM client
python src/llm/llm_client.py

# Test full pipeline
python src/pipeline/ambedkargpt.py
```

## 📊 Architecture Overview

### 1. Semantic Chunking (Algorithm 1)

```python
# Implements equation: cosine_distance(d(ci), d(ci+k)) < θ
# Where d(c) is the sentence embedding
```

**Process:**
1. Split document into sentences
2. Merge with buffer context (size b)
3. Embed merged sentences
4. Calculate cosine distances
5. Group by semantic similarity (distance < θ)
6. Split oversized chunks with overlap

### 2. Knowledge Graph Construction

**Nodes:** Entities extracted via spaCy NER
**Edges:** Relationships via dependency parsing
**Communities:** Detected using Leiden/Louvain algorithm

### 3. Local Graph RAG (Equation 4)

```
D_retrieved = Top_k({v ∈ V, g ∈ G | 
              sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
```

Retrieves entities and chunks based on:
- Entity-query similarity > τ_e
- Chunk-entity relevance > τ_d

### 4. Global Graph RAG (Equation 5)

```
D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} 
              (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))))
```

Retrieves through:
1. Top-K community summaries
2. Chunks from those communities
3. Scored and ranked points

### 5. Community Summarization (Equation 3)

```
S(C_i) = LLM_summarize(⋃_{v∈V_i} s(v) ∪ ⋃_{(v_j,v_k)∈E_i} s(v_j, v_k))
```

Generates summaries of communities for global search.

## 🎓 Sample Questions

Try these questions with the system:

1. "Who was Dr. B.R. Ambedkar?"
2. "What was Ambedkar's role in drafting the Indian Constitution?"
3. "What did Ambedkar write about caste discrimination?"
4. "How did Ambedkar contribute to social reform?"
5. "What is Ambedkar's view on Buddhism?"

## 🔧 Troubleshooting

### Ollama Not Running

```bash
# Start Ollama service
ollama serve

# In another terminal, verify:
ollama list
```

### Memory Issues

- Reduce `buffer_size` in config.yaml
- Reduce `max_chunk_tokens`
- Use smaller LLM (mistral:7b instead of llama3:8b)

### Slow Performance

- Use GPU if available
- Reduce `top_k` values in config
- Pre-build index once and reuse with `--load`

### Import Errors

```bash
# Ensure you're in the project root
cd ambedkargpt

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 📚 Technical Details

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Speed:** ~3000 sentences/sec on CPU

### NER Model
- **Model:** spaCy `en_core_web_sm`
- **Entities:** PERSON, ORG, GPE, DATE, WORK_OF_ART, etc.

### Community Detection
- **Algorithms:** Leiden (preferred) or Louvain
- **Resolution:** 1.0 (configurable)

### LLM
- **Default:** Llama-3-8B
- **Alternatives:** Mistral-7B, Gemma2-9B
- **Context:** 4096 tokens max

## 📈 Performance Metrics

On a typical 94-page document:
- **Chunking:** ~2-5 minutes
- **Entity Extraction:** ~3-8 minutes
- **Graph Construction:** ~1-2 minutes
- **Community Detection:** ~30 seconds
- **Query Time:** ~2-5 seconds

## 🤝 Contributing

This is an assignment implementation. For improvements:
1. Test with different buffer sizes
2. Experiment with community detection parameters
3. Try different LLM models
4. Optimize embedding models


## 🎯 Assignment Completion Checklist

- [x] Semantic chunking with Algorithm 1
- [x] Knowledge graph with entities and relationships
- [x] Community detection (Leiden/Louvain)
- [x] Local Graph RAG search (Equation 4)
- [x] Global Graph RAG search (Equation 5)
- [x] LLM integration (Ollama)
- [x] Answer generation with citations
- [x] Modular, production-ready code
- [x] Configuration via YAML
- [x] Demo script for live presentation
- [x] Comprehensive documentation

## 📞 Contact

For questions about this implementation:
- Submit issues or questions via email
- Refer to SEMRAG research paper for algorithm details

---

**Note:** This system requires a local LLM running via Ollama. Make sure Ollama is installed and the model is pulled before running the demo.
