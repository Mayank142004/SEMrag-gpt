# Quick Start Guide - AmbedkarGPT SEMRAG

## 🚀 Get Running in 5 Minutes

### Step 1: Install Ollama (if not installed)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

### Step 2: Pull the LLM Model

```bash
ollama pull llama3:8b
# This will download ~4.7GB
```

### Step 3: Setup Python Environment

```bash
cd ambedkargpt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Place Your PDF

```bash
# Put Ambedkar_book.pdf in the data directory
cp /path/to/Ambedkar_book.pdf data/
```

### Step 5: Run the Demo

```bash
# Build index and run demo (first time)
python demo.py --pdf data/Ambedkar_book.pdf

# This will:
# 1. Load and process the PDF (~3-5 min)
# 2. Build knowledge graph (~2-4 min)
# 3. Create community summaries (~1-2 min)
# 4. Answer 3 demo questions

# For subsequent runs, load the pre-built index:
python demo.py --load

# Interactive Q&A mode:
python demo.py --load --interactive
```

## 📝 Sample Questions to Try

1. "Who was Dr. B.R. Ambedkar?"
2. "What was Ambedkar's role in drafting the Indian Constitution?"
3. "What did Ambedkar write about caste discrimination?"
4. "Tell me about Ambedkar's education and academic achievements"
5. "What is Ambedkar's perspective on Buddhism?"

## ⚡ Quick Commands

```bash
# Test individual components
python src/chunking/semantic_chunker.py
python src/graph/entity_extractor.py
python src/retrieval/local_search.py

# Run tests
python tests/test_chunking.py

# Check Ollama status
ollama list
ollama serve  # If not running
```

## 🎯 For the Interview

**Before the interview:**
1. Build the index once (takes 5-10 minutes)
2. Test with `python demo.py --load`
3. Prepare 5 questions about Ambedkar
4. Check Ollama is running: `ollama list`
5. Have backup ready (system works with MockLLM if Ollama fails)

**During the interview:**
```bash
# Quick demo
python demo.py --load

# Interactive Q&A
python demo.py --load --interactive

# Show specific component
python src/chunking/semantic_chunker.py
```

## 🔧 Common Issues & Fixes

**Issue:** "Ollama not found"
```bash
# Check if Ollama is running
ollama serve

# In another terminal
ollama list
```

**Issue:** "Model not found"
```bash
ollama pull llama3:8b
```

**Issue:** "Import errors"
```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Issue:** "Out of memory"
```bash
# Edit config.yaml, reduce these:
buffer_size: 3          # was 5
max_chunk_tokens: 512   # was 1024
top_k: 3               # was 5
```

## 📊 What to Show

1. **Architecture Overview**
   - Show folder structure
   - Explain each component
   - Point to paper equations in code

2. **Semantic Chunking Demo**
   ```bash
   python src/chunking/semantic_chunker.py
   ```
   - Show buffer merging
   - Explain cosine similarity
   - Point to Algorithm 1 implementation

3. **Knowledge Graph**
   ```bash
   python src/graph/graph_builder.py
   ```
   - Show entities extracted
   - Show relationships
   - Show community detection

4. **Retrieval Methods**
   - Explain Local RAG (Equation 4)
   - Explain Global RAG (Equation 5)
   - Show how they combine

5. **Live Q&A**
   ```bash
   python demo.py --load --interactive
   ```
   - Answer questions
   - Show citations
   - Explain answer generation

## 📚 Files to Reference

- `README.md` - Overall documentation
- `IMPLEMENTATION.md` - Technical details
- `config.yaml` - Hyperparameters
- `src/pipeline/ambedkargpt.py` - Main pipeline
- Paper equations - Point to exact locations in code

## ✅ Pre-Demo Checklist

- [ ] Ollama running: `ollama serve`
- [ ] Model downloaded: `ollama list` shows llama3:8b
- [ ] Index built: `data/processed/` directory exists
- [ ] Test run successful: `python demo.py --load`
- [ ] Laptop charged & stable internet
- [ ] Code editor open with key files
- [ ] Terminal ready with commands
- [ ] Sample questions prepared

## 🎓 Key Points to Emphasize

1. **Complete SEMRAG Implementation**
   - Not simplified RAG
   - All paper components implemented
   - Equations 1-5 directly in code

2. **Production Quality**
   - Modular architecture
   - Error handling
   - Configuration management
   - Documentation

3. **Demonstrates Understanding**
   - Can explain each component
   - Can modify parameters
   - Can troubleshoot issues
   - Knows trade-offs

Good luck! 🚀
