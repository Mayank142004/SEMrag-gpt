"""
SEMRAG Pipeline

Main pipeline integrating all components:
1. PDF ingestion
2. Semantic chunking (Algorithm 1)
3. Knowledge graph construction
4. Community detection and summarization
5. Local and Global Graph RAG retrieval
6. LLM-based answer generation
"""

import os
import sys
import yaml
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
import pypdf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.semantic_chunker import SemanticChunker
from graph.entity_extractor import EntityExtractor
from graph.graph_builder import KnowledgeGraphBuilder
from graph.summarizer import CommunitySummarizer
from retrieval.local_search import LocalGraphRAG
from retrieval.global_search import GlobalGraphRAG
from llm.llm_client import create_llm_client
from llm.answer_generator import AnswerGenerator


class SEMRAGPipeline:
    """
    Complete SEMRAG pipeline implementation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize SEMRAG pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.chunker = None
        self.entity_extractor = None
        self.graph_builder = None
        self.community_summarizer = None
        self.local_rag = None
        self.global_rag = None
        self.llm = None
        self.answer_generator = None
        
        # Data storage
        self.chunks = []
        self.extraction_results = []
        self.communities = {}
        self.community_summaries = []
        
        print("SEMRAG Pipeline initialized")
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        print(f"\nLoading PDF: {pdf_path}")
        
        text_parts = []
        
        with open(pdf_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            
            print(f"  Pages: {num_pages}")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        print(f"  Extracted {len(full_text)} characters")
        
        return full_text
    
    def build_index(self, text: str):
        """
        Build the complete index (Steps 1-4).
        
        Args:
            text: Input document text
        """
        print("\n" + "="*60)
        print("STEP 1: SEMANTIC CHUNKING")
        print("="*60)
        
        # Initialize chunker
        chunking_config = self.config['chunking']
        self.chunker = SemanticChunker(
            embedding_model=chunking_config['embedding_model'],
            buffer_size=chunking_config['buffer_size'],
            cosine_threshold=chunking_config['cosine_threshold'],
            max_tokens=chunking_config['max_chunk_tokens'],
            overlap_tokens=chunking_config['overlap_tokens']
        )
        
        # Perform chunking
        self.chunks = self.chunker.chunk_document(text)
        
        print("\n" + "="*60)
        print("STEP 2: ENTITY EXTRACTION")
        print("="*60)
        
        # Initialize entity extractor
        kg_config = self.config['knowledge_graph']
        self.entity_extractor = EntityExtractor(model_name=kg_config['ner_model'])
        
        # Extract entities and relationships
        self.extraction_results = self.entity_extractor.extract_from_chunks(self.chunks)
        
        print("\n" + "="*60)
        print("STEP 3: KNOWLEDGE GRAPH CONSTRUCTION")
        print("="*60)
        
        # Build knowledge graph
        self.graph_builder = KnowledgeGraphBuilder()
        self.graph_builder.build_from_extractions(self.extraction_results, self.chunks)
        
        # Detect communities
        self.communities = self.graph_builder.detect_communities(
            algorithm=kg_config['community_algorithm'],
            resolution=kg_config['resolution']
        )
        
        print("\n" + "="*60)
        print("STEP 4: COMMUNITY SUMMARIZATION")
        print("="*60)
        
        # Initialize LLM for summarization
        llm_config = self.config['llm']
        self.llm = create_llm_client(
            model_name=llm_config['model_name'],
            base_url=llm_config['base_url'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        # Generate community summaries
        self.community_summarizer = CommunitySummarizer(self.llm)
        self.community_summaries = self.community_summarizer.summarize_all_communities(
            self.communities,
            self.graph_builder,
            self.chunks
        )
        
        print("\n" + "="*60)
        print("STEP 5: INDEXING FOR RETRIEVAL")
        print("="*60)
        
        # Initialize Local RAG
        local_config = self.config['retrieval']['local']
        self.local_rag = LocalGraphRAG(
            embedding_model=chunking_config['embedding_model'],
            tau_e=local_config['tau_e'],
            tau_d=local_config['tau_d'],
            top_k=local_config['top_k']
        )
        
        # Index entities and chunks for local search
        all_entities = list(self.graph_builder.entity_to_chunks.keys())
        self.local_rag.index_entities(all_entities)
        self.local_rag.index_chunks(self.chunks)
        self.local_rag.set_entity_chunk_links(self.graph_builder.entity_to_chunks)
        
        # Initialize Global RAG
        global_config = self.config['retrieval']['global']
        self.global_rag = GlobalGraphRAG(
            embedding_model=chunking_config['embedding_model'],
            top_k_communities=global_config['top_k_communities'],
            top_k_points=global_config['top_k_points']
        )
        
        # Index community summaries and chunks for global search
        self.global_rag.index_community_summaries(self.community_summaries)
        self.global_rag.index_chunks(self.chunks)
        self.global_rag.set_communities(self.communities, self.graph_builder.entity_to_chunks)
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator(self.llm)
        
        print("\n" + "="*60)
        print("INDEX BUILD COMPLETE!")
        print("="*60)
        print(f"  Chunks: {len(self.chunks)}")
        print(f"  Entities: {len(all_entities)}")
        print(f"  Relationships: {len(self.graph_builder.relationships)}")
        print(f"  Communities: {len(set(self.communities.values()))}")
    
    def query(self, question: str, history: Optional[str] = None) -> Dict:
        """
        Answer a question using SEMRAG.
        
        Args:
            question: User question
            history: Optional conversation history
            
        Returns:
            Answer dictionary
        """
        print("\n" + "="*60)
        print(f"QUERY: {question}")
        print("="*60)
        
        # Local Graph RAG Search
        print("\nLocal Graph RAG Search...")
        local_results = self.local_rag.search(question, history)
        
        # Global Graph RAG Search
        print("\nGlobal Graph RAG Search...")
        global_results = self.global_rag.search(question)
        
        # Generate answer
        print("\nGenerating answer...")
        result = self.answer_generator.generate_answer(
            question,
            local_results,
            global_results
        )
        
        return result
    
    def save_index(self, output_dir: str = "./data/processed"):
        """
        Save the built index to disk.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving index to {output_dir}...")
        
        # Save chunks
        with open(f"{output_dir}/chunks.json", 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save graph
        with open(f"{output_dir}/knowledge_graph.pkl", 'wb') as f:
            pickle.dump({
                'graph': self.graph_builder.graph,
                'entity_to_chunks': self.graph_builder.entity_to_chunks,
                'communities': self.communities,
                'community_summaries': self.community_summaries
            }, f)
        
        print("  Index saved successfully!")
    
    def load_index(self, input_dir: str = "./data/processed"):
        """
        Load a previously built index.
        
        Args:
            input_dir: Input directory
        """
        print(f"\nLoading index from {input_dir}...")
        
        # Load chunks
        with open(f"{input_dir}/chunks.json", 'r') as f:
            self.chunks = json.load(f)
        
        # Load graph
        with open(f"{input_dir}/knowledge_graph.pkl", 'rb') as f:
            data = pickle.load(f)
            self.graph_builder = KnowledgeGraphBuilder()
            self.graph_builder.graph = data['graph']
            self.graph_builder.entity_to_chunks = data['entity_to_chunks']
            self.communities = data['communities']
            self.community_summaries = data['community_summaries']
        
        # Re-initialize retrieval components
        chunking_config = self.config['chunking']
        local_config = self.config['retrieval']['local']
        global_config = self.config['retrieval']['global']
        llm_config = self.config['llm']
        
        # Local RAG
        self.local_rag = LocalGraphRAG(
            embedding_model=chunking_config['embedding_model'],
            tau_e=local_config['tau_e'],
            tau_d=local_config['tau_d'],
            top_k=local_config['top_k']
        )
        all_entities = list(self.graph_builder.entity_to_chunks.keys())
        self.local_rag.index_entities(all_entities)
        self.local_rag.index_chunks(self.chunks)
        self.local_rag.set_entity_chunk_links(self.graph_builder.entity_to_chunks)
        
        # Global RAG
        self.global_rag = GlobalGraphRAG(
            embedding_model=chunking_config['embedding_model'],
            top_k_communities=global_config['top_k_communities'],
            top_k_points=global_config['top_k_points']
        )
        self.global_rag.index_community_summaries(self.community_summaries)
        self.global_rag.index_chunks(self.chunks)
        self.global_rag.set_communities(self.communities, self.graph_builder.entity_to_chunks)
        
        # LLM and answer generator
        self.llm = create_llm_client(
            model_name=llm_config['model_name'],
            base_url=llm_config['base_url'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        self.answer_generator = AnswerGenerator(self.llm)
        
        print("  Index loaded successfully!")
    
    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "num_chunks": len(self.chunks),
            "num_entities": len(self.graph_builder.entity_to_chunks) if self.graph_builder else 0,
            "num_relationships": len(self.graph_builder.relationships) if self.graph_builder else 0,
            "num_communities": len(set(self.communities.values())) if self.communities else 0,
            "graph_stats": self.graph_builder.get_statistics() if self.graph_builder else {}
        }
        return stats


def demo():
    """Demo the SEMRAG pipeline."""
    # Sample text
    sample_text = """
    Dr. Bhimrao Ramji Ambedkar, popularly known as Babasaheb Ambedkar, was an Indian jurist, 
    economist, social reformer and political leader who headed the committee drafting the 
    Constitution of India from the Constituent Assembly debates, served as Law and Justice minister 
    in the first cabinet of Jawaharlal Nehru, and inspired the Dalit Buddhist movement after 
    renouncing Hinduism.
    
    Born into a poor Mahar family, Ambedkar spent his whole life fighting against social discrimination, 
    the system of Chaturvarna – the categorization of Hindu society into four varnas – and the Indian caste system. 
    He also wrote the book Annihilation of Caste in 1936.
    """
    
    # Create pipeline
    pipeline = SEMRAGPipeline()
    
    # Build index
    pipeline.build_index(sample_text)
    
    # Query
    question = "What did Ambedkar do?"
    result = pipeline.query(question)
    
    # Display
    print("\n" + pipeline.answer_generator.format_response(result))


if __name__ == "__main__":
    demo()
