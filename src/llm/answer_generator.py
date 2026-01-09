"""
Answer Generator

Generates answers to user queries using:
- Retrieved context from Local and Global RAG
- LLM for generation
- Citations to source chunks
"""

from typing import List, Dict, Optional


class AnswerGenerator:
    """
    Generates grounded answers using retrieved knowledge and LLM.
    """
    
    def __init__(self, llm_client):
        """
        Initialize answer generator.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
    
    def generate_answer(
        self,
        query: str,
        local_results: List[Dict],
        global_results: List[Dict],
        max_context_length: int = 4000
    ) -> Dict:
        """
        Generate answer using both local and global retrieval results.
        
        Args:
            query: User query
            local_results: Results from Local Graph RAG
            global_results: Results from Global Graph RAG
            max_context_length: Maximum context length
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build context from retrieved results
        context = self._build_context(local_results, global_results, max_context_length)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate answer
        answer = self.llm.generate(
            prompt,
            max_tokens=500,
            temperature=0.1,
            system_prompt=self._get_system_prompt()
        )
        
        # Extract citations
        citations = self._extract_citations(local_results, global_results)
        
        return {
            "query": query,
            "answer": answer.strip(),
            "citations": citations,
            "num_local_sources": len(local_results),
            "num_global_sources": len(global_results)
        }
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return """You are an AI assistant specialized in Dr. B.R. Ambedkar's works and philosophy.

Your task is to answer questions based ONLY on the provided context. Follow these rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Be concise and accurate
4. Cite specific information when possible
5. Do not make up information or use external knowledge"""
    
    def _build_context(
        self,
        local_results: List[Dict],
        global_results: List[Dict],
        max_length: int
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            local_results: Local RAG results
            global_results: Global RAG results
            max_length: Maximum context length
            
        Returns:
            Context string
        """
        context_parts = []
        current_length = 0
        
        # Add local results (entity-focused)
        if local_results:
            context_parts.append("=== RELEVANT ENTITIES AND TEXT ===\n")
            
            for i, result in enumerate(local_results):
                entities = result.get("entities", [])
                text = result.get("chunk_text", "")
                
                snippet = f"\n[Source {i+1}] (Entities: {', '.join(entities[:3])})\n{text}\n"
                
                if current_length + len(snippet) > max_length:
                    break
                
                context_parts.append(snippet)
                current_length += len(snippet)
        
        # Add global results (community-focused)
        if global_results and current_length < max_length:
            context_parts.append("\n=== BROADER CONTEXT ===\n")
            
            for i, result in enumerate(global_results):
                text = result.get("chunk_text", "")
                comm_id = result.get("community_id", "N/A")
                
                snippet = f"\n[Source {len(local_results) + i + 1}] (Community {comm_id})\n{text}\n"
                
                if current_length + len(snippet) > max_length:
                    break
                
                context_parts.append(snippet)
                current_length += len(snippet)
        
        return "".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Prompt string
        """
        prompt = f"""Context from Dr. B.R. Ambedkar's works:

{context}

Question: {query}

Answer (based only on the context above):"""
        
        return prompt
    
    def _extract_citations(
        self,
        local_results: List[Dict],
        global_results: List[Dict]
    ) -> List[Dict]:
        """
        Extract citation information from results.
        
        Args:
            local_results: Local RAG results
            global_results: Global RAG results
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Add local citations
        for i, result in enumerate(local_results):
            citation = {
                "source_id": i + 1,
                "type": "local",
                "chunk_id": result.get("chunk_id"),
                "score": result.get("score"),
                "entities": result.get("entities", []),
                "text_preview": result.get("chunk_text", "")[:100] + "..."
            }
            citations.append(citation)
        
        # Add global citations
        for i, result in enumerate(global_results):
            citation = {
                "source_id": len(local_results) + i + 1,
                "type": "global",
                "chunk_id": result.get("chunk_id"),
                "score": result.get("score"),
                "community_id": result.get("community_id"),
                "text_preview": result.get("chunk_text", "")[:100] + "..."
            }
            citations.append(citation)
        
        return citations
    
    def format_response(self, result: Dict) -> str:
        """
        Format the generation result for display.
        
        Args:
            result: Generation result dictionary
            
        Returns:
            Formatted response string
        """
        output = []
        output.append("="*60)
        output.append(f"QUESTION: {result['query']}")
        output.append("="*60)
        output.append(f"\nANSWER:\n{result['answer']}\n")
        
        if result['citations']:
            output.append("\nSOURCES:")
            for citation in result['citations']:
                output.append(f"\n[{citation['source_id']}] {citation['type'].upper()}")
                output.append(f"    Chunk {citation['chunk_id']} (Score: {citation['score']:.3f})")
                
                if citation['type'] == 'local' and citation.get('entities'):
                    output.append(f"    Entities: {', '.join(citation['entities'][:3])}")
                elif citation['type'] == 'global':
                    output.append(f"    Community: {citation.get('community_id')}")
                
                output.append(f"    Preview: {citation['text_preview']}")
        
        output.append("\n" + "="*60)
        return "\n".join(output)


def demo():
    """Demo answer generator."""
    # Mock LLM
    class MockLLM:
        def generate(self, prompt, **kwargs):
            return "Dr. B.R. Ambedkar was a prominent social reformer who drafted the Indian Constitution and fought against caste discrimination."
    
    # Sample results
    local_results = [
        {
            "chunk_id": 0,
            "chunk_text": "Dr. B.R. Ambedkar was a social reformer and jurist who drafted the Constitution.",
            "score": 0.85,
            "entities": ["Dr. B.R. Ambedkar", "Constitution"]
        }
    ]
    
    global_results = [
        {
            "chunk_id": 1,
            "chunk_text": "Ambedkar fought against caste discrimination and advocated for social equality.",
            "score": 0.78,
            "community_id": 0
        }
    ]
    
    # Generate answer
    generator = AnswerGenerator(MockLLM())
    result = generator.generate_answer(
        "What did Ambedkar do?",
        local_results,
        global_results
    )
    
    # Format and print
    print(generator.format_response(result))


if __name__ == "__main__":
    demo()
