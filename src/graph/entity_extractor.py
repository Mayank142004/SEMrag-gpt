"""
Entity Extraction Module

Extracts entities and relationships from text chunks using spaCy NER
and dependency parsing as described in Section 3.2.2 of the SEMRAG paper.
"""

import spacy
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import re


class EntityExtractor:
    """
    Extracts entities and relationships from text chunks.
    
    Uses:
    - spaCy NER for entity extraction
    - Dependency parsing for relationship extraction
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize entity extractor.
        
        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with text, label, and position
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            entities.append(entity)
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            text: Input text
            
        Returns:
            List of relationship tuples (subject, relation, object)
        """
        doc = self.nlp(text)
        relationships = []
        
        # Extract entity positions
        entities = {ent.text: ent for ent in doc.ents}
        
        # Use dependency parsing to find relationships
        for token in doc:
            # Look for verbs that connect entities
            if token.pos_ == "VERB":
                # Find subject
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                # Find objects
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                
                for subj in subjects:
                    for obj in objects:
                        # Check if subject and object are entities or contain entities
                        subj_text = self._get_entity_text(subj, entities)
                        obj_text = self._get_entity_text(obj, entities)
                        
                        if subj_text and obj_text:
                            relationship = {
                                "subject": subj_text,
                                "relation": token.lemma_,
                                "object": obj_text,
                                "sentence": token.sent.text
                            }
                            relationships.append(relationship)
        
        return relationships
    
    def _get_entity_text(self, token, entities: Dict) -> str:
        """
        Get entity text from token or its subtree.
        
        Args:
            token: spaCy token
            entities: Dictionary of entity texts
            
        Returns:
            Entity text if found, else empty string
        """
        # Check if token itself is an entity
        for ent_text in entities:
            if ent_text.lower() in token.text.lower():
                return ent_text
        
        # Check subtree
        subtree_text = " ".join([t.text for t in token.subtree])
        for ent_text in entities:
            if ent_text.lower() in subtree_text.lower():
                return ent_text
        
        # Return head noun phrase if it exists
        if token.pos_ in ("NOUN", "PROPN", "PRON"):
            return token.text
        
        return ""
    
    def extract_from_chunk(self, chunk: Dict) -> Dict:
        """
        Extract entities and relationships from a chunk.
        
        Args:
            chunk: Chunk dictionary with 'text' field
            
        Returns:
            Dictionary with entities and relationships
        """
        text = chunk["text"]
        
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text)
        
        return {
            "chunk_id": chunk["chunk_id"],
            "entities": entities,
            "relationships": relationships
        }
    
    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract entities and relationships from all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of extraction results
        """
        results = []
        
        print("Extracting entities and relationships from chunks...")
        for chunk in chunks:
            result = self.extract_from_chunk(chunk)
            results.append(result)
        
        return results
    
    def aggregate_entities(self, extraction_results: List[Dict]) -> Dict[str, List[int]]:
        """
        Aggregate entities across chunks.
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            Dictionary mapping entity text to list of chunk IDs
        """
        entity_to_chunks = defaultdict(list)
        
        for result in extraction_results:
            chunk_id = result["chunk_id"]
            for entity in result["entities"]:
                entity_text = entity["text"]
                entity_to_chunks[entity_text].append(chunk_id)
        
        return dict(entity_to_chunks)
    
    def aggregate_relationships(self, extraction_results: List[Dict]) -> List[Dict]:
        """
        Aggregate unique relationships across chunks.
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            List of unique relationships
        """
        unique_relationships = []
        seen = set()
        
        for result in extraction_results:
            for rel in result["relationships"]:
                # Create unique key
                key = (rel["subject"], rel["relation"], rel["object"])
                if key not in seen:
                    seen.add(key)
                    unique_relationships.append(rel)
        
        return unique_relationships


def demo():
    """Demo entity extraction."""
    sample_chunks = [
        {
            "chunk_id": 0,
            "text": "Dr. B.R. Ambedkar was born in Mhow, India. He was a jurist and economist."
        },
        {
            "chunk_id": 1,
            "text": "Ambedkar drafted the Indian Constitution. He believed in social equality."
        }
    ]
    
    extractor = EntityExtractor()
    results = extractor.extract_from_chunks(sample_chunks)
    
    print("\n" + "="*60)
    print("ENTITY EXTRACTION RESULTS:")
    print("="*60)
    for result in results:
        print(f"\nChunk {result['chunk_id']}:")
        print(f"  Entities: {[e['text'] for e in result['entities']]}")
        print(f"  Relationships: {len(result['relationships'])}")
        for rel in result['relationships'][:2]:
            print(f"    - {rel['subject']} --[{rel['relation']}]--> {rel['object']}")


if __name__ == "__main__":
    demo()
