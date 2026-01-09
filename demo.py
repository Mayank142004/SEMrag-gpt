#!/usr/bin/env python3
"""
AmbedkarGPT Demo Script

This script demonstrates the SEMRAG system for answering questions
about Dr. B.R. Ambedkar's works.

Usage:
    python demo.py --pdf <path_to_ambedkar_book.pdf>
    python demo.py --load  # Load pre-built index
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.ambedkargpt import SEMRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="AmbedkarGPT SEMRAG Demo")
    parser.add_argument("--pdf", type=str, help="Path to Ambedkar book PDF")
    parser.add_argument("--load", action="store_true", help="Load pre-built index")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("\n" + "="*80)
    print(" " * 25 + "AMBEDKARGPT - SEMRAG SYSTEM")
    print("="*80)
    
    pipeline = SEMRAGPipeline(config_path=args.config)
    
    # Build or load index
    if args.load:
        print("\nLoading pre-built index...")
        pipeline.load_index()
    elif args.pdf:
        print(f"\nBuilding index from PDF: {args.pdf}")
        text = pipeline.load_pdf(args.pdf)
        pipeline.build_index(text)
        
        # Save for future use
        pipeline.save_index()
    else:
        print("\nError: Provide either --pdf or --load")
        parser.print_help()
        return
    
    # Display statistics
    stats = pipeline.get_statistics()
    print("\n" + "="*80)
    print("INDEX STATISTICS:")
    print("="*80)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Demo questions
    demo_questions = [
        "Who was Dr. B.R. Ambedkar?",
        "What was Ambedkar's role in drafting the Constitution?",
        "What did Ambedkar write about caste discrimination?",
        "How did Ambedkar contribute to social reform?",
        "What is Ambedkar's view on Buddhism?"
    ]
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*80)
        print("INTERACTIVE Q&A MODE (type 'quit' to exit)")
        print("="*80)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                # Query the system
                result = pipeline.query(question)
                
                # Display answer
                print("\n" + pipeline.answer_generator.format_response(result))
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    else:
        # Demo mode with sample questions
        print("\n" + "="*80)
        print("DEMO MODE - Sample Questions")
        print("="*80)
        
        for i, question in enumerate(demo_questions[:3], 1):
            print(f"\n{'='*80}")
            print(f"QUESTION {i}: {question}")
            print('='*80)
            
            try:
                result = pipeline.query(question)
                print(f"\nANSWER:\n{result['answer']}\n")
                
                print(f"SOURCES: {result['num_local_sources']} local + {result['num_global_sources']} global")
                
                # Show top 2 citations
                for citation in result['citations'][:2]:
                    print(f"  [{citation['source_id']}] {citation['type'].upper()} - Chunk {citation['chunk_id']} (score: {citation['score']:.3f})")
                
            except Exception as e:
                print(f"Error processing question: {e}")
        
        print("\n" + "="*80)
        print("Demo complete! Run with --interactive for Q&A mode")
        print("="*80)


if __name__ == "__main__":
    main()
