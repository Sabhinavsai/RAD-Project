"""
Example script demonstrating programmatic usage of the SLM-RAG Chatbot components.
This shows how to use the chatbot without the Streamlit UI.
"""

import json
from prompt_analyzer import PromptAnalyzer
from rag_engine import RAGEngine
from utils import format_analysis_for_display

def main():
    print("=" * 70)
    print("SLM-RAG Chatbot - Programmatic Usage Example")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing components...")
    analyzer = PromptAnalyzer()
    rag_engine = RAGEngine()
    print("✓ Components initialized successfully!\n")
    
    # Example 1: Simple prompt analysis
    print("-" * 70)
    print("EXAMPLE 1: Prompt Analysis")
    print("-" * 70)
    
    prompt1 = "What is machine learning and how does it work?"
    print(f"Prompt: {prompt1}\n")
    
    analysis1 = analyzer.analyze(prompt1)
    print("Analysis Results:")
    print(json.dumps(analysis1, indent=2))
    print()
    
    # Example 2: RAG response generation
    print("-" * 70)
    print("EXAMPLE 2: RAG Response Generation")
    print("-" * 70)
    
    prompt2 = "Explain RAG systems"
    print(f"Prompt: {prompt2}\n")
    
    analysis2 = analyzer.analyze(prompt2)
    print("Prompt Analysis:")
    print(f"  Intent: {analysis2['intent']}")
    print(f"  Domain: {analysis2['domain']}")
    print(f"  Complexity: {analysis2['complexity']}")
    print()
    
    response = rag_engine.generate_response(
        prompt=prompt2,
        analysis=analysis2,
        temperature=0.7,
        max_tokens=300,
        top_k=3
    )
    
    print("Generated Response:")
    print(response)
    print()
    
    print("Retrieved Documents:")
    for i, doc in enumerate(rag_engine.last_retrieved_docs, 1):
        print(f"\n  Document {i} (Score: {doc['score']:.3f}):")
        print(f"  {doc['content'][:100]}...")
    print()
    
    # Example 3: Adding custom documents
    print("-" * 70)
    print("EXAMPLE 3: Adding Custom Documents")
    print("-" * 70)
    
    custom_docs = [
        {
            "content": "Prompt engineering is the practice of crafting effective inputs for AI models to get desired outputs. It involves understanding model capabilities and limitations.",
            "metadata": {"category": "Prompt Engineering", "source": "custom"}
        },
        {
            "content": "Few-shot learning allows models to learn from just a few examples by providing sample inputs and outputs in the prompt.",
            "metadata": {"category": "Few-shot Learning", "source": "custom"}
        }
    ]
    
    rag_engine.add_documents(custom_docs)
    print(f"✓ Added {len(custom_docs)} custom documents to knowledge base\n")
    
    # Query with custom documents
    prompt3 = "What is prompt engineering?"
    print(f"Prompt: {prompt3}\n")
    
    analysis3 = analyzer.analyze(prompt3)
    response3 = rag_engine.generate_response(
        prompt=prompt3,
        analysis=analysis3,
        temperature=0.5,
        max_tokens=200,
        top_k=2
    )
    
    print("Response (using custom documents):")
    print(response3)
    print()
    
    # Example 4: Detailed analysis breakdown
    print("-" * 70)
    print("EXAMPLE 4: Detailed Analysis Breakdown")
    print("-" * 70)
    
    prompt4 = "Can you help me create a Python script for data analysis in 2024?"
    print(f"Prompt: {prompt4}\n")
    
    analysis4 = analyzer.analyze(prompt4)
    
    print("Complete Analysis:")
    print(f"  Original Prompt: {analysis4['original_prompt']}")
    print(f"  Intent: {analysis4['intent']}")
    print(f"  Keywords: {', '.join(analysis4['keywords'])}")
    print(f"  Entities: {len(analysis4['entities'])} found")
    for entity in analysis4['entities']:
        print(f"    - {entity['type']}: {entity['value']}")
    print(f"  Complexity: {analysis4['complexity']}")
    print(f"  Domain: {analysis4['domain']}")
    print(f"  Question Type: {analysis4['question_type']}")
    print(f"  Sentiment: {analysis4['sentiment']}")
    print(f"  Requires Context: {analysis4['requires_context']}")
    print(f"  Token Count: {analysis4['tokens_count']}")
    print()
    
    # Example 5: Conversation flow
    print("-" * 70)
    print("EXAMPLE 5: Multi-turn Conversation")
    print("-" * 70)
    
    conversation = [
        "What is deep learning?",
        "How is it different from machine learning?",
        "Can you give me an example?"
    ]
    
    for i, prompt in enumerate(conversation, 1):
        print(f"\nTurn {i}:")
        print(f"User: {prompt}")
        
        analysis = analyzer.analyze(prompt)
        response = rag_engine.generate_response(
            prompt=prompt,
            analysis=analysis,
            temperature=0.7,
            max_tokens=200,
            top_k=2
        )
        
        print(f"Assistant: {response}")
        
        # Add to conversation context
        rag_engine.add_conversation_to_knowledge(prompt, response)
    
    print()
    
    # Example 6: Exporting analysis
    print("-" * 70)
    print("EXAMPLE 6: Export Analysis as JSON")
    print("-" * 70)
    
    prompt6 = "Explain transformers in NLP"
    analysis6 = analyzer.analyze(prompt6)
    
    # Save to file
    output_file = "analysis_output.json"
    with open(output_file, 'w') as f:
        json.dump(analysis6, f, indent=2)
    
    print(f"✓ Analysis saved to {output_file}")
    print()
    
    # Example 7: Batch processing
    print("-" * 70)
    print("EXAMPLE 7: Batch Processing Multiple Prompts")
    print("-" * 70)
    
    prompts = [
        "What is AI?",
        "Explain neural networks",
        "What are transformers?"
    ]
    
    print(f"Processing {len(prompts)} prompts...\n")
    
    batch_results = []
    for prompt in prompts:
        analysis = analyzer.analyze(prompt)
        batch_results.append({
            "prompt": prompt,
            "intent": analysis['intent'],
            "domain": analysis['domain'],
            "complexity": analysis['complexity']
        })
    
    print("Batch Results:")
    for i, result in enumerate(batch_results, 1):
        print(f"\n  {i}. {result['prompt']}")
        print(f"     Intent: {result['intent']}, Domain: {result['domain']}, Complexity: {result['complexity']}")
    
    print()
    
    # Example 8: Performance metrics
    print("-" * 70)
    print("EXAMPLE 8: Performance Metrics")
    print("-" * 70)
    
    import time
    
    prompt8 = "What is natural language processing?"
    
    # Measure analysis time
    start = time.time()
    analysis8 = analyzer.analyze(prompt8)
    analysis_time = time.time() - start
    
    # Measure generation time
    start = time.time()
    response8 = rag_engine.generate_response(
        prompt=prompt8,
        analysis=analysis8,
        temperature=0.7,
        max_tokens=200,
        top_k=3
    )
    generation_time = time.time() - start
    
    print(f"Performance Metrics:")
    print(f"  Analysis Time: {analysis_time:.3f} seconds")
    print(f"  Generation Time: {generation_time:.3f} seconds")
    print(f"  Total Time: {(analysis_time + generation_time):.3f} seconds")
    print(f"  Documents Retrieved: {len(rag_engine.last_retrieved_docs)}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Analyzed {len(prompts) + 6} prompts")
    print(f"✓ Generated {len(prompts) + 6} responses")
    print(f"✓ Added {len(custom_docs)} custom documents")
    print(f"✓ Knowledge base size: {len(rag_engine.documents)} documents")
    print()
    print("Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
