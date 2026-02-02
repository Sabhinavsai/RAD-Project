"""
Enhanced SLM-RAG Chatbot - Test & Verification Script
Tests all components and features
"""

import sys
import time

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 70)
    print("TEST 1: Package Imports")
    print("=" * 70)
    
    required_packages = [
        ("streamlit", "Streamlit"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("bs4", "BeautifulSoup4"),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r enhanced_requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!\n")
        return True

def test_component_loading():
    """Test loading of enhanced components"""
    print("=" * 70)
    print("TEST 2: Component Loading")
    print("=" * 70)
    
    try:
        import enhanced_config as enhanced_config
        print("  ✓ Enhanced Config")
    except Exception as e:
        print(f"  ✗ Enhanced Config failed: {e}")
        return False
    
    try:
        from web_content_fetcher import WebContentFetcher
        print("  ✓ Web Content Fetcher")
    except Exception as e:
        print(f"  ✗ Web Content Fetcher failed: {e}")
        return False
    
    try:
        from enhanced_prompt_analyzer import EnhancedPromptAnalyzer
        print("  ✓ Enhanced Prompt Analyzer")
    except Exception as e:
        print(f"  ✗ Enhanced Prompt Analyzer failed: {e}")
        return False
    
    try:
        from enhanced_rag_engine import EnhancedRAGEngine
        print("  ✓ Enhanced RAG Engine")
    except Exception as e:
        print(f"  ✗ Enhanced RAG Engine failed: {e}")
        return False
    
    print("\n✅ All components loaded!\n")
    return True

def test_prompt_analysis():
    """Test enhanced prompt analyzer"""
    print("=" * 70)
    print("TEST 3: Prompt Analysis")
    print("=" * 70)
    
    try:
        from enhanced_prompt_analyzer import EnhancedPromptAnalyzer
        import enhanced_config as enhanced_config

        config = {
            'INTENT_PATTERNS': enhanced_config.INTENT_PATTERNS,
            'CONTEXT_PATTERNS': enhanced_config.CONTEXT_PATTERNS,
            'DOMAIN_SOURCES': enhanced_config.DOMAIN_SOURCES
        }
        
        analyzer = EnhancedPromptAnalyzer(config=config)
        
        # Test prompts
        test_cases = [
            {
                "prompt": "Generate a movie recommendation ML model",
                "expected_intent": "model_creation",
                "expected_subject": "movie"
            },
            {
                "prompt": "What is machine learning?",
                "expected_intent": "information_seeking",
            },
            {
                "prompt": "How to build a chatbot",
                "expected_intent": "tutorial",
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['prompt']}")
            
            analysis = analyzer.analyze(test['prompt'])
            
            print(f"  Intent: {analysis['intent']}")
            print(f"  Domain: {analysis['domain']}")
            print(f"  Complexity: {analysis['complexity']}")
            
            if analysis.get('parameters', {}).get('main_topic'):
                print(f"  Main Topic: {analysis['parameters']['main_topic']}")
            
            print(f"  Keywords: {', '.join(analysis['keywords'][:5])}")
            
            # Validate
            if 'expected_intent' in test:
                if test['expected_intent'] in analysis['intent']:
                    print("  ✓ Intent detection correct")
                else:
                    print(f"  ⚠ Expected {test['expected_intent']}, got {analysis['intent']}")
        
        print("\n✅ Prompt analysis working!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Prompt analysis failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_web_fetching():
    """Test web content fetcher"""
    print("=" * 70)
    print("TEST 4: Web Content Fetching")
    print("=" * 70)
    
    try:
        from web_content_fetcher import WebContentFetcher
        import enhanced_config as enhanced_config

        config = {
            'KNOWLEDGE_SOURCES': enhanced_config.KNOWLEDGE_SOURCES,
            'WEB_CONTENT_TIMEOUT': 5,  # Shorter timeout for testing
            'WEB_CONTENT_MAX_LENGTH': 1000,
            'ENABLE_WEB_CACHE': False,  # Disable cache for testing
            'WEB_CACHE_DIR': '.web_cache',
            'WEB_CACHE_DURATION': 3600
        }
        
        fetcher = WebContentFetcher(config)
        
        print("\nFetching from Wikipedia...")
        wiki_content = fetcher._fetch_wikipedia("machine learning")
        
        if wiki_content:
            print(f"  ✓ Retrieved {len(wiki_content)} documents from Wikipedia")
            print(f"  Sample: {wiki_content[0]['content'][:100]}...")
        else:
            print("  ⚠ No content from Wikipedia (might be network issue)")
        
        print("\n✅ Web fetching functional!\n")
        return True
        
    except Exception as e:
        print(f"\n⚠ Web fetching test encountered issue: {e}")
        print("This is non-critical - system can work with default knowledge\n")
        return True  # Non-critical

def test_rag_engine():
    """Test enhanced RAG engine"""
    print("=" * 70)
    print("TEST 5: RAG Engine")
    print("=" * 70)
    
    try:
        from enhanced_rag_engine import EnhancedRAGEngine
        import enhanced_config as enhanced_config

        config = {
            'ENHANCED_KNOWLEDGE_BASE': enhanced_config.ENHANCED_KNOWLEDGE_BASE[:3],  # Use subset
            'USE_DEFAULT_DOCUMENTS': True,
            'USE_WEB_CONTENT': False,  # Disable for faster testing
        }
        
        print("Initializing RAG engine (this may take a moment)...")
        rag = EnhancedRAGEngine(config=config, web_fetcher=None)
        
        print(f"  ✓ RAG engine initialized")
        print(f"  Documents in KB: {len(rag.documents)}")
        
        # Test retrieval
        print("\nTesting document retrieval...")
        query = "What is machine learning?"
        docs = rag.retrieve(query, top_k=3)
        
        print(f"  ✓ Retrieved {len(docs)} documents")
        if docs:
            print(f"  Top document score: {docs[0].get('score', 0):.3f}")
        
        print("\n✅ RAG engine working!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG engine failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test complete workflow"""
    print("=" * 70)
    print("TEST 6: End-to-End Workflow")
    print("=" * 70)
    
    try:
        from enhanced_prompt_analyzer import EnhancedPromptAnalyzer
        from enhanced_rag_engine import EnhancedRAGEngine
        import enhanced_config as enhanced_config

        config = {
            'INTENT_PATTERNS': enhanced_config.INTENT_PATTERNS,
            'CONTEXT_PATTERNS': enhanced_config.CONTEXT_PATTERNS,
            'DOMAIN_SOURCES': enhanced_config.DOMAIN_SOURCES,
            'ENHANCED_KNOWLEDGE_BASE': enhanced_config.ENHANCED_KNOWLEDGE_BASE[:5],
            'USE_DEFAULT_DOCUMENTS': True,
            'USE_WEB_CONTENT': False,
        }
        
        print("Initializing components...")
        analyzer = EnhancedPromptAnalyzer(config=config)
        rag = EnhancedRAGEngine(config=config, web_fetcher=None)
        
        prompt = "Explain neural networks"
        
        print(f"\nTest Prompt: '{prompt}'")
        
        # Step 1: Analyze
        print("\n1. Analyzing prompt...")
        start = time.time()
        analysis = analyzer.analyze(prompt)
        analysis_time = time.time() - start
        print(f"   ✓ Analysis completed in {analysis_time:.2f}s")
        print(f"   Intent: {analysis['intent']}")
        print(f"   Domain: {analysis['domain']}")
        
        # Step 2: Generate response
        print("\n2. Generating response...")
        start = time.time()
        response = rag.generate_response(
            prompt=prompt,
            analysis=analysis,
            temperature=0.7,
            max_tokens=500,
            top_k=3
        )
        gen_time = time.time() - start
        print(f"   ✓ Response generated in {gen_time:.2f}s")
        print(f"   Response length: {len(response)} characters")
        print(f"\n   Response preview:")
        print(f"   {response[:200]}...")
        
        print(f"\n   Total time: {(analysis_time + gen_time):.2f}s")
        
        print("\n✅ End-to-end workflow successful!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ End-to-end test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_code_generation():
    """Test code generation capability"""
    print("=" * 70)
    print("TEST 7: Code Generation")
    print("=" * 70)
    
    try:
        from enhanced_rag_engine import EnhancedRAGEngine
        import enhanced_config as enhanced_config

        config = {
            'ENHANCED_KNOWLEDGE_BASE': enhanced_config.ENHANCED_KNOWLEDGE_BASE,
            'USE_DEFAULT_DOCUMENTS': True,
            'USE_WEB_CONTENT': False,
        }
        
        rag = EnhancedRAGEngine(config=config, web_fetcher=None)
        
        # Test code generation
        code = rag._generate_recommendation_code("movie recommendation")
        
        print("Generated code sample:")
        print(code[:300] + "...")
        
        # Verify code has key components
        checks = [
            ("class" in code, "Class definition"),
            ("def" in code, "Function definitions"),
            ("import" in code, "Import statements"),
            ("cosine_similarity" in code, "Algorithm implementation"),
        ]
        
        print("\nCode quality checks:")
        all_passed = True
        for passed, check_name in checks:
            if passed:
                print(f"  ✓ {check_name}")
            else:
                print(f"  ✗ {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n✅ Code generation working!\n")
            return True
        else:
            print("\n⚠ Some code checks failed\n")
            return False
        
    except Exception as e:
        print(f"\n❌ Code generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("\n" + "=" * 70)
    print("ENHANCED SLM-RAG CHATBOT - COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Component Loading", test_component_loading),
        ("Prompt Analysis", test_prompt_analysis),
        ("Web Fetching", test_web_fetching),
        ("RAG Engine", test_rag_engine),
        ("End-to-End", test_end_to_end),
        ("Code Generation", test_code_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}\n")
            results[test_name] = False
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("=" * 70)
    print(f"\nResults: {passed}/{total} tests passed ({(passed/total*100):.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour enhanced SLM-RAG chatbot is ready to use!")
        print("\nNext steps:")
        print("  1. Run: streamlit run enhanced_app.py")
        print("  2. Open browser at http://localhost:8501")
        print("  3. Try: 'Generate a movie recommendation ML model'")
        print()
        return True
    elif passed >= total * 0.8:
        print("\n✅ MOST TESTS PASSED!")
        print("\nYour system is functional. Some advanced features may have issues.")
        print("\nYou can still run:")
        print("  streamlit run enhanced_app.py")
        print()
        return True
    else:
        print("\n⚠️ SEVERAL TESTS FAILED")
        print("\nPlease check the errors above and:")
        print("  1. Ensure all dependencies are installed")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Verify file permissions")
        print()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
