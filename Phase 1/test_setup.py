"""
Test script to verify the SLM-RAG Chatbot installation and basic functionality.
Run this after installing dependencies to ensure everything is working.
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    required_packages = [
        ("streamlit", "Streamlit"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name} - NOT FOUND")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n[X] Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All packages installed successfully!")
        return True

def test_component_loading():
    """Test if components can be loaded"""
    print("\nTesting component loading...")
    
    try:
        from prompt_analyzer import PromptAnalyzer
        print("  [OK] PromptAnalyzer imported")
    except Exception as e:
        print(f"  [FAIL] PromptAnalyzer failed: {e}")
        return False
    
    try:
        from rag_engine import RAGEngine
        print("  [OK] RAGEngine imported")
    except Exception as e:
        print(f"  [FAIL] RAGEngine failed: {e}")
        return False
    
    try:
        from utils import format_analysis_for_display
        print("  [OK] Utils imported")
    except Exception as e:
        print(f"  [FAIL] Utils failed: {e}")
        return False
    
    print("\n[OK] All components loaded successfully!")
    return True

def test_basic_functionality():
    """Test basic functionality without loading heavy models"""
    print("\nTesting basic functionality...")
    
    try:
        from prompt_analyzer import PromptAnalyzer
        
        # Initialize with rule-based analysis (no model loading)
        analyzer = PromptAnalyzer()
        
        # Test analysis
        test_prompt = "What is machine learning?"
        analysis = analyzer._analyze_rule_based(test_prompt)
        
        assert 'intent' in analysis, "Intent not found in analysis"
        assert 'keywords' in analysis, "Keywords not found in analysis"
        assert 'entities' in analysis, "Entities not found in analysis"
        assert 'complexity' in analysis, "Complexity not found in analysis"
        
        print("  [OK] Prompt analysis working")
        
    except Exception as e:
        print(f"  [FAIL] Prompt analysis failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model (lighter than full LLM)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(["test sentence"])
        
        assert embeddings.shape[0] == 1, "Embedding shape incorrect"
        
        print("  [OK] Embedding model working")
        
    except Exception as e:
        print(f"  [FAIL] Embedding model failed: {e}")
        return False
    
    print("\n[OK] Basic functionality working!")
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    import os
    
    required_files = [
        "app.py",
        "prompt_analyzer.py",
        "rag_engine.py",
        "utils.py",
        "config.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file} - NOT FOUND")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n[X] Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n[OK] All required files present!")
        return True

def test_torch_device():
    """Test PyTorch device availability"""
    print("\nTesting PyTorch device...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA available - GPU: {gpu_name}")
        else:
            device = "cpu"
            print(f"  [INFO] CUDA not available - Using CPU")
        
        print(f"  Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] PyTorch device test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("SLM-RAG Chatbot - Installation Test")
    print("=" * 70)
    print()
    
    results = {
        "File Structure": test_file_structure(),
        "Package Imports": test_imports(),
        "Component Loading": test_component_loading(),
        "PyTorch Device": test_torch_device(),
        "Basic Functionality": test_basic_functionality(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results.items():
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n[OK] All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open browser at http://localhost:8501")
        print("  3. Start chatting!")
    else:
        print("\n[WARN] Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  1. Ensure all packages are installed: pip install -r requirements.txt")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Verify all files are present in the directory")
    
    print()
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
