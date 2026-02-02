# Complete Installation & Usage Guide

## Enhanced SLM-RAG Chatbot v2.0

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space (for models)
- **Internet**: Required for web scraping feature

### Step-by-Step Installation

#### 1. Download the Project

Save all files in a directory:
```
enhanced-slm-rag/
├── enhanced_app.py
├── enhanced_config.py
├── enhanced_prompt_analyzer.py
├── enhanced_rag_engine.py
├── web_content_fetcher.py
├── enhanced_requirements.txt
├── test_enhanced.py
├── ENHANCED_README.md
└── QUICKSTART_ENHANCED.md
```

#### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r enhanced_requirements.txt
```

**Important Notes:**
- First installation takes 10-15 minutes
- Downloads ~3-5 GB of AI models
- Requires stable internet connection
- May show warnings - these are normal

#### 4. Verify Installation

```bash
python test_enhanced.py
```

This runs comprehensive tests. You should see:
```
✅ Package Imports ................ PASS
✅ Component Loading .............. PASS
✅ Prompt Analysis ................ PASS
✅ Web Fetching ................... PASS
✅ RAG Engine ..................... PASS
✅ End-to-End ..................... PASS
✅ Code Generation ................ PASS

🎉 ALL TESTS PASSED!
```

---

## 🚀 Running the Application

### Basic Usage

```bash
streamlit run enhanced_app.py
```

The app opens automatically at `http://localhost:8501`

### Advanced Usage

**Custom Port:**
```bash
streamlit run enhanced_app.py --server.port 8502
```

**Headless Mode:**
```bash
streamlit run enhanced_app.py --server.headless true
```

---

## 💬 Using the Chatbot

### Example Queries

#### 1. ML Model Generation

**Query:**
```
Generate a movie recommendation ML model
```

**What You Get:**
- 📋 **Title**: Movie Recommendation ML Model
- 🎯 **Context**: What recommendation systems are and why they matter
- 🧮 **Algorithms**: Collaborative Filtering, Matrix Factorization, Neural CF
- 📊 **Dataset Info**: MovieLens, Netflix Prize, data requirements
- 💻 **Code**: 100+ lines of working Python code
- 📖 **Explanation**: Step-by-step breakdown of how it works
- 📚 **Sources**: Wikipedia, Britannica, knowledge base

**Generated Code Includes:**
```python
class MovieRecommender:
    def __init__(self):
        # Initialization
    
    def fit(self, ratings_df):
        # Training logic
    
    def recommend(self, user_id, n_recommendations=10):
        # Recommendation logic
```

#### 2. Simple Questions

**Query:**
```
What is deep learning?
```

**What You Get:**
- Clear explanation
- Key concepts
- Examples
- References from multiple sources

#### 3. Tutorial Requests

**Query:**
```
How to build a sentiment analysis model
```

**What You Get:**
- Prerequisites
- Step-by-step guide
- Code examples
- Best practices

#### 4. Comparisons

**Query:**
```
Compare Random Forest and Neural Networks
```

**What You Get:**
- Side-by-side comparison
- Pros and cons
- Use cases
- Performance considerations

---

## ⚙️ Configuration

### Sidebar Settings

#### Model Settings

**Temperature** (0.0 - 1.0)
- **0.0-0.3**: Very focused, deterministic
- **0.4-0.6**: Balanced
- **0.7-0.9**: Creative, varied
- **Recommendation**: 0.7 for general use

**Max Tokens** (500 - 3000)
- Controls response length
- More tokens = longer, detailed responses
- **Recommendation**: 
  - 1000 for quick answers
  - 2000 for code generation
  - 3000 for comprehensive tutorials

#### RAG Settings

**Top K Documents** (1 - 10)
- Number of sources to consult
- More = comprehensive but slower
- **Recommendation**:
  - 3 for simple questions
  - 5 for complex queries
  - 8-10 for research

**Enable Web Search**
- ✅ On: Fetches latest info from web
- ❌ Off: Uses only default knowledge
- **Recommendation**: Keep ON for best results

#### Display Options

**Show Prompt Analysis**
- See how AI understands your query
- View extracted parameters
- Check confidence scores

**Show Web Sources**
- See which websites were used
- View source URLs
- Check relevance scores

**Show Processing Steps**
- Real-time progress updates
- Useful for debugging
- Can be disabled for cleaner UI

---

## 🎨 Advanced Configuration

### Customizing Web Sources

Edit `enhanced_config.py`:

```python
KNOWLEDGE_SOURCES = {
    "wikipedia": {
        "base_url": "https://en.wikipedia.org/wiki/",
        "api_url": "https://en.wikipedia.org/w/api.php",
        "enabled": True,  # Set to False to disable
        "priority": 1
    },
    "britannica": {
        "base_url": "https://www.britannica.com/",
        "enabled": True,
        "priority": 2
    },
    "infoplease": {
        "base_url": "https://www.infoplease.com/",
        "enabled": True,
        "priority": 3
    }
}
```

### Adding Custom Knowledge

Add to `ENHANCED_KNOWLEDGE_BASE` in `enhanced_config.py`:

```python
{
    "content": """Your custom knowledge here.
    Can be multiple paragraphs with detailed information.""",
    "metadata": {
        "category": "Your Category",
        "source": "your_source",
        "priority": "high"
    }
}
```

### API Integration (Optional)

For enhanced capabilities, add API keys:

```python
# Gemini API
GEMINI_API_KEY = "your-api-key-here"

# HuggingFace API
HUGGINGFACE_API_KEY = "your-api-key-here"
```

### Performance Tuning

```python
# Caching
ENABLE_WEB_CACHE = True
WEB_CACHE_DURATION = 3600  # 1 hour

# Web fetching
MAX_WEB_PAGES_PER_QUERY = 3
WEB_CONTENT_TIMEOUT = 10  # seconds

# Response generation
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Model failed to load"

**Symptoms:**
- Warning about using rule-based analysis
- Responses still work but less sophisticated

**Solutions:**
- System automatically falls back - this is okay
- For better quality: ensure 8GB+ RAM
- Try smaller model: Edit `enhanced_config.py`
  ```python
  PROMPT_ANALYSIS_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  ```

#### 2. Slow Performance

**CPU Mode (No GPU):**
- Prompt Analysis: 2-5 seconds
- Web Fetch: 3-8 seconds (first time), instant (cached)
- Response Generation: 15-30 seconds

**Solutions:**
- Reduce max_tokens to 1000
- Decrease top_k to 3
- Enable caching (default: ON)
- Close other applications

**GPU Mode (With CUDA):**
- 3-5x faster than CPU
- Install CUDA version of PyTorch:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

#### 3. Web Scraping Errors

**Symptoms:**
- "Could not fetch web content"
- Still gets responses from default knowledge

**Solutions:**
- Check internet connection
- Increase timeout in config
- Disable problematic sources
- Use cached content (automatic)

#### 4. Out of Memory

**Symptoms:**
- System crashes
- "Out of memory" error

**Solutions:**
- Reduce max_tokens to 500-800
- Use CPU mode (automatic fallback)
- Clear cache:
  ```python
  import os
  import shutil
  shutil.rmtree('.web_cache', ignore_errors=True)
  shutil.rmtree('.embedding_cache', ignore_errors=True)
  ```
- Restart application

#### 5. Import Errors

**Symptoms:**
- "Module not found"
- Import failures

**Solutions:**
```bash
# Reinstall dependencies
pip install --upgrade -r enhanced_requirements.txt

# If issues persist, try:
pip install --no-cache-dir -r enhanced_requirements.txt
```

---

## 📊 Performance Optimization

### For Fastest Responses

```python
# In enhanced_config.py
MAX_WEB_PAGES_PER_QUERY = 1
DEFAULT_TOP_K = 2
DEFAULT_MAX_TOKENS = 800
ENABLE_WEB_CACHE = True
```

**In UI:**
- Temperature: 0.5
- Max Tokens: 800
- Top K: 2-3

### For Best Quality

```python
# In enhanced_config.py
MAX_WEB_PAGES_PER_QUERY = 3
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 2000
USE_WEB_CONTENT = True
```

**In UI:**
- Temperature: 0.7
- Max Tokens: 2000
- Top K: 5-8
- Enable Web Search: ✅

### For Code Generation

```python
# In enhanced_config.py
DEFAULT_MAX_TOKENS = 2500
DEFAULT_TOP_K = 5
```

**In UI:**
- Temperature: 0.5-0.6 (more focused)
- Max Tokens: 2000-2500
- Top K: 5
- Enable Web Search: ✅

---

## 💾 Data Management

### Cache Locations

```
.web_cache/          # Web content cache
.embedding_cache/    # Embedding cache
conversations/       # Saved conversations
```

### Cache Management

**View Cache Size:**
```python
import os
for dir in ['.web_cache', '.embedding_cache']:
    if os.path.exists(dir):
        size = sum(os.path.getsize(os.path.join(dir, f)) 
                  for f in os.listdir(dir))
        print(f"{dir}: {size / 1024 / 1024:.2f} MB")
```

**Clear Cache:**
```bash
# Windows
rmdir /s /q .web_cache .embedding_cache

# macOS/Linux
rm -rf .web_cache .embedding_cache
```

**Export Conversations:**
- Click "Export" in sidebar
- Download as JSON
- Can be reimported later

---

## 🎓 Best Practices

### Query Formulation

**✅ Good Queries:**
- "Generate a movie recommendation ML model"
- "How to implement gradient descent in Python"
- "Compare supervised and unsupervised learning with examples"

**❌ Avoid:**
- "ML" (too vague)
- Single word queries
- Extremely long queries (>200 words)

### Using Features

1. **Enable Analysis Display** when learning
2. **Enable Web Search** for current topics
3. **Export conversations** for important sessions
4. **Adjust temperature** based on task:
   - Factual: 0.3-0.5
   - Creative: 0.7-0.9

### Workflow Tips

1. Start with simple queries to warm up cache
2. Refine queries based on analysis feedback
3. Use follow-up questions for depth
4. Export valuable conversations

---

## 🔒 Security & Privacy

### Data Privacy

- ✅ All processing happens locally
- ✅ No data sent to external servers (except web scraping)
- ✅ Conversations stored locally only
- ✅ No tracking or analytics

### Web Scraping

- Only fetches from reputable sources
- Respects robots.txt
- Caches to minimize requests
- Educational use only

---

## 🆘 Getting Help

### Resources

1. **Documentation**: `ENHANCED_README.md`
2. **Quick Start**: `QUICKSTART_ENHANCED.md`
3. **Config Guide**: `enhanced_config.py` (commented)
4. **Test Suite**: `python test_enhanced.py`

### Self-Diagnosis

Run the test suite:
```bash
python test_enhanced.py
```

Check logs:
- Streamlit terminal output
- Error messages in UI
- Analysis display

---

## 🎉 You're All Set!

Start the application:
```bash
streamlit run enhanced_app.py
```

Try this first query:
```
Generate a movie recommendation ML model
```

Enjoy your enhanced AI chatbot! 🚀
