# Enhanced SLM-RAG Chatbot v2.0

An advanced AI chatbot that combines Small Language Models (SLM), Retrieval Augmented Generation (RAG), and web content integration to provide comprehensive, structured responses with code generation capabilities.

## 🌟 Key Features

### Phase 1 Features (Current)
- ✅ **Multi-Source RAG**: Combines default knowledge base with real-time web content
- ✅ **Web Integration**: Fetches content from Wikipedia, Britannica, and Infoplease
- ✅ **Advanced Prompt Analysis**: Sophisticated intent detection and parameter extraction
- ✅ **Structured Output Generation**: Organized responses with clear sections
- ✅ **Code Generation**: Creates working code examples for ML models and algorithms
- ✅ **Smart Caching**: Caches web content to improve performance
- ✅ **Confidence Scoring**: Shows confidence levels for intent and domain detection

### Coming in Phase 2
- 🔄 **Dataset Generation**: Automatic creation of sample datasets
- 🔄 **Interactive Coding**: Real-time code execution and testing
- 🔄 **Model Training**: End-to-end ML model training capabilities

## 🎯 What Makes This Different?

### Example Query: "Generate a movie recommendation ML model"

**Traditional Chatbot Response:**
> "A movie recommendation system uses collaborative filtering..."

**Enhanced SLM-RAG Response:**
```
# Movie Recommendation ML Model

## 📋 Context
Movie recommendation systems analyze user preferences and viewing history...

## 🧮 Algorithms and Approaches
1. Collaborative Filtering
2. Content-Based Filtering
3. Matrix Factorization
4. Neural Collaborative Filtering

## 📊 Dataset Information
**Dataset Requirements:**
- User-item interaction data
- Popular Datasets: MovieLens, Netflix Prize

## 💻 Sample Implementation
[Complete working Python code]

## 📖 Detailed Explanation
[Step-by-step breakdown of how it works]

## 📚 Sources
Wikipedia, Britannica
```

## 🚀 Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r enhanced_requirements.txt
```

2. **Run the Application**
```bash
streamlit run enhanced_app.py
```

3. **Access the Interface**
Open your browser to `http://localhost:8501`

### First Run
On first run, models will download (~3-5 GB). This may take a few minutes.

## 📖 Usage Guide

### Basic Usage

Simply type your question or request. The system handles:

1. **Information Seeking**
   - "What is machine learning?"
   - "Explain neural networks"

2. **Code Generation**
   - "Generate a movie recommendation ML model"
   - "Create a classification algorithm"
   - "Build a clustering system"

3. **Tutorials**
   - "How to implement gradient descent"
   - "Guide to building a chatbot"

4. **Analysis**
   - "Compare supervised vs unsupervised learning"
   - "What's the difference between CNN and RNN?"

### Advanced Features

#### Parameter Extraction
When you request "Generate a movie recommendation ML model", the system:
1. Extracts: `subject="movie recommendation"`, `type="ML model"`
2. Searches web for relevant content
3. Generates structured output with code

#### Web Content Integration
- Automatically searches Wikipedia, Britannica, and Infoplease
- Caches results for faster subsequent queries
- Combines web content with default knowledge base
- Shows source attribution

#### Structured Outputs
For generation requests, outputs include:
- **Title**: Clear heading
- **Context**: Background information
- **Algorithms**: Relevant approaches
- **Dataset Info**: Data requirements
- **Sample Code**: Working implementation
- **Detailed Explanation**: Step-by-step breakdown
- **Sources**: References used

## ⚙️ Configuration

### Basic Settings (enhanced_config.py)

```python
# Enable/disable web scraping
USE_WEB_CONTENT = True
USE_DEFAULT_DOCUMENTS = True

# Web sources
KNOWLEDGE_SOURCES = {
    "wikipedia": {"enabled": True, "priority": 1},
    "britannica": {"enabled": True, "priority": 2},
    "infoplease": {"enabled": True, "priority": 3}
}

# Performance
MAX_WEB_PAGES_PER_QUERY = 3
WEB_CONTENT_TIMEOUT = 10
ENABLE_WEB_CACHE = True
```

### API Integration (Optional)

```python
# Add your API keys for enhanced capabilities
GEMINI_API_KEY = "your-key-here"
HUGGINGFACE_API_KEY = "your-key-here"
```

## 🏗️ Architecture

```
User Input
    ↓
Enhanced Prompt Analyzer
    ├─ Intent Detection (with confidence)
    ├─ Parameter Extraction (subject, task, type)
    ├─ Context Requirements Analysis
    └─ Search Query Generation
    ↓
Enhanced RAG Engine
    ├─ Default Document Retrieval
    ├─ Web Content Fetching
    │   ├─ Wikipedia API
    │   ├─ Britannica Scraping
    │   └─ Infoplease Scraping
    ├─ Document Re-ranking
    └─ Context Building
    ↓
Response Generator
    ├─ Structured Output (for generation requests)
    │   ├─ Title
    │   ├─ Context
    │   ├─ Algorithms
    │   ├─ Dataset Info
    │   ├─ Sample Code
    │   ├─ Detailed Explanation
    │   └─ Sources
    └─ Standard Output (for other requests)
    ↓
User Interface
```

## 📁 Project Structure

```
enhanced-slm-rag/
├── enhanced_app.py                 # Main Streamlit application
├── enhanced_config.py              # Configuration settings
├── enhanced_prompt_analyzer.py     # Advanced prompt analysis
├── enhanced_rag_engine.py          # RAG with web integration
├── web_content_fetcher.py          # Web scraping module
├── enhanced_requirements.txt       # Dependencies
├── README.md                       # This file
│
├── .web_cache/                     # Web content cache (auto-created)
├── .embedding_cache/               # Embedding cache (auto-created)
└── conversations/                  # Saved conversations (auto-created)
```

## 🔧 Customization

### Adding New Knowledge Sources

In `enhanced_config.py`:

```python
KNOWLEDGE_SOURCES = {
    "your_source": {
        "base_url": "https://yoursource.com",
        "enabled": True,
        "priority": 4
    }
}
```

Then implement in `web_content_fetcher.py`:

```python
def _fetch_your_source(self, query: str):
    # Your scraping logic here
    pass
```

### Adding Custom Knowledge

In `enhanced_config.py`, add to `ENHANCED_KNOWLEDGE_BASE`:

```python
{
    "content": "Your knowledge content here...",
    "metadata": {
        "category": "Your Category",
        "source": "your_source",
        "priority": "high"
    }
}
```

### Customizing Code Generation

In `enhanced_rag_engine.py`, modify the code generation methods:

```python
def _generate_your_code_type(self, subject: str) -> str:
    return '''
    # Your code template here
    '''
```

## 🎨 UI Customization

### Sidebar Settings
- **Temperature**: Control creativity (0.0 = precise, 1.0 = creative)
- **Max Tokens**: Response length
- **Top K Documents**: Number of sources to use
- **Enable Web Search**: Toggle web integration
- **Display Options**: Show/hide analysis, sources, processing steps

### Export Options
- Export conversations as JSON
- Include full analysis history
- Preserve metadata and sources

## 🔍 Examples

### Example 1: ML Model Generation

**Input:**
```
Generate a movie recommendation ML model
```

**Output Includes:**
- Title and context about recommendation systems
- Algorithms: Collaborative Filtering, Matrix Factorization, etc.
- Dataset info: MovieLens, Netflix Prize
- Complete working Python code (100+ lines)
- Detailed explanation of how it works
- Sources from Wikipedia and other knowledge bases

### Example 2: Tutorial Request

**Input:**
```
How to build a sentiment analysis model
```

**Output Includes:**
- Step-by-step guide
- Required libraries
- Data preprocessing steps
- Model training code
- Evaluation metrics
- Best practices

### Example 3: Comparison

**Input:**
```
Compare Random Forest and Neural Networks for classification
```

**Output Includes:**
- Side-by-side comparison
- Use cases for each
- Pros and cons
- Code examples for both
- Performance considerations

## 🐛 Troubleshooting

### Model Loading Issues

**Issue**: "Model failed to load"
**Solution**: The system falls back to rule-based analysis. You can:
- Use a smaller model (TinyLlama)
- Ensure you have 8GB+ RAM
- Install CUDA for GPU support

### Web Scraping Errors

**Issue**: "Could not fetch web content"
**Solution**: 
- Check internet connection
- Verify timeout settings in config
- Use cached content (enabled by default)
- Disable problematic sources in config

### Slow Performance

**Solutions**:
- Reduce `max_tokens` (500-1000)
- Decrease `top_k` (2-3)
- Enable caching (default: enabled)
- Use GPU if available

### Out of Memory

**Solutions**:
- Use CPU mode (automatic fallback)
- Reduce batch size in config
- Clear cache directories
- Use smaller model

## 📊 Performance

### Typical Response Times

**CPU Mode:**
- Prompt Analysis: 1-2 seconds
- Web Fetch: 2-5 seconds (first time), <0.1s (cached)
- Document Retrieval: <0.5 seconds
- Response Generation: 10-30 seconds
- **Total**: 15-40 seconds

**GPU Mode (CUDA):**
- Prompt Analysis: 0.3-0.5 seconds
- Web Fetch: 2-5 seconds (first time), <0.1s (cached)
- Document Retrieval: <0.2 seconds
- Response Generation: 2-5 seconds
- **Total**: 5-12 seconds

### Caching Benefits
- **First query**: Full web fetch + processing
- **Repeat query**: Instant (from cache)
- **Cache duration**: 1 hour (configurable)

## 🔐 Privacy & Security

- **Local Processing**: All AI processing happens locally
- **No Data Sent**: User data stays on your machine (unless using external APIs)
- **Cache Control**: Web cache stored locally, can be cleared
- **No Tracking**: No analytics or usage tracking

## 🤝 Contributing

This is an educational project. Areas for contribution:
- Additional knowledge sources
- Better code generation templates
- Improved prompt analysis
- Multi-language support
- Enhanced error handling

## 📄 License

This project is provided for educational purposes.

## 🙏 Acknowledgments

- **HuggingFace**: For transformers and models
- **Streamlit**: For the web framework
- **Wikipedia, Britannica, Infoplease**: For knowledge content
- **Microsoft**: For Phi-2 model
- **Sentence Transformers**: For embeddings
- **FAISS**: For vector similarity search

## 📞 Support

### Documentation
- Full architecture details in code comments
- Configuration options in `enhanced_config.py`
- Example usage in `README.md`

### Common Issues
- See Troubleshooting section above
- Check model compatibility on HuggingFace
- Verify all dependencies are installed

## 🔄 Changelog

### Version 2.0 (Current - Phase 1)
- ✅ Web content integration
- ✅ Multi-source RAG
- ✅ Enhanced prompt analysis
- ✅ Structured output generation
- ✅ Code generation capabilities
- ✅ Smart caching
- ✅ Confidence scoring

### Version 1.0 (Original)
- Basic SLM analysis
- Simple RAG retrieval
- Template responses

---

**Made with ❤️ using AI and Open Source**

For questions, issues, or contributions, please refer to the documentation and code comments.
