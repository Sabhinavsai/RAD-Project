# SLM-RAG Chatbot

A sophisticated chatbot built with Small Language Models (SLM) for prompt analysis and Retrieval Augmented Generation (RAG) for contextual response generation.

## 🌟 Features

- **Intelligent Prompt Analysis**: Uses SLM to break down and understand user prompts
- **RAG-powered Responses**: Generates accurate responses using retrieved relevant documents
- **Interactive UI**: Clean Streamlit interface with real-time analysis
- **JSON Output**: Structured breakdown of prompts including intent, entities, keywords
- **Configurable**: Adjustable temperature, token limits, and retrieval parameters
- **Conversation Export**: Save and export conversation history
- **Visual Analytics**: Real-time display of prompt analysis metrics

## 📋 Architecture

```
User Input → SLM (Prompt Analysis) → JSON Structure → RAG Engine → Response
                                                            ↓
                                                    Vector Database
                                                            ↓
                                                    Retrieved Docs
```

### Components

1. **Prompt Analyzer** (`prompt_analyzer.py`)
   - Uses Microsoft Phi-2 or similar SLM
   - Extracts: Intent, Entities, Keywords, Domain, Sentiment
   - Falls back to rule-based analysis if model unavailable

2. **RAG Engine** (`rag_engine.py`)
   - Embedding: sentence-transformers/all-MiniLM-L6-v2
   - Vector Store: FAISS for similarity search
   - Generation: Microsoft Phi-2 or template-based

3. **Streamlit App** (`app.py`)
   - User interface
   - Real-time analysis display
   - Conversation management

4. **Utilities** (`utils.py`)
   - Helper functions
   - Data processing
   - Export/Import capabilities

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for GPU usage)
- (Optional) CUDA-capable GPU for faster inference

### Setup Steps

1. **Clone or download this project**

```bash
cd slm-rag-chatbot
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Note**: First run will download models (~2-3GB). This may take a few minutes.

## 🎯 Usage

### Basic Usage

1. **Start the application**

```bash
streamlit run app.py
```

2. **Access the interface**
   - Browser will open automatically at `http://localhost:8501`
   - If not, manually navigate to that URL

3. **Interact with the chatbot**
   - Type your question in the chat input
   - View real-time prompt analysis
   - See retrieved documents (optional)
   - Get contextual responses

### Configuration

Use the sidebar to adjust:

- **Temperature**: Controls response randomness (0.0 = deterministic, 1.0 = creative)
- **Max Tokens**: Maximum length of generated response
- **Top K Documents**: Number of documents to retrieve
- **Analysis Display**: Show/hide prompt analysis
- **JSON Display**: Show/hide raw JSON breakdown

### Features

#### Prompt Analysis

Every user message is analyzed to extract:

```json
{
  "intent": "information_seeking",
  "keywords": ["artificial", "intelligence", "machine", "learning"],
  "entities": [
    {"type": "proper_noun", "value": "Python"},
    {"type": "number", "value": "2024"}
  ],
  "complexity": "medium",
  "domain": "technology",
  "question_type": "factual",
  "sentiment": "neutral",
  "requires_context": false
}
```

#### RAG Response Generation

1. Analyzes your prompt
2. Retrieves relevant documents from knowledge base
3. Generates contextual response using retrieved information
4. Shows source documents with relevance scores

## 📚 Customization

### Adding Your Own Documents

Edit `rag_engine.py` and modify the `_initialize_knowledge_base()` method:

```python
def _initialize_knowledge_base(self):
    sample_docs = [
        {
            "content": "Your document content here",
            "metadata": {"category": "YourCategory", "source": "your_source"}
        },
        # Add more documents...
    ]
    self.add_documents(sample_docs)
```

Or add documents programmatically:

```python
from rag_engine import RAGEngine

rag = RAGEngine()
new_docs = [
    {
        "content": "New information",
        "metadata": {"category": "custom"}
    }
]
rag.add_documents(new_docs)
```

### Changing Models

#### For Prompt Analysis (SLM)

In `prompt_analyzer.py`, change the model:

```python
analyzer = PromptAnalyzer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

Recommended SLMs:
- `microsoft/phi-2` (2.7B - default)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B - faster)
- `stabilityai/stablelm-2-zephyr-1_6b` (1.6B)

#### For Response Generation

In `rag_engine.py`, change the model:

```python
rag = RAGEngine(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    generation_model="microsoft/phi-2"
)
```

### Adding Custom Analysis Rules

Modify the rule-based analysis in `prompt_analyzer.py`:

```python
def _detect_intent(self, prompt: str) -> str:
    # Add your custom intent detection logic
    if "your_keyword" in prompt:
        return "your_custom_intent"
    # ... existing code
```

## 📊 Example Interactions

### Example 1: Factual Question

**User**: "What is machine learning?"

**Analysis**:
```json
{
  "intent": "information_seeking",
  "domain": "technology",
  "complexity": "simple",
  "question_type": "factual"
}
```

**Response**: [Retrieved from knowledge base with ML definition]

### Example 2: Complex Query

**User**: "Can you explain how RAG systems work and why they're better than traditional chatbots?"

**Analysis**:
```json
{
  "intent": "explanation",
  "domain": "technology",
  "complexity": "complex",
  "question_type": "explanatory"
}
```

**Response**: [Multi-document retrieval with comprehensive explanation]

## 🔧 Troubleshooting

### Model Loading Issues

If models fail to load:

1. **Check available RAM**: Models need 4-8GB
2. **Use smaller models**: Try TinyLlama instead of Phi-2
3. **CPU-only mode**: System will automatically fall back to CPU
4. **Rule-based fallback**: If model fails, rule-based analysis is used

### Performance Issues

1. **Reduce max_tokens**: Lower from 500 to 200-300
2. **Decrease top_k**: Use 1-2 documents instead of 3-5
3. **Use GPU**: Install CUDA and `torch` with CUDA support

### CUDA/GPU Issues

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
slm-rag-chatbot/
│
├── app.py                  # Main Streamlit application
├── prompt_analyzer.py      # SLM-based prompt analysis
├── rag_engine.py          # RAG implementation
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── conversations/        # Saved conversations (created on first save)
├── feedback/            # User feedback logs (created on feedback)
└── .streamlit/          # Streamlit config (optional)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Add more sophisticated entity extraction
- Implement conversation memory
- Add support for document upload
- Integrate with external APIs
- Improve response quality
- Add multilingual support

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **Hugging Face**: For transformers and model hosting
- **Streamlit**: For the web framework
- **Microsoft**: For Phi-2 model
- **Sentence Transformers**: For embedding models
- **FAISS**: For vector similarity search

## 📞 Support

For issues or questions:

1. Check the Troubleshooting section
2. Review the example code
3. Check model documentation on Hugging Face

## 🔄 Updates

### Version 1.0.0
- Initial release
- SLM-based prompt analysis
- RAG response generation
- Streamlit UI
- JSON export functionality

---

**Happy Chatting! 🤖**
