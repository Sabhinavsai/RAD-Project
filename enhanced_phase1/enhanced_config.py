"""
Enhanced Configuration file for SLM-RAG Chatbot
Supports multiple AI providers and web content retrieval
"""

# ============================================================================
# API CONFIGURATIONS
# ============================================================================

# Gemini API Configuration
GEMINI_API_KEY = ""  # Add your Gemini API key here
GEMINI_MODEL = "gemini-pro"

# HuggingFace API Configuration
HUGGINGFACE_API_KEY = ""  # Add your HuggingFace API key here
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# LangChain Configuration
USE_LANGCHAIN = True
LANGCHAIN_MODEL_PROVIDER = "huggingface"  # Options: "huggingface", "gemini", "local"

# ============================================================================
# WEB CONTENT SOURCES
# ============================================================================

# Reliable knowledge sources for web scraping
KNOWLEDGE_SOURCES = {
    "wikipedia": {
        "base_url": "https://en.wikipedia.org/wiki/",
        "api_url": "https://en.wikipedia.org/w/api.php",
        "enabled": True,
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
    },
    "archive": {
        "base_url": "https://archive.org/",
        "enabled": False,
        "priority": 4
    }
}

# ============================================================================
# RAG ENGINE CONFIGURATION
# ============================================================================

# Use both default documents and web content
USE_DEFAULT_DOCUMENTS = True
USE_WEB_CONTENT = True

# Web scraping settings
MAX_WEB_PAGES_PER_QUERY = 3
WEB_CONTENT_TIMEOUT = 10  # seconds
WEB_CONTENT_MAX_LENGTH = 5000  # characters

# Document retrieval settings
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# ============================================================================
# PROMPT ANALYSIS CONFIGURATION
# ============================================================================

# Enhanced intent detection
INTENT_PATTERNS = {
    "code_generation": ["generate", "create", "build", "develop", "code", "script", "program"],
    "model_creation": ["model", "ml model", "machine learning", "ai model", "train"],
    "explanation": ["explain", "describe", "what is", "how does", "tell me about"],
    "recommendation": ["recommend", "suggest", "advice", "best"],
    "tutorial": ["tutorial", "guide", "how to", "step by step", "learn"],
    "analysis": ["analyze", "compare", "evaluate", "assess"],
    "documentation": ["document", "documentation", "reference", "api"]
}

# Context extraction patterns
CONTEXT_PATTERNS = {
    "algorithms": ["algorithm", "method", "approach", "technique"],
    "datasets": ["dataset", "data", "training data", "examples"],
    "implementation": ["implementation", "code", "example", "demo"],
    "workflow": ["workflow", "process", "pipeline", "steps"]
}

# ============================================================================
# TEXT GENERATION CONFIGURATION
# ============================================================================

# Output structure templates
OUTPUT_TEMPLATE = {
    "code_generation": {
        "sections": ["title", "context", "algorithms", "dataset", "code", "explanation"],
        "include_dataset": False  # Phase 2 feature
    },
    "model_creation": {
        "sections": ["title", "context", "algorithms", "dataset_info", "sample_code", "detailed_explanation"],
        "include_dataset": False  # Phase 2 feature
    },
    "explanation": {
        "sections": ["title", "overview", "key_concepts", "examples", "references"],
        "include_dataset": False
    },
    "tutorial": {
        "sections": ["title", "introduction", "prerequisites", "steps", "code_examples", "summary"],
        "include_dataset": False
    }
}

# Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.3
}

# Default UI-friendly aliases for generation settings
DEFAULT_TEMPERATURE = GENERATION_CONFIG.get("temperature", 0.7)
DEFAULT_MAX_TOKENS = GENERATION_CONFIG.get("max_tokens", 2000)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Primary models
PROMPT_ANALYSIS_MODEL = "microsoft/phi-2"
RESPONSE_GENERATION_MODEL = "microsoft/phi-2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fallback models
FALLBACK_MODELS = {
    "analysis": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "gpt2"],
    "generation": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "gpt2"]
}

# ============================================================================
# ENHANCED KNOWLEDGE BASE
# ============================================================================

# Domain-specific knowledge sources
DOMAIN_SOURCES = {
    "machine_learning": {
        "keywords": ["ml", "machine learning", "model", "training", "neural network"],
        "sources": ["wikipedia", "britannica"],
        "default_docs": True
    },
    "data_science": {
        "keywords": ["data", "analytics", "statistics", "visualization"],
        "sources": ["wikipedia", "infoplease"],
        "default_docs": True
    },
    "programming": {
        "keywords": ["code", "programming", "python", "javascript", "development"],
        "sources": ["wikipedia"],
        "default_docs": True
    },
    "general": {
        "keywords": [],
        "sources": ["wikipedia", "britannica", "infoplease"],
        "default_docs": True
    }
}

# ============================================================================
# ENHANCED DEFAULT KNOWLEDGE BASE
# ============================================================================

ENHANCED_KNOWLEDGE_BASE = [
    # Machine Learning
    {
        "content": """Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key algorithms include:
        
        1. Supervised Learning: Linear Regression, Logistic Regression, Decision Trees, Random Forests, SVM, Neural Networks
        2. Unsupervised Learning: K-Means Clustering, Hierarchical Clustering, PCA, Association Rules
        3. Reinforcement Learning: Q-Learning, Deep Q-Networks, Policy Gradients
        
        Common datasets: MNIST, CIFAR-10, ImageNet, Titanic, Iris, Boston Housing
        
        Popular frameworks: TensorFlow, PyTorch, Scikit-learn, Keras""",
        "metadata": {"category": "Machine Learning", "source": "knowledge_base", "priority": "high"}
    },
    
    # Recommendation Systems
    {
        "content": """Recommendation Systems are ML models that suggest items to users based on their preferences and behavior. Types include:
        
        1. Collaborative Filtering: User-based, Item-based, Matrix Factorization
        2. Content-Based Filtering: Using item features
        3. Hybrid Systems: Combining multiple approaches
        
        Key algorithms:
        - Alternating Least Squares (ALS)
        - Singular Value Decomposition (SVD)
        - Neural Collaborative Filtering
        - Deep Learning approaches (Neural Networks)
        
        Popular datasets: MovieLens, Netflix Prize, Amazon Reviews, Last.fm
        
        Implementation libraries: Surprise, LightFM, TensorFlow Recommenders, implicit""",
        "metadata": {"category": "Recommendation Systems", "source": "knowledge_base", "priority": "high"}
    },
    
    # Deep Learning
    {
        "content": """Deep Learning uses neural networks with multiple layers to learn hierarchical representations of data. Architectures include:
        
        1. Feedforward Networks: Basic neural networks for classification/regression
        2. Convolutional Neural Networks (CNN): For image processing
        3. Recurrent Neural Networks (RNN/LSTM/GRU): For sequential data
        4. Transformers: For NLP and beyond
        5. Autoencoders: For dimensionality reduction and generation
        6. GANs: For generative tasks
        
        Frameworks: PyTorch, TensorFlow, JAX, MXNet
        Common datasets: ImageNet, COCO, WikiText, Common Crawl""",
        "metadata": {"category": "Deep Learning", "source": "knowledge_base", "priority": "high"}
    },
    
    # NLP
    {
        "content": """Natural Language Processing (NLP) enables computers to understand and generate human language. Key tasks:
        
        1. Text Classification: Sentiment analysis, spam detection
        2. Named Entity Recognition: Identifying entities in text
        3. Machine Translation: Translation between languages
        4. Question Answering: Answering questions based on context
        5. Text Generation: Creating coherent text
        
        Modern approaches: BERT, GPT, T5, RoBERTa, XLNet
        Datasets: GLUE, SQuAD, IMDB, AG News, WikiText""",
        "metadata": {"category": "NLP", "source": "knowledge_base", "priority": "high"}
    },
    
    # Computer Vision
    {
        "content": """Computer Vision enables machines to interpret and understand visual information. Applications:
        
        1. Image Classification: Categorizing images
        2. Object Detection: YOLO, R-CNN, SSD, RetinaNet
        3. Segmentation: U-Net, Mask R-CNN, DeepLab
        4. Face Recognition: FaceNet, ArcFace
        5. Image Generation: StyleGAN, DALL-E, Stable Diffusion
        
        Datasets: ImageNet, COCO, Pascal VOC, CelebA, Places365
        Libraries: OpenCV, PIL, scikit-image, torchvision""",
        "metadata": {"category": "Computer Vision", "source": "knowledge_base", "priority": "high"}
    },
    
    # Data Preprocessing
    {
        "content": """Data Preprocessing is crucial for ML model success. Steps include:
        
        1. Data Cleaning: Handling missing values, outliers, duplicates
        2. Feature Scaling: Normalization, Standardization, Min-Max scaling
        3. Feature Engineering: Creating new features, polynomial features
        4. Encoding: One-hot encoding, Label encoding, Target encoding
        5. Feature Selection: Correlation analysis, RFE, L1 regularization
        
        Tools: Pandas, NumPy, Scikit-learn preprocessing, Feature-engine""",
        "metadata": {"category": "Data Preprocessing", "source": "knowledge_base", "priority": "medium"}
    },
    
    # Model Evaluation
    {
        "content": """Model Evaluation metrics help assess ML model performance:
        
        Classification Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - ROC-AUC, PR-AUC
        - Confusion Matrix
        
        Regression Metrics:
        - MSE, RMSE, MAE
        - R-squared, Adjusted R-squared
        - MAPE
        
        Cross-validation: K-fold, Stratified K-fold, Leave-one-out
        Libraries: Scikit-learn metrics, MLflow, TensorBoard""",
        "metadata": {"category": "Model Evaluation", "source": "knowledge_base", "priority": "medium"}
    }
]

# ============================================================================
# CACHING AND PERFORMANCE
# ============================================================================

# Enable caching for web content
ENABLE_WEB_CACHE = True
WEB_CACHE_DURATION = 3600  # seconds (1 hour)
WEB_CACHE_DIR = ".web_cache"

# Enable caching for embeddings
ENABLE_EMBEDDING_CACHE = True
EMBEDDING_CACHE_DIR = ".embedding_cache"

# Performance settings
BATCH_SIZE = 8
MAX_PARALLEL_REQUESTS = 3

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

# Enhanced logging
VERBOSE_LOGGING = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "rag_chatbot.log"

# Debug mode
DEBUG_MODE = False
SAVE_INTERMEDIATE_RESULTS = True

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Enhanced UI text
UI_TEXT = {
    "welcome_message": """👋 Welcome to Enhanced SLM-RAG Chatbot!

I can help you with:
- 🤖 ML Model Development (e.g., "Generate a movie recommendation ML model")
- 💻 Code Generation
- 📚 Detailed Explanations
- 📊 Data Analysis Guidance
- 🎓 Learning Tutorials

Ask me anything!""",
    "processing_message": "🔍 Analyzing your request...",
    "web_search_message": "🌐 Searching web sources...",
    "generating_message": "✍️ Generating comprehensive response...",
}

# Display options
SHOW_WEB_SOURCES = True
SHOW_CONFIDENCE_SCORES = True
SHOW_PROCESSING_STEPS = True
# UI default for showing analysis
SHOW_ANALYSIS_DEFAULT = True

# ============================================================================
# ERROR HANDLING
# ============================================================================

ERROR_MESSAGES = {
    "model_load_failed": "⚠️ Using fallback model. Quality may vary.",
    "web_fetch_failed": "⚠️ Could not fetch web content. Using default knowledge.",
    "api_error": "⚠️ API error occurred. Using local models.",
    "generation_failed": "❌ Generation failed. Please try rephrasing your request.",
}

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Phase 1 Features
ENABLE_WEB_SCRAPING = True
ENABLE_ENHANCED_ANALYSIS = True
ENABLE_STRUCTURED_OUTPUT = True
ENABLE_CODE_GENERATION = True

# Phase 2 Features (Coming Soon)
ENABLE_DATASET_GENERATION = False
ENABLE_MODEL_TRAINING = False
ENABLE_INTERACTIVE_CODING = False

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_FORMATS = ["json", "markdown", "python", "html"]
DEFAULT_EXPORT_FORMAT = "markdown"
INCLUDE_SOURCES = True
INCLUDE_METADATA = True
