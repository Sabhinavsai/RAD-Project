"""
Configuration file for SLM-RAG Chatbot
Modify these settings to customize your chatbot behavior
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Small Language Model for Prompt Analysis
# Options: 
#   - "microsoft/phi-2" (2.7B, best quality, slower)
#   - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B, faster)
#   - "stabilityai/stablelm-2-zephyr-1_6b" (1.6B, balanced)
PROMPT_ANALYSIS_MODEL = "microsoft/phi-2"

# Model for Response Generation (same options as above)
RESPONSE_GENERATION_MODEL = "microsoft/phi-2"

# Embedding Model for Document Retrieval
# Options:
#   - "sentence-transformers/all-MiniLM-L6-v2" (fast, good quality)
#   - "sentence-transformers/all-mpnet-base-v2" (slower, better quality)
#   - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# DEFAULT GENERATION PARAMETERS
# ============================================================================

# Temperature: Controls randomness (0.0 = deterministic, 1.0 = very creative)
DEFAULT_TEMPERATURE = 0.7

# Max tokens in generated response
DEFAULT_MAX_TOKENS = 500

# Number of documents to retrieve for RAG
DEFAULT_TOP_K = 3

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Show prompt analysis by default
SHOW_ANALYSIS_DEFAULT = True

# Show JSON breakdown by default
SHOW_JSON_DEFAULT = False

# Page title
PAGE_TITLE = "SLM-RAG Chatbot"

# Page icon (emoji)
PAGE_ICON = "🤖"

# Sidebar initial state ("expanded" or "collapsed")
SIDEBAR_STATE = "expanded"

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Maximum number of keywords to extract
MAX_KEYWORDS = 10

# Maximum number of entities to display
MAX_ENTITIES_DISPLAY = 5

# Complexity thresholds (word count)
COMPLEXITY_THRESHOLDS = {
    "simple": 10,      # < 10 words = simple
    "medium": 30,      # 10-30 words = medium
    "complex": 999999  # > 30 words = complex
}

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

# Vector database dimension (auto-set based on embedding model)
VECTOR_DIMENSION = None  # Auto-detected

# Similarity score threshold (0-1, higher = more relevant)
SIMILARITY_THRESHOLD = 0.0

# Maximum number of documents in knowledge base
MAX_KNOWLEDGE_BASE_SIZE = 10000

# Batch size for document processing
DOCUMENT_BATCH_SIZE = 10

# ============================================================================
# CONVERSATION CONFIGURATION
# ============================================================================

# Maximum conversation history to keep
MAX_CONVERSATION_HISTORY = 100

# Save conversations automatically
AUTO_SAVE_CONVERSATIONS = False

# Conversation save directory
CONVERSATION_SAVE_DIR = "conversations"

# Feedback save directory
FEEDBACK_SAVE_DIR = "feedback"

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Use GPU if available
USE_GPU = True

# Torch data type ("float32" or "float16")
# float16 is faster but requires GPU
TORCH_DTYPE = "float16"  # Will fallback to float32 on CPU

# Enable model caching
ENABLE_MODEL_CACHE = True

# Cache directory
CACHE_DIR = ".cache"

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable rule-based fallback if model loading fails
ENABLE_RULE_BASED_FALLBACK = True

# Enable conversation context in knowledge base
ENABLE_CONVERSATION_CONTEXT = True

# Enable export functionality
ENABLE_EXPORT = True

# Enable feedback collection
ENABLE_FEEDBACK = True

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Token limit for context window
CONTEXT_TOKEN_LIMIT = 1024

# Minimum similarity score to include document
MIN_SIMILARITY_SCORE = 0.1

# Enable verbose logging
VERBOSE_LOGGING = False

# Generation parameters
GENERATION_CONFIG = {
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# Stop sequences for generation
STOP_SEQUENCES = ["User:", "Human:", "\n\n\n"]

# ============================================================================
# DOMAIN-SPECIFIC KEYWORDS (for domain detection)
# ============================================================================

DOMAIN_KEYWORDS = {
    "technology": ["computer", "software", "code", "programming", "ai", "ml", 
                   "tech", "algorithm", "data", "digital", "cyber"],
    "science": ["science", "research", "experiment", "theory", "hypothesis",
                "biology", "chemistry", "physics", "study"],
    "business": ["business", "market", "sales", "revenue", "company", 
                 "strategy", "finance", "profit", "customer"],
    "health": ["health", "medical", "doctor", "disease", "treatment", 
               "patient", "medicine", "hospital", "diagnosis"],
    "education": ["learn", "study", "education", "teaching", "course", 
                  "school", "university", "student", "professor"],
    "general": []
}

# ============================================================================
# INTENT KEYWORDS (for intent detection)
# ============================================================================

INTENT_KEYWORDS = {
    "information_seeking": ["what", "who", "where", "when", "which"],
    "assistance_request": ["how to", "can you", "please", "help me", "assist"],
    "creation": ["create", "generate", "write", "make", "build", "design"],
    "explanation": ["explain", "describe", "tell me about", "clarify"],
    "analysis": ["analyze", "compare", "evaluate", "assess", "examine"],
}

# ============================================================================
# SENTIMENT KEYWORDS
# ============================================================================

SENTIMENT_KEYWORDS = {
    "positive": ["good", "great", "excellent", "amazing", "wonderful", 
                 "love", "like", "fantastic", "awesome", "perfect"],
    "negative": ["bad", "terrible", "awful", "hate", "dislike", "problem", 
                 "issue", "wrong", "poor", "worst"],
}

# ============================================================================
# CUSTOM KNOWLEDGE BASE
# ============================================================================

# Define your custom documents here
CUSTOM_KNOWLEDGE_BASE = [
    {
        "content": "Add your custom document content here",
        "metadata": {"category": "custom", "source": "manual"}
    },
    # Add more documents as needed
]

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

# Export format options
EXPORT_FORMATS = ["json", "txt", "csv"]

# Default export format
DEFAULT_EXPORT_FORMAT = "json"

# Include metadata in exports
INCLUDE_METADATA_IN_EXPORT = True

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    "model_load_failed": "Failed to load model. Using rule-based analysis.",
    "generation_failed": "Failed to generate response. Please try again.",
    "retrieval_failed": "Failed to retrieve documents. Using template response.",
    "no_documents": "No relevant documents found in knowledge base.",
}

# ============================================================================
# UI TEXT
# ============================================================================

UI_TEXT = {
    "welcome_message": "👋 Welcome! I'm an AI assistant powered by SLM and RAG. Ask me anything!",
    "processing_message": "🤔 Analyzing your question...",
    "generating_message": "✍️ Generating response...",
    "retrieving_message": "📚 Retrieving relevant information...",
}
