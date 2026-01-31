# Architecture Documentation

## System Overview

The SLM-RAG Chatbot combines two powerful AI techniques:
1. **SLM (Small Language Model)**: For prompt understanding and analysis
2. **RAG (Retrieval Augmented Generation)**: For accurate, context-aware responses

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                     (Streamlit Web App)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ User Input
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt Analyzer (SLM)                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Model: Microsoft Phi-2 / TinyLlama                    │    │
│  │  Task: Break down and understand user prompt           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Extracts:                                                       │
│  • Intent (what user wants)                                     │
│  • Entities (people, places, numbers)                           │
│  • Keywords (important terms)                                   │
│  • Domain (topic area)                                          │
│  • Complexity (simple/medium/complex)                           │
│  • Sentiment (positive/neutral/negative)                        │
│                                                                  │
│  Output: Structured JSON                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ Analysis JSON
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Engine                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Step 1: Document Retrieval                   │  │
│  │  ┌────────────────────────────────────────────────┐      │  │
│  │  │  Query Embedding                                │      │  │
│  │  │  (sentence-transformers/all-MiniLM-L6-v2)      │      │  │
│  │  └────────────┬───────────────────────────────────┘      │  │
│  │               │                                            │  │
│  │               ▼                                            │  │
│  │  ┌────────────────────────────────────────────────┐      │  │
│  │  │  Vector Database (FAISS)                       │      │  │
│  │  │  - Stores document embeddings                  │      │  │
│  │  │  - Fast similarity search                      │      │  │
│  │  │  - Returns top-k relevant docs                 │      │  │
│  │  └────────────┬───────────────────────────────────┘      │  │
│  └───────────────┼────────────────────────────────────────────┘  │
│                  │                                                │
│                  │ Retrieved Documents                            │
│                  ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Step 2: Response Generation                  │  │
│  │  ┌────────────────────────────────────────────────┐      │  │
│  │  │  Context Construction                          │      │  │
│  │  │  - Combine retrieved documents                 │      │  │
│  │  │  - Add prompt analysis                         │      │  │
│  │  │  - Format for generation                       │      │  │
│  │  └────────────┬───────────────────────────────────┘      │  │
│  │               │                                            │  │
│  │               ▼                                            │  │
│  │  ┌────────────────────────────────────────────────┐      │  │
│  │  │  Language Model                                │      │  │
│  │  │  (Microsoft Phi-2 / Template-based)            │      │  │
│  │  │  - Generates contextual response               │      │  │
│  │  └────────────┬───────────────────────────────────┘      │  │
│  └───────────────┼────────────────────────────────────────────┘  │
└──────────────────┼──────────────────────────────────────────────┘
                   │
                   │ Generated Response
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface                              │
│                  (Display Response)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Prompt Analyzer (prompt_analyzer.py)

**Purpose**: Understand and break down user prompts into structured data.

**Technologies**:
- Primary: Microsoft Phi-2 (2.7B parameter SLM)
- Fallback: Rule-based NLP

**Process**:
1. Receive user input
2. Tokenize and process with SLM
3. Extract structured information
4. Return JSON with analysis

**Output Structure**:
```json
{
  "original_prompt": "string",
  "intent": "information_seeking|assistance_request|creation|explanation|analysis",
  "keywords": ["keyword1", "keyword2", ...],
  "entities": [
    {"type": "entity_type", "value": "entity_value"}
  ],
  "complexity": "simple|medium|complex",
  "domain": "technology|science|business|health|education|general",
  "question_type": "factual|explanatory|opinion|creative|other",
  "sentiment": "positive|neutral|negative",
  "requires_context": boolean,
  "tokens_count": integer
}
```

**Key Features**:
- **Intent Detection**: Understands what the user wants to accomplish
- **Entity Extraction**: Identifies important items (names, numbers, dates)
- **Keyword Extraction**: Pulls out relevant terms for retrieval
- **Domain Classification**: Categorizes the topic area
- **Complexity Analysis**: Determines how sophisticated the query is
- **Fallback Mechanism**: Uses rules if model fails to load

### 2. RAG Engine (rag_engine.py)

**Purpose**: Retrieve relevant information and generate accurate responses.

**Technologies**:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS (Facebook AI Similarity Search)
- Generation: Microsoft Phi-2 or template-based

**Process**:

#### Retrieval Phase:
1. Convert user query to embedding vector
2. Search vector database for similar documents
3. Rank by similarity score
4. Return top-k most relevant documents

#### Generation Phase:
1. Construct context from retrieved documents
2. Combine with prompt analysis
3. Format prompt for language model
4. Generate response
5. Post-process and validate

**Knowledge Base**:
- Stores document embeddings as vectors
- Enables fast similarity search (O(log n))
- Supports incremental updates
- Can store unlimited documents

**Retrieval Algorithm**:
```
similarity_score = 1 / (1 + euclidean_distance)
```

### 3. User Interface (app.py)

**Purpose**: Provide interactive web interface for users.

**Technologies**:
- Streamlit for web framework
- Session state for conversation history

**Features**:
- Chat interface with message history
- Real-time prompt analysis display
- Configuration sidebar
- Conversation export
- Retrieved document visualization

**Session Management**:
- Maintains conversation history
- Stores analysis results
- Preserves user settings
- Handles component initialization

### 4. Utilities (utils.py)

**Purpose**: Support functions for the main components.

**Features**:
- Conversation saving/loading
- Text formatting and truncation
- JSON validation
- Similarity calculations
- Feedback logging
- System statistics

## Data Flow

### Complete Request Flow:

```
1. User types message
   ↓
2. Message sent to Prompt Analyzer
   ↓
3. SLM analyzes prompt → JSON
   ↓
4. JSON sent to RAG Engine
   ↓
5. RAG retrieves relevant documents
   ↓
6. Context built from documents + analysis
   ↓
7. Language model generates response
   ↓
8. Response displayed to user
   ↓
9. Conversation stored in session
```

### Vector Database Operations:

```
Document Addition:
1. Document content → Embedding model
2. Embedding vector → FAISS index
3. Document metadata → Document store

Document Retrieval:
1. Query → Embedding model
2. Query vector → FAISS search
3. Top-k indices → Document lookup
4. Documents + scores → Response
```

## Performance Characteristics

### Model Sizes:
- **Phi-2**: ~2.7B parameters (~5.4GB on disk)
- **TinyLlama**: ~1.1B parameters (~2.2GB on disk)
- **Embedding Model**: ~22M parameters (~90MB on disk)

### Inference Times (CPU):
- Prompt Analysis: 2-5 seconds
- Document Retrieval: <100ms
- Response Generation: 5-15 seconds

### Inference Times (GPU):
- Prompt Analysis: 0.5-1 second
- Document Retrieval: <50ms
- Response Generation: 1-3 seconds

### Memory Requirements:
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- GPU: 6GB+ VRAM (optional)

## Scalability

### Knowledge Base:
- Supports 10,000+ documents
- FAISS enables efficient search
- Memory usage: ~1MB per 100 documents

### Concurrent Users:
- Single instance: 1-5 users
- Horizontal scaling: Load balancer + multiple instances
- Shared vector store: Redis or similar

### Optimization Strategies:
1. **Model Quantization**: Reduce precision (FP16)
2. **Caching**: Store frequent queries
3. **Batch Processing**: Group similar requests
4. **Async Operations**: Non-blocking retrieval

## Security Considerations

### Data Privacy:
- Conversations stored locally
- No external API calls (unless configured)
- User data not shared

### Input Validation:
- Sanitize user inputs
- Length limits enforced
- Injection attack prevention

### Model Safety:
- Content filtering available
- Harmful content detection
- User feedback mechanism

## Extension Points

### Adding New Models:
1. Modify `config.py`
2. Update model initialization
3. Test compatibility

### Custom Knowledge Base:
1. Prepare documents in JSON format
2. Add to `rag_engine.py`
3. Rebuild vector index

### API Integration:
1. Create REST API wrapper
2. Implement authentication
3. Add rate limiting

### External Data Sources:
1. Implement data connectors
2. Add to retrieval pipeline
3. Update document metadata

## Monitoring and Debugging

### Logging:
- Model loading status
- Retrieval metrics
- Generation statistics
- Error tracking

### Metrics to Track:
- Response time
- Retrieval accuracy
- User satisfaction
- Error rates

### Debug Mode:
- Enable verbose logging
- Display intermediate results
- Show model outputs
- Track memory usage

## Future Enhancements

### Planned Features:
1. Multi-modal support (images, audio)
2. Conversation memory
3. User preferences learning
4. Advanced analytics
5. Mobile app version

### Research Directions:
1. Fine-tuning on domain data
2. Hybrid retrieval methods
3. Query reformulation
4. Response ranking
