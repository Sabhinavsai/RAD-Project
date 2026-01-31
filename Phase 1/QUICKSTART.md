# Quick Start Guide

## Get Started in 3 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (UI framework)
- Transformers (for SLM models)
- Sentence-transformers (for embeddings)
- FAISS (for vector search)
- PyTorch (deep learning framework)

**Note**: First run downloads models (~2-3GB). Be patient!

### Step 2: Run the App (10 seconds)

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

### Step 3: Start Chatting!

1. Type a question in the chat box
2. Watch the prompt analysis in real-time
3. Get intelligent, context-aware responses

## Example Questions to Try

**Simple Questions:**
- "What is AI?"
- "Explain machine learning"
- "What is RAG?"

**Complex Questions:**
- "How do RAG systems differ from traditional chatbots?"
- "Compare deep learning and machine learning"
- "Explain the architecture of transformers"

**Creative Requests:**
- "Create a plan for learning AI"
- "Suggest resources for NLP"

## Understanding the Interface

### Main Chat Area
- Type questions here
- See responses with retrieved context
- View conversation history

### Sidebar Settings

**Model Configuration:**
- **Temperature**: 0.0 (precise) to 1.0 (creative)
- **Max Tokens**: Response length (100-2000)

**RAG Configuration:**
- **Top K Documents**: How many docs to retrieve (1-10)

**Display Options:**
- ✅ **Show Prompt Analysis**: See how your prompt is understood
- ✅ **Show JSON Breakdown**: See raw analysis data

**Actions:**
- **Clear Conversation**: Start fresh
- **Export Conversation**: Download as JSON

### Analysis Display

When enabled, you'll see:
- **Intent**: What you're trying to do
- **Entities**: Important items found
- **Complexity**: Simple/Medium/Complex
- **Keywords**: Main topics
- **Domain**: Subject area
- **Sentiment**: Tone of your message

## Tips for Best Results

1. **Be Specific**: "Explain how transformers work in NLP" > "Tell me about transformers"

2. **Use Keywords**: The system retrieves based on keywords, so include relevant terms

3. **Ask Follow-ups**: The system can handle context from previous messages

4. **Adjust Settings**: 
   - Lower temperature (0.3-0.5) for factual answers
   - Higher temperature (0.7-0.9) for creative responses

5. **Check Retrieved Docs**: See what information was used to answer

## Common Issues & Solutions

### "Model loading failed"
- **Solution**: System falls back to rule-based analysis (still works!)
- Or use a smaller model: Edit `prompt_analyzer.py` line 15

### "Slow responses"
- **Solution**: Reduce max_tokens to 200-300
- Or reduce top_k to 1-2

### "Out of memory"
- **Solution**: Use CPU mode (automatic)
- Or restart the application

## Next Steps

1. **Add Your Own Documents**: Edit `rag_engine.py` line 57-90
2. **Customize Analysis**: Modify rules in `prompt_analyzer.py`
3. **Change Models**: Try different SLMs for your use case
4. **Export Conversations**: Save interesting chats for later

## Need Help?

Check the full README.md for:
- Detailed architecture explanation
- Advanced customization options
- Troubleshooting guide
- Model recommendations

---

**You're ready to go! Start chatting and explore the capabilities! 🚀**
