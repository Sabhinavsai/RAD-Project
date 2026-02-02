# Quick Setup Guide - Enhanced SLM-RAG Chatbot

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (3 minutes)

```bash
pip install -r enhanced_requirements.txt
```

**Note**: First installation downloads AI models (~3-5 GB). Please be patient!

### Step 2: Configure (Optional - 1 minute)

Edit `enhanced_config.py` if you want to:
- Add API keys (Gemini, HuggingFace)
- Disable certain web sources
- Adjust caching behavior
- Change model settings

**For basic use, no configuration needed!**

### Step 3: Run (10 seconds)

```bash
streamlit run enhanced_app.py
```

Browser opens automatically at `http://localhost:8501`

### Step 4: Test It Out!

Try these example prompts:

#### 1. Simple Question
```
What is machine learning?
```

#### 2. Code Generation
```
Generate a movie recommendation ML model
```

#### 3. Tutorial Request
```
How to build a sentiment analysis model
```

#### 4. Comparison
```
Compare Random Forest and Neural Networks
```

## ✨ What You'll See

For "Generate a movie recommendation ML model":

1. **Prompt Analysis** (if enabled):
   - Intent: model_creation
   - Subject: movie recommendation
   - Confidence: 95%

2. **Processing Steps**:
   - ✓ Analyzing request
   - ✓ Fetching from Wikipedia
   - ✓ Retrieving documents
   - ✓ Generating code

3. **Structured Response**:
   - 📋 Context
   - 🧮 Algorithms
   - 📊 Dataset Info
   - 💻 Complete Python Code (100+ lines)
   - 📖 Detailed Explanation
   - 📚 Sources

## 🎯 Key Features to Try

### 1. Web Integration
- Toggle "Enable Web Search" in sidebar
- Watch it fetch from Wikipedia, Britannica
- See sources at bottom of response

### 2. Prompt Analysis
- Enable "Show Prompt Analysis"
- See how AI understands your query
- View extracted parameters

### 3. Code Generation
Ask for any ML model:
- Classification models
- Regression models
- Clustering algorithms
- Recommendation systems
- Neural networks

### 4. Export Functionality
- Click "Export" in sidebar
- Download conversation as JSON
- Includes full analysis history

## ⚙️ Sidebar Controls

**Model Settings:**
- **Temperature** (0.0-1.0): 
  - Lower = more focused/precise
  - Higher = more creative
  - Recommended: 0.7

- **Max Tokens** (500-3000):
  - Controls response length
  - More tokens = longer response
  - Recommended: 2000 for code generation

**RAG Settings:**
- **Top K Documents** (1-10):
  - Number of sources to use
  - More = comprehensive but slower
  - Recommended: 5

**Display Options:**
- Show Prompt Analysis
- Show Web Sources
- Show Processing Steps

## 💡 Pro Tips

### 1. Be Specific
❌ "Tell me about ML"
✅ "Generate a movie recommendation ML model"

### 2. Use Keywords
- "Generate", "Create", "Build" for code
- "Explain", "How does" for explanations
- "Compare" for comparisons
- "Tutorial" or "Guide" for step-by-step

### 3. Follow-Up Questions
The system remembers context:
```
You: Generate a movie recommendation model
Bot: [Generates model]
You: How do I evaluate this model?
Bot: [Provides evaluation methods]
```

### 4. Adjust Settings
- For factual Q&A: Temperature = 0.3, Top K = 3
- For creative tasks: Temperature = 0.8, Top K = 5
- For code: Temperature = 0.5, Max Tokens = 2000+

## 🐛 Common Issues & Fixes

### Issue: Slow on First Run
**Fix**: Models are downloading. Subsequent runs are faster.

### Issue: Out of Memory
**Fix**: 
- Reduce max_tokens to 1000
- Use CPU mode (automatic)
- Close other applications

### Issue: Web Fetch Failed
**Fix**:
- Check internet connection
- Content is cached, works offline after first fetch
- Disable web search in sidebar if needed

### Issue: Response Quality
**Fix**:
- Try different temperature values
- Increase top_k for more context
- Rephrase your question

## 📊 Performance Expectations

**First Query** (with web fetch):
- 20-40 seconds (CPU)
- 8-15 seconds (GPU)

**Subsequent Queries** (cached):
- 10-20 seconds (CPU)
- 3-8 seconds (GPU)

**Without Web Search**:
- 5-15 seconds (CPU)
- 2-5 seconds (GPU)

## 🎓 Learning Path

### Beginner: Start Here
1. Ask simple questions
2. Enable "Show Prompt Analysis"
3. Try code generation prompts
4. Experiment with temperature

### Intermediate
1. Use web search feature
2. Try complex multi-part queries
3. Export conversations
4. Adjust all settings

### Advanced
1. Modify `enhanced_config.py`
2. Add custom knowledge sources
3. Customize code templates
4. Integrate APIs

## 🔄 What's Next?

### Phase 2 (Coming Soon)
- **Dataset Generation**: Automatic sample data creation
- **Model Training**: End-to-end ML workflows
- **Interactive Coding**: Real-time code execution

### Your Feedback Matters!
- What works well?
- What could be better?
- What features do you want?

## 📚 Additional Resources

- **Full Documentation**: `ENHANCED_README.md`
- **Architecture Details**: In code comments
- **Configuration Guide**: `enhanced_config.py`
- **Web Scraping**: `web_content_fetcher.py`

## 🎉 You're Ready!

Run this command and start chatting:
```bash
streamlit run enhanced_app.py
```

Happy coding! 🚀
