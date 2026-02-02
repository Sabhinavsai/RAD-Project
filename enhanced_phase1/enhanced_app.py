"""
Enhanced SLM-RAG Chatbot Application
Features:
- Web content integration
- Structured output generation
- Code generation
- Multi-source retrieval
"""

import streamlit as st
import json
from datetime import datetime
from enhanced_config import *
from enhanced_prompt_analyzer import EnhancedPromptAnalyzer
from enhanced_rag_engine import EnhancedRAGEngine
from web_content_fetcher import WebContentFetcher

# Page configuration
st.set_page_config(
    page_title="Enhanced SLM-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stAlert {
    padding: 1rem;
    border-radius: 0.5rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.code-section {
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Initialize components (lazy loading)
@st.cache_resource
def load_components():
    """Load and cache components"""
    config = {
        'KNOWLEDGE_SOURCES': KNOWLEDGE_SOURCES,
        'WEB_CONTENT_TIMEOUT': WEB_CONTENT_TIMEOUT,
        'WEB_CONTENT_MAX_LENGTH': WEB_CONTENT_MAX_LENGTH,
        'ENABLE_WEB_CACHE': ENABLE_WEB_CACHE,
        'WEB_CACHE_DIR': WEB_CACHE_DIR,
        'WEB_CACHE_DURATION': WEB_CACHE_DURATION,
        'INTENT_PATTERNS': INTENT_PATTERNS,
        'CONTEXT_PATTERNS': CONTEXT_PATTERNS,
        'DOMAIN_SOURCES': DOMAIN_SOURCES,
        'ENHANCED_KNOWLEDGE_BASE': ENHANCED_KNOWLEDGE_BASE,
        'USE_DEFAULT_DOCUMENTS': USE_DEFAULT_DOCUMENTS,
        'USE_WEB_CONTENT': USE_WEB_CONTENT,
        'MAX_WEB_PAGES_PER_QUERY': MAX_WEB_PAGES_PER_QUERY,
    }
    
    web_fetcher = WebContentFetcher(config)
    analyzer = EnhancedPromptAnalyzer(config=config)
    rag_engine = EnhancedRAGEngine(
        config=config,
        web_fetcher=web_fetcher
    )
    
    return analyzer, rag_engine, web_fetcher

# Load components
with st.spinner("🚀 Loading Enhanced RAG System..."):
    analyzer, rag_engine, web_fetcher = load_components()

# Sidebar
with st.sidebar:
    st.title("⚙️ Enhanced Settings")
    
    st.subheader("🔧 Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1,
                           help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Tokens", 500, 3000, DEFAULT_MAX_TOKENS, 100,
                          help="Maximum length of response")
    
    st.subheader("📚 RAG Configuration")
    top_k = st.slider("Top K Documents", 1, 10, DEFAULT_TOP_K, 1,
                     help="Number of documents to retrieve")
    
    use_web = st.checkbox("Enable Web Search", value=USE_WEB_CONTENT,
                         help="Fetch content from Wikipedia, Britannica, etc.")
    
    st.subheader("📊 Display Options")
    show_analysis = st.checkbox("Show Prompt Analysis", value=SHOW_ANALYSIS_DEFAULT)
    show_web_sources = st.checkbox("Show Web Sources", value=SHOW_WEB_SOURCES)
    show_processing = st.checkbox("Show Processing Steps", value=SHOW_PROCESSING_STEPS)
    
    st.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.analysis_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export", use_container_width=True):
            if st.session_state.messages:
                conversation_data = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages,
                    "analysis_history": st.session_state.analysis_history
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(conversation_data, indent=2),
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("KB Documents", len(rag_engine.documents))
    
    # Info
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Enhanced SLM-RAG Chatbot**
        
        Features:
        - 🌐 Web content integration
        - 💻 Code generation
        - 📊 Structured outputs
        - 🎯 Multi-source retrieval
        
        Version: 2.0 (Enhanced)
        """)

# Main content
st.title("🤖 Enhanced SLM-RAG Chatbot")
st.markdown("*AI-Powered Chatbot with Web Integration & Code Generation*")

# Welcome message
if len(st.session_state.messages) == 0:
    st.info(UI_TEXT['welcome_message'])

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show analysis for user messages
        if message["role"] == "user" and show_analysis and idx < len(st.session_state.analysis_history):
            with st.expander("📊 Prompt Analysis", expanded=False):
                analysis = st.session_state.analysis_history[idx]
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intent", analysis.get("intent", "N/A"))
                with col2:
                    st.metric("Domain", analysis.get("domain", "N/A"))
                with col3:
                    st.metric("Complexity", analysis.get("complexity", "N/A"))
                with col4:
                    confidence = analysis.get("intent_confidence", 0.5)
                    st.metric("Confidence", f"{confidence:.0%}")
                
                # Parameters
                if analysis.get('parameters', {}).get('main_topic'):
                    st.markdown("**📋 Extracted Parameters:**")
                    params = analysis['parameters']
                    st.write(f"- **Topic:** {params.get('main_topic', 'N/A')}")
                    st.write(f"- **Task:** {params.get('task_type', 'N/A')}")
                    if params.get('subject'):
                        st.write(f"- **Subject:** {params.get('subject', 'N/A')}")
                
                # Keywords
                if analysis.get('keywords'):
                    st.markdown(f"**🔑 Keywords:** {', '.join(analysis['keywords'][:8])}")

# Chat input
if prompt := st.chat_input("Ask me anything... (e.g., 'Generate a movie recommendation ML model')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and respond
    with st.chat_message("assistant"):
        # Progress tracking
        if show_processing:
            status_container = st.empty()
        
        # Step 1: Analyze prompt
        if show_processing:
            status_container.info("🔍 Analyzing your request...")
        
        analysis_result = analyzer.analyze(prompt)
        st.session_state.analysis_history.append(analysis_result)
        
        # Display analysis
        if show_analysis:
            with st.expander("📊 Prompt Analysis", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intent", analysis_result.get("intent", "N/A"))
                with col2:
                    st.metric("Domain", analysis_result.get("domain", "N/A"))
                with col3:
                    st.metric("Complexity", analysis_result.get("complexity", "N/A"))
                with col4:
                    confidence = analysis_result.get("intent_confidence", 0.5)
                    st.metric("Confidence", f"{confidence:.0%}")
                
                # Parameters
                if analysis_result.get('parameters', {}).get('main_topic'):
                    st.markdown("**📋 Extracted Parameters:**")
                    params = analysis_result['parameters']
                    st.write(f"- **Topic:** {params.get('main_topic', 'N/A')}")
                    st.write(f"- **Task:** {params.get('task_type', 'N/A')}")
                    if params.get('subject'):
                        st.write(f"- **Subject:** {params.get('subject', 'N/A')}")
                
                # Keywords
                if analysis_result.get('keywords'):
                    st.markdown(f"**🔑 Keywords:** {', '.join(analysis_result['keywords'][:8])}")
        
        # Step 2: Generate response
        if show_processing:
            status_container.info("✍️ Generating comprehensive response...")
        
        try:
            # Temporarily update config for this request
            rag_engine.config['USE_WEB_CONTENT'] = use_web
            
            response = rag_engine.generate_response(
                prompt=prompt,
                analysis=analysis_result,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k
            )
            
            # Clear status
            if show_processing:
                status_container.empty()
            
            # Display response
            st.markdown(response)
            
            # Show sources
            if show_web_sources and rag_engine.last_retrieved_docs:
                with st.expander("📚 Sources Used", expanded=False):
                    for i, doc in enumerate(rag_engine.last_retrieved_docs[:5], 1):
                        source = doc.get('metadata', {}).get('source', 'Unknown')
                        score = doc.get('score', 0)
                        
                        st.markdown(f"**{i}. {source}** (Relevance: {score:.2%})")
                        
                        # Show URL if available
                        url = doc.get('metadata', {}).get('url')
                        if url:
                            st.markdown(f"   🔗 [{url}]({url})")
                        
                        # Preview
                        content = doc.get('content', '')
                        st.text(content[:150] + "..." if len(content) > 150 else content)
                        st.divider()
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            response = "I apologize, but I encountered an error generating the response. Please try again or rephrase your question."
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Enhanced SLM-RAG Chatbot v2.0 | Powered by AI with Web Integration 🌐</small><br>
    <small>Multi-source retrieval • Structured outputs • Code generation</small>
</div>
""", unsafe_allow_html=True)
