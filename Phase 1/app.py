import streamlit as st
import json
from datetime import datetime
from prompt_analyzer import PromptAnalyzer
from rag_engine import RAGEngine
from utils import save_conversation, load_conversation_history

# Page configuration
st.set_page_config(
    page_title="SLM-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'prompt_analyzer' not in st.session_state:
    st.session_state.prompt_analyzer = PromptAnalyzer()
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)
    
    st.subheader("RAG Configuration")
    top_k = st.slider("Top K Documents", 1, 10, 3, 1)
    
    st.subheader("Analysis Display")
    show_analysis = st.checkbox("Show Prompt Analysis", value=True)
    show_json = st.checkbox("Show JSON Breakdown", value=False)
    
    st.divider()
    
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.analysis_history = []
        st.rerun()
    
    if st.button("Export Conversation", type="secondary"):
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
                mime="application/json"
            )

# Main content
st.title("🤖 SLM-RAG Chatbot")
st.markdown("*Powered by Small Language Model + Retrieval Augmented Generation*")

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show analysis for user messages if enabled
        if message["role"] == "user" and show_analysis and idx < len(st.session_state.analysis_history):
            with st.expander("📊 Prompt Analysis"):
                analysis = st.session_state.analysis_history[idx]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent", analysis.get("intent", "N/A"))
                with col2:
                    st.metric("Entities Found", len(analysis.get("entities", [])))
                with col3:
                    st.metric("Complexity", analysis.get("complexity", "N/A"))
                
                if show_json:
                    st.json(analysis)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the prompt
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your prompt..."):
            # Step 1: Analyze the prompt with SLM
            analysis_result = st.session_state.prompt_analyzer.analyze(prompt)
            st.session_state.analysis_history.append(analysis_result)
            
            # Display analysis if enabled
            if show_analysis:
                with st.expander("📊 Prompt Analysis", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", analysis_result.get("intent", "N/A"))
                    with col2:
                        st.metric("Entities Found", len(analysis_result.get("entities", [])))
                    with col3:
                        st.metric("Complexity", analysis_result.get("complexity", "N/A"))
                    
                    if show_json:
                        st.json(analysis_result)
        
        with st.spinner("Generating response..."):
            # Step 2: Generate response using RAG
            response = st.session_state.rag_engine.generate_response(
                prompt=prompt,
                analysis=analysis_result,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k
            )
            
            st.markdown(response)
            
            # Show retrieved documents if available
            if hasattr(st.session_state.rag_engine, 'last_retrieved_docs'):
                with st.expander("📚 Retrieved Documents"):
                    for i, doc in enumerate(st.session_state.rag_engine.last_retrieved_docs[:top_k], 1):
                        st.markdown(f"**Document {i}** (Score: {doc.get('score', 0):.3f})")
                        st.text(doc.get('content', '')[:200] + "...")
                        st.divider()
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>Built with Streamlit | SLM for Analysis | RAG for Generation</small>
    </div>
    """,
    unsafe_allow_html=True
)
