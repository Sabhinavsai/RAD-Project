import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Optional FAISS import; provide a numpy fallback when FAISS isn't available (Windows common case)
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

class RAGEngine:
    """
    Retrieval Augmented Generation engine that retrieves relevant documents
    and generates contextual responses.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model: str = "microsoft/phi-2"
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embedding_model: Model for creating embeddings
            generation_model: Model for generating responses
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing RAG Engine on {self.device}...")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize generation model
        print(f"Loading generation model: {generation_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                generation_model,
                trust_remote_code=True
            )
            self.gen_model = AutoModelForCausalLM.from_pretrained(
                generation_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.gen_model = self.gen_model.to(self.device)
                
            self.model_loaded = True
            print("Generation model loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load generation model: {e}")
            print("Using template-based responses")
            self.model_loaded = False
        
        # Initialize vector store or numpy fallback
        self.vector_store = None
        self._embeddings = None  # numpy array fallback when faiss missing
        self.documents = []
        self.last_retrieved_docs = []
        
        # Load sample knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with sample documents (can be replaced with your data)"""
        
        sample_docs = [
            {
                "content": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes machine learning, deep learning, and natural language processing.",
                "metadata": {"category": "AI", "source": "knowledge_base"}
            },
            {
                "content": "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming. It uses algorithms to identify patterns.",
                "metadata": {"category": "ML", "source": "knowledge_base"}
            },
            {
                "content": "Natural Language Processing (NLP) helps computers understand, interpret, and generate human language. It powers chatbots, translation, and text analysis.",
                "metadata": {"category": "NLP", "source": "knowledge_base"}
            },
            {
                "content": "Deep Learning uses neural networks with multiple layers to process complex patterns. It's used in image recognition, speech recognition, and autonomous vehicles.",
                "metadata": {"category": "Deep Learning", "source": "knowledge_base"}
            },
            {
                "content": "RAG (Retrieval Augmented Generation) combines information retrieval with text generation to provide accurate, contextual responses based on relevant documents.",
                "metadata": {"category": "RAG", "source": "knowledge_base"}
            },
            {
                "content": "Small Language Models (SLMs) are compact AI models that offer efficiency and faster inference while maintaining good performance for specific tasks.",
                "metadata": {"category": "SLM", "source": "knowledge_base"}
            },
            {
                "content": "Vector databases store embeddings of documents, enabling fast similarity search for retrieval tasks in RAG systems.",
                "metadata": {"category": "Vector DB", "source": "knowledge_base"}
            },
            {
                "content": "Streamlit is an open-source Python framework for building data apps and ML interfaces quickly with minimal code.",
                "metadata": {"category": "Streamlit", "source": "knowledge_base"}
            },
            {
                "content": "Transformers are neural network architectures that use self-attention mechanisms. They're the foundation of modern NLP models like BERT and GPT.",
                "metadata": {"category": "Transformers", "source": "knowledge_base"}
            },
            {
                "content": "Fine-tuning involves training a pre-trained model on specific data to adapt it for particular tasks or domains.",
                "metadata": {"category": "Fine-tuning", "source": "knowledge_base"}
            }
        ]
        
        self.add_documents(sample_docs)
        print(f"Knowledge base initialized with {len(sample_docs)} documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
        """
        self.documents.extend(documents)

        # Create embeddings (ensure numpy array)
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(contents)
        emb = np.array(embeddings)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        emb = emb.astype('float32')

        # Create or update FAISS index if available, otherwise keep numpy embeddings
        if HAS_FAISS:
            dimension = emb.shape[1]
            if self.vector_store is None:
                self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(emb)
        else:
            if self._embeddings is None:
                self._embeddings = emb
            else:
                self._embeddings = np.vstack([self._embeddings, emb])

        print(f"Added {len(documents)} documents to knowledge base")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        q = np.array(query_embedding).astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)

        results = []
        if HAS_FAISS and self.vector_store is not None:
            distances, indices = self.vector_store.search(
                q,
                min(top_k, len(self.documents))
            )

            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(1 / (1 + distance))  # Convert distance to similarity
                    results.append(doc)
        else:
            # Numpy fallback: cosine similarity search
            if self._embeddings is None:
                return []

            emb = self._embeddings
            # normalize
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
            sims = (emb_norm @ q_norm.T).reshape(-1)

            k = min(top_k, len(self.documents))
            idxs = np.argsort(-sims)[:k]
            for idx in idxs:
                doc = self.documents[idx].copy()
                doc['score'] = float(sims[idx])
                results.append(doc)

        self.last_retrieved_docs = results
        return results
    
    def generate_response(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_k: int = 3
    ) -> str:
        """
        Generate a response using RAG.
        
        Args:
            prompt: User prompt
            analysis: Prompt analysis from SLM
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve(prompt, top_k=top_k)
        
        # Construct context from retrieved documents
        context = self._build_context(relevant_docs)
        
        # Generate response
        if self.model_loaded:
            response = self._generate_with_model(
                prompt=prompt,
                context=context,
                analysis=analysis,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = self._generate_template_response(
                prompt=prompt,
                context=context,
                analysis=analysis
            )
        
        return response
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not documents:
            return "No relevant information found in knowledge base."
        
        context_parts = ["Relevant Information:"]
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"\n{i}. {doc['content']}")
        
        return "\n".join(context_parts)
    
    def _generate_with_model(
        self,
        prompt: str,
        context: str,
        analysis: Dict[str, Any],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using the language model"""
        
        # Create generation prompt
        generation_prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately.

Context:
{context}

User Question: {prompt}

Intent: {analysis.get('intent', 'unknown')}
Domain: {analysis.get('domain', 'general')}

Provide a clear, accurate, and helpful response based on the context above:"""

        try:
            # Tokenize
            inputs = self.tokenizer(
                generation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.gen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response = full_response[len(generation_prompt):].strip()
            
            # Clean up response
            if not response:
                response = self._generate_template_response(prompt, context, analysis)
            
            return response
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return self._generate_template_response(prompt, context, analysis)
    
    def _generate_template_response(
        self,
        prompt: str,
        context: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate template-based response when model is not available"""
        
        intent = analysis.get('intent', 'general_query')
        domain = analysis.get('domain', 'general')
        
        if not self.last_retrieved_docs:
            return f"I understand you're asking about {domain}. However, I don't have specific information in my knowledge base to answer this question accurately. Could you provide more details or rephrase your question?"
        
        # Build response from retrieved documents
        response_parts = []
        
        if intent == "information_seeking":
            response_parts.append(f"Based on the available information about {domain}:")
        elif intent == "explanation":
            response_parts.append(f"Let me explain {domain}:")
        else:
            response_parts.append("Here's what I found:")
        
        # Add relevant information
        for doc in self.last_retrieved_docs[:2]:
            response_parts.append(f"\n\n{doc['content']}")
        
        # Add closing
        if len(self.last_retrieved_docs) > 2:
            response_parts.append(f"\n\nI found {len(self.last_retrieved_docs)} relevant pieces of information. Would you like me to elaborate on any specific aspect?")
        
        return "".join(response_parts)
    
    def add_conversation_to_knowledge(self, user_message: str, assistant_message: str):
        """
        Add conversation history to knowledge base for context-aware responses.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        doc = {
            "content": f"Q: {user_message}\nA: {assistant_message}",
            "metadata": {
                "category": "conversation_history",
                "source": "chat"
            }
        }
        self.add_documents([doc])
    
    def clear_knowledge_base(self):
        """Clear the knowledge base"""
        self.documents = []
        self.vector_store = None
        self.last_retrieved_docs = []
        self._initialize_knowledge_base()
    
    def export_knowledge_base(self) -> str:
        """Export knowledge base as JSON"""
        return json.dumps(self.documents, indent=2)
    
    def import_knowledge_base(self, json_data: str):
        """Import knowledge base from JSON"""
        documents = json.loads(json_data)
        self.clear_knowledge_base()
        self.add_documents(documents)
