import json
import re
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PromptAnalyzer:
    """
    Uses a Small Language Model to analyze and break down user prompts.
    Extracts intent, entities, keywords, and generates structured JSON output.
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """
        Initialize the prompt analyzer with a small language model.
        
        Args:
            model_name: HuggingFace model name (default: microsoft/phi-2)
                       Alternative options: TinyLlama/TinyLlama-1.1B-Chat-v1.0,
                       stabilityai/stablelm-2-zephyr-1_6b
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to rule-based analysis")
            self.model = None
            self.tokenizer = None
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the prompt and return structured breakdown.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Dictionary containing analysis results
        """
        if self.model is not None:
            return self._analyze_with_slm(prompt)
        else:
            return self._analyze_rule_based(prompt)
    
    def _analyze_with_slm(self, prompt: str) -> Dict[str, Any]:
        """Use SLM for prompt analysis"""
        
        # Create analysis prompt for the SLM
        analysis_prompt = f"""Analyze the following user prompt and extract structured information.

User Prompt: "{prompt}"

Provide a JSON response with the following structure:
{{
    "intent": "the main intent or goal",
    "keywords": ["key", "words", "from", "prompt"],
    "entities": [{{"type": "entity_type", "value": "entity_value"}}],
    "complexity": "simple|medium|complex",
    "domain": "the domain or topic0 area",
    "question_type": "factual|opinion|instruction|creative|other",
    "sentiment": "positive|neutral|negative",
    "requires_context": true/false
}}

JSON Response:"""

        try:
            # Tokenize and generate
            inputs = self.tokenizer(
                analysis_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback to rule-based if JSON parsing fails
                analysis = self._analyze_rule_based(prompt)
            
            # Add metadata
            analysis['timestamp'] = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            analysis['original_prompt'] = prompt
            analysis['tokens_count'] = len(self.tokenizer.encode(prompt))
            
            return analysis
            
        except Exception as e:
            print(f"Error in SLM analysis: {e}")
            return self._analyze_rule_based(prompt)
    
    def _analyze_rule_based(self, prompt: str) -> Dict[str, Any]:
        """Fallback rule-based analysis when SLM is not available"""
        
        prompt_lower = prompt.lower()
        
        # Intent detection
        intent = self._detect_intent(prompt_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(prompt)
        
        # Extract entities
        entities = self._extract_entities(prompt)
        
        # Determine complexity
        complexity = self._determine_complexity(prompt)
        
        # Detect domain
        domain = self._detect_domain(prompt_lower)
        
        # Question type
        question_type = self._detect_question_type(prompt_lower)
        
        # Sentiment
        sentiment = self._detect_sentiment(prompt_lower)
        
        return {
            "original_prompt": prompt,
            "intent": intent,
            "keywords": keywords,
            "entities": entities,
            "complexity": complexity,
            "domain": domain,
            "question_type": question_type,
            "sentiment": sentiment,
            "requires_context": self._requires_context(prompt_lower),
            "tokens_count": len(prompt.split()),
            "analysis_method": "rule_based"
        }
    
    def _detect_intent(self, prompt: str) -> str:
        """Detect the main intent of the prompt"""
        if any(word in prompt for word in ['what', 'who', 'where', 'when', 'which']):
            return "information_seeking"
        elif any(word in prompt for word in ['how to', 'can you', 'please', 'help me']):
            return "assistance_request"
        elif any(word in prompt for word in ['create', 'generate', 'write', 'make']):
            return "creation"
        elif any(word in prompt for word in ['explain', 'describe', 'tell me about']):
            return "explanation"
        elif any(word in prompt for word in ['analyze', 'compare', 'evaluate']):
            return "analysis"
        else:
            return "general_query"
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract important keywords from prompt"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', prompt.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return top keywords (by frequency and length)
        return list(dict.fromkeys(keywords))[:10]
    
    def _extract_entities(self, prompt: str) -> List[Dict[str, str]]:
        """Extract named entities (simple version)"""
        entities = []
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', prompt)
        for num in numbers:
            entities.append({"type": "number", "value": num})
        
        # Extract dates (simple patterns)
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', prompt)
        for date in date_patterns:
            entities.append({"type": "date", "value": date})
        
        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prompt)
        for noun in proper_nouns[:5]:  # Limit to 5
            entities.append({"type": "proper_noun", "value": noun})
        
        return entities
    
    def _determine_complexity(self, prompt: str) -> str:
        """Determine prompt complexity"""
        word_count = len(prompt.split())
        sentence_count = len(re.split(r'[.!?]+', prompt))
        
        if word_count < 10 and sentence_count <= 1:
            return "simple"
        elif word_count < 30 and sentence_count <= 3:
            return "medium"
        else:
            return "complex"
    
    def _detect_domain(self, prompt: str) -> str:
        """Detect the domain or topic area"""
        domain_keywords = {
            "technology": ["computer", "software", "code", "programming", "ai", "ml", "tech"],
            "science": ["science", "research", "experiment", "theory", "hypothesis"],
            "business": ["business", "market", "sales", "revenue", "company", "strategy"],
            "health": ["health", "medical", "doctor", "disease", "treatment", "patient"],
            "education": ["learn", "study", "education", "teaching", "course", "school"],
            "general": []
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                return domain
        return "general"
    
    def _detect_question_type(self, prompt: str) -> str:
        """Detect the type of question"""
        if any(word in prompt for word in ['what', 'who', 'where', 'when']):
            return "factual"
        elif any(word in prompt for word in ['how', 'why']):
            return "explanatory"
        elif any(word in prompt for word in ['should', 'would', 'do you think']):
            return "opinion"
        elif any(word in prompt for word in ['create', 'write', 'generate', 'make']):
            return "creative"
        else:
            return "other"
    
    def _detect_sentiment(self, prompt: str) -> str:
        """Detect sentiment of the prompt"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'problem', 'issue']
        
        pos_count = sum(1 for word in positive_words if word in prompt)
        neg_count = sum(1 for word in negative_words if word in prompt)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _requires_context(self, prompt: str) -> bool:
        """Determine if prompt requires conversation context"""
        context_indicators = ['this', 'that', 'these', 'those', 'it', 'they', 'previous', 'earlier', 'above']
        return any(indicator in prompt for indicator in context_indicators)
    
    def get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of the analysis"""
        summary = f"""
Prompt Analysis Summary:
- Intent: {analysis['intent']}
- Domain: {analysis['domain']}
- Complexity: {analysis['complexity']}
- Question Type: {analysis['question_type']}
- Keywords: {', '.join(analysis['keywords'][:5])}
- Entities Found: {len(analysis['entities'])}
- Sentiment: {analysis['sentiment']}
"""
        return summary
