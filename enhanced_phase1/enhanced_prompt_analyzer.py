"""
Enhanced Prompt Analyzer with Advanced Intent Detection
Supports complex queries and structured output generation
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import torch

class EnhancedPromptAnalyzer:
    """
    Advanced prompt analysis with support for:
    - Code generation requests
    - ML model creation
    - Tutorial requests
    - Multi-part queries
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2", config: Dict[str, Any] = None):
        """
        Initialize enhanced prompt analyzer
        
        Args:
            model_name: HuggingFace model name
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load intent and context patterns from config
        self.intent_patterns = self.config.get('INTENT_PATTERNS', {})
        self.context_patterns = self.config.get('CONTEXT_PATTERNS', {})
        
        print(f"Loading Enhanced Prompt Analyzer: {model_name} on {self.device}")
        # Lazy import transformers to avoid import-time failures when package isn't available
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
            transformers_available = True
        except Exception:
            transformers_available = False

        if transformers_available:
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

                self.model_loaded = True
                print("✓ Model loaded successfully!")

            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                print("→ Using enhanced rule-based analysis")
                self.model = None
                self.tokenizer = None
                self.model_loaded = False
        else:
            print("⚠️ transformers package not available (AutoTokenizer import failed)")
            print("→ Using enhanced rule-based analysis")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Comprehensive prompt analysis
        
        Args:
            prompt: User input
            
        Returns:
            Detailed analysis dictionary
        """
        # Start with rule-based analysis (always reliable)
        analysis = self._analyze_rule_based(prompt)
        
        # Enhance with parameter extraction
        analysis['parameters'] = self._extract_parameters(prompt)
        
        # Determine required sections
        analysis['required_sections'] = self._determine_required_sections(analysis)
        
        # Extract context requirements
        analysis['context_requirements'] = self._extract_context_requirements(prompt)
        
        # Determine web search needs
        analysis['web_search_queries'] = self._generate_search_queries(prompt, analysis)
        
        return analysis
    
    def _analyze_rule_based(self, prompt: str) -> Dict[str, Any]:
        """Enhanced rule-based analysis"""
        
        prompt_lower = prompt.lower()
        
        # Detect primary intent with enhanced patterns
        intent, intent_confidence = self._detect_intent_enhanced(prompt_lower)
        
        # Extract keywords with context awareness
        keywords = self._extract_keywords_enhanced(prompt)
        
        # Extract entities with categorization
        entities = self._extract_entities_enhanced(prompt)
        
        # Determine complexity
        complexity = self._determine_complexity(prompt)
        
        # Detect domain with confidence
        domain, domain_confidence = self._detect_domain_enhanced(prompt_lower)
        
        # Question type
        question_type = self._detect_question_type(prompt_lower)
        
        # Sentiment
        sentiment = self._detect_sentiment(prompt_lower)
        
        # Detect if this is a code/model generation request
        is_generation_request = self._is_generation_request(prompt_lower)
        
        return {
            "original_prompt": prompt,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "keywords": keywords,
            "entities": entities,
            "complexity": complexity,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "question_type": question_type,
            "sentiment": sentiment,
            "is_generation_request": is_generation_request,
            "requires_context": self._requires_context(prompt_lower),
            "requires_code": self._requires_code(prompt_lower),
            "requires_explanation": self._requires_explanation(prompt_lower),
            "tokens_count": len(prompt.split()),
            "analysis_method": "enhanced_rule_based"
        }
    
    def _detect_intent_enhanced(self, prompt: str) -> Tuple[str, float]:
        """
        Enhanced intent detection with confidence scoring
        
        Returns:
            Tuple of (intent, confidence)
        """
        intent_scores = {}
        
        for intent_type, keywords in self.intent_patterns.items():
            score = sum(1 for kw in keywords if kw in prompt)
            if score > 0:
                intent_scores[intent_type] = score
        
        if not intent_scores:
            return "general_query", 0.5
        
        # Get intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence (0-1)
        confidence = min(max_score / 3.0, 1.0)
        
        return best_intent, confidence
    
    def _extract_keywords_enhanced(self, prompt: str) -> List[str]:
        """Enhanced keyword extraction with relevance scoring"""
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'me', 'my', 'you', 'your', 'i', 'we', 'us', 'our'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', prompt.lower())
        
        # Filter and score keywords
        keyword_scores = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                # Score based on length and uniqueness
                score = len(word) * (1 / (words.count(word) + 1))
                keyword_scores[word] = keyword_scores.get(word, 0) + score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:15]]
    
    def _extract_entities_enhanced(self, prompt: str) -> List[Dict[str, str]]:
        """Enhanced entity extraction with categorization"""
        
        entities = []
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', prompt)
        for num in numbers:
            entities.append({"type": "number", "value": num, "category": "numeric"})
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}\b'  # Years
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, prompt)
            for date in dates:
                entities.append({"type": "date", "value": date, "category": "temporal"})
        
        # Extract capitalized words (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prompt)
        for noun in proper_nouns[:10]:
            entities.append({"type": "proper_noun", "value": noun, "category": "named_entity"})
        
        # Extract technical terms
        tech_terms = re.findall(r'\b(?:ML|AI|NLP|CNN|RNN|LSTM|API|GPU|CPU|RAM)\b', prompt)
        for term in tech_terms:
            entities.append({"type": "technical_term", "value": term, "category": "technical"})
        
        # Extract file extensions
        file_exts = re.findall(r'\.(?:py|js|java|cpp|html|css|json|csv|txt)\b', prompt)
        for ext in file_exts:
            entities.append({"type": "file_extension", "value": ext, "category": "technical"})
        
        return entities
    
    def _determine_complexity(self, prompt: str) -> str:
        """Determine prompt complexity with finer granularity"""
        
        word_count = len(prompt.split())
        sentence_count = len(re.split(r'[.!?]+', prompt))
        
        # Check for technical terms
        tech_terms = len(re.findall(r'\b(?:algorithm|model|dataset|training|neural|learning|classification)\b', prompt.lower()))
        
        # Calculate complexity score
        complexity_score = (word_count / 10) + (sentence_count * 2) + (tech_terms * 3)
        
        if complexity_score < 5:
            return "simple"
        elif complexity_score < 15:
            return "medium"
        elif complexity_score < 30:
            return "complex"
        else:
            return "very_complex"
    
    def _detect_domain_enhanced(self, prompt: str) -> Tuple[str, float]:
        """
        Enhanced domain detection with confidence
        
        Returns:
            Tuple of (domain, confidence)
        """
        domain_scores = {}
        
        domain_keywords = self.config.get('DOMAIN_SOURCES', {})
        
        for domain, info in domain_keywords.items():
            keywords = info.get('keywords', [])
            score = sum(1 for kw in keywords if kw in prompt)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "general", 0.5
        
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        confidence = min(max_score / 3.0, 1.0)
        
        return best_domain, confidence
    
    def _detect_question_type(self, prompt: str) -> str:
        """Detect question type"""
        
        if any(word in prompt for word in ['what', 'who', 'where', 'when', 'which']):
            return "factual"
        elif 'how' in prompt and 'to' in prompt:
            return "procedural"
        elif any(word in prompt for word in ['why', 'explain', 'describe']):
            return "explanatory"
        elif any(word in prompt for word in ['should', 'would', 'recommend']):
            return "advisory"
        elif any(word in prompt for word in ['create', 'generate', 'build', 'make']):
            return "creative"
        elif any(word in prompt for word in ['compare', 'difference', 'versus', 'vs']):
            return "comparative"
        else:
            return "other"
    
    def _detect_sentiment(self, prompt: str) -> str:
        """Detect sentiment"""
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'best', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'problem', 'issue', 'wrong', 'worst']
        
        pos_count = sum(1 for word in positive_words if word in prompt)
        neg_count = sum(1 for word in negative_words if word in prompt)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _requires_context(self, prompt: str) -> bool:
        """Check if prompt requires context"""
        
        context_indicators = [
            'this', 'that', 'these', 'those', 'it', 'they',
            'previous', 'earlier', 'above', 'mentioned', 'said'
        ]
        return any(indicator in prompt for indicator in context_indicators)
    
    def _requires_code(self, prompt: str) -> bool:
        """Check if response should include code"""
        
        code_indicators = [
            'code', 'script', 'program', 'implementation', 'example',
            'python', 'javascript', 'java', 'function', 'class'
        ]
        return any(indicator in prompt for indicator in code_indicators)
    
    def _requires_explanation(self, prompt: str) -> bool:
        """Check if detailed explanation is needed"""
        
        explanation_indicators = [
            'explain', 'describe', 'how', 'why', 'what is',
            'tell me', 'detail', 'understand', 'work'
        ]
        return any(indicator in prompt for indicator in explanation_indicators)
    
    def _is_generation_request(self, prompt: str) -> bool:
        """Check if this is a generation request"""
        
        generation_indicators = [
            'generate', 'create', 'build', 'make', 'develop',
            'write', 'design', 'implement', 'code'
        ]
        return any(indicator in prompt for indicator in generation_indicators)
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Extract main parameters from the prompt
        
        For "Generate a movie recommendation ML model":
        - Main topic: "movie recommendation ML model"
        - Task: "generate"
        - Subject: "movie recommendation"
        - Type: "ML model"
        """
        parameters = {
            "main_topic": "",
            "task_type": "",
            "subject": "",
            "model_type": "",
            "additional_requirements": []
        }
        
        prompt_lower = prompt.lower()
        
        # Extract task type
        for task in ['generate', 'create', 'build', 'develop', 'make', 'write']:
            if task in prompt_lower:
                parameters["task_type"] = task
                break
        
        # Extract model type
        if 'ml model' in prompt_lower or 'machine learning model' in prompt_lower:
            parameters["model_type"] = "ml_model"
        elif 'deep learning' in prompt_lower:
            parameters["model_type"] = "deep_learning"
        elif 'neural network' in prompt_lower:
            parameters["model_type"] = "neural_network"
        elif 'model' in prompt_lower:
            parameters["model_type"] = "model"
        
        # Extract subject (what the model is about)
        # Pattern: "X recommendation", "X classification", "X prediction"
        subject_patterns = [
            r'(\w+(?:\s+\w+)*?)\s+(?:recommendation|classification|prediction|detection|recognition)',
            r'(?:for|about|on)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                parameters["subject"] = match.group(1).strip()
                break
        
        # Main topic is the combination
        if parameters["subject"] and parameters["model_type"]:
            parameters["main_topic"] = f"{parameters['subject']} {parameters['model_type']}"
        else:
            # Extract the part after generate/create
            if parameters["task_type"]:
                pattern = rf'{parameters["task_type"]}\s+(?:a|an|the)?\s*(.+?)(?:\.|$)'
                match = re.search(pattern, prompt_lower)
                if match:
                    parameters["main_topic"] = match.group(1).strip()
        
        return parameters
    
    def _determine_required_sections(self, analysis: Dict[str, Any]) -> List[str]:
        """Determine which sections are needed in the response"""
        
        intent = analysis.get('intent', '')
        is_generation = analysis.get('is_generation_request', False)
        requires_code = analysis.get('requires_code', False)
        
        sections = ["title"]
        
        if intent == "model_creation" or is_generation:
            sections.extend([
                "context",
                "algorithms",
                "dataset_info",
                "sample_code",
                "detailed_explanation"
            ])
        elif intent == "code_generation":
            sections.extend([
                "context",
                "code",
                "explanation"
            ])
        elif intent == "explanation":
            sections.extend([
                "overview",
                "key_concepts",
                "examples"
            ])
        elif intent == "tutorial":
            sections.extend([
                "introduction",
                "prerequisites",
                "steps",
                "code_examples"
            ])
        else:
            sections.extend([
                "context",
                "main_content"
            ])
            if requires_code:
                sections.append("code_examples")
        
        return sections
    
    def _extract_context_requirements(self, prompt: str) -> Dict[str, bool]:
        """Extract what context is needed"""
        
        prompt_lower = prompt.lower()
        
        requirements = {
            "algorithms": any(kw in prompt_lower for kw in self.context_patterns.get('algorithms', [])),
            "datasets": any(kw in prompt_lower for kw in self.context_patterns.get('datasets', [])),
            "implementation": any(kw in prompt_lower for kw in self.context_patterns.get('implementation', [])),
            "workflow": any(kw in prompt_lower for kw in self.context_patterns.get('workflow', []))
        }
        
        return requirements
    
    def _generate_search_queries(self, prompt: str, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate web search queries based on the prompt
        
        For "Generate movie recommendation ML model":
        Returns: [
            "movie recommendation algorithms",
            "movie recommendation datasets",
            "collaborative filtering movie recommendation"
        ]
        """
        queries = []
        
        parameters = analysis.get('parameters', {})
        main_topic = parameters.get('main_topic', '')
        subject = parameters.get('subject', '')
        
        if not subject and not main_topic:
            # Fallback to keywords
            keywords = analysis.get('keywords', [])
            if keywords:
                queries.append(' '.join(keywords[:3]))
        else:
            # Generate targeted queries
            if subject:
                queries.append(f"{subject} algorithms")
                queries.append(f"{subject} datasets")
                queries.append(f"{subject} machine learning")
                queries.append(f"how to build {subject} system")
        
        # Add context-specific queries
        context_req = analysis.get('context_requirements', {})
        if context_req.get('algorithms'):
            queries.append(f"{subject or main_topic} algorithms explained")
        if context_req.get('datasets'):
            queries.append(f"{subject or main_topic} datasets")
        
        # Remove duplicates and limit
        queries = list(dict.fromkeys(queries))[:4]
        
        return queries
