import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pickle

def save_conversation(messages: List[Dict[str, str]], filename: str = None):
    """
    Save conversation history to a file.
    
    Args:
        messages: List of message dictionaries
        filename: Optional filename (auto-generated if not provided)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
    
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    filepath = os.path.join("conversations", filename)
    
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def load_conversation_history(filepath: str) -> List[Dict[str, str]]:
    """
    Load conversation history from a file.
    
    Args:
        filepath: Path to conversation file
        
    Returns:
        List of message dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("messages", [])

def format_analysis_for_display(analysis: Dict[str, Any]) -> str:
    """
    Format analysis results for display.
    
    Args:
        analysis: Analysis dictionary
        
    Returns:
        Formatted string
    """
    formatted = "📊 Prompt Analysis\n"
    formatted += "=" * 50 + "\n\n"
    
    formatted += f"🎯 Intent: {analysis.get('intent', 'N/A')}\n"
    formatted += f"📚 Domain: {analysis.get('domain', 'N/A')}\n"
    formatted += f"⚡ Complexity: {analysis.get('complexity', 'N/A')}\n"
    formatted += f"❓ Question Type: {analysis.get('question_type', 'N/A')}\n"
    formatted += f"😊 Sentiment: {analysis.get('sentiment', 'N/A')}\n\n"
    
    if analysis.get('keywords'):
        formatted += f"🔑 Keywords: {', '.join(analysis['keywords'][:5])}\n"
    
    if analysis.get('entities'):
        formatted += f"🏷️ Entities Found: {len(analysis['entities'])}\n"
        for entity in analysis['entities'][:3]:
            formatted += f"   - {entity['type']}: {entity['value']}\n"
    
    formatted += f"\n📝 Token Count: {analysis.get('tokens_count', 0)}\n"
    
    return formatted

def create_prompt_template(
    system_message: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Create a formatted prompt template.
    
    Args:
        system_message: System instruction
        user_message: User's message
        context: Optional context information
        
    Returns:
        Formatted prompt
    """
    prompt_parts = []
    
    if system_message:
        prompt_parts.append(f"System: {system_message}")
    
    if context:
        prompt_parts.append(f"\nContext: {context}")
    
    prompt_parts.append(f"\nUser: {user_message}")
    prompt_parts.append("\nAssistant:")
    
    return "\n".join(prompt_parts)

def truncate_text(text: str, max_length: int = 200, ellipsis: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        ellipsis: Ellipsis string to append
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from text.
    
    Args:
        text: Input text
        
    Returns:
        List of code blocks with language and content
    """
    import re
    
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_blocks = []
    for lang, code in matches:
        code_blocks.append({
            "language": lang or "text",
            "content": code.strip()
        })
    
    return code_blocks

def validate_json(json_string: str) -> bool:
    """
    Validate if a string is valid JSON.
    
    Args:
        json_string: String to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word-based similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def create_feedback_log(
    user_message: str,
    assistant_response: str,
    feedback_type: str,
    feedback_text: str = None
):
    """
    Create a feedback log entry.
    
    Args:
        user_message: User's message
        assistant_response: Assistant's response
        feedback_type: Type of feedback (positive/negative)
        feedback_text: Optional feedback text
    """
    os.makedirs("feedback", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback/feedback_{timestamp}.json"
    
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "assistant_response": assistant_response,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)

def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics.
    
    Returns:
        Dictionary with system stats
    """
    import psutil
    
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    return stats

def format_time_elapsed(seconds: float) -> str:
    """
    Format elapsed time in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length-len(ext)] + ext
    
    return filename

def batch_process_documents(
    documents: List[str],
    batch_size: int = 10
) -> List[List[str]]:
    """
    Split documents into batches for processing.
    
    Args:
        documents: List of documents
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(documents), batch_size):
        batches.append(documents[i:i + batch_size])
    
    return batches

def merge_analysis_results(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge multiple analysis results.
    
    Args:
        results: List of analysis dictionaries
        
    Returns:
        Merged analysis
    """
    if not results:
        return {}
    
    merged = {
        "intents": [],
        "all_keywords": [],
        "all_entities": [],
        "domains": [],
        "complexity_scores": []
    }
    
    for result in results:
        if 'intent' in result:
            merged['intents'].append(result['intent'])
        if 'keywords' in result:
            merged['all_keywords'].extend(result['keywords'])
        if 'entities' in result:
            merged['all_entities'].extend(result['entities'])
        if 'domain' in result:
            merged['domains'].append(result['domain'])
        if 'complexity' in result:
            merged['complexity_scores'].append(result['complexity'])
    
    # Remove duplicates
    merged['all_keywords'] = list(set(merged['all_keywords']))
    
    return merged
