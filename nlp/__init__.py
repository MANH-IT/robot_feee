"""
__init__.py - Khởi tạo package NLP, export các class chính
"""

from .config import NLPConfig
from .data_structs import GrammarRule, ParsedCommand, DialogueContext
from .stt import VoskSTT, MockSTT
from .dependency_parser import DependencyParser, MockDependencyParser
from .adaptive_grammar import AdaptiveGrammar
from .intent_classifier import IntentClassifier, IntentSNN
from .utils import preprocess_text, extract_action_from_intent, format_response

__all__ = [
    # Config
    'NLPConfig',
    
    # Data structures
    'GrammarRule',
    'ParsedCommand',
    'DialogueContext',
    
    # STT
    'VoskSTT',
    'MockSTT',
    
    # Parser
    'DependencyParser',
    'MockDependencyParser',
    
    # Grammar
    'AdaptiveGrammar',
    
    # Intent
    'IntentClassifier',
    'IntentSNN',
    
    # Utils
    'preprocess_text',
    'extract_action_from_intent',
    'format_response'
]
