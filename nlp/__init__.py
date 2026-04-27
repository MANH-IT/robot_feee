# -*- coding: utf-8 -*-
"""
__init__.py - Khoi tao module NLP cho robot FEEE
"""

from .config import NLPConfig, default_config
from .intent_classifier import IntentClassifier
from .knowledge_retriever import KnowledgeRetriever
from .nlp_processor import NLPProcessor, default_processor

# Cac doi tuong duoc export ra ngoai
__all__ = [
    'NLPConfig',
    'default_config',
    'IntentClassifier',
    'KnowledgeRetriever',
    'NLPProcessor',
    'default_processor'
]

# Thong tin module
__version__ = '3.0.0'
__author__ = 'Robot EEEC Team'