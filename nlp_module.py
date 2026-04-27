"""
nlp_module.py - Giao diện chính cho NLP module
"""

import os
from typing import Optional, Tuple

from nlp import (
    NLPConfig,
    VoskSTT,
    MockSTT,
    DependencyParser,
    MockDependencyParser,
    AdaptiveGrammar,
    IntentClassifier,
    ParsedCommand,
    extract_action_from_intent
)


class NLPModule:
    """Module xử lý ngôn ngữ tự nhiên tổng hợp"""
    
    def __init__(self, use_mock: bool = False):
        """
        Khởi tạo NLP module
        
        Args:
            use_mock: Dùng mock (input() thay cho microphone, rule-based parsing)
        """
        self.config = NLPConfig()
        
        # Khởi tạo STT
        if use_mock:
            self.stt = MockSTT()
        else:
            try:
                self.stt = VoskSTT(self.config)
            except Exception as e:
                print(f"[NLP] Cannot load Vosk: {e}, using mock")
                self.stt = MockSTT()
        
        # Khởi tạo Dependency Parser
        if use_mock:
            self.parser = MockDependencyParser()
        else:
            try:
                self.parser = DependencyParser(self.config)
            except Exception as e:
                print(f"[NLP] Cannot load Stanza: {e}, using mock")
                self.parser = MockDependencyParser()
        
        # Khởi tạo Adaptive Grammar
        self.grammar = AdaptiveGrammar(self.config)
        
        # Khởi tạo Intent Classifier
        self.intent_classifier = IntentClassifier(self.config)
        
        print("[NLPModule] Initialized successfully")
    
    def listen_once(self) -> Optional[str]:
        """Lắng nghe một câu lệnh"""
        return self.stt.listen_once()
    
    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[ParsedCommand]]:
        """
        Phân tích câu lệnh
        
        Args:
            text: Câu lệnh dạng text
        
        Returns:
            action: Hành động robot (MOVE_LEFT, STOP, ...)
            command: Đối tượng ParsedCommand (chứa đầy đủ thông tin)
        """
        if not text:
            return None, None
        
        # Phân tích dependency
        dep_graph, dep_info = self.parser.parse(text)
        
        # Phân loại ý định
        intent, intent_conf = self.intent_classifier.predict_with_confidence(text)
        
        # Trích xuất thực thể
        entities = self.intent_classifier.extract_entities(text, intent)
        
        # Tìm luật grammar phù hợp
        grammar_action, grammar_weight, rule_id = self.grammar.match(dep_graph)
        
        # Chọn action (ưu tiên grammar nếu có)
        if grammar_action and grammar_weight > 0.5:
            action = grammar_action
        else:
            action = extract_action_from_intent(intent, entities, self.config.INTENT_TO_ACTION)
        
        # Tạo ParsedCommand object
        command = ParsedCommand(
            original_text=text,
            intent=intent,
            action=action,
            entities=entities,
            dependency_graph=dep_graph,
            confidence=max(intent_conf, grammar_weight) if grammar_weight else intent_conf,
            matched_rule_id=rule_id
        )
        
        print(f"[NLP] Parsed: '{text}' -> intent={intent}, action={action}")
        
        return action, command
    
    def update_feedback(self, command: ParsedCommand, success: bool):
        """Cập nhật grammar dựa trên phản hồi"""
        self.grammar.update_from_feedback(command, success)
    
    def get_grammar_summary(self):
        """Lấy tóm tắt các luật grammar (debug)"""
        return self.grammar.get_rules_summary()
