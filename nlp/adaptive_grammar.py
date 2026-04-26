"""
adaptive_grammar.py - Graph-based adaptive grammar (học từ phản hồi)
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json
import os

from .config import NLPConfig
from .data_structs import GrammarRule, ParsedCommand


class AdaptiveGrammar:
    """
    Ngữ pháp thích nghi dạng đồ thị.
    Lưu trữ các pattern dạng đồ thị con và cập nhật trọng số dựa trên phản hồi.
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.rules: List[GrammarRule] = []
        self.next_rule_id = 0
        
        # Khởi tạo các luật mặc định
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Tạo các luật mặc định"""
        default_patterns = [
            # Điều khiển hướng
            (['rẽ', 'trái'], ['rẽ', 'trái'], "MOVE_LEFT"),
            (['rẽ', 'phải'], ['rẽ', 'phải'], "MOVE_RIGHT"),
            (['đi', 'thẳng'], ['đi', 'thẳng'], "MOVE_STRAIGHT"),
            
            # Dừng
            (['dừng', 'lại'], ['dừng'], "STOP"),
            
            # Bám theo
            (['bám', 'theo'], ['bám', 'theo'], "FOLLOW"),
        ]
        
        for words, pattern_words, action in default_patterns:
            graph = nx.DiGraph()
            for i in range(len(pattern_words) - 1):
                graph.add_edge(pattern_words[i], pattern_words[i+1], rel='next')
            self.add_rule(graph, action, weight=1.0)
    
    def add_rule(self, pattern_graph: nx.DiGraph, action: str, weight: float = 1.0) -> int:
        """Thêm luật mới"""
        rule = GrammarRule(
            id=self.next_rule_id,
            pattern_graph=pattern_graph,
            action=action,
            weight=weight
        )
        self.rules.append(rule)
        self.next_rule_id += 1
        return rule.id
    
    def match(self, dependency_graph: nx.DiGraph) -> Tuple[Optional[str], float, Optional[int]]:
        """
        Tìm luật phù hợp nhất với đồ thị dependency
        
        Returns:
            action: Hành động tìm được
            confidence: Độ tin cậy (trọng số)
            rule_id: ID của luật được chọn
        """
        best_action = None
        best_weight = -1
        best_rule_id = None
        
        for rule in self.rules:
            # Kiểm tra subgraph isomorphism đơn giản
            if self._is_subgraph(rule.pattern_graph, dependency_graph):
                if rule.weight > best_weight:
                    best_weight = rule.weight
                    best_action = rule.action
                    best_rule_id = rule.id
        
        return best_action, best_weight, best_rule_id
    
    def _is_subgraph(self, pattern: nx.DiGraph, target: nx.DiGraph) -> bool:
        """Kiểm tra pattern có là đồ thị con của target không"""
        # Kiểm tra tất cả cạnh trong pattern đều có trong target
        for u, v, data in pattern.edges(data=True):
            if not target.has_edge(u, v):
                # Thử tìm với tên node khác (trường hợp từ đồng nghĩa)
                found = False
                for tu, tv in target.edges():
                    if (self._is_similar(u, tu) and 
                        self._is_similar(v, tv) and
                        data.get('rel') == target.edges[tu, tv].get('rel')):
                        found = True
                        break
                if not found:
                    return False
        return True
    
    def _is_similar(self, word1: str, word2: str) -> bool:
        """Kiểm tra hai từ có tương tự nhau không (có thể mở rộng với synonym)"""
        # Đơn giản: so sánh trực tiếp
        # Sau này có thể dùng PhoBERT để tìm từ đồng nghĩa
        return word1.lower() == word2.lower()
    
    def update_from_feedback(self, command: ParsedCommand, success: bool):
        """Cập nhật trọng số dựa trên phản hồi"""
        if command.matched_rule_id is not None:
            for rule in self.rules:
                if rule.id == command.matched_rule_id:
                    rule.update_weight(success, self.config.LEARNING_RATE)
                    print(f"[Grammar] Updated rule {rule.id}: weight={rule.weight:.2f}")
                    break
        else:
            # Không có luật nào khớp, tạo luật mới từ đồ thị dependency
            if success:
                new_id = self.add_rule(command.dependency_graph, command.action)
                print(f"[Grammar] Added new rule {new_id} for action {command.action}")
    
    def get_rules_summary(self) -> List[Dict]:
        """Lấy tóm tắt các luật (cho debug)"""
        return [
            {
                'id': r.id,
                'action': r.action,
                'weight': r.weight,
                'success_rate': r.get_confidence(),
                'num_nodes': r.pattern_graph.number_of_nodes(),
                'num_edges': r.pattern_graph.number_of_edges()
            }
            for r in self.rules
        ]
    
    def save_rules(self, filepath: str):
        """Lưu các luật ra file JSON"""
        rules_data = []
        for rule in self.rules:
            # Chuyển đồ thị thành list edges
            edges = [(u, v, data.get('rel', '')) 
                     for u, v, data in rule.pattern_graph.edges(data=True)]
            rules_data.append({
                'id': rule.id,
                'edges': edges,
                'action': rule.action,
                'weight': rule.weight,
                'success_count': rule.success_count,
                'fail_count': rule.fail_count
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)
        print(f"[Grammar] Saved {len(rules_data)} rules to {filepath}")
    
    def load_rules(self, filepath: str):
        """Load luật từ file JSON"""
        if not os.path.exists(filepath):
            print(f"[Grammar] No rules file found at {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        self.rules.clear()
        for data in rules_data:
            graph = nx.DiGraph()
            for u, v, rel in data['edges']:
                graph.add_edge(u, v, rel=rel)
            
            rule = GrammarRule(
                id=data['id'],
                pattern_graph=graph,
                action=data['action'],
                weight=data.get('weight', 1.0),
                success_count=data.get('success_count', 0),
                fail_count=data.get('fail_count', 0)
            )
            self.rules.append(rule)
            self.next_rule_id = max(self.next_rule_id, rule.id + 1)
        
        print(f"[Grammar] Loaded {len(self.rules)} rules from {filepath}")
