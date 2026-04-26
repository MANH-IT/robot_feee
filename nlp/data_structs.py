"""
data_structs.py - Định nghĩa các lớp dữ liệu cho NLP module
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import networkx as nx


@dataclass
class GrammarRule:
    """Một luật ngữ pháp dưới dạng đồ thị"""
    id: int
    pattern_graph: nx.DiGraph      # Đồ thị dependency pattern
    action: str                    # Hành động robot tương ứng
    weight: float = 1.0            # Trọng số (học được)
    created_at: datetime = field(default_factory=datetime.now)
    success_count: int = 0
    fail_count: int = 0
    
    def get_confidence(self) -> float:
        """Tính độ tin cậy dựa trên tỷ lệ thành công"""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def update_weight(self, success: bool, lr: float = 0.1):
        """Cập nhật trọng số dựa trên phản hồi"""
        if success:
            self.weight = min(2.0, self.weight + lr)
            self.success_count += 1
        else:
            self.weight = max(0.0, self.weight - lr)
            self.fail_count += 1


@dataclass
class ParsedCommand:
    """Kết quả parse câu lệnh"""
    original_text: str
    intent: str                    # ý định từ SNN
    action: str                    # hành động robot
    entities: Dict[str, str]       # {object, direction, value, ...}
    dependency_graph: nx.DiGraph   # đồ thị dependency gốc
    confidence: float              # độ tin cậy (0-1)
    matched_rule_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Chuyển thành dict để gửi qua web"""
        return {
            'text': self.original_text,
            'intent': self.intent,
            'action': self.action,
            'entities': self.entities,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DialogueContext:
    """Ngữ cảnh hội thoại (cho multi-turn)"""
    session_id: str
    last_command: Optional[ParsedCommand] = None
    last_response: Optional[str] = None
    turn_count: int = 0
    history: List[Dict] = field(default_factory=list)
    
    def add_exchange(self, command: ParsedCommand, response: str):
        """Thêm một lượt hội thoại"""
        self.history.append({
            'command': command.to_dict(),
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        self.last_command = command
        self.last_response = response
        self.turn_count += 1
