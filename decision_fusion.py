"""
decision_fusion.py - Kết hợp dữ liệu Vision và NLP để đưa ra quyết định cuối cùng
"""

class DecisionFusion:
    """Module hợp nhất quyết định dựa trên trọng số và độ ưu tiên"""
    
    def __init__(self):
        print("[DecisionFusion] Initialized")
    
    def decide(self, nlp_action, vision_objects):
        """
        Đưa ra quyết định cuối cùng
        
        Args:
            nlp_action: Hành động từ module NLP (ví dụ: MOVE_LEFT)
            vision_objects: Danh sách đối tượng từ module Vision
            
        Returns:
            final_action: Hành động robot thực tế sau khi kiểm tra an toàn
        """
        # Logic đơn giản: Nếu có vật cản quá gần, ghi đè lệnh di chuyển bằng STOP
        if nlp_action.startswith("MOVE"):
            for obj in vision_objects:
                # Giả sử behavior 'stopped' hoặc khoảng cách quá gần (logic demo)
                if obj.get('behavior') == 'stopped' and obj.get('confidence', 0) > 0.8:
                    print(f"[Fusion] Obstacle detected (ID {obj.get('id')}), OVERRIDING to STOP")
                    return "STOP_SAFETY"
        
        return nlp_action

    def fuse(self, nlp_action, vision_objects):
        """Alias cho decide để tương thích ngược"""
        return self.decide(nlp_action, vision_objects)
