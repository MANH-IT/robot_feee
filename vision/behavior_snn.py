"""
behavior_snn.py - SNN behavior classifier (nhận diện hành vi từ trajectory)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from .config import VisionConfig
from .data_structs import ObjectInfo, Trajectory


class SpikingNeuron(nn.Module):
    """Neuron LIF (Leaky Integrate-and-Fire) đơn giản cho SNN"""
    
    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.beta = beta  # decay rate
        self.threshold = threshold
        
    def forward(self, input_current, mem):
        """
        Args:
            input_current: Dòng vào tại timestep hiện tại
            mem: Điện thế màng trước đó
        Returns:
            spike: Xung đầu ra (0 hoặc 1)
            mem_new: Điện thế màng mới
        """
        mem = self.beta * mem + input_current
        spike = (mem >= self.threshold).float()
        mem = mem - spike * self.threshold  # reset
        return spike, mem


class BehaviorSNN(nn.Module):
    """
    Mạng SNN phân loại hành vi từ trajectory.
    Input: trajectory features [steps, batch, features]
    Output: behavior class (normal, fast, stopped, turning)
    """
    
    def __init__(self, config: VisionConfig = None):
        super().__init__()
        self.config = config or VisionConfig()
        
        input_dim = self.config.TRAJ_FEATURE_DIM  # 5 features
        hidden_dim = 32
        output_dim = len(self.config.SNN_BEHAVIORS)  # 4 classes
        self.num_steps = self.config.SNN_NUM_STEPS
        
        # Các lớp Fully Connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Neuron LIF
        self.lif1 = SpikingNeuron(beta=0.9)
        self.lif2 = SpikingNeuron(beta=0.9)
        
        self.behavior_names = self.config.SNN_BEHAVIORS
        
    def forward(self, x):
        """
        Forward pass qua SNN
        
        Args:
            x: Tensor shape [steps, batch, features]
        Returns:
            outputs: Tensor shape [steps, batch, classes] - spike trains
        """
        batch_size = x.shape[1]
        
        # Khởi tạo điện thế màng trên cùng device với x
        mem1 = torch.zeros(batch_size, 32, device=x.device)
        mem2 = torch.zeros(batch_size, len(self.behavior_names), device=x.device)
        
        spike_outputs = []
        
        for step in range(self.num_steps):
            # Lớp 1
            cur1 = self.fc1(x[step])
            spike1, mem1 = self.lif1(cur1, mem1)
            
            # Lớp 2
            cur2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(cur2, mem2)
            
            spike_outputs.append(spike2)
        
        return torch.stack(spike_outputs, dim=0)
    
    def predict(self, features: np.ndarray) -> str:
        """
        Dự đoán hành vi từ features
        
        Args:
            features: numpy array shape [steps, features]
        Returns:
            behavior: Tên hành vi dự đoán
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [steps, 1, features]
            
            outputs = self.forward(x)  # [steps, batch, classes]
            
            # Tổng số spike qua các timestep
            spike_counts = outputs.sum(dim=0)  # [batch, classes]
            pred_class = spike_counts.argmax(dim=1).item()
            
            return self.behavior_names[pred_class]
    
    def classify_from_trajectory(self, trajectory: List[Tuple[float, float]]) -> str:
        """
        Phân loại hành vi từ trajectory (rule-based fallback khi chưa train)
        
        Hiện tại dùng rule-based để có kết quả ngay.
        Sau khi train xong, sẽ thay bằng predict().
        """
        if len(trajectory) < 3:
            return "unknown"
        
        # Tính vận tốc trung bình (pixel/frame)
        velocities = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            v = np.sqrt(dx*dx + dy*dy)
            velocities.append(v)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        
        # Tính góc thay đổi hướng trung bình
        angles = []
        for i in range(2, len(trajectory)):
            v1 = (trajectory[i-1][0] - trajectory[i-2][0],
                  trajectory[i-1][1] - trajectory[i-2][1])
            v2 = (trajectory[i][0] - trajectory[i-1][0],
                  trajectory[i][1] - trajectory[i-1][1])
            
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1, min(1, dot / (norm1 * norm2)))
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        avg_angle = np.mean(angles) if angles else 0
        
        # Rule-based classification
        if avg_velocity < self.config.SPEED_STOPPED:
            return "stopped"
        elif avg_angle > self.config.ANGLE_TURNING:
            return "turning"
        elif avg_velocity > self.config.SPEED_FAST:
            return "fast"
        else:
            return "normal"
    
    def classify_batch(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> Dict[int, str]:
        """Phân loại hành vi cho nhiều object"""
        results = {}
        for obj_id, traj in trajectories.items():
            results[obj_id] = self.classify_from_trajectory(traj)
        return results
    
    def save_model(self, path: str):
        """Lưu model đã train"""
        torch.save(self.state_dict(), path)
        print(f"[SNN] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model đã train"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"[SNN] Model loaded from {path}")
