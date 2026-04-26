"""
__init__.py - Khởi tạo package vision, export các class chính
"""

from .config import VisionConfig
from .data_structs import ObjectInfo, Trajectory, TrajectoryPoint
from .detector import YOLODetector
from .tracker import ByteTracker
from .behavior_snn import BehaviorSNN, SpikingNeuron
from .utils import (
    draw_bbox,
    draw_trajectory,
    encode_trajectory_to_spike,
    resize_frame,
    frame_to_jpeg
)

__all__ = [
    # Config
    'VisionConfig',
    
    # Data structures
    'ObjectInfo',
    'Trajectory',
    'TrajectoryPoint',
    
    # Modules
    'YOLODetector',
    'ByteTracker',
    'BehaviorSNN',
    'SpikingNeuron',
    
    # Utils
    'draw_bbox',
    'draw_trajectory',
    'encode_trajectory_to_spike',
    'resize_frame',
    'frame_to_jpeg'
]
