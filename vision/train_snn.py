import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple
from vision.behavior_snn import BehaviorSNN, SpikingNeuron
from vision.utils import encode_trajectory_to_spike
from vision.config import VisionConfig

# Surrogate Gradient function
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # FastSigmoid surrogate gradient
        grad_input = grad_input / (10 * torch.abs(input) + 1.0)**2
        return grad_input

spike_fn = SurrogateGradient.apply

# Overwrite SpikingNeuron for training (to use surrogate gradient)
class TrainableSpikingNeuron(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
        
    def forward(self, input_current, mem):
        mem = self.beta * mem + input_current
        spike = spike_fn(mem - 1.0) # threshold = 1.0
        mem = mem - spike * 1.0 # reset
        return spike, mem

# Modified BehaviorSNN for training
class TrainableBehaviorSNN(BehaviorSNN):
    def __init__(self, config=None):
        super().__init__(config)
        # Thay thế LIF neurons bằng bản trainable
        self.lif1 = TrainableSpikingNeuron(beta=0.9)
        self.lif2 = TrainableSpikingNeuron(beta=0.9)

def generate_synthetic_data(num_samples=1000):
    """Tạo dữ liệu giả lập cho 4 hành vi"""
    data = []
    labels = []
    
    for _ in range(num_samples):
        behavior_idx = np.random.randint(0, 4)
        trajectory = [(0.0, 0.0)]
        
        # 0: normal, 1: fast, 2: stopped, 3: turning
        steps = 30
        x, y = 0.0, 0.0
        vx, vy = np.random.uniform(2, 5), np.random.uniform(2, 5)
        
        if behavior_idx == 1: # fast
            vx *= 4
            vy *= 4
        elif behavior_idx == 2: # stopped
            vx, vy = 0.1, 0.1
            
        for i in range(steps):
            if behavior_idx == 3: # turning
                angle = i * 0.2
                x += np.cos(angle) * 5
                y += np.sin(angle) * 5
            else:
                x += vx + np.random.normal(0, 0.5)
                y += vy + np.random.normal(0, 0.5)
            trajectory.append((x, y))
            
        spikes = encode_trajectory_to_spike(trajectory, num_steps=20, method='delta')
        data.append(spikes)
        labels.append(behavior_idx)
        
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

def train():
    config = VisionConfig()
    model = TrainableBehaviorSNN(config)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Generating data...")
    x_train, y_train = generate_synthetic_data(2000)
    x_test, y_test = generate_synthetic_data(400)
    
    print("Starting training...")
    for epoch in range(50):
        model.train()
        # [batch, steps, features] -> [steps, batch, features]
        inputs = x_train.transpose(0, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs) # [steps, batch, classes]
        
        # Sum spikes over time
        spike_counts = outputs.sum(dim=0) # [batch, classes]
        loss = criterion(spike_counts, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test.transpose(0, 1)).sum(dim=0)
                pred = test_outputs.argmax(dim=1)
                acc = (pred == y_test).float().mean()
                print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_model(config.SNN_MODEL_PATH)
    print(f"Training complete. Model saved to {config.SNN_MODEL_PATH}")

if __name__ == "__main__":
    train()
