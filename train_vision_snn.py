import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from vision.behavior_snn import BehaviorSNN
from vision.config import VisionConfig

def generate_synthetic_data(num_samples=2000, steps=20, features=5, num_classes=4):
    """
    Tạo dataset giả lập cho SNN học hỏi.
    features = [dx, dy, velocity, angle, acceleration]
    classes = [normal(0), fast(1), stopped(2), turning(3)]
    """
    print(f"[Data] Generating {num_samples} synthetic trajectory samples...")
    # X: [steps, batch, features]
    X = torch.randn(steps, num_samples, features) * 0.1
    
    # Y: [batch]
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Nhúng pattern để SNN học
    for i in range(num_samples):
        label = y[i].item()
        if label == 0:   # normal
            X[:, i, 2] = torch.abs(torch.randn(steps) * 0.2 + 0.5)
            X[:, i, 3] = torch.randn(steps) * 0.1
        elif label == 1: # fast
            X[:, i, 2] = torch.abs(torch.randn(steps) * 0.5 + 2.0)
            X[:, i, 4] = torch.abs(torch.randn(steps) * 0.3 + 0.5)
        elif label == 2: # stopped
            X[:, i, 0:3] = torch.abs(torch.randn(steps, 3) * 0.05)
        elif label == 3: # turning
            X[:, i, 2] = torch.abs(torch.randn(steps) * 0.3 + 0.4)
            X[:, i, 3] = torch.abs(torch.randn(steps) * 0.8 + 1.5)
            
    return X, y

def train():
    config = VisionConfig()
    model = BehaviorSNN(config)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Sinh dữ liệu
    X_train, y_train = generate_synthetic_data(num_samples=1600)
    X_test, y_test = generate_synthetic_data(num_samples=400)
    
    epochs = 60
    batch_size = 64
    
    print("\n[Train] Starting Spiking Neural Network (SNN) training...")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        # Mini-batch training
        for i in range(0, len(y_train), batch_size):
            x_batch = X_train[:, i:i+batch_size, :]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward qua SNN
            outputs = model(x_batch) # [steps, batch, classes]
            
            # Tính tổng số spike (tần số bắn xung)
            spike_counts = outputs.sum(dim=0) # [batch, classes]
            
            # Tính Loss và Backprop
            loss = criterion(spike_counts, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = spike_counts.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            
        acc = correct / len(y_train)
        
        if (epoch + 1) % 10 == 0:
            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_counts = val_outputs.sum(dim=0)
                val_pred = val_counts.argmax(dim=1)
                val_acc = (val_pred == y_test).sum().item() / len(y_test)
                
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {total_loss/batch_size:.4f} | Train Acc: {acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
            
    print("-" * 50)
    print(f"Training completed in {time.time() - start_time:.1f} seconds.")
    
    # Đảm bảo thư mục models tồn tại
    os.makedirs(os.path.dirname(config.SNN_MODEL_PATH), exist_ok=True)
    
    # Lưu weights
    model.save_model(config.SNN_MODEL_PATH)
    print(f"\n✅ [Train] Model weights successfully exported to: {config.SNN_MODEL_PATH}")

if __name__ == "__main__":
    train()
