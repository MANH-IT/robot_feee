"""
train_nlp_snn.py - Huấn luyện Spiking Neural Network (SNN) cho Intent Classification
Phiên bản: 3.0
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

from nlp.intent_classifier import SpikingIntentNeuron, IntentClassifier
from nlp.config import default_config

class IntentDataset(Dataset):
    """
    Tạo dataset giả lập cho Intent Classification.
    """
    def __init__(self, num_samples=2500, num_steps=10, feature_dim=128):
        self.config = default_config
        self.intent_classes = self.config.intent_classes
        
        # Danh sách từ khóa mồi cho từng intent
        self.templates = {
            'di_chuyen': ["đi thẳng", "tiến lên", "lùi lại", "rẽ trái", "quay phải", "di chuyển", "đi tiếp đi"],
            'dung_lai': ["dừng lại", "đứng im", "phanh gấp", "stop", "dừng ngay", "đừng đi nữa"],
            'bam_theo': ["đi theo tôi", "bám theo người kia", "theo sau tôi", "theo tôi", "đi theo"],
            'chao_hoi': ["xin chào", "chào robot", "hello", "hi", "chào buổi sáng", "chào bạn"],
            'hoi_thong_tin': ["phòng 303 ở đâu", "cho tôi hỏi", "khoa cntt ở tầng mấy", "thông tin trường"],
            'cam_on': ["cảm ơn", "cảm ơn bạn nhé", "thanks", "tuyệt vời, cảm ơn"],
            'tam_biet': ["tạm biệt", "bye bye", "hẹn gặp lại", "chào tạm biệt"],
            'khac': ["hôm nay trời đẹp", "hát một bài đi", "bạn tên là gì", "thời tiết thế nào"]
        }
        
        print(f"[Data] Đang sinh {num_samples} mẫu câu lệnh tự động...")
        self.samples = []
        
        intents_list = list(self.templates.keys())
        
        # Dùng hàm của Classifier để bảo đảm khớp logic Inference
        classifier = IntentClassifier(config=self.config)
        
        for i in range(num_samples):
            intent_name = intents_list[i % len(intents_list)]
            text = np.random.choice(self.templates[intent_name])
            
            label_idx = self.intent_classes.index(intent_name) if intent_name in self.intent_classes else len(self.intent_classes)-1
            
            # Trích xuất đặc trưng THỰC TẾ từ chuỗi text thay vì fake ngẫu nhiên
            base_feature = classifier.vectorizer.transform(text)
            
            features_seq = base_feature.unsqueeze(0).repeat(num_steps, 1) # [steps, features]
            features_seq = features_seq + (torch.randn(num_steps, feature_dim) * 0.1)
            
            self.samples.append((features_seq, label_idx))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1]


def train():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    print("🚀 Bắt đầu huấn luyện SNN cho NLP Intent Classification (100 Epochs)")
    print("=" * 60)
    
    config = default_config
    num_classes = len(config.intent_classes)
    
    model = SpikingIntentNeuron(
        input_dim=config.snn_input_dim,
        hidden_dim=config.snn_hidden_dim,
        num_classes=num_classes,
        num_steps=config.snn_num_steps
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    dataset = IntentDataset(num_samples=2500, num_steps=config.snn_num_steps, feature_dim=config.snn_input_dim)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    epochs = 100
    best_acc = 0.0
    best_state = None
    start_time = time.time()
    
    # Lịch sử để vẽ biểu đồ
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.permute(1, 0, 2)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            spike_counts = outputs.sum(dim=0)
            
            loss = criterion(spike_counts, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = spike_counts.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x.permute(1, 0, 2)
                outputs = model(batch_x)
                spike_counts = outputs.sum(dim=0)
                
                loss = criterion(spike_counts, batch_y)
                val_loss += loss.item()
                
                pred = spike_counts.argmax(dim=1)
                val_correct += (pred == batch_y).sum().item()
                val_total += batch_y.size(0)
                
                if epoch == epochs - 1: # Lưu lại để vẽ confusion matrix ở epoch cuối
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            
    print("=" * 60)
    print(f"🎉 Huấn luyện hoàn tất sau {time.time() - start_time:.1f} giây!")
    print(f"Độ chính xác tốt nhất (Validation): {best_acc*100:.2f}%")
    
    # Lưu Model
    os.makedirs('models', exist_ok=True)
    save_path = 'models/snn_intent.pth'
    torch.save(best_state, save_path)
    
    # ================= VẼ BIỂU ĐỒ BÁO CÁO =================
    os.makedirs('plots', exist_ok=True)
    
    # 1. Biểu đồ Loss & Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Loss over Epochs (SNN Intent)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red')
    plt.title('Accuracy over Epochs (SNN Intent)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/nlp_training_metrics.png')
    print("📊 Đã lưu biểu đồ Training Metrics tại: plots/nlp_training_metrics.png")
    
    # 2. Biểu đồ Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.intent_classes, 
                yticklabels=config.intent_classes)
    plt.title('Confusion Matrix - SNN Intent Classification')
    plt.ylabel('True Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/nlp_confusion_matrix.png')
    print("📊 Đã lưu biểu đồ Confusion Matrix tại: plots/nlp_confusion_matrix.png")
    
if __name__ == "__main__":
    train()
