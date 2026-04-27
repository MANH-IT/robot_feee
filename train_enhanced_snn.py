import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from typing import List, Dict

from enhanced_vectorizer import EnhancedTextVectorizer
from optimized_snn_config import OPTIMIZED_SNN_CONFIG

class SNNDataset(Dataset):
    """Dataset cho SNN với vector hóa nâng cao"""
    
    def __init__(self, samples: List[Dict], vectorizer: EnhancedTextVectorizer):
        self.samples = samples
        self.vectorizer = vectorizer
        self.intent_to_idx = {
            'di_chuyen': 0, 'dung_lai': 1, 'bam_theo': 2,
            'hoi_thong_tin': 3, 'chao_hoi': 4, 'cam_on': 5,
            'tam_biet': 6, 'khac': 7
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.vectorizer.transform(sample['text'])
        label = self.intent_to_idx[sample['intent']]
        return features, torch.LongTensor([label])


class EnhancedSpikingIntentNeuron(nn.Module):
    """SNN nâng cấp với dropout và batch normalization"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.num_steps = config['num_steps']
        
        import snntorch as snn
        from snntorch import surrogate
        
        spike_grad = surrogate.fast_sigmoid(slope=config['surrogate_slope'])
        
        self.fc1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.bn1 = nn.BatchNorm1d(config['hidden_dim'])
        self.dropout1 = nn.Dropout(config['dropout_rate'])
        self.lif1 = snn.Leaky(beta=config['beta'], spike_grad=spike_grad, output=True)
        
        self.fc2 = nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2)
        self.bn2 = nn.BatchNorm1d(config['hidden_dim'] // 2)
        self.dropout2 = nn.Dropout(config['dropout_rate'])
        self.lif2 = snn.Leaky(beta=config['beta'], spike_grad=spike_grad, output=True)
        
        self.fc3 = nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        self.lif3 = snn.Leaky(beta=config['beta'], spike_grad=spike_grad, output=True)
    
    def forward(self, x_sequence):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spike_outputs = []
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x_sequence[step])
            # BatchNorm needs shape [N, C] or [N, C, L]
            # cur1 shape: [batch, hidden_dim]
            if cur1.shape[0] > 1: # Batch norm requires > 1 samples
                cur1 = self.bn1(cur1)
            spike1, mem1 = self.lif1(cur1, mem1)
            spike1 = self.dropout1(spike1)
            
            cur2 = self.fc2(spike1)
            if cur2.shape[0] > 1:
                cur2 = self.bn2(cur2)
            spike2, mem2 = self.lif2(cur2, mem2)
            spike2 = self.dropout2(spike2)
            
            cur3 = self.fc3(spike2)
            spike3, mem3 = self.lif3(cur3, mem3)
            
            spike_outputs.append(spike3)
        
        return torch.stack(spike_outputs, dim=0)


def train_enhanced_snn():
    """Huấn luyện SNN nâng cấp"""
    
    print("=" * 60)
    print("ĐÀO TẠO SNN NÂNG CẤP - PHIÊN BẢN 4.0")
    print("=" * 60)
    
    # Tạo dataset
    from train_nlp_snn_enhanced import EnhancedIntentDataset
    
    print("\n[1/5] Tạo dữ liệu huấn luyện...")
    all_samples = EnhancedIntentDataset.create_dataset(samples_per_intent=2000)
    
    # Tạo vectorizer nâng cấp
    vectorizer = EnhancedTextVectorizer(
        output_dim=OPTIMIZED_SNN_CONFIG['input_dim'],
        use_bigrams=True
    )
    
    # Tạo dataset PyTorch
    dataset = SNNDataset(all_samples, vectorizer)
    
    # Chia train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=OPTIMIZED_SNN_CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=OPTIMIZED_SNN_CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")
    
    # Khởi tạo model
    print("\n[2/5] Khởi tạo SNN model...")
    model = EnhancedSpikingIntentNeuron(OPTIMIZED_SNN_CONFIG)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   Device: {device}")
    
    # Optimizer và scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZED_SNN_CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Huấn luyện
    print("\n[3/5] Bắt đầu huấn luyện...")
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(OPTIMIZED_SNN_CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features_seq = features.unsqueeze(0).repeat(OPTIMIZED_SNN_CONFIG['num_steps'], 1, 1)
            features_seq = features_seq.to(device)
            labels = labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(features_seq)
            spike_counts = outputs.sum(dim=0)
            loss = criterion(spike_counts, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = spike_counts.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features_seq = features.unsqueeze(0).repeat(OPTIMIZED_SNN_CONFIG['num_steps'], 1, 1)
                features_seq = features_seq.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(features_seq)
                spike_counts = outputs.sum(dim=0)
                loss = criterion(spike_counts, labels)
                
                val_loss += loss.item()
                _, predicted = spike_counts.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Lưu history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Cập nhật learning rate
        scheduler.step(avg_val_loss)
        
        # In kết quả
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{OPTIMIZED_SNN_CONFIG['epochs']}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': OPTIMIZED_SNN_CONFIG
            }, 'models/snn_intent_enhanced.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= OPTIMIZED_SNN_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping tại epoch {epoch + 1}")
            break
    
    print(f"\n[4/5] Huấn luyện hoàn tất!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}% tại epoch {best_epoch}")
    print(f"   Model saved: models/snn_intent_enhanced.pth")
    
    # Vẽ biểu đồ
    print("\n[5/5] Vẽ biểu đồ đánh giá...")
    plot_training_results(history)
    plot_confusion_matrix(model, val_loader, device)
    
    return model, history


def plot_training_results(history):
    """Vẽ biểu đồ training và validation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Lưu
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/enhanced_nlp_training_metrics.png', dpi=150)
    print("   Saved: plots/enhanced_nlp_training_metrics.png")
    plt.close()


def plot_confusion_matrix(model, val_loader, device):
    """Vẽ ma trận nhầm lẫn"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features_seq = features.unsqueeze(0).repeat(OPTIMIZED_SNN_CONFIG['num_steps'], 1, 1)
            features_seq = features_seq.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(features_seq)
            spike_counts = outputs.sum(dim=0)
            _, predicted = spike_counts.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    intent_names = ['di_chuyen', 'dung_lai', 'bam_theo', 'hoi_thong_tin', 
                    'chao_hoi', 'cam_on', 'tam_biet', 'khac']
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=intent_names, yticklabels=intent_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Enhanced SNN')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/enhanced_nlp_confusion_matrix.png', dpi=150)
    print("   Saved: plots/enhanced_nlp_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    train_enhanced_snn()
