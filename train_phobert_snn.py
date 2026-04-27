import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

from train_nlp_snn_enhanced import EnhancedIntentDataset
from nlp.intent_classifier import PhoBERTVectorizer, SpikingIntentNeuron
from optimized_snn_config import OPTIMIZED_SNN_CONFIG

# Cap nhat config cho PhoBERT
CONFIG = OPTIMIZED_SNN_CONFIG.copy()
CONFIG['input_dim'] = 768
CONFIG['epochs'] = 50

class PrecomputedSNNDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def extract_phobert_features(samples, vectorizer, batch_size=32):
    print(f"Dang trich xuat dac trung PhoBERT cho {len(samples)} mau...")
    features = []
    labels = []
    intent_to_idx = {
        'di_chuyen': 0, 'dung_lai': 1, 'bam_theo': 2,
        'hoi_thong_tin': 3, 'chao_hoi': 4, 'cam_on': 5,
        'tam_biet': 6, 'khac': 7
    }
    
    start_time = time.time()
    for i, sample in enumerate(samples):
        feat = vectorizer.transform(sample['text'])
        features.append(feat)
        labels.append(torch.LongTensor([intent_to_idx[sample['intent']]]))
        
        if (i+1) % 500 == 0:
            print(f"  Da xu ly {i+1}/{len(samples)} mau...")
            
    print(f"Hoan tat trich xuat sau {time.time() - start_time:.2f} giay.")
    return torch.stack(features), torch.stack(labels)

def train_phobert_snn():
    print("=" * 60)
    print("DAO TAO SNN VOI PHOBERT EMBEDDINGS (768-dim)")
    print("=" * 60)
    
    # 1. Tao du lieu (giam xuong 500 mau/intent de tiet kiem thoi gian rut trich)
    print("\n[1/4] Tao du lieu huan luyen...")
    all_samples = EnhancedIntentDataset.create_dataset(samples_per_intent=500)
    
    # 2. Trich xuat vector PhoBERT
    print("\n[2/4] Khoi tao PhoBERT va trich xuat dac trung (chay offline)...")
    vectorizer = PhoBERTVectorizer()
    
    X_features, y_labels = extract_phobert_features(all_samples, vectorizer)
    
    dataset = PrecomputedSNNDataset(X_features, y_labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    # 3. Khoi tao model
    print("\n[3/4] Khoi tao SNN model (input_dim=768)...")
    model = SpikingIntentNeuron(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=8,  # 8 intents
        num_steps=CONFIG['num_steps']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Huan luyen
    print("\n[4/4] Bat dau huan luyen SNN...")
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features_seq = features.unsqueeze(0).repeat(CONFIG['num_steps'], 1, 1)
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
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features_seq = features.unsqueeze(0).repeat(CONFIG['num_steps'], 1, 1)
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
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, 'models/snn_intent_phobert.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= 15:
            print(f"\nEarly stopping tai epoch {epoch + 1}")
            break
            
    print(f"\n[XONG] Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Da luu SNN + PhoBERT tai: models/snn_intent_phobert.pth")

if __name__ == "__main__":
    train_phobert_snn()
