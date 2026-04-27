import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

class DNASequenceDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X_seq = data['X_seq']  # (4 channels, 25000 length)
        self.X_struct = data['X_struct']  # (4 channels, 130 length)
        self.y = data['y']
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.X_seq[idx])
        structure = torch.from_numpy(self.X_struct[idx])
        label = torch.tensor(self.y[idx], dtype=torch.float)
        return sequence, structure, label

class SequenceCNN(nn.Module):
    def __init__(self, seq_length=25000):
        super(SequenceCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=30, padding=14)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=40, padding=19)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=50, padding=24)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # 计算卷积后的长度
        def conv_output_length(length, kernel_size, padding, stride=1):
            return (length + 2*padding - kernel_size) // stride + 1
        
        l = seq_length
        l = conv_output_length(l, 30, 14) // 2  # conv1 + pool
        l = conv_output_length(l, 40, 19) // 2  # conv2 + pool
        l = conv_output_length(l, 50, 24) // 2  # conv3 + pool
        
        self.fc = nn.Linear(128 * l, 128)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        x = torch.relu(self.fc(x))
        return x

class StructureCNN(nn.Module):
    def __init__(self, struct_length=130):
        super(StructureCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        

        l = struct_length
        l = l // 2  # conv1 + pool
        l = l // 2  # conv2 + pool
        l = l // 2  # conv3 + pool
        
        self.fc = nn.Linear(128 * l, 128)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        x = torch.relu(self.fc(x))
        return x

class CombinedModel(nn.Module):
    def __init__(self, seq_length=25000, struct_length=130):
        super(CombinedModel, self).__init__()
        self.seq_cnn = SequenceCNN(seq_length)
        self.struct_cnn = StructureCNN(struct_length)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seq, struct):
        seq_features = self.seq_cnn(seq)
        struct_features = self.struct_cnn(struct)
        combined = torch.cat([seq_features, struct_features], dim=1)
        x = self.dropout(combined)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

def evaluate_model(model, data_loader, criterion, device, test_set=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for seq, struct, labels in data_loader:
            seq, struct, labels = seq.to(device), struct.to(device), labels.to(device)
            outputs = model(seq.float(), struct.float())
            loss = criterion(outputs, labels.unsqueeze(1))
            
            running_loss += loss.item() * seq.size(0)
            preds = (outputs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    if test_set:
        print("\nTest Set Performance:")
    else:
        print("\nValidation Set Performance:")
    print(classification_report(all_labels, all_preds, target_names=['OtherTE', 'Helitron']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return avg_loss, accuracy, precision, recall, f1

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epochs=10, model_name='model'):
    best_val_f1 = 0.0
    metrics = {
        'train_loss': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': []
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for seq, struct, labels in progress:
            seq, struct, labels = seq.to(device), struct.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq.float(), struct.float())
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * seq.size(0)
            progress.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        metrics['train_loss'].append(epoch_loss)
        

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device, test_set=False)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        metrics['val_f1'].append(val_f1)
        scheduler.step(val_loss)
        

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
            model, test_loader, criterion, device, test_set=True)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        metrics['test_precision'].append(test_precision)
        metrics['test_recall'].append(test_recall)
        metrics['test_f1'].append(test_f1)
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'best_{model_name}_oxy.pth')
            print(f'New best model saved with val_f1: {best_val_f1:.4f}')
    

    plot_training_metrics(metrics, model_name)
    
    return model, metrics

def plot_training_metrics(metrics, model_name):
    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.plot(metrics['test_loss'], label='Test Loss')
    plt.legend()
    plt.title('Loss Curve')
    

    plt.subplot(2, 3, 2)
    plt.plot(metrics['val_acc'], label='Val Accuracy')
    plt.plot(metrics['test_acc'], label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    

    plt.subplot(2, 3, 3)
    plt.plot(metrics['val_precision'], label='Val Precision')
    plt.plot(metrics['test_precision'], label='Test Precision')
    plt.legend()
    plt.title('Precision Curve')
    

    plt.subplot(2, 3, 4)
    plt.plot(metrics['val_recall'], label='Val Recall')
    plt.plot(metrics['test_recall'], label='Test Recall')
    plt.legend()
    plt.title('Recall Curve')
    

    plt.subplot(2, 3, 5)
    plt.plot(metrics['val_f1'], label='Val F1 Score')
    plt.plot(metrics['test_f1'], label='Test F1 Score')
    plt.legend()
    plt.title('F1 Score Curve')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_metrics.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    

    train_dataset = DNASequenceDataset('train_fusarium_oxysporum_separate.npz')
    val_dataset = DNASequenceDataset('val_fusarium_oxysporum_separate.npz')
    test_dataset = DNASequenceDataset('test_fusarium_oxysporum_separate.npz')
    #val_dataset = DNASequenceDataset('val_fusarium_oxysporum_separate.npz')
    #test_dataset = DNASequenceDataset('test_fusarium_oxysporum_separate.npz') 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    

    model = CombinedModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print('Training combined model...')
    model, metrics = train_model(
        model, train_loader, val_loader, test_loader, 
        criterion, optimizer, device, args.epochs, 'combined_model'
    )
    

    model.load_state_dict(torch.load('best_combined_model_oxy.pth'))
    #model.load_state_dict(torch.load('best_combined_model.pth'))
    print('\nFinal Model Performance:')
    

    val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
        model, val_loader, criterion, device, test_set=False)
    print(f'\nValidation Set:')
    print(f'Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')
    print(f'Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
    

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
        model, test_loader, criterion, device, test_set=True)
    print(f'\nTest Set:')
    print(f'Loss: {test_loss:.4f} | Acc: {test_acc:.4f}')
    print(f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}')

if __name__ == '__main__':
    main()
