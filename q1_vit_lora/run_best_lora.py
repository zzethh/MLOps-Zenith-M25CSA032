import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm

def train_best_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainloader, testloader = get_dataloaders(batch_size=64)
    model = get_model(use_lora=True, lora_r=8, lora_alpha=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 10
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch} Train", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        train_loss /= len(trainloader)
        
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc=f"Epoch {epoch} Val", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * correct / total
        val_loss /= len(testloader)
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.3f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.3f}, Val Acc {val_acc:.2f}%")
        
    torch.save(model.state_dict(), "best_model_ViT-S_LoRA_r8_alpha8.pth")
    
    # Plotting Train/Val Loss and Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, epochs + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', marker='x')
    axes[0].set_title('Training and Validation Loss (LoRA Rank=8, Alpha=8)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='x')
    axes[1].set_title('Training and Validation Accuracy (LoRA Rank=8, Alpha=8)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("../report/q1_best_lora_graphs.png", dpi=150)
    print("Saved q1_best_lora_graphs.png")
    
    # Dump table format for easy markdown pasting
    print("\nMarkdown Table for 10 Epochs:")
    for i in range(10):
        print(f"| {i+1} | {history['train_loss'][i]:.3f} | {history['val_loss'][i]:.3f} | {history['train_acc'][i]:.2f}% | {history['val_acc'][i]:.2f}% |")

if __name__ == "__main__":
    train_best_lora()
