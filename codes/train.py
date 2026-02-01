import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import numpy as np
from dataset import get_dataloaders
from model import SimpleCNN, count_flops
from utils import plot_grad_flow, plot_weight_flow, plot_confusion_matrix, plot_class_accuracy, plot_prediction_grid

CONFIG = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "project": "lab2_cifar10",
    "save_dir": "./checkpoints"
}

def train():
    wandb.init(project=CONFIG["project"], config=CONFIG, name="CIFAR10_Final_Run")
    
    train_loader, test_loader = get_dataloaders(CONFIG["batch_size"])
    model = SimpleCNN().to(CONFIG["device"])
    
    flops = count_flops(model)
    print(f"Model FLOPs: {flops/1e9:.4f} GFLOPs")
    wandb.log({"GFLOPs": flops/1e9})

    wandb.watch(model, log="all", log_freq=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10}")
    print("-" * 65)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if i % 200 == 0:
                grad_fig = plot_grad_flow(model.named_parameters())
                weight_fig = plot_weight_flow(model.named_parameters())
                
                wandb.log({
                    "gradients": wandb.Image(grad_fig),
                    "weights": wandb.Image(weight_fig)
                }, commit=False)

            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct_val / total_val
        
        print(f"{epoch+1:^6} | {avg_train_loss:^12.4f} | {train_acc:^10.2f} | {avg_val_loss:^10.4f} | {val_acc:^10.2f}")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

    print("-" * 65)
    print("Training Complete. Calculating Final Test Accuracy...")

    model.eval()
    all_preds = []
    all_labels = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_test_acc = 100 * correct_test / total_test
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    wandb.log({"test_accuracy": final_test_acc})

    print("Generating report visualizations...")

    cm_img = plot_confusion_matrix(all_labels, all_preds, classes)
    wandb.log({"confusion_matrix": wandb.Image(cm_img)})
    
    acc_img = plot_class_accuracy(all_labels, all_preds, classes)
    wandb.log({"class_accuracy": wandb.Image(acc_img)})
    
    pred_img = plot_prediction_grid(model, test_loader, classes, CONFIG["device"])
    wandb.log({"predictions": wandb.Image(pred_img)})

    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])
    
    save_path = os.path.join(CONFIG["save_dir"], "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
    
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train()