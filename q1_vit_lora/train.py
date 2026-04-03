import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import optuna
import argparse
from tqdm import tqdm
from dataset import get_dataloaders
from model import get_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    lora_grads = []
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, targets)
        loss.backward()
        
        
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if 'lora' in name and param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        lora_grads.append(grad_norm ** 0.5)
            
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    avg_grad = sum(lora_grads)/len(lora_grads) if lora_grads else 0
    return total_loss / len(loader), 100. * correct / total, avg_grad

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    class_correct = [0]*100
    class_total = [0]*100
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
                
    return total_loss / len(loader), 100. * correct / total, class_correct, class_total

def run_experiment(use_lora=False, lora_r=8, lora_alpha=8, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_name = f"ViT-S_Base" if not use_lora else f"ViT-S_LoRA_r{lora_r}_alpha{lora_alpha}"
    wandb.init(project="dlops-ass5-q1", name=run_name, reinit=True)
    
    trainloader, testloader = get_dataloaders(batch_size=64)
    model = get_model(use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_parameters": trainable_params})
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0
    best_class_acc = None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, avg_lora_grad = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc, class_correct, class_total = val_epoch(model, testloader, criterion, device)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.3f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.3f}, Val Acc {val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "avg_lora_gradient_norm": avg_lora_grad
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_class_acc = [(100. * c / t) if t > 0 else 0 for c, t in zip(class_correct, class_total)]
            torch.save(model.state_dict(), f"best_model_{run_name}.pth")
            
    wandb.run.summary["best_val_accuracy"] = best_acc
    if best_class_acc:
        data = [[str(i), acc] for i, acc in enumerate(best_class_acc)]
        table = wandb.Table(data=data, columns=["Class ID", "Accuracy"])
        wandb.log({"Class-wise Test Accuracy": wandb.plot.bar(table, "Class ID", "Accuracy", title="Class-wise Accuracy")})
        
    wandb.finish()
    return best_acc

def optuna_objective(trial):
    lora_r = trial.suggest_categorical("lora_r", [2, 4, 8])
    lora_alpha = trial.suggest_categorical("lora_alpha", [2, 4, 8])
    return run_experiment(use_lora=True, lora_r=lora_r, lora_alpha=lora_alpha, epochs=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_optuna", action="store_true")
    args = parser.parse_args()
    
    if args.run_baseline:
        print("Running Baseline without LoRA...")
        run_experiment(use_lora=False, epochs=10)
        
    if args.run_optuna:
        print("Running Optuna tuning for LoRA hyperparameters...")
        search_space = {"lora_r": [2, 4, 8], "lora_alpha": [2, 4, 8]}
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=9)
        
        print("Best Optuna Trial:")
        print(study.best_trial.value)
        print("Best Optuna Params:")
        print(study.best_trial.params)
