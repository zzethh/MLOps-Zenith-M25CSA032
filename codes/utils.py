import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import io
from PIL import Image

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n.replace(".weight", ""))
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="teal", label="Max Gradient")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=1.0, lw=1, color="navy", label="Avg Gradient")
    ax.axhline(0, color="k", lw=2)
    ax.set_xticks(range(len(ave_grads)))
    ax.set_xticklabels(layers, rotation="vertical")
    ax.set_xlim(left=-0.5, right=len(ave_grads) - 0.5)
    ax.set_xlabel("Model Layers")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_title("Gradient Flow (Layer-wise Gradients)")
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image

def plot_weight_flow(named_parameters):
    ave_weights = []
    max_weights = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n.replace(".weight", ""))
            ave_weights.append(p.data.abs().mean().cpu().item())
            max_weights.append(p.data.abs().max().cpu().item())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(max_weights)), max_weights, alpha=0.3, lw=1, color="purple", label="Max Weight")
    ax.bar(np.arange(len(max_weights)), ave_weights, alpha=1.0, lw=1, color="magenta", label="Avg Weight")
    ax.axhline(0, color="k", lw=2)
    ax.set_xticks(range(len(ave_weights)))
    ax.set_xticklabels(layers, rotation="vertical")
    ax.set_xlim(left=-0.5, right=len(ave_weights) - 0.5)
    ax.set_xlabel("Model Layers")
    ax.set_ylabel("Weight Magnitude")
    ax.set_title("Weight Flow (Layer-wise Parameters)")
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, square=True)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image

def plot_class_accuracy(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(classes))
    bars = plt.bar(classes, class_acc * 100, color=colors)
    
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image

def plot_prediction_grid(model, test_loader, classes, device, num_images=25):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)
    
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(12, 12))
    for idx in range(num_images):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        title_color = "green" if preds[idx] == labels[idx] else "red"
        ax.set_title(f"P: {classes[preds[idx]]}\nA: {classes[labels[idx]]}", 
                     color=title_color, fontsize=9, fontweight='bold')
    
    plt.suptitle("Predictions vs Actuals (Green=Correct, Red=Incorrect)", y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image