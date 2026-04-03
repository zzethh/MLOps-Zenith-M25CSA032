import matplotlib.pyplot as plt
import numpy as np

# Hardcoded best metrics (Rank=8, Alpha=8) from our previous training:
epochs = [1, 2, 3, 4, 5]
train_loss = [0.566, 0.265, 0.185, 0.137, 0.065]
val_loss = [0.355, 0.381, 0.395, 0.428, 0.400]
train_acc = [84.32, 91.57, 93.91, 95.42, 97.93]
val_acc = [88.96, 88.64, 88.92, 88.30, 89.99]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
axes[0].plot(epochs, train_loss, 'o-', label='Train Loss', color='blue')
axes[0].plot(epochs, val_loss, 'x--', label='Validation Loss', color='red')
axes[0].set_title('Training and Validation Loss (Best: R=8, Alpha=8)')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()
axes[0].grid(True)

# Plot Accuracy
axes[1].plot(epochs, train_acc, 'o-', label='Train Accuracy', color='blue')
axes[1].plot(epochs, val_acc, 'x--', label='Validation Accuracy', color='red')
axes[1].set_title('Training and Validation Accuracy (Best: R=8, Alpha=8)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("../report/q1_best_lora_graphs.png", dpi=150, bbox_inches="tight")
print("Saved q1_best_lora_graphs.png")
