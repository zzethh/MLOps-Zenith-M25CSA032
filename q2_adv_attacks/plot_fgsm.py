import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    return testloader

device = 'cpu'
model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
model.eval()

testloader = load_data()
criterion = torch.nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=criterion,
    optimizer=None,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

x_test, y_test = [], []
for data, target in testloader:
    x_test.append(data.numpy()[0])
    y_test.append(target.numpy()[0])
    if len(x_test) == 10:
        break
x_test = np.array(x_test)
y_test = np.array(y_test)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
fig.suptitle("FGSM Visual Comparison (eps=0.15): Clean vs Adversarial (IBM ART)", fontsize=14)

attack_art = FastGradientMethod(estimator=classifier, eps=0.15)
x_art_adv = attack_art.generate(x=x_test)

for i in range(10):
    axes[0, i].imshow(x_test[i].transpose(1, 2, 0))
    axes[0, i].set_title("Clean", fontsize=10)
    axes[0, i].axis("off")
    
    axes[1, i].imshow(x_art_adv[i].transpose(1, 2, 0))
    axes[1, i].set_title("Adv (ART)", fontsize=10)
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("../report/fgsm_visual_comparison.png", dpi=150, bbox_inches="tight")
print("Saved fgsm_visual_comparison.png")
