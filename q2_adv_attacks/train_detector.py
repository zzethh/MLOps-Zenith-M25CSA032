import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

class AdvDetectorDataset(Dataset):
    def __init__(self, clean_data, adv_data, clean_labels, adv_labels):
        self.data = torch.cat([clean_data, adv_data], dim=0)
        self.labels = torch.cat([clean_labels, adv_labels], dim=0)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_adv_data(model, testloader, attack_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=None,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    
    if attack_type == 'PGD':
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=10)
    elif attack_type == 'BIM':
        attack = BasicIterativeMethod(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=10)
        
    clean_images = []
    adv_images = []
    clean_binary_labels = []
    adv_binary_labels = []
    
    for data, target in testloader:
        clean_images.append(data)
        clean_binary_labels.extend([0]*data.size(0))
        
        x_adv = attack.generate(x=data.numpy())
        adv_images.append(torch.from_numpy(x_adv))
        adv_binary_labels.extend([1]*data.size(0))
        
    clean_images = torch.cat(clean_images, dim=0)
    adv_images = torch.cat(adv_images, dim=0)
    clean_binary_labels = torch.tensor(clean_binary_labels)
    adv_binary_labels = torch.tensor(adv_binary_labels)
    
    table_adv = wandb.Table(columns=["Attack Framework", "Clean Image", "Adversarial Image"])
    for i in range(10):
        c_img_w = wandb.Image(clean_images[i].numpy().transpose(1,2,0))
        a_img_w = wandb.Image(adv_images[i].numpy().transpose(1,2,0))
        table_adv.add_data(f"IBM ART {attack_type}", c_img_w, a_img_w)
    wandb.log({f"Iterative Samples Table ({attack_type})": table_adv})    
        
    return clean_images, adv_images, clean_binary_labels, adv_binary_labels

def train_detector(attack_type):
    wandb.init(project="dlops-ass5-q2", name=f"Detector_{attack_type}", reinit=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_model = models.resnet18(num_classes=10)
    try:
        base_model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
    except FileNotFoundError:
        print("Please train resnet18 first")
        return
        
    base_model.to(device)
    base_model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Generating data using {attack_type}...")
    c_img, a_img, c_lbl, a_lbl = generate_adv_data(base_model, loader, attack_type)
    
    full_dataset = AdvDetectorDataset(c_img, a_img, c_lbl, a_lbl)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    detector = models.resnet34(pretrained=False)
    detector.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    detector.maxpool = nn.Identity()
    detector.fc = nn.Linear(detector.fc.in_features, 2)
    detector.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=1e-3)
    
    print(f"Training Detector for {attack_type}...")
    for epoch in range(1, 11):
        detector.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = detector(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        detector.eval()
        test_loss, t_correct, t_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = detector(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                t_total += targets.size(0)
                t_correct += predicted.eq(targets).sum().item()
                
        print(f"Epoch {epoch}: Train Acc {100.*correct/total:.2f}% | Test Acc {100.*t_correct/t_total:.2f}%")
        wandb.log({"epoch": epoch, "train_acc": 100.*correct/total, "test_acc": 100.*t_correct/t_total})
    wandb.finish()

if __name__ == '__main__':
    train_detector('PGD')
    train_detector('BIM')
