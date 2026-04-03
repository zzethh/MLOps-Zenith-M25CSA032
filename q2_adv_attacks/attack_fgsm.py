import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import wandb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

def fgsm_attack_scratch(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    return testloader

def run_attacks():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(num_classes=10)
    try:
        model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
    except FileNotFoundError:
        print("Model weights not found. Please train first.")
        return
        
    model.to(device)
    model.eval()
    
    testloader = load_data()
    criterion = nn.CrossEntropyLoss()
    
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    wandb.init(project="dlops-ass5-q2", name="FGSM_Attacks", reinit=True)
    table_fgsm = wandb.Table(columns=["Attack Framework", "Epsilon", "Clean Image", "Adversarial Image", "True Label", "Predicted Label"])
    
    print("--- Scratch FGSM ---")
    scratch_accuracies = []
    for eps in epsilons:
        correct = 0
        total = 0
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                total += 1
                continue
                
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            perturbed_data = fgsm_attack_scratch(data, eps, data_grad)
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
            
            if total < 10 and eps == 0.15:
                clean_img = wandb.Image(data.squeeze().cpu().detach().numpy().transpose(1,2,0))
                adv_img = wandb.Image(perturbed_data.squeeze().cpu().detach().numpy().transpose(1,2,0))
                table_fgsm.add_data("Scratch FGSM", eps, clean_img, adv_img, target.item(), final_pred.item())
                
            total += 1
            if total > 500:
                break
        acc = correct / float(total)
        scratch_accuracies.append(acc)
        print(f"Scratch FGSM Epsilon: {eps}\tTest Accuracy = {correct} / {total} = {acc}")

    print("--- ART FGSM ---")
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=None,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    
    art_accuracies = []
    x_test, y_test = [], []
    for data, target in testloader:
        x_test.append(data.numpy()[0])
        y_test.append(target.numpy()[0])
        if len(x_test) == 500:
            break
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    for eps in epsilons:
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        x_test_adv = attack.generate(x=x_test)
        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        art_accuracies.append(accuracy)
        print(f"ART FGSM Epsilon: {eps}\tTest Accuracy = {accuracy}")
        
        if eps == 0.15:
            for i in range(10):
                c_img = wandb.Image(x_test[i].transpose(1,2,0))
                a_img = wandb.Image(x_test_adv[i].transpose(1,2,0))
                t_lbl = y_test[i]
                p_lbl = np.argmax(predictions[i])
                table_fgsm.add_data("IBM ART FGSM", eps, c_img, a_img, t_lbl, p_lbl)
                
    wandb.log({"Adversarial Samples Table": table_fgsm})
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle("FGSM Visual Comparison (ε=0.15): Clean vs Adversarial", fontsize=14)
    attack_scratch = FastGradientMethod(estimator=classifier, eps=0.15)
    x_scratch_adv = attack_scratch.generate(x=x_test[:10])
    attack_art = FastGradientMethod(estimator=classifier, eps=0.15)
    x_art_adv = attack_art.generate(x=x_test[:10])
    for i in range(10):
        axes[0, i].imshow(x_test[i].transpose(1, 2, 0))
        axes[0, i].set_title("Clean", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(x_art_adv[i].transpose(1, 2, 0))
        axes[1, i].set_title("Adv (ART)", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("../report/fgsm_visual_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fgsm_visual_comparison.png")

    wandb.finish()

if __name__ == '__main__':
    run_attacks()
