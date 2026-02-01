import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_flops(model, input_size=(1, 3, 32, 32)):
    total_flops = 0
    
    def conv_flops(m, i, o):
        b, c_in, h, w = i[0].shape
        c_out, h_out, w_out = o.shape[1:]
        k_h, k_w = m.kernel_size
        return 2 * c_in * c_out * k_h * k_w * h_out * w_out

    def linear_flops(m, i, o):
        return 2 * m.in_features * m.out_features

    model_flops = 0
    def hook(m, i, o):
        nonlocal model_flops
        if isinstance(m, nn.Conv2d):
            model_flops += conv_flops(m, i, o)
        elif isinstance(m, nn.Linear):
            model_flops += linear_flops(m, i, o)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))

    dummy = torch.randn(input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy)

    for h in hooks: h.remove()
    return model_flops