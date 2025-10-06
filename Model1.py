"""
================================================================================
MODEL LOG: 
================================================================================
1. Targets: Build a working, convolutional model skeleton using Conv/BN/GAP blocks and to keep the total parameters under the tight 8,000 constraint.

2. Results: Total parameters: 7,426. Best Train Acc: 97.22%. Best Test Acc: 98.390%. Last 3 Avg Test Acc: 98.377%.

3. Analysis: The model was extremely stable but failed due to slow convergence. The Max LR(0.01) was far too low, resulting in the model running out of time before reaching its potential. The low accuracy across the board showed the network was simply under-trained. Hence, decided that the target for the next experiment must be to slightly increase the learning rate to see if a small speed increase would be enough.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# =========================================
# 1. Device Setup (with MPS for Mac Silicon)
# =========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# =========================================
# 2. Model Definition (OptimizedNetV8)
# =========================================
class OptimizedNetV8(nn.Module):
    def __init__(self, dropout_rate=0.05):
        super(OptimizedNetV8, self).__init__()
        
        # --- Block 1 (Initial) --- 28x28 -> 28x28 (8 channels)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # --- Block 2 (Feature Expansion) --- 28x28 -> 28x28 (16 channels)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 1 --- MaxPool reduces size: 28x28x16 -> 14x14x16
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # --- Block 3 (Mid-Level) --- 14x14 -> 14x14 (16 channels)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1) 
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 2 --- MaxPool reduces size: 14x14x16 -> 7x7x16
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # --- Block 4 (Deep Features) --- 7x7 -> 7x7 (24 channels)
        self.conv4 = nn.Conv2d(16, 24, 3, padding=1) 
        self.bn4 = nn.BatchNorm2d(24)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # --- Final Channel Reduction (1x1 Conv) --- 7x7x24 -> 7x7x10 (10 classes)
        self.conv_final = nn.Conv2d(24, 10, 1) 
        
        # --- Global Average Pooling (Classification) --- 10x7x7 -> 10x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(x) 
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Final Channel Reduction
        x = F.relu(self.conv_final(x)) 
        
        # Global Average Pooling (Classification)
        x = self.gap(x)
        
        # Flatten and apply Log Softmax
        x = x.view(x.size(0), -1) 
        return F.log_softmax(x, dim=1)

# =========================================
# 3. Data Loading and Preprocessing
# =========================================
def load_mnist_data():
    # Mean and Std Dev for MNIST (used for normalization)
    # The calculated values are approx (0.1307,) and (0.3081,)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use a small batch size for training to aid MPS/GPU memory efficiency
    # Larger batch size (e.g., 128) can be used on powerful GPUs
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# =========================================
# 4. Training and Testing Functions
# =========================================
def train_model(model, device, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Update LR based on scheduler
        scheduler.step()

        # Get prediction and calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(
            f'Train Loss={loss.item():.4f} LR={scheduler.get_last_lr()[0]:.6f} Acc={100*correct/processed:.2f}%'
        )

    return 100. * correct / processed

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.3f}%)'
    )
    return test_acc

# =========================================
# 5. Main Execution
# =========================================
def run_experiment():
    
    # --- Configuration ---
    EPOCHS = 15
    MAX_LR = 0.01 # Peak Learning Rate for OneCycleLR
    
    # --- Data and Model Initialization ---
    train_loader, test_loader = load_mnist_data()
    model = OptimizedNetV8(dropout_rate=0.02).to(device) # Lowered dropout to 0.02
    
    # Calculate and check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 OptimizedNetV8 Parameters: {total_params:,}")
    print(f"✅ Parameter constraint: {'PASS' if total_params < 8000 else 'FAIL'} (< 8,000)")
    
    # Optimizer (SGD with momentum is often better for CNN convergence)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) 

    # OneCycleLR Scheduler (Crucial for fast high-accuracy training)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS, 
        pct_start=0.3, # Peak LR reached at 30% of training
        div_factor=10, 
        final_div_factor=10
    )

    # --- Training Loop ---
    test_accuracies = []
    print(f"\n🚀 Starting training for {EPOCHS} epochs with OneCycleLR (Max LR: {MAX_LR})...")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_acc = train_model(model, device, train_loader, optimizer, scheduler)
        test_acc = test_model(model, device, test_loader)
        test_accuracies.append(test_acc)

    # --- Final Analysis ---
    if len(test_accuracies) >= 3:
        final_accs = test_accuracies[-3:]
        avg_final_acc = sum(final_accs) / 3
        
        print("\n==============================================")
        print("✅ FINAL 3 EPOCHS ANALYSIS")
        print(f"Test Accuracies: {final_accs}")
        print(f"Average Accuracy: {avg_final_acc:.3f}%")
        
        if avg_final_acc >= 99.4:
            print("🏆 STATUS: SUCCESS! Target (>=99.4%) achieved consistently.")
        else:
            print("⚠️ STATUS: CLOSE, but the consistent 99.4% target was NOT met.")
        print("==============================================")


if __name__ == '__main__':
    run_experiment()
