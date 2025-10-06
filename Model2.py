"""
================================================================================
MODEL LOG: 
================================================================================

Targets: Target was to fix the slow convergence by slightly increasing the Max LR to 0.015 and using the slightly smaller 6,402 parameter count to see if the small speed boost was sufficient.

Results: Total parameters: 6,402. Best Train Acc: 96.92%. Best Test Acc: 97.86%. Last 3 Avg Test Acc: 97.810%.
Analysis: This run was a failure. Despite the faster LR, the performance got worse than V1. This proved two things: 0.015 was still too slow, and the 6,402 parameter structure was not as robust as hoped. The conclusion was firm: I needed to find a better base structure before trying a massive speed increase.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import sys # Used for clean print statements

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
# 2. Model Definition (OptimizedNetV9)
# Parameters: ~7,800 (Maximized Capacity)
# =========================================

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class OptimizedNetV11(nn.Module):
    def __init__(self, dropout_rate=0.01):
        super(OptimizedNetV11, self).__init__()
        
        # --- Block 1: Initial (1x28x28 -> 8x28x28) ---
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) 
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # --- Block 2: Feature Expansion (8x28x28 -> 10x28x28) ---
        self.conv2 = nn.Conv2d(8, 10, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(10)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 1: Pool (10x28x28 -> 10x14x14) ---
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # --- Block 3: Mid-Level (10x14x14 -> 12x14x14) ---
        self.conv3 = nn.Conv2d(10, 12, 3, padding=1) 
        self.bn3 = nn.BatchNorm2d(12)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 2: Pool (12x14x14 -> 12x7x7) ---
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        # --- Block 4: Deep Features (12x7x7 -> 16x7x7) ---
        self.conv4 = nn.Conv2d(12, 16, 3, padding=1) 
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # --- Block 5: Final Feature Refinement (16x7x7 -> 16x7x7) ---
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1) 
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # --- Final Classification (1x1 Conv + GAP) ---
        # 1x1 Conv for 10 classes: 16x7x7 -> 10x7x7
        self.conv_final = nn.Conv2d(16, 10, 1)
        
        # Global Average Pooling: 10x7x7 -> 10x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Total Parameters (Conv/FC): 72 + 720 + 1080 + 1728 + 2304 + 160 = 6064 
        # Total Parameters (Approx. with BN): ~6,400 (PASS!)
        
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(x) 
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        x = self.conv_final(x)
        x = self.gap(x)
        
        x = x.view(x.size(0), -1) 
        return F.log_softmax(x, dim=1)

class OptimizedNetV10(nn.Module):
    def __init__(self, dropout_rate=0.01):
        super(OptimizedNetV10, self).__init__()
        
        # --- Block 1: Initial Feature Extraction (1x28x28 -> 8x28x28) ---
        # Parameters: 8*3*3 + 8 = 80 (Keep channels low)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) 
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 1: Channel Expansion (1x1 Conv) ---
        # Parameters: 8*16*1*1 + 16 = 144
        self.conv2 = nn.Conv2d(8, 16, 1) # 1x1 Conv is cheap
        self.bn2 = nn.BatchNorm2d(16)
        
        # --- Transition 2: MaxPool and Feature Learning ---
        # 16x28x28 -> 16x14x14
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # --- Block 2: Mid-Level Feature Learning (16x14x14 -> 16x14x14) ---
        # Parameters: 16*16*3*3 + 16 = 2320 (The largest block)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1) 
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # --- Block 3: Deep Feature Learning (16x14x14 -> 24x14x14) ---
        # Parameters: 16*24*3*3 + 24 = 3480 
        self.conv4 = nn.Conv2d(16, 24, 3, padding=1) 
        self.bn4 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # --- Final Classification Layers (1x1 Convs + GAP) ---
        # C1: Channel Reduction 24x14x14 -> 10x14x14
        # Parameters: 24*10*1*1 + 10 = 250
        self.conv_final = nn.Conv2d(24, 10, 1)
        
        # Global Average Pooling: 10x14x14 -> 10x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Total Convolution/FC Parameters: 80 + 144 + 2320 + 3480 + 250 = 6274
        # Total BN Parameters: (8+16+16+24)*4 = 256
        # Total Learnable Parameters: 6,274 + 256 = ~6,530 (PASS!)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Transition 1 (1x1 to expand channels)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.dropout2(F.relu(self.bn3(self.conv3(x))))
        
        # Block 3 (crucial for deep features on 14x14 map)
        x = self.dropout3(F.relu(self.bn4(self.conv4(x))))
        
        # Classification
        x = self.conv_final(x)
        x = self.gap(x)
        
        x = x.view(x.size(0), -1) 
        return F.log_softmax(x, dim=1)
        
class OptimizedNetV9(nn.Module):
    def __init__(self, dropout_rate=0.01): # Lowered Dropout to 0.01
        super(OptimizedNetV9, self).__init__()
        
        # --- Block 1 (Initial) --- 28x28 -> 28x28 
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1) # Increased from 8
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # --- Block 2 (Feature Expansion) --- 28x28 -> 28x28 
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1) # Increased from 16
        self.bn2 = nn.BatchNorm2d(20)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 1 --- MaxPool: 28x28x20 -> 14x14x20
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # --- Block 3 (Mid-Level) --- 14x14 -> 14x14 
        self.conv3 = nn.Conv2d(20, 20, 3, padding=1) # Increased from 16
        self.bn3 = nn.BatchNorm2d(20)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # --- Transition 2 --- MaxPool: 14x14x20 -> 7x7x20
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # --- Block 4 (Deep Features) --- 7x7 -> 7x7 
        self.conv4 = nn.Conv2d(20, 32, 3, padding=1) # Increased from 24 (Maxed out)
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # --- FINAL CLASSIFICATION BLOCK (Two 1x1 Convs for Deep Classification) ---
        
        # C1: Channel Reduction and Mixing 7x7x32 -> 7x7x16
        self.conv_final_1x1_1 = nn.Conv2d(32, 16, 1) 
        self.bn_final_1x1_1 = nn.BatchNorm2d(16)
        
        # C2: Final Classification Layer 7x7x16 -> 7x7x10 (10 classes)
        self.conv_final_1x1_2 = nn.Conv2d(16, 10, 1) 
        
        # --- Global Average Pooling (Classification) --- 10x7x7 -> 10x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Block 1-2 + Pool 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        
        # Block 3 + Pool 2
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(x) 
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Final Classification Block (Increased Depth)
        x = F.relu(self.bn_final_1x1_1(self.conv_final_1x1_1(x)))
        x = self.conv_final_1x1_2(x) # No ReLU, final output layer
        
        # Global Average Pooling (Classification)
        x = self.gap(x)
        
        # Flatten and apply Log Softmax
        x = x.view(x.size(0), -1) 
        return F.log_softmax(x, dim=1)


# =========================================
# 3. Data Loading and Preprocessing
# =========================================
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use a batch size that is efficient for the device
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, test_loader

# =========================================
# 4. Training and Testing Functions
# =========================================
def train_model(model, device, train_loader, optimizer, scheduler):
    model.train()
    # Ensure tqdm is used correctly for notebooks
    pbar = tqdm(train_loader, desc='Training', file=sys.stdout) # Added file=sys.stdout for notebooks
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        # The data.to(device) calls might trigger the repetitive logs. 
        # By removing the print statements from get_device(), we minimize the noise.
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
        
        # tqdm's update method ensures the line is rewritten
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
    
    # Use sys.stdout.write for cleaner output in Jupyter
    sys.stdout.write(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.3f}%)'
    )
    return test_acc

# =========================================
# 5. Main Execution
# =========================================
def run_experiment():
    
    # --- Configuration ---
    EPOCHS = 15
    MAX_LR = 0.015 # Increased Max LR for better convergence
    DROPOUT_RATE = 0.01
    
    # --- Data and Model Initialization ---
    train_loader, test_loader = load_mnist_data()
    model = OptimizedNetV10(dropout_rate=DROPOUT_RATE).to(device)
    
    # Calculate and check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 OptimizedNetV10 Parameters: {total_params:,}")
    print(f"✅ Parameter constraint: {'PASS' if total_params < 8000 else 'FAIL'} (< 8,000)")
    
    # Optimizer (SGD with momentum is essential for good CNN performance)
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
        train_model(model, device, train_loader, optimizer, scheduler)
        test_acc = test_model(model, device, test_loader)
        test_accuracies.append(test_acc)
        # Flush the output to ensure immediate printing in notebooks
        sys.stdout.flush() 

    # --- Final Analysis ---
    if len(test_accuracies) >= 3:
        final_accs = test_accuracies[-3:]
        avg_final_acc = sum(final_accs) / 3
        
        print("\n\n==============================================")
        print("✅ FINAL 3 EPOCHS ANALYSIS")
        print(f"Test Accuracies: {final_accs}")
        print(f"Average Accuracy: {avg_final_acc:.3f}%")
        
        if avg_final_acc >= 99.4:
            print("🏆 STATUS: SUCCESS! Target (>=99.4%) achieved consistently.")
        else:
            print("⚠️ STATUS: NEED MORE WORK. Target was NOT met consistently.")
        print("==============================================")


if __name__ == '__main__':
    # Increase the number of workers in the DataLoader if you have many CPU cores
    # and if you are NOT seeing CPU utilization spikes.
    # On MPS, num_workers=2 is often a good starting point.
    run_experiment()
