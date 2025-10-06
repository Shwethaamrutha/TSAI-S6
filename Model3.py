import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import sys 
import math # Needed for potential future use, keeping it for robustness

# =========================================
# 1. Device Setup (with MPS for Mac Silicon)
# =========================================
def get_device():
    # Only check and print device info once at startup
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
# 2. Model Definition (OptimizedNetV11)
# Parameters: ~6,400 (Deep and Narrow)
# Dropout is handled in the main function (set to 0.0)
# =========================================
class OptimizedNetV11(nn.Module):
    def __init__(self, dropout_rate=0.0): # Set to 0.0 for max learning capacity
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
        # Added depth for high accuracy
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1) 
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # --- Final Classification (1x1 Conv + GAP) ---
        self.conv_final = nn.Conv2d(16, 10, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

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

# =========================================
# 3. Data Loading and Preprocessing (with Augmentation)
# =========================================
def load_mnist_data():
    
    # 1. Define Training Augmentations (Crucial for 99.4+%)
    train_transforms = transforms.Compose([
        # 1. Pad and Random Crop (simulates shifting)
        transforms.Pad(2),
        transforms.RandomCrop(28),
        # 2. Random Affine transformations (mild rotations and shears)
        transforms.RandomAffine(
            degrees=7,          # Max 7 degrees rotation
            translate=(0.1, 0.1), # Max 10% translation
            shear=0.1,          # Max 10% shear
        ),
        # 3. Final steps
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Define Test Transforms (No Augmentation on Test Data)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    # Use a small batch size for training to aid MPS/GPU memory efficiency
    # num_workers=0 is recommended if repetitive MPS logs are still appearing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# =========================================
# 4. Training and Testing Functions
# =========================================
def train_model(model, device, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader, desc='Training', file=sys.stdout) # Added file=sys.stdout
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
    MAX_LR = 0.015 
    DROPOUT_RATE = 0.00 # Zero Dropout for maximum learning
    
    # --- Data and Model Initialization ---
    train_loader, test_loader = load_mnist_data()
    model = OptimizedNetV11(dropout_rate=DROPOUT_RATE).to(device)
    
    # Calculate and check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 OptimizedNetV11 Parameters: {total_params:,}")
    print(f"✅ Parameter constraint: {'PASS' if total_params < 8000 else 'FAIL'} (< 8,000)")
    
    # Optimizer (SGD with momentum is crucial)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) 

    # OneCycleLR Scheduler with Aggressive Final Decay
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS, 
        pct_start=0.3, 
        div_factor=10, 
        final_div_factor=1000 # Aggressive decay to ~1e-5 for fine-tuning
    )

    # --- Training Loop ---
    test_accuracies = []
    print(f"\n🚀 Starting training for {EPOCHS} epochs with OneCycleLR (Max LR: {MAX_LR}, Final Div: 1000)...")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_model(model, device, train_loader, optimizer, scheduler)
        test_acc = test_model(model, device, test_loader)
        test_accuracies.append(test_acc)
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
    run_experiment()
