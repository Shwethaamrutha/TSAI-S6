"""
================================================================================
FINAL MODEL LOG: 
================================================================================

🎯 TARGET:
Achieve a CONSISTENT 99.4% test accuracy (3-epoch average) by applying the final fix: increasing Momentum to **0.98** to solve the stability/fluctuation issue observed in V33, using the proven optimal structure and schedule.

📈 RESULTS:
- Total Parameters: 6,260
- Best Test Accuracy: 99.440%
- Final 3-Epoch Average: 99.420%
- Constraints: 🏆 SUCCESS! Target Exceeded.

🧠 ANALYSIS:
The final configuration was a success. The combination of the aggressive Max LR (0.05), Hybrid Augmentation, and the high **Momentum (0.98)** provided the necessary speed, generalization, and most importantly, **stability**. Dropout was deliberately excluded (0.0) as the small network capacity required maximum feature learning, which Hybrid Augmentation provided without the capacity limitation of Dropout.

================================================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import sys 

# =========================================
# 1. Device Setup (Omitted for brevity)
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
# 2. Model Definition (OptimizedNetV36 - 6,260 Parameters)
# =========================================
class OptimizedNetV36(nn.Module):
    # Original V11 Architecture (6,260 params)
    def __init__(self): 
        super(OptimizedNetV36, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1); self.bn1 = nn.BatchNorm2d(8) 
        self.conv2 = nn.Conv2d(8, 10, 3, padding=1); self.bn2 = nn.BatchNorm2d(10) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 12, 3, padding=1); self.bn3 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.conv4 = nn.Conv2d(12, 16, 3, padding=1); self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1); self.bn5 = nn.BatchNorm2d(16)
        self.conv_final = nn.Conv2d(16, 10, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x) 
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv_final(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) 
        return F.log_softmax(x, dim=1)

# =========================================
# 3. Data Loading (Hybrid Augmentation)
# =========================================
def load_mnist_data(device):
    
    # Hybrid Augmentation: Rotation (7 degrees) + Translation (0.1)
    train_transforms = transforms.Compose([
        transforms.Pad(2), transforms.RandomCrop(28),
        transforms.RandomAffine(
            degrees=7,          
            translate=(0.1, 0.1), 
        ),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    pin_mem = True if device.type != 'mps' else False
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=pin_mem)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_mem)
    
    return train_loader, test_loader

# =========================================
# 4. Training and Testing Functions (Omitted for brevity)
# =========================================
def train_model(model, device, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader, desc='Training', file=sys.stdout)
    correct = 0; processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(); output = model(data); loss = F.nll_loss(output, target); loss.backward(); optimizer.step()
        scheduler.step()
        pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(f'Train Loss={loss.item():.4f} LR={scheduler.get_last_lr()[0]:.6f} Acc={100*correct/processed:.2f}%')
    return 100. * correct / processed

def test_model(model, device, test_loader):
    model.eval(); test_loss = 0; correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data); test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset); test_acc = 100. * correct / len(test_loader.dataset)
    sys.stdout.write(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.3f}%)')
    return test_acc

# =========================================
# 5. Main Execution (V33 Configuration with Momentum Tweak)
# =========================================
def run_experiment():
    
    # --- Configuration ---
    EPOCHS = 15              # Reverted to 15 epochs
    MAX_LR = 0.05           
    DROPOUT_RATE = 0.0 
    PCT_START = 0.05         
    WEIGHT_DECAY = 0.00005   
    MOMENTUM = 0.98          # <<< Increased Momentum
    
    # --- Data and Model Initialization ---
    train_loader, test_loader = load_mnist_data(device) # Hybrid Augmentation
    model = OptimizedNetV36().to(device) # 6,260 params
    
    # Calculate and check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 OptimizedNetV36 Parameters: {total_params:,}")
    print(f"✅ Parameter constraint: {'PASS' if total_params < 8000 else 'FAIL'} (< 8,000)")
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) 

    # OneCycleLR Scheduler 
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS, 
        pct_start=PCT_START, 
        div_factor=10, 
        final_div_factor=100 
    )

    # --- Training Loop ---
    test_accuracies = []
    print(f"\n🚀 Starting training for {EPOCHS} epochs (Max LR: {MAX_LR}, Momentum: {MOMENTUM}, WD: {WEIGHT_DECAY}, Dropout: {DROPOUT_RATE}, PCT Start: {PCT_START}, Final Div: 100)...")
    
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
