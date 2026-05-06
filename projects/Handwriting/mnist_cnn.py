import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 100:.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

def test_model_per_class(model, test_loader, criterion, device):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    class_loss = [0.0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, predicted = output.max(1)
            
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
                class_loss[label] += loss.item()
    
    class_accuracy = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    class_avg_loss = [class_loss[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    
    return class_accuracy, class_avg_loss

def visualize_per_class(class_accuracy, class_avg_loss):
    digits = list(range(10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.RdYlGn(np.array(class_accuracy) / 100.0)
    
    bars1 = ax1.bar(digits, class_accuracy, color=colors, edgecolor='black')
    ax1.set_xlabel('Digit', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy per Digit', fontsize=14, fontweight='bold')
    ax1.set_xticks(digits)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=np.mean(class_accuracy), color='red', linestyle='--', label=f'Mean: {np.mean(class_accuracy):.2f}%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars1, class_accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    bars2 = ax2.bar(digits, class_avg_loss, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Average Loss', fontsize=12)
    ax2.set_title('Loss per Digit', fontsize=14, fontweight='bold')
    ax2.set_xticks(digits)
    ax2.axhline(y=np.mean(class_avg_loss), color='red', linestyle='--', label=f'Mean: {np.mean(class_avg_loss):.4f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, loss in zip(bars2, class_avg_loss):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{loss:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('MNIST CNN Performance per Digit', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('digit_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('\nVisualization saved to digit_performance.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print('\nTraining CNN on MNIST dataset...\n')
    
    num_epochs = 10
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 40)
        
        train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        class_accuracy, class_loss = test_model_per_class(model, test_loader, criterion, device)
        test_accuracy = np.mean(class_accuracy)
        
        print(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%\n')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
    
    print(f'Training completed! Best Test Accuracy: {best_accuracy:.2f}%')
    print('\nGenerating per-class visualization...')
    class_accuracy, class_avg_loss = test_model_per_class(model, test_loader, criterion, device)
    visualize_per_class(class_accuracy, class_avg_loss)

if __name__ == '__main__':
    main()
