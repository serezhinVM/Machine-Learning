import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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

def load_model(model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_digit(image_path, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = output.argmax(1).item()
        confidence = probabilities[0][predicted_digit].item()
    
    return predicted_digit, confidence

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage: python inference.py <image_path>')
        print('Example: python inference.py digit.png')
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        model, device = load_model()
        print(f'Loaded model on device: {device}')
        
        digit, confidence = predict_digit(image_path, model, device)
        print(f'Predicted digit: {digit}')
        print(f'Confidence: {confidence * 100:.2f}%')
    except FileNotFoundError:
        print('Error: best_model.pth not found. Train the model first!')
