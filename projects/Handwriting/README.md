# MNIST Handwritten Digit Recognition with CNN (PyTorch)

## Project Structure
- `mnist_cnn.py` - Training script with CNN model
- `inference.py` - Inference script for predicting digits
- `requirements.txt` - Python dependencies
- `best_model.pth` - Saved trained model (after training)
- `data/` - MNIST dataset (downloaded automatically)

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python mnist_cnn.py
```

The model uses a 3-layer CNN architecture:
- Conv1: 1 → 32 channels
- Conv2: 32 → 64 channels  
- Conv3: 64 → 128 channels
- Fully connected: 128*3*3 → 256 → 10

Training runs for 10 epochs with Adam optimizer.

## Inference
```bash
python inference.py path_to_your_digit_image.png
```

The image will be converted to 28x28 grayscale and predicted.
