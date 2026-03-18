# River Detection Model

This project trains a Convolutional Neural Network (CNN) using TensorFlow.js to classify images as containing a river or not. It uses **transfer learning** with a MobileNet backbone for high accuracy even on small datasets, and includes built-in memory management, data augmentation, early stopping, and learning-rate scheduling.

## Technologies Used
- **Node.js** - JavaScript runtime for executing scripts
- **TensorFlow.js** - Machine learning library for training and inference
- **MobileNet v1** - Pretrained feature extractor used via transfer learning
- **glob** - Library to match file paths for dataset loading

## Project Structure
```
|-- dataset/
|   |-- rivers/
|   |-- no-rivers/
|-- trained_model/
|-- train.js
|-- predict.js
|-- README.md
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Maya03Patil/river-object-detection.git
   cd river-object-detection
   ```
2. Install dependencies:
   ```sh
   npm install
   ```

## Dataset Preparation
- Place labelled images inside `dataset/` in subfolders (e.g. `rivers/`, `no-rivers/`).
- Supported formats: `.jpg`, `.png`, `.jpeg`.
- The folder names become the class labels (sorted alphabetically).

## Training the Model
```sh
npm run train
# or
node train.js
```

### Training features
- **Transfer learning** – Uses a frozen MobileNet v1 (0.25 alpha) as a feature extractor with a custom classification head. Falls back to a custom CNN if MobileNet cannot be downloaded.
- **Data augmentation** – Random horizontal/vertical flips, brightness jitter, contrast adjustment, and Gaussian noise (applied to training set only).
- **Early stopping** – Stops training when validation loss stops improving (patience = 8 epochs).
- **Learning-rate reduction** – Halves the learning rate when validation loss plateaus for 4 consecutive epochs.
- **L2 regularization & dropout** – Reduces overfitting on small datasets.
- **Memory management** – Uses `tf.tidy()` and explicit tensor disposal to prevent memory leaks.

The trained model is saved to the `trained_model/` directory.

## Making Predictions
```sh
# Single image
node predict.js path/to/image.jpg

# Multiple images (batch mode)
node predict.js img1.jpg img2.png img3.jpeg

# Default (uses test.jpg)
npm run predict
```

The output shows per-class confidence scores with a visual bar chart and a final verdict.

## Prediction Logic
- If the confidence for "rivers" is **above 60%**, it outputs: `River detected`.
- Otherwise, it outputs: `No river detected`.

## License
This project is open-source. Feel free to modify and improve it!

