# River Detection Model

This project trains a Convolutional Neural Network (CNN) using TensorFlow.js to classify images as containing a river or not. The trained model can then be used to predict whether a given image contains a river.

## Technologies Used
- **Node.js** - JavaScript runtime for executing scripts
- **TensorFlow.js** - Machine learning library for training and inference
- **fs (File System)** - Node.js module for handling files
- **path** - Node.js module for working with file paths
- **glob** - Library to match file paths for dataset loading

## Project Structure
```
|-- dataset/
|   |-- river/
|   |-- no_river/
|-- trained_model/
|-- train.js
|-- predict.js
|-- README.md
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/river-detection.git
   cd river-detection
   ```
2. Install dependencies:
   ```sh
   npm install @tensorflow/tfjs-node glob
   ```

## Dataset Preparation
- Create a `dataset/` folder with two subfolders: `river/` and `no_river/`.
- Add labeled images to each category (`.jpg`, `.png`, or `.jpeg`).

## Training the Model
Run the following command to train the model:
```sh
node train.js
```
- The model will be saved in the `trained_model/` directory after training.

## Making Predictions
Run the following command to classify an image:
```sh
node predict.js 
```
Example:
```sh
node predict.js 
```
The output will indicate whether the image contains a river based on confidence scores.

## Prediction Logic
- If the confidence for "river" is **above 70%**, it outputs: `✅ Detected: River`.
- Otherwise, it outputs: `❌ Detected: No-River`.

## License
This project is open-source. Feel free to modify and improve it!

