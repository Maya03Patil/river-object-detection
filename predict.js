const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

// Load model
async function loadModel() {
    const modelPath = "file://" + path.resolve(__dirname, "trained_model/model.json");
    return await tf.loadLayersModel(modelPath);
}

// Preprocess image
async function preprocessImage(imagePath) {
    const buffer = fs.readFileSync(imagePath);
    return tf.node.decodeImage(buffer, 3)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims(0);
}

// Make prediction
async function predict(imagePath) {
    const model = await loadModel();
    const inputTensor = await preprocessImage(imagePath);

    // Run prediction
    const predictions = await model.predict(inputTensor).dataSync();
    console.log("Confidence Scores:", predictions);

    // Load category labels
    const categories = fs.readdirSync(path.join(__dirname, "dataset")).sort();
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = predictions[maxIndex];

    console.log(`(Confidence: ${(confidence * 100).toFixed(2)}%)`);

    // **Updated Logic**
    if (maxIndex === categories.indexOf("rivers") && confidence > 0.70) {
        console.log("✅ Detected: River");
    } else {
        console.log("❌ Detected: No-River");
    }
}

// Test with an image
predict("test.jpg").catch(console.error);
