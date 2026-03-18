const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const imageSize = 224;
let cachedModel = null;

/**
 * Load the trained model (cached after first load for repeated predictions).
 */
async function loadModel() {
    if (cachedModel) return cachedModel;
    const modelPath = "file://" + path.resolve(__dirname, "trained_model/model.json");
    cachedModel = await tf.loadLayersModel(modelPath);
    return cachedModel;
}

/**
 * Preprocess a single image for inference. Uses tf.tidy to prevent memory leaks.
 * @param {string} imagePath - Path to the image file
 */
function preprocessImage(imagePath) {
    return tf.tidy(() => {
        const buffer = fs.readFileSync(imagePath);
        return tf.node.decodeImage(buffer, 3)
            .resizeBilinear([imageSize, imageSize])
            .toFloat()
            .div(255.0)
            .expandDims(0);
    });
}

/**
 * Run prediction on a single image and print results.
 * @param {string} imagePath - Path to the image file
 */
async function predict(imagePath) {
    if (!fs.existsSync(imagePath)) {
        console.error(`File not found: ${imagePath}`);
        process.exit(1);
    }

    const model = await loadModel();
    const inputTensor = preprocessImage(imagePath);

    // Run prediction inside tidy and extract data before disposing
    const outputTensor = model.predict(inputTensor);
    const predictions = outputTensor.dataSync();

    // Dispose tensors after extracting data
    inputTensor.dispose();
    outputTensor.dispose();

    // Load category labels (sorted to match training order)
    const categories = fs.readdirSync(path.join(__dirname, "dataset")).sort();

    console.log(`\nImage: ${imagePath}`);
    console.log("---");

    // Print per-class confidence scores
    categories.forEach((cat, i) => {
        const score = (predictions[i] * 100).toFixed(2);
        const bar = "#".repeat(Math.round(predictions[i] * 30));
        console.log(`  ${cat.padEnd(12)} ${score.padStart(6)}%  ${bar}`);
    });

    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = predictions[maxIndex];

    console.log("---");

    if (maxIndex === categories.indexOf("rivers") && confidence > 0.60) {
        console.log(`Result: River detected (${(confidence * 100).toFixed(2)}% confidence)`);
    } else {
        console.log(`Result: No river detected (${categories[maxIndex]} at ${(confidence * 100).toFixed(2)}% confidence)`);
    }

    return { category: categories[maxIndex], confidence, predictions };
}

/**
 * Run predictions on multiple images (batch mode).
 * @param {string[]} imagePaths - Array of image file paths
 */
async function predictBatch(imagePaths) {
    console.log(`Running predictions on ${imagePaths.length} image(s)...\n`);

    const results = [];
    for (const imagePath of imagePaths) {
        const result = await predict(imagePath);
        results.push({ imagePath, ...result });
        console.log("");
    }

    // Summary
    if (results.length > 1) {
        console.log("=== Summary ===");
        const riverCount = results.filter((r) => r.category === "rivers" && r.confidence > 0.60).length;
        console.log(`Rivers detected: ${riverCount}/${results.length}`);
    }

    return results;
}

// Parse CLI arguments: node predict.js <image1> [image2] ...
const args = process.argv.slice(2);
const imagePaths = args.length > 0 ? args : ["test.jpg"];

predictBatch(imagePaths).catch(console.error);
