const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const glob = require("glob");

const datasetPath = path.join(__dirname, "dataset");
const imageSize = 224;
const batchSize = 8; // Increased batch size
const epochs = 25; // More training for better learning

async function loadDataset() {
    const categories = fs.readdirSync(datasetPath);
    let images = [];
    let labels = [];

    categories.forEach((category, index) => {
        const categoryPath = path.join(datasetPath, category);
        const files = glob.sync(`${categoryPath}/*.{jpg,png,jpeg}`);

        files.forEach((file) => {
            images.push(file);
            labels.push(index);
        });
    });

    return { images, labels, categories };
}

async function preprocessImage(imagePath) {
    const buffer = await fs.promises.readFile(imagePath);
    let tensor = tf.node.decodeImage(buffer, 3) // Ensure RGB
        .resizeBilinear([imageSize, imageSize])
        .toFloat()
        .div(tf.scalar(255));

    // 🔥 **Data Augmentation**
    if (Math.random() > 0.5) tensor = tensor.reverse(1); // Random horizontal flip
    if (Math.random() > 0.5) tensor = tensor.reverse(0); // Random vertical flip
    if (Math.random() > 0.5) tensor = tensor.mul(tf.scalar(1.1)).clipByValue(0, 1); // Slight brightness increase

    return tensor;
}

async function trainModel() {
    const { images, labels, categories } = await loadDataset();
    const numClasses = categories.length;

    console.log(`Loaded ${images.length} images from ${numClasses} categories:`, categories);

    // Convert labels to one-hot encoding with correct dtype
    const yTrain = tf.oneHot(tf.tensor1d(labels, "int32"), numClasses).toFloat();

    // Load and preprocess images
    const processedImages = await Promise.all(images.map(preprocessImage));
    const xTrain = tf.stack(processedImages);

    // ✅ **Improved CNN Architecture**
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({ inputShape: [imageSize, imageSize, 3], filters: 32, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.flatten());
    
    // 🔥 **Additional Dense Layer for Better Feature Extraction**
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dropout(0.5));

    model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

    model.compile({
        optimizer: tf.train.adam(0.0001),  // 🔥 Lowered learning rate
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    console.log("Training model...");
    await model.fit(xTrain, yTrain, {
        epochs,
        batchSize,
        validationSplit: 0.2,  // Helps prevent overfitting
        shuffle: true
    });

    // Save the trained model
    await model.save(`file://${path.join(__dirname, "trained_model")}`);
    console.log("✅ Model trained and saved to /trained_model/");
}

trainModel().catch(console.error);
