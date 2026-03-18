const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const glob = require("glob");

const datasetPath = path.join(__dirname, "dataset");
const imageSize = 224;
const batchSize = 16;
const epochs = 40;
const initialLearningRate = 0.001;

/**
 * Load dataset file paths and labels, then shuffle them together.
 */
async function loadDataset() {
    const categories = fs.readdirSync(datasetPath).sort();
    const images = [];
    const labels = [];

    categories.forEach((category, index) => {
        const categoryPath = path.join(datasetPath, category);
        const files = glob.sync(`${categoryPath}/*.{jpg,png,jpeg}`);

        files.forEach((file) => {
            images.push(file);
            labels.push(index);
        });
    });

    // Shuffle dataset consistently (Fisher-Yates)
    for (let i = images.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [images[i], images[j]] = [images[j], images[i]];
        [labels[i], labels[j]] = [labels[j], labels[i]];
    }

    return { images, labels, categories };
}

/**
 * Preprocess and augment a single image. Uses tf.tidy to avoid memory leaks.
 * @param {string} imagePath - Path to the image file
 * @param {boolean} augment - Whether to apply data augmentation
 */
function preprocessImage(imagePath, augment = true) {
    return tf.tidy(() => {
        const buffer = fs.readFileSync(imagePath);
        let tensor = tf.node.decodeImage(buffer, 3)
            .resizeBilinear([imageSize, imageSize])
            .toFloat()
            .div(255.0);

        if (augment) {
            // Random horizontal flip
            if (Math.random() > 0.5) {
                tensor = tensor.reverse(1);
            }
            // Random vertical flip
            if (Math.random() > 0.5) {
                tensor = tensor.reverse(0);
            }
            // Random brightness adjustment (+/- 15%)
            const brightnessFactor = 1.0 + (Math.random() * 0.3 - 0.15);
            tensor = tensor.mul(brightnessFactor).clipByValue(0, 1);
            // Random contrast adjustment
            const contrastFactor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
            const mean = tensor.mean();
            tensor = mean.add(tensor.sub(mean).mul(contrastFactor)).clipByValue(0, 1);
            // Add slight Gaussian noise
            if (Math.random() > 0.5) {
                const noise = tf.randomNormal(tensor.shape, 0, 0.02);
                tensor = tensor.add(noise).clipByValue(0, 1);
            }
        }

        return tensor;
    });
}

/**
 * Load images in batches to manage memory, returning a stacked tensor.
 * @param {string[]} imagePaths - Array of image file paths
 * @param {boolean} augment - Whether to apply augmentation
 */
async function loadImagesAsTensor(imagePaths, augment) {
    const batchChunkSize = 32;
    const allTensors = [];

    for (let i = 0; i < imagePaths.length; i += batchChunkSize) {
        const chunk = imagePaths.slice(i, i + batchChunkSize);
        const chunkTensors = chunk.map((p) => preprocessImage(p, augment));
        allTensors.push(...chunkTensors);
    }

    const stacked = tf.stack(allTensors);
    // Dispose individual tensors after stacking
    allTensors.forEach((t) => t.dispose());
    return stacked;
}

/**
 * Build a transfer-learning model using MobileNet as a frozen feature extractor,
 * with a custom classification head on top.
 * @param {number} numClasses - Number of output classes
 */
async function buildTransferModel(numClasses) {
    // Load MobileNet (lightweight 0.25 alpha variant for speed)
    const mobilenet = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );

    // Freeze all MobileNet layers so only the new head trains
    for (const layer of mobilenet.layers) {
        layer.trainable = false;
    }

    // Use the output of the last MobileNet conv layer as features
    const featureOutput = mobilenet.output;

    // Add custom classification head
    let x = tf.layers.globalAveragePooling2d({}).apply(featureOutput);
    x = tf.layers.dense({
        units: 128,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }).apply(x);
    x = tf.layers.batchNormalization({}).apply(x);
    x = tf.layers.dropout({ rate: 0.4 }).apply(x);
    x = tf.layers.dense({
        units: 64,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }).apply(x);
    x = tf.layers.dropout({ rate: 0.3 }).apply(x);
    const output = tf.layers.dense({
        units: numClasses,
        activation: "softmax",
    }).apply(x);

    const model = tf.model({ inputs: mobilenet.input, outputs: output });
    return model;
}

/**
 * Build a lightweight custom CNN (fallback if transfer learning is not desired).
 * Uses L2 regularization, global average pooling, and proper layer ordering.
 * @param {number} numClasses - Number of output classes
 */
function buildCustomCNN(numClasses) {
    const model = tf.sequential();

    // Block 1
    model.add(tf.layers.conv2d({
        inputShape: [imageSize, imageSize, 3],
        filters: 32,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Block 2
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Block 3
    model.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Block 4
    model.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.batchNormalization());

    // Global average pooling (more efficient than flatten, reduces overfitting)
    model.add(tf.layers.globalAveragePooling2d({}));

    // Dense head (decreasing units: 128 -> output)
    model.add(tf.layers.dense({
        units: 128,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

    return model;
}

/**
 * Train the model with early stopping, learning-rate reduction, and proper
 * memory management.
 */
async function trainModel() {
    const { images, labels, categories } = await loadDataset();
    const numClasses = categories.length;

    console.log(`Loaded ${images.length} images from ${numClasses} categories:`, categories);
    console.log(`Training config: epochs=${epochs}, batchSize=${batchSize}, lr=${initialLearningRate}`);

    // Split into train / validation sets (80/20) before any augmentation
    const splitIndex = Math.floor(images.length * 0.8);
    const trainImages = images.slice(0, splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const valImages = images.slice(splitIndex);
    const valLabels = labels.slice(splitIndex);

    console.log(`Training set: ${trainImages.length} images | Validation set: ${valImages.length} images`);

    // Load and preprocess images (augment training set only)
    console.log("Loading and preprocessing images...");
    const xTrain = await loadImagesAsTensor(trainImages, true);
    const xVal = await loadImagesAsTensor(valImages, false);
    const yTrain = tf.oneHot(tf.tensor1d(trainLabels, "int32"), numClasses).toFloat();
    const yVal = tf.oneHot(tf.tensor1d(valLabels, "int32"), numClasses).toFloat();

    // Build the model using transfer learning (MobileNet backbone)
    console.log("Loading MobileNet backbone for transfer learning...");
    let model;
    try {
        model = await buildTransferModel(numClasses);
        console.log("Using transfer learning with MobileNet.");
    } catch (err) {
        console.warn("Could not load MobileNet, falling back to custom CNN:", err.message);
        model = buildCustomCNN(numClasses);
        console.log("Using custom CNN architecture.");
    }

    // Compile with Adam optimizer
    const optimizer = tf.train.adam(initialLearningRate);
    model.compile({
        optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    model.summary();

    // Custom early stopping + reduce-LR-on-plateau via callbacks
    let bestValLoss = Infinity;
    let bestValAcc = 0;
    const patience = 8;
    let patienceCounter = 0;
    let currentLR = initialLearningRate;
    const lrReductionFactor = 0.5;
    const lrPatience = 4;
    let lrPatienceCounter = 0;
    const minLR = 1e-6;

    const customCallbacks = {
        onEpochEnd: async (epoch, logs) => {
            const valLoss = logs.val_loss;
            const valAcc = logs.val_acc;

            // Track best accuracy
            if (valAcc > bestValAcc) {
                bestValAcc = valAcc;
            }

            // Reduce learning rate on plateau
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                lrPatienceCounter = 0;
            } else {
                patienceCounter++;
                lrPatienceCounter++;

                if (lrPatienceCounter >= lrPatience && currentLR > minLR) {
                    currentLR = Math.max(currentLR * lrReductionFactor, minLR);
                    optimizer.learningRate = currentLR;
                    console.log(`  -> Reducing learning rate to ${currentLR.toExponential(2)}`);
                    lrPatienceCounter = 0;
                }
            }

            console.log(
                `  Epoch ${epoch + 1}: val_loss=${valLoss.toFixed(4)}, val_acc=${valAcc.toFixed(4)}, ` +
                `best_val_acc=${bestValAcc.toFixed(4)}, lr=${currentLR.toExponential(2)}, patience=${patienceCounter}/${patience}`
            );

            // Early stopping
            if (patienceCounter >= patience) {
                console.log(`Early stopping triggered at epoch ${epoch + 1}`);
                model.stopTraining = true;
            }
        },
    };

    console.log("Training model...");
    await model.fit(xTrain, yTrain, {
        epochs,
        batchSize,
        validationData: [xVal, yVal],
        shuffle: true,
        callbacks: customCallbacks,
    });

    // Save the trained model
    const savePath = path.join(__dirname, "trained_model");
    await model.save(`file://${savePath}`);
    console.log(`Model trained and saved to ${savePath}/`);
    console.log(`Best validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`);

    // Clean up tensors
    xTrain.dispose();
    xVal.dispose();
    yTrain.dispose();
    yVal.dispose();
    console.log(`Final tensor count: ${tf.memory().numTensors}`);
}

trainModel().catch(console.error);
