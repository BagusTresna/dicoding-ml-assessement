const tf = require('@tensorflow/tfjs-node');
const path = require('path');

let model;

// Fungsi untuk memuat model
const loadModel = async() => {
    if (!model) {
        model = await tf.loadGraphModel(`file://${path.join(__dirname, '../models/model.json')}`);
    }
    return model;
};

// Fungsi untuk memprediksi gambar
const predictImage = async(image) => {
    try {
        const model = await loadModel();

        // Load dan persiapkan gambar untuk prediksi
        const tensorImage = tf.node.decodeImage(image, 3);
        const resizedImage = tf.image.resizeBilinear(tensorImage, [224, 224]);
        const input = resizedImage.expandDims(0).div(tf.scalar(255));

        // Prediksi
        const prediction = await model.predict(input).data();
        const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

        return { prediction: result, suggestion: suggestion };
    } catch (error) {
        throw new Error('Error while predicting the image');
    }
};

module.exports = { predictImage };