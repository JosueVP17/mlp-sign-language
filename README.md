# Sign Language Recognition with Multilayer Perceptron

A machine learning project that uses a Multilayer Perceptron (MLP) neural network to classify American Sign Language (ASL) alphabet gestures from images. The project includes both image classification and real-time webcam recognition.

![ASL Alphabet](https://storage.googleapis.com/kagglesdsdata/datasets/3258/5337/amer_sign2.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251210%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251210T231512Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=199c0116793ed138c1db0dba747682ca49a72d90c6ce3e8e23c678c84e23aadc262313dfd51f9ea8d9a249a6fc71c32992f89b62b3746ff63e14b5e7903b7a792e2bcba2dfed11237ffa04d0b9f5dd697ac5b4e684866e130f68bab9e27e3d31a93c63a558e11c6a858d35d6f750bf0c49e718dc05f4572705a116b3427a1c761d0e19dc8b6767c4da3a29084c20f3dfa680ccc3927d1eb5559e8927d8a998e80293f03cf0c1ae72b0335e36516d571f2f1ceb6045e2b0fd4309b2a2d0a9958b043263a0957d52b6d9d78ecee0febd5ade65dd19d9ee422ea27c782e8f27c4208bde2515356eb60d5ce0ab07bf7e2db19ebfa6996ac2c59bfa229caf4de84a21)

## Dataset

This project uses the **Sign Language MNIST** dataset from Kaggle, which contains grayscale images of American Sign Language letters (A-Z, excluding J and Z which require motion).

- **Source**: [Kaggle - Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Training samples**: 27,455 images
- **Test samples**: 7,172 images
- **Image size**: 28×28 pixels
- **Format**: CSV files with 785 columns (1 label + 784 pixel values)
- **Classes**: 24 (letters A-Y, excluding J and Z)

## Features

- **MLP Neural Network**: Custom implementation using scikit-learn
- **Feature Extraction**: Histogram of Oriented Gradients (HOG) for robust feature representation
- **Model Persistence**: Trained model saved as `mlp_model.pkl` for reuse
- **Photo Prediction**: Capture and classify single images from webcam
- **Real-time Recognition**: Live webcam integration for sign language detection

## Project Structure

```
MLP_SignLanguage/
├── mlp.ipynb                 # Main notebook with model training and evaluation
├── photo_prediction.py       # Capture and predict from single photo
├── real_time.py             # Real-time webcam sign language recognition
├── mlp_model.pkl            # Trained MLP model (generated after training)
├── sign_mnist_train.csv     # Training dataset
├── sign_mnist_test.csv      # Test dataset
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/JosueVP17/mlp-sign-language.git
cd MLP_SignLanguage
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download dataset from Kaggle.

### Key Dependencies

- `scikit-learn` - Machine learning model
- `scikit-image` - HOG feature extraction
- `opencv-python` - Image processing and webcam integration
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` & `seaborn` - Visualization

## Usage

### 1. Train the Model

Open and run the Jupyter notebook to train the MLP model:

```bash
jupyter notebook mlp.ipynb
```

The notebook includes:

- Data loading and preprocessing
- HOG feature extraction with optimal parameters
- MLP model training and hyperparameter tuning
- Model evaluation and performance metrics
- Model serialization to `mlp_model.pkl`

### 2. Photo Prediction

Capture a single photo for prediction:

```bash
python photo_prediction.py
```

- The webcam will activate
- Press **SPACE** to capture an image
- The prediction will be displayed in the console
- Press **ESC** to exit without capturing

### 3. Real-time Recognition

Run the real-time sign language recognition:

```bash
python real_time.py
```

- The webcam will activate and display predictions in real-time
- The predicted letter and class will be shown on screen
- Press **ESC** to exit

## Model Architecture

### HOG Feature Extraction

The model uses Histogram of Oriented Gradients (HOG) with the following parameters optimized for 28×28 images:

- **Orientations**: 9 (gradient directions split into 9 bins from 0° to 180°)
- **Pixels per cell**: (4, 4) - divides image into 7×7 grid of cells
- **Cells per block**: (2, 2) - groups cells for normalization
- **Block normalization**: L2-Hys (L2-norm followed by clipping and renormalization)

This configuration produces a robust feature vector that captures edge orientations and shapes essential for character recognition.

### MLP Configuration

The Multilayer Perceptron is configured with:

- Input layer: HOG feature vector
- Hidden layers: One layer with sigmoid activation
- Output layer: 24 classes (one per letter)
- Optimizer: Adam

## Performance

The model achieves high accuracy on the test set. Detailed metrics including:

- Confusion matrix
- Classification report

## License

This project is for educational purposes. The dataset is provided by Kaggle under their terms of use.

## Acknowledgments

- Dataset: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Original dataset creator: tecperson
- American Sign Language alphabet images from Kaggle dataset resources
