# Sign Language Recognition with Multilayer Perceptron

A machine learning project that uses a Multilayer Perceptron (MLP) neural network to classify American Sign Language (ASL) alphabet gestures from images. The project includes both image classification and real-time webcam recognition.

![ASL Alphabet](https://storage.googleapis.com/kagglesdsdata/datasets/3258/5337/amer_sign2.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251201T201049Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=45c4b5a18e1cedb8c59236c433510ad193dee235bd2b8b5847ff1a593f512f6be78d8f79889799f4884bcee3fd69fb6d6e44f881bf2554641a5664a0fe822446d0b22b0896ff04a65bf4ace757ff27520ef7c1bc239033136443a9093a1d5ef4d9803f007abe850fb57237e19482ef872993f18da2f7135059d22f41a1e95e3a0d2c0053c05f4de8cd1f7f5085a22e9a539ae6a3ee124c3b23c7993d2875332c8e3320d18ff21a6b94408fadf6eb0cd1159a69926bfef1d88a77e5811da63f9b2a496ffb19efecc883d8c0352cd2db17c22f8313e9d1765b8b6ba31ff9b84b287bd2f2d28746322ee7cc67c2e02b3e942ac957db54f5c2f17b50b5c756a1a231)

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