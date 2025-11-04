# Realtime Skin Care Analyst

A machine learning-powered skin type classification system that can analyze skin types in real-time through video analysis and static image processing.

## Overview

This project uses deep learning to classify skin types into three categories:
- **Dry skin** (`dry_data`)
- **Normal skin** (`normal_data`) 
- **Oily skin** (`oily_data`)

The system combines computer vision techniques for face detection with a pre-trained EfficientNet model for accurate skin type prediction.

## Files

### `skin_type_pred.ipynb`

The main Jupyter notebook containing the complete machine learning pipeline for skin type classification.

#### Key Features:
- **Data Loading & Preprocessing**: Automatically loads and processes skin type images from organized directory structure
- **Model Architecture**: Uses EfficientNetV2B0 (or EfficientNetB0 as fallback) as feature extractor with custom classification head
- **Training Pipeline**: Complete model training with data augmentation and validation
- **Real-time Analysis**: Live webcam-based skin type prediction with face detection
- **Static Image Testing**: Batch processing capabilities for analyzing individual images

#### Notebook Sections:

1. **Environment Setup**
   - TensorFlow installation and version verification
   - Required library imports (OpenCV, matplotlib, sklearn, etc.)

2. **Data Management**
   - Automatic dataset loading from `data/raw/skin_type/` directory
   - Train/validation split with stratification
   - Class distribution analysis and imbalance detection
   - Fallback to dummy data if no images found

3. **Data Preprocessing**
   - Image augmentation using `ImageDataGenerator`
   - Batch processing with 224x224 target size
   - Normalization and RGB color mode

4. **Model Architecture**
   - **Robust Model Creation**: Automatically tries TensorFlow Hub EfficientNetV2, falls back to local `tf.keras.applications` on failure
   - Feature extraction layer (frozen pretrained weights)
   - Custom classification head with regularization:
     - BatchNormalization
     - Dense layer (64 units) with L1/L2 regularization
     - Dropout (45%)
     - Softmax output layer (3 classes)

5. **Training**
   - Model compilation with categorical crossentropy loss
   - Adamax optimizer (learning rate: 0.001)
   - 20 epochs with validation monitoring

6. **Evaluation**
   - Training/validation/test accuracy and loss metrics
   - Confusion matrix visualization
   - Classification report with precision/recall/F1-score
   - Training history plots

7. **Model Persistence**
   - Saves trained model as `model/skin_type_classification_model.h5`
   - Saves class labels as `model/class_labels.pkl`

8. **Real-time Analysis Features**
   - **Face Detection**: Uses OpenCV Haar cascades for face detection
   - **Image Preprocessing**: Automatic resizing, color conversion, normalization
   - **Live Video Analysis**: Real-time webcam processing with bounding boxes and predictions
   - **Static Image Testing**: Batch processing of individual images
   - **Confidence Scoring**: Displays prediction confidence for all classes

#### Usage Examples:

**Training the Model:**
```python
# The notebook handles everything automatically
# Just run cells 1-11 to train the model
```

**Real-time Analysis:**
```python
# Run the video analysis cell (cell 22)
results = analyze_skin_from_video(loaded_model, class_labels, face_cascade, duration=60, notebook=True)
```

**Testing with Static Images:**
```python
# Test with your own image
test_with_static_image(loaded_model, class_labels, face_cascade, "path/to/your/image.jpg")
```

#### Technical Requirements:
- Python 3.7+
- TensorFlow 2.9.1+
- OpenCV for computer vision
- matplotlib for visualization
- scikit-learn for metrics
- TensorFlow Hub (optional, has fallback)

#### Error Handling:
- **TensorFlow Hub Issues**: Automatically falls back to local EfficientNet models
- **Missing Data**: Creates dummy data structure for testing when no images found
- **Camera Access**: Graceful handling of camera connection issues
- **Face Detection**: Continues processing even when no faces detected

#### Expected Data Structure:
```
data/raw/skin_type/
├── train/
│   ├── dry_data/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── normal_data/
│   │   └── ...
│   └── oily_data/
│       └── ...
└── validation/
    ├── dry_data/
    ├── normal_data/
    └── oily_data/
```

#### Output Files:
- `model/skin_type_classification_model.h5` - Trained model weights
- `model/class_labels.pkl` - Class label mappings
- Training plots and confusion matrices (displayed in notebook)
