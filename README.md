# African Wildlife Classification System

A comprehensive machine learning application for classifying African wildlife images using both traditional ML and deep learning approaches with proper validation methodology.

## Related Work

This implementation is based on the research and analysis from the [Africa Wildlife Recognition Kaggle Notebook](https://www.kaggle.com/code/georgeokello/africa-wildlife-recognition) by **George Okello**.

## Features

- **Comprehensive EDA** with sample images, statistics, and quality assessment
- **Traditional ML Models** with advanced feature engineering and selection
- **Deep Learning Models** including custom CNN and ResNet-18 transfer learning
- **Proper Validation** using train/validation/test splits or cross-validation
- **Interactive Prediction Interface** with debugging capabilities
- **Professional Visualizations** and performance comparisons
- **Pipeline Management** ensuring consistent transformations

## System Architecture

```
├── Traditional ML Pipeline
│   ├── Feature Extraction (HOG, LBP, Statistical, Texture)
│   ├── Feature Selection (Univariate, RFE, Tree-based)
│   ├── Dimensionality Reduction (PCA)
│   └── Models (SVM, Random Forest, Ensemble Methods)
│
├── Deep Learning Pipeline
│   ├── Custom CNN with Residual Connections
│   ├── ResNet-18 Transfer Learning
│   └── Advanced Data Augmentation
│
└── Validation Strategy
    ├── Train/Validation/Test Split (70/15/15)
    └── Optional Cross-Validation (5-fold)
```

## Dataset Structure

The application expects the following directory structure:

```
african-wildlife/
├── buffalo/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── elephant/
│   ├── image1.jpg
│   └── ...
├── rhino/
│   └── ...
└── zebra/
    └── ...
```

## Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd african-wildlife-classification
```

2. **Prepare your dataset:**
```bash
# Create the dataset directory structure
mkdir -p african-wildlife/{buffalo,elephant,rhino,zebra}
# Add your images to the respective directories
```

3. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

4. **Access the application:**
   - Open your browser and go to `http://localhost:8501`

### Option 2: Local Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd african-wildlife-classification
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare dataset structure** (as shown above)

5. **Run the application:**
```bash
streamlit run streamlit_app.py
```

## Configuration

The system can be configured via `config.py`:

```python
# Data split configuration
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15  
TEST_SIZE = 0.15

# Cross-validation option
USE_CROSS_VALIDATION = False
CV_FOLDS = 5

# Deep learning configuration
DEEP_IMG_SIZE = 224
DEEP_BATCH_SIZE = 32
CNN_EPOCHS = 15
```

## Usage Guide

### 1. System Overview
- Check dataset structure and validation strategy
- Run comprehensive EDA to understand your data
- Review sample images and quality metrics

### 2. Traditional ML Training
- Configure training parameters (optimized hyperparameters, feature selection)
- Train individual models (SVM, Random Forest, etc.)
- Optional ensemble training for improved performance

### 3. Deep Learning Training
- Choose between Custom CNN or ResNet-18 transfer learning
- Configure epochs and monitor training progress
- Automatic early stopping with validation monitoring

### 4. Model Comparison
- View comprehensive performance rankings
- Compare traditional ML vs deep learning approaches
- Generate detailed performance reports

### 5. Prediction Interface
- Upload new images or select from dataset
- Get predictions with confidence scores
- Debug mode for pipeline troubleshooting

## Model Performance

The system typically achieves:
- **Traditional ML**: 75-85% accuracy depending on feature selection
- **Deep Learning**: 85-95% accuracy with proper augmentation
- **Best Results**: Usually achieved with ensemble methods or fine-tuned CNNs

## Validation Methodology

The application implements proper ML validation practices:

- **No Data Leakage**: Clear separation between train/validation/test sets
- **Honest Evaluation**: Test set never used for model selection
- **Proper Pipeline**: Same transformations applied during training and prediction
- **Cross-Validation Option**: For more robust performance estimates

## API Reference

### Core Components

- **DataManager**: Dataset loading and EDA functionality
- **FeatureExtractor**: Traditional ML feature extraction with pipeline management
- **MLModelTrainer**: Traditional ML model training and evaluation
- **DeepLearningTrainer**: CNN and transfer learning implementation
- **SessionManager**: State management and persistence

### Key Methods

```python
# Feature extraction with proper pipeline
feature_extractor = FeatureExtractor()
X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.prepare_dataset()

# Model training
trainer = MLModelTrainer()
results = trainer.train_individual_models(X_train, X_val, X_test, y_train, y_val, y_test)

# Prediction with validation
prediction, probabilities, status = safe_make_prediction_with_debug(
    image, model_name, feature_extractor, ml_results, debug_mode=True
)
```

## Docker Deployment Details

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Application port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)
- `PYTHONPATH`: Python path configuration

### Volume Mounts
- `./african-wildlife:/app/african-wildlife:ro` - Dataset (read-only)
- `./saved_models:/app/saved_models` - Model persistence
- `./cache:/app/cache` - Temporary cache
- `./reports:/app/reports` - Generated reports

### Health Checks
The container includes health checks to ensure the application is running properly.

## Troubleshooting

### Common Issues

1. **Feature Dimension Mismatch**
   - Use debug mode in prediction interface
   - Check pipeline configuration
   - Ensure proper feature extractor fitting

2. **Memory Issues**
   - Reduce batch size in configuration
   - Limit feature extraction samples
   - Use smaller image sizes

3. **Dataset Issues**
   - Verify directory structure
   - Check image file formats
   - Review EDA quality assessment

### Debug Mode

Enable debug mode in the prediction interface to see:
- Step-by-step feature transformation
- Pipeline component status
- Expected vs actual feature counts
- Detailed error messages

## Performance Optimization

- **CPU Usage**: Adjust `N_JOBS` in configuration
- **Memory**: Modify batch sizes and image sizes
- **Training Speed**: Use GPU support for deep learning
- **Storage**: Clear cache directory periodically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Requirements

- Python 3.9+
- 4GB+ RAM recommended
- GPU optional but recommended for deep learning
- Dataset with balanced classes for best results

## License

This project is open source. Please refer to the license file for details.

## Acknowledgments

- **George Okello** for the original research and Kaggle notebook
- Streamlit team for the excellent web framework
- scikit-learn and PyTorch communities for ML libraries
- African wildlife conservation efforts that make this research meaningful

## Citation

If you use this system in your research, please cite:

```
African Wildlife Classification System
Original Research: George Okello - Africa Wildlife Recognition
Kaggle Notebook: https://www.kaggle.com/code/georgeokello/africa-wildlife-recognition
```

---

For questions, issues, or contributions, please use the GitHub issue tracker or contact the maintainers.