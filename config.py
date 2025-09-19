"""
Configuration settings for African Wildlife Classification System
Updated to use only cross-validation without train/test splits
"""

import os
import torch
from joblib import cpu_count

class Config:
    """Central configuration class for the application"""

    # Dataset Configuration
    DATA_DIR = "african-wildlife"
    CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']
    VALID_EXTS = ['.jpg', '.jpeg', '.png']

    # Cross-validation Configuration - ONLY APPROACH
    USE_CROSS_VALIDATION = True  # Always True now
    CV_FOLDS = 5
    CV_SCORING = 'accuracy'

    RANDOM_STATE = 42

    # Feature Engineering Configuration
    N_FEATURES = 800
    IMG_SIZE = (128, 128)

    # Parallel Processing Configuration
    N_JOBS = min(cpu_count(), 8)
    BATCH_SIZE = min(100, max(10, cpu_count() * 10))

    # Deep Learning Configuration
    DEEP_IMG_SIZE = 224
    DEEP_BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Training Configuration
    CNN_EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5

    # Feature Selection Configuration
    FEATURE_SELECTION_METHODS = {
        'univariate': {'k': 1000},
        'rfe': {'n_features': 800, 'step': 200},
        'tree_importance': {'threshold': 'median'},
        'lasso': {'alpha': 0.01, 'threshold': 'median'}
    }

    # PCA Configuration
    PCA_VARIANCE_THRESHOLDS = [0.95, 0.90, 0.85, 0.80]

    # Session State Keys
    SESSION_KEYS = {
        'data_loaded': 'data_loaded',
        'models_trained': 'ml_models_trained',
        'cnn_trained': 'cnn_trained',
        'resnet_trained': 'resnet_trained',
        'ensemble_trained': 'ensemble_trained',
        'feature_extractor': 'feature_extractor',
        'ml_results': 'ml_results',
        'cnn_results': 'cnn_results',
        'resnet_results': 'resnet_results',
        'ensemble_results': 'ensemble_results',
        'eda_completed': 'eda_completed',
        'eda_results': 'eda_results'
    }

    # Visualization Configuration
    COLORS = {
        'primary': '#2E8B57',
        'secondary': '#667eea',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }

    # Report Configuration
    REPORT_CONFIG = {
        'figure_size': (12, 8),
        'dpi': 300,
        'style': 'seaborn-v0_8-whitegrid',
        'color_palette': 'viridis'
    }

    @classmethod
    def validate_data_splits(cls):
        """Validate that cross-validation configuration is correct"""
        if cls.CV_FOLDS < 2:
            raise ValueError("Cross-validation requires at least 2 folds")

# Utility functions for compatibility
def get_data_path():
    """Get the data directory path"""
    return Config.DATA_DIR

def get_class_path(class_name):
    """Get the path for a specific class"""
    return os.path.join(Config.DATA_DIR, class_name)

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['saved_models', 'cache', 'reports', 'temp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_data_directory():
    """Validate that the data directory structure is correct"""
    if not os.path.exists(Config.DATA_DIR):
        return False, f"Data directory '{Config.DATA_DIR}' not found"

    for class_name in Config.CLASSES:
        class_path = get_class_path(class_name)
        if not os.path.exists(class_path):
            return False, f"Class directory '{class_path}' not found"

    return True, "Data directory structure is valid"

# Configuration objects for backward compatibility with utils.py
class DATA_CONFIG:
    classes = Config.CLASSES
    valid_extensions = Config.VALID_EXTS

class MODEL_CONFIG:
    random_state = Config.RANDOM_STATE

class FEATURE_CONFIG:
    target_size = Config.IMG_SIZE
    hog_orientations = 9

class UI_CONFIG:
    primary_color = Config.COLORS['primary']
    secondary_color = Config.COLORS['secondary']
    success_color = Config.COLORS['success']
    warning_color = Config.COLORS['warning']
    danger_color = Config.COLORS['danger']
    info_color = Config.COLORS['info']

class DEEP_LEARNING_CONFIG:
    device = Config.DEVICE
    deep_img_size = Config.DEEP_IMG_SIZE
    batch_size = Config.DEEP_BATCH_SIZE

class PROCESSING_CONFIG:
    n_jobs = Config.N_JOBS
    batch_size = Config.BATCH_SIZE
    feature_selection_methods = Config.FEATURE_SELECTION_METHODS
    pca_variance_thresholds = Config.PCA_VARIANCE_THRESHOLDS

class VISUALIZATION_CONFIG:
    style = Config.REPORT_CONFIG['style']
    color_palette = Config.REPORT_CONFIG['color_palette']
    confusion_matrix_cmap = 'Blues'
    performance_cmap = 'viridis'
    dashboard_height = 600