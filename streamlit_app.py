"""
Main Streamlit Application for African Wildlife Classification System
Updated with proper train/validation/test splitting
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# Import custom modules (these would need to be created separately)
try:
    from config import Config
    from session_manager import SessionManager
    from data_manager import DataManager
    from feature_engineering import FeatureExtractor, FeatureOptimizer
    from ml_models import MLModelTrainer
    from deep_learning_models import DeepLearningTrainer
    from visualization_utils import ProfessionalVisualizer
except ImportError:
    st.error("Required modules not found. Please ensure all custom modules are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Wildlife Classification System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    .nav-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class WildlifeClassificationApp:
    """Main application class with proper validation strategy"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.data_manager = DataManager()
        self.visualizer = ProfessionalVisualizer()

        # Initialize session state
        self.session_manager.initialize_session_state()

        # Load previous session if available
        if st.sidebar.button("🔄 Load Previous Session"):
            if self.session_manager.load_session():
                st.success("Previous session loaded successfully!")
                st.rerun()

        # Auto-save session periodically
        if hasattr(st.session_state, 'last_autosave'):
            if time.time() - st.session_state.last_autosave > 300:  # Every 5 minutes
                self.session_manager.save_session()
                st.session_state.last_autosave = time.time()
        else:
            st.session_state.last_autosave = time.time()

    def display_header(self):
        """Display professional header with validation strategy info"""
        st.markdown('<h1 class="main-header">🦁 African Wildlife Classification System</h1>',
                   unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Machine Learning with Proper Validation</h3>',
                   unsafe_allow_html=True)

        # Display validation strategy info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                validation_type = f"{Config.CV_FOLDS}-Fold CV"
            else:
                validation_type = "Train/Val/Test"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{validation_type}</h3>
                <p>Validation</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            device_info = "GPU" if hasattr(Config, 'DEVICE') and Config.DEVICE.type == "cuda" else "CPU"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{device_info}</h3>
                <p>Device</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            n_jobs = getattr(Config, 'N_JOBS', 'Auto')
            st.markdown(f"""
            <div class="metric-card">
                <h3>{n_jobs}</h3>
                <p>CPU Cores</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Show data split percentages
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                split_info = f"{(1-Config.TEST_SIZE)*100:.0f}/{Config.TEST_SIZE*100:.0f}"
            else:
                train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
                val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
                test_size = getattr(Config, 'TEST_SIZE', 0.15)
                split_info = f"{train_size*100:.0f}/{val_size*100:.0f}/{test_size*100:.0f}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{split_info}</h3>
                <p>Data Split %</p>
            </div>
            """, unsafe_allow_html=True)

    def display_sidebar_status(self):
        """Display comprehensive sidebar status with validation info"""
        st.sidebar.markdown("### 📊 System Status")

        status = self.session_manager.get_training_status()

        for task, completed in status.items():
            if completed:
                st.sidebar.markdown(f"✅ {task}")
            else:
                st.sidebar.markdown(f"⏳ {task}")

        # Validation strategy info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Validation Strategy")

        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.sidebar.info(f"Using {Config.CV_FOLDS}-fold cross-validation")
            st.sidebar.text(f"Train+Val: {(1-Config.TEST_SIZE)*100:.0f}%")
            st.sidebar.text(f"Test: {Config.TEST_SIZE*100:.0f}%")
        else:
            st.sidebar.info("Using train/validation/test split")
            train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
            val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
            test_size = getattr(Config, 'TEST_SIZE', 0.15)
            st.sidebar.text(f"Train: {train_size*100:.0f}%")
            st.sidebar.text(f"Validation: {val_size*100:.0f}%")
            st.sidebar.text(f"Test: {test_size*100:.0f}%")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ Quick Actions")

        if st.sidebar.button("💾 Save Session"):
            self.session_manager.save_session()
            st.sidebar.success("Session saved!")

        if st.sidebar.button("🗑️ Clear All Data"):
            if st.sidebar.checkbox("Confirm deletion"):
                self.session_manager.clear_session()
                st.sidebar.success("All data cleared!")
                st.rerun()

        # Dataset info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Dataset Info")

        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()
        if dataset_exists:
            total_images = sum([info['count'] if isinstance(info, dict) else info
                              for info in class_counts.values()])
            st.sidebar.success(f"Dataset: {total_images} images")

            for cls, info in class_counts.items():
                count = info['count'] if isinstance(info, dict) else info
                st.sidebar.text(f"{cls.title()}: {count}")
        else:
            st.sidebar.error("Dataset not found")

    def show_system_overview(self):
        """Display comprehensive system overview with validation explanation"""
        st.header("🏠 System Overview")

        # Dataset check
        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()

        if not dataset_exists:
            st.error("Dataset not found! Please ensure the 'african-wildlife' directory exists with the required structure.")
            st.info("Expected structure: african-wildlife/{buffalo,elephant,rhino,zebra}/")
            return

        st.success(message)

        # Validation strategy explanation
        st.subheader("🔍 Validation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                st.markdown(f"""
                <div class="info-box">
                <strong>Cross-Validation Approach</strong><br>
                • {Config.CV_FOLDS}-fold cross-validation for model selection<br>
                • Separate test set ({Config.TEST_SIZE*100:.0f}%) for final evaluation<br>
                • No data leakage - test set never used for decisions<br>
                • More robust performance estimates
                </div>
                """, unsafe_allow_html=True)
            else:
                train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
                val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
                test_size = getattr(Config, 'TEST_SIZE', 0.15)
                st.markdown(f"""
                <div class="info-box">
                <strong>Train/Validation/Test Split</strong><br>
                • Training set ({train_size*100:.0f}%) for model training<br>
                • Validation set ({val_size*100:.0f}%) for model selection<br>
                • Test set ({test_size*100:.0f}%) for final evaluation<br>
                • No data leakage - clear separation of concerns
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="success-box">
            <strong>Why This Matters</strong><br>
            • Prevents overfitting to test data<br>
            • Provides honest performance estimates<br>
            • Enables proper model selection<br>
            • Follows ML best practices<br>
            • Results are scientifically valid
            </div>
            """, unsafe_allow_html=True)

        # EDA section
        st.subheader("📊 Exploratory Data Analysis")

        if st.session_state.get('eda_completed', False):
            st.success("✅ EDA completed successfully!")

            # Display EDA results if available
            if 'eda_results' in st.session_state:
                eda_results = st.session_state.eda_results

                # Show quick statistics
                dataset_info = eda_results.get('dataset_info', {})
                total_images = dataset_info.get('total_images', 0)

                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Total Images", f"{total_images:,}")
                with col4:
                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    st.metric("Classes", len(classes))
                with col5:
                    balance_ratio = max([info['count'] if isinstance(info, dict) else info
                                       for info in class_counts.values()]) / \
                                  min([info['count'] if isinstance(info, dict) else info
                                      for info in class_counts.values()])
                    st.metric("Balance Ratio", f"{balance_ratio:.1f}")

                # Visualize dataset
                self.visualizer.create_dataset_analysis_dashboard(dataset_info)

            if st.button("🔄 Re-run EDA"):
                with st.spinner("Performing comprehensive EDA..."):
                    self.data_manager.perform_comprehensive_eda()
                    self.session_manager.save_session()
                    st.rerun()
        else:
            st.info("EDA not completed yet")
            if st.button("🚀 Start EDA", type="primary"):
                with st.spinner("Performing comprehensive EDA..."):
                    self.data_manager.perform_comprehensive_eda()
                    self.session_manager.save_session()
                    st.rerun()

    def show_traditional_ml(self):
        """Display traditional ML training interface with proper validation"""
        st.header("🔧 Traditional Machine Learning")

        # Check prerequisites
        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        # Validation strategy reminder
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info(f"Using {Config.CV_FOLDS}-fold cross-validation for model selection and hyperparameter tuning")
        else:
            st.info("Using separate validation set for model selection - test set reserved for final evaluation")

        # Training configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            use_optimized = st.checkbox(
                "Use Optimized Hyperparameters",
                value=True,
                help="Use pre-tuned hyperparameters for better performance"
            )

        with col2:
            perform_feature_selection = st.checkbox(
                "Advanced Feature Selection",
                value=True,
                help="Compare multiple feature selection methods using validation data"
            )

        with col3:
            train_ensembles = st.checkbox(
                "Train Ensemble Methods",
                value=True,
                help="Train comprehensive ensemble models"
            )

        # Display current status
        if st.session_state.get('ml_models_trained', False):
            st.success("✅ Traditional ML models already trained with proper validation!")

            if st.button("🔄 Retrain Models"):
                st.session_state.ml_models_trained = False
                st.session_state.ml_results = None
                st.rerun()

            # Display existing results
            if 'ml_results' in st.session_state and st.session_state.ml_results:
                self.display_ml_results(st.session_state.ml_results)

            return

        # Training button
        if st.button("🚀 Start Traditional ML Training", type="primary"):
            self.train_traditional_ml(use_optimized, perform_feature_selection, train_ensembles)

    def train_traditional_ml(self, use_optimized, perform_feature_selection, train_ensembles):
        """Train traditional ML models with proper train/validation/test splits"""

        start_time = time.time()

        try:
            # Feature extraction
            st.header("🔍 Feature Extraction")

            if 'feature_extractor' not in st.session_state or st.session_state.feature_extractor is None:
                feature_extractor = FeatureExtractor()

                with st.spinner("Extracting features with proper data splitting..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.prepare_dataset()

                if X_train is None:
                    st.error("Feature extraction failed!")
                    return

                if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                    st.success(f"✅ Features extracted! Training+CV: {X_train.shape[0]}, Test: {X_test.shape[0]}")
                else:
                    st.success(f"✅ Features extracted! Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

                st.info(f"Feature dimension: {X_train.shape[1]}")

                st.session_state.feature_extractor = feature_extractor
                st.session_state.X_train = X_train
                st.session_state.X_val = X_val
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_val = y_val
                st.session_state.y_test = y_test
            else:
                feature_extractor = st.session_state.feature_extractor
                X_train = st.session_state.X_train
                X_val = st.session_state.X_val
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_val = st.session_state.y_val
                y_test = st.session_state.y_test
                st.info("✅ Using cached features")

            # Feature optimization
            X_train_final, X_val_final, X_test_final = X_train, X_val, X_test

            if perform_feature_selection:
                st.header("🎯 Advanced Feature Selection")

                if 'feature_optimizer' not in st.session_state:
                    optimizer = FeatureOptimizer()

                    X_train_selected, X_val_selected, X_test_selected, best_selector = optimizer.compare_feature_selection_methods(
                        X_train, X_val, X_test, y_train, y_val, y_test
                    )

                    X_train_final, X_val_final, X_test_final, final_features = optimizer.apply_pca_analysis(
                        X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test
                    )

                    # Store optimization results
                    original_features = X_train.shape[1]
                    reduction_pct = (1 - final_features / original_features) * 100
                    st.session_state.feature_reduction_info = reduction_pct

                    st.session_state.feature_optimizer = optimizer
                    feature_extractor.selector = best_selector
                else:
                    optimizer = st.session_state.feature_optimizer
                    st.info("✅ Using cached feature optimization")

            # Model training with proper validation
            st.header("🤖 Model Training")

            trainer = MLModelTrainer()

            with st.spinner("Training individual models with proper validation..."):
                individual_results = trainer.train_individual_models(
                    X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, use_optimized
                )

            # Ensemble training
            if train_ensembles:
                st.subheader("🎯 Ensemble Methods")
                with st.spinner("Training ensemble methods..."):
                    ensemble_results = trainer.train_ensemble_methods(
                        X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, individual_results
                    )
                    # Combine individual and ensemble results
                    all_results = {**individual_results, **ensemble_results}
            else:
                all_results = individual_results

            # Store results
            st.session_state.ml_results = all_results
            st.session_state.ml_models_trained = True
            st.session_state.feature_extractor = feature_extractor

            # Add training event
            training_time = time.time() - start_time
            best_test_acc = max([r['accuracy'] for r in all_results.values() if r['model'] is not None])
            self.session_manager.add_training_event(
                'Traditional ML Training',
                f"{len(all_results)} models",
                best_test_acc,
                training_time
            )

            # Display results
            self.display_ml_results(all_results)

            # Save session
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    def display_ml_results(self, all_results):
        """Display comprehensive ML results with validation metrics"""

        st.header("📊 Training Results")

        # Initialize trainer for result display
        trainer = MLModelTrainer()

        # Display results with proper validation metrics
        best_model_info = trainer.display_results(all_results)

        if best_model_info:
            best_name, best_result = best_model_info

            # Display confusion matrix (uses test set)
            trainer.display_confusion_matrix(best_result, best_name)

            # Create comprehensive visualizations
            st.subheader("📈 Performance Visualizations")
            self.visualizer.create_model_comparison_dashboard(all_results)

        # Performance summary table
        st.subheader("📋 Detailed Performance Table")

        summary_data = []
        for name, result in all_results.items():
            if result['model'] is not None and result['accuracy'] > 0:
                row = {
                    'Model': name,
                    'Test Accuracy': f"{result['accuracy']:.4f}",
                    'Test Accuracy (%)': f"{result['accuracy']*100:.2f}%",
                    'Training Time': f"{result.get('training_time', 0):.2f}s",
                }

                # Add validation metrics if available
                if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                    if 'cv_mean' in result:
                        row['CV Mean'] = f"{result['cv_mean']:.4f}"
                        row['CV Std'] = f"{result['cv_std']:.4f}"
                else:
                    if 'val_accuracy' in result:
                        row['Val Accuracy'] = f"{result['val_accuracy']:.4f}"

                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('Test Accuracy', ascending=False)
            st.dataframe(df, use_container_width=True)

        # Validation explanation
        st.info("Note: Test accuracy represents final unbiased performance. Validation metrics were used for model selection only.")

    def show_deep_learning(self):
        """Display deep learning training interface with proper validation"""
        st.header("🧠 Deep Learning Models")

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        # Validation strategy reminder
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info("Deep learning uses validation split from training data (simplified CV)")
        else:
            st.info("Using proper train/validation/test split for deep learning")

        # Configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            model_choice = st.selectbox(
                "Select Model",
                ["Custom CNN", "ResNet-18 Transfer"],
                help="Choose between custom CNN or transfer learning"
            )

        with col2:
            epochs = st.slider("Training Epochs", 5, 30, 15)

        with col3:
            device = getattr(Config, 'DEVICE', 'cpu')
            batch_size = getattr(Config, 'DEEP_BATCH_SIZE', 32)
            img_size = getattr(Config, 'DEEP_IMG_SIZE', 224)
            patience = getattr(Config, 'PATIENCE', 5)

            st.markdown(f"""
            <div class="status-card">
            <strong>Device:</strong> {device}<br>
            <strong>Batch Size:</strong> {batch_size}<br>
            <strong>Image Size:</strong> {img_size}px<br>
            <strong>Early Stopping:</strong> {patience} epochs
            </div>
            """, unsafe_allow_html=True)

        # Check training status
        model_key = 'cnn_trained' if model_choice == "Custom CNN" else 'resnet_trained'
        result_key = 'cnn_results' if model_choice == "Custom CNN" else 'resnet_results'

        if st.session_state.get(model_key, False):
            st.success(f"✅ {model_choice} already trained with proper validation!")

            if st.button(f"🔄 Retrain {model_choice}"):
                st.session_state[model_key] = False
                st.session_state[result_key] = None
                st.rerun()

            # Display existing results
            if result_key in st.session_state and st.session_state[result_key]:
                self.display_deep_learning_results(st.session_state[result_key])

            return

        # Training button
        if st.button(f"🚀 Start {model_choice} Training", type="primary"):
            self.train_deep_learning_model(model_choice, epochs)

    def train_deep_learning_model(self, model_choice, epochs):
        """Train deep learning model with proper validation"""

        start_time = time.time()

        try:
            trainer = DeepLearningTrainer()

            if model_choice == "Custom CNN":
                with st.spinner("Training Custom CNN with proper validation..."):
                    results = trainer.train_custom_cnn(epochs)

                if results:
                    st.session_state.cnn_results = results
                    st.session_state.cnn_trained = True

                    # Add training event
                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Deep Learning Training',
                        'Custom CNN',
                        results['best_accuracy'],  # This is now test accuracy
                        training_time
                    )

                    self.display_deep_learning_results(results)

            else:  # ResNet-18
                with st.spinner("Training ResNet-18 with proper validation..."):
                    results = trainer.train_resnet_transfer(epochs)

                if results:
                    st.session_state.resnet_results = results
                    st.session_state.resnet_trained = True

                    # Add training event
                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Deep Learning Training',
                        'ResNet-18',
                        results['best_accuracy'],  # This is now test accuracy
                        training_time
                    )

                    self.display_deep_learning_results(results)

            # Save session
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Deep learning training failed: {str(e)}")
            st.exception(e)

    def display_deep_learning_results(self, results):
        """Display deep learning training results with validation info"""

        st.header("📊 Deep Learning Results")

        # Training summary with validation info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Test Accuracy", f"{results['best_accuracy']:.4f}")

        with col2:
            st.metric("Test Accuracy %", f"{results['best_accuracy']*100:.2f}%")

        with col3:
            st.metric("Training Epochs", results['final_epoch'])

        # Show validation accuracy if available
        if 'val_accuracy' in results:
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Validation Accuracy", f"{results['val_accuracy']:.4f}")
            with col5:
                st.metric("Val-Test Gap", f"{abs(results['val_accuracy'] - results['best_accuracy']):.4f}")

        # Training curves
        st.subheader("📈 Training Progress")
        self.visualizer.create_training_progress_dashboard(results)

        # Model evaluation
        if 'model' in results:
            trainer = DeepLearningTrainer()
            eval_results = trainer.evaluate_model(results['model'], results['model_name'])

        # Validation explanation
        st.info("Note: Best accuracy represents final test set performance. Validation was used for early stopping and model selection.")

    def show_model_comparison(self):
        """Display comprehensive model comparison with proper validation context"""
        st.header("📊 Comprehensive Model Comparison")

        # Validation context
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info("All results show final test set performance. Cross-validation was used for model selection.")
        else:
            st.info("All results show final test set performance. Validation sets were used for model selection.")

        # Collect all results
        all_results = {}

        # Traditional ML results
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results
            for name, result in ml_results.items():
                if result['model'] is not None and result['accuracy'] > 0:
                    all_results[f"ML_{name}"] = {
                        'accuracy': result['accuracy'],  # This is test accuracy
                        'training_time': result.get('training_time', 0),
                        'model_type': 'Traditional ML'
                    }

        # Deep learning results
        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            all_results["Deep_Custom_CNN"] = {
                'accuracy': cnn_result['best_accuracy'],  # This is test accuracy
                'training_time': 0,
                'model_type': 'Deep Learning'
            }

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            all_results["Deep_ResNet18"] = {
                'accuracy': resnet_result['best_accuracy'],  # This is test accuracy
                'training_time': 0,
                'model_type': 'Deep Learning'
            }

        if not all_results:
            st.warning("No trained models found. Please train some models first!")
            return

        # Create comprehensive comparison dashboard
        self.visualizer.create_model_comparison_dashboard(all_results)

        # Ranking table
        st.subheader("🏆 Model Performance Ranking")

        ranking_data = []
        for name, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            clean_name = name.replace('ML_', '').replace('Deep_', '').replace('_', ' ')
            ranking_data.append({
                'Rank': len(ranking_data) + 1,
                'Model': clean_name,
                'Type': result['model_type'],
                'Test Accuracy': f"{result['accuracy']:.4f}",
                'Test Accuracy (%)': f"{result['accuracy']*100:.2f}%"
            })

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True)

        # Performance insights
        st.subheader("💡 Performance Insights")

        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        ml_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Traditional ML'}
        dl_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Deep Learning'}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="success-box">
            <strong>🏆 Best Overall Performance</strong><br>
            Model: {best_model[0].replace('ML_', '').replace('Deep_', '').replace('_', ' ')}<br>
            Test Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if ml_models and dl_models:
                best_ml = max(ml_models.values(), key=lambda x: x['accuracy'])['accuracy']
                best_dl = max(dl_models.values(), key=lambda x: x['accuracy'])['accuracy']

                if best_dl > best_ml:
                    advantage = "Deep Learning"
                    diff = (best_dl - best_ml) * 100
                else:
                    advantage = "Traditional ML"
                    diff = (best_ml - best_dl) * 100

                st.markdown(f"""
                <div class="info-box">
                <strong>📈 Method Comparison</strong><br>
                {advantage} performs better<br>
                Advantage: {diff:.2f} percentage points
                </div>
                """, unsafe_allow_html=True)

        # Generate report
        if st.button("📄 Generate Performance Report"):
            report_fig = self.visualizer.create_performance_report(all_results)

    def show_prediction_interface(self):
        """Display prediction interface"""
        st.header("🔮 Model Prediction Interface")

        # Check available models
        available_models = []

        if st.session_state.get('ml_models_trained', False):
            available_models.extend(['ML_SVM', 'ML_Random Forest', 'ML_Logistic Regression'])

        if st.session_state.get('cnn_trained', False):
            available_models.append('Custom_CNN')

        if st.session_state.get('resnet_trained', False):
            available_models.append('ResNet18')

        if not available_models:
            st.warning("No trained models available. Please train some models first!")
            return

        # Model selection
        selected_model = st.selectbox("Select Model for Prediction", available_models)

        # Input method
        input_method = st.radio("Choose Input Method", ["Upload Image", "Select from Dataset"])

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("🎯 Make Prediction"):
                    self.make_prediction(image, selected_model)

        else:
            # Select from dataset
            classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
            selected_class = st.selectbox("Select class", classes)
            data_dir = getattr(Config, 'DATA_DIR', 'african-wildlife')
            cls_path = os.path.join(data_dir, selected_class)

            if os.path.exists(cls_path):
                valid_exts = getattr(Config, 'VALID_EXTS', ['.jpg', '.jpeg', '.png'])
                files = [f for f in os.listdir(cls_path)
                        if os.path.splitext(f)[1].lower() in valid_exts]

                if files:
                    selected_image = st.selectbox("Select image", files[:20])
                    img_path = os.path.join(cls_path, selected_image)

                    image = Image.open(img_path)
                    st.image(image, caption=f"Selected from {selected_class}", use_container_width=True)

                    if st.button("🎯 Make Prediction"):
                        self.make_prediction(image, selected_model, true_class=selected_class)

    def make_prediction(self, image, model_name, true_class=None):
        """Make prediction on a single image"""

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Prediction Result")

                if model_name.startswith("ML_"):
                    # Traditional ML prediction
                    if 'feature_extractor' not in st.session_state:
                        st.error("Feature extractor not available")
                        return

                    # Save image temporarily
                    temp_path = "temp_pred_image.jpg"
                    image.save(temp_path)

                    try:
                        # Extract features
                        feature_extractor = st.session_state.feature_extractor
                        features = feature_extractor.extract_single_image_features(temp_path)

                        if features is None:
                            st.error("Could not extract features from image")
                            return

                        # Scale and select features
                        features_scaled = feature_extractor.scaler.transform([features])
                        if feature_extractor.selector:
                            features_selected = feature_extractor.selector.transform(features_scaled)
                        else:
                            features_selected = features_scaled

                        # Get model and predict
                        model_key = model_name.replace("ML_", "")
                        if 'ml_results' in st.session_state and model_key in st.session_state.ml_results:
                            model = st.session_state.ml_results[model_key]['model']

                            if model is None:
                                st.error(f"Model {model_key} is not available")
                                return

                            prediction = model.predict(features_selected)[0]

                            # Get probabilities if available
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(features_selected)[0]
                            else:
                                probabilities = None
                        else:
                            st.error(f"Model {model_key} not found in results")
                            return

                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                else:
                    # Deep learning prediction
                    if model_name == "Custom_CNN":
                        if 'cnn_results' not in st.session_state:
                            st.error("Custom CNN model not available")
                            return
                        model = st.session_state.cnn_results['model']
                    else:  # ResNet18
                        if 'resnet_results' not in st.session_state:
                            st.error("ResNet-18 model not available")
                            return
                        model = st.session_state.resnet_results['model']

                    # Prepare image
                    img_size = getattr(Config, 'DEEP_IMG_SIZE', 224)
                    device = getattr(Config, 'DEVICE', 'cpu')

                    val_transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    input_tensor = val_transform(image).unsqueeze(0).to(device)

                    # Predict
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
                        prediction = outputs.argmax(1).item()

                # Display results
                classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                predicted_class = classes[prediction]
                st.metric("Predicted Class", predicted_class.title())

                if true_class:
                    is_correct = predicted_class == true_class
                    st.metric("Result", "✅ Correct" if is_correct else "❌ Incorrect")

            with col2:
                if probabilities is not None:
                    st.subheader("📊 Class Probabilities")

                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    prob_df = pd.DataFrame({
                        'Class': [cls.title() for cls in classes],
                        'Probability': probabilities
                    })

                    # Create probability chart
                    import plotly.express as px
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        title="Prediction Confidence",
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Confidence assessment
                    max_prob = np.max(probabilities)
                    if max_prob > 0.8:
                        st.markdown('<div class="success-box">High Confidence Prediction</div>',
                                   unsafe_allow_html=True)
                    elif max_prob > 0.6:
                        st.markdown('<div class="info-box">Medium Confidence Prediction</div>',
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">Low Confidence Prediction</div>',
                                   unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

    def run(self):
        """Main application runner"""

        # Display header
        self.display_header()

        # Display sidebar
        self.display_sidebar_status()

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Navigation")

        page = st.sidebar.selectbox(
            "Choose a section",
            ["🏠 System Overview", "🔧 Traditional ML", "🧠 Deep Learning",
             "📊 Model Comparison", "🔮 Prediction Interface"],
            key="navigation"
        )

        # Route to appropriate page
        if page == "🏠 System Overview":
            self.show_system_overview()
        elif page == "🔧 Traditional ML":
            self.show_traditional_ml()
        elif page == "🧠 Deep Learning":
            self.show_deep_learning()
        elif page == "📊 Model Comparison":
            self.show_model_comparison()
        elif page == "🔮 Prediction Interface":
            self.show_prediction_interface()

# Main execution
def main():
    """Main function to run the application"""

    try:
        app = WildlifeClassificationApp()
        app.run()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

        # Provide recovery options
        st.markdown("---")
        st.subheader("🔧 Recovery Options")

        if st.button("🔄 Restart Application"):
            st.rerun()

        if st.button("🗑️ Clear All Data"):
            session_manager = SessionManager()
            session_manager.clear_session()
            st.success("All data cleared. Please refresh the page.")

if __name__ == "__main__":
    main()