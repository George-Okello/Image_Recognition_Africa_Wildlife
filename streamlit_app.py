"""
Complete Streamlit Application for African Wildlife Classification System
With integrated deep learning visualizations
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

# Import custom modules
try:
    from config import Config
    from session_manager import SessionManager
    from data_manager import DataManager
    from feature_engineering import FeatureExtractor, FeatureOptimizer
    from ml_models import MLModelTrainer
    from deep_learning_models import (
        DeepLearningTrainer, DeepLearningVisualizer,
        create_model_ensemble, analyze_model_complexity
    )
    from visualization_utils import ProfessionalVisualizer
    from utils import (
        validate_feature_pipeline, debug_feature_transformation,
        display_pipeline_status, safe_make_prediction_with_debug,
        StyleManager
    )
except ImportError as e:
    st.error(f"Required modules not found: {e}")
    st.error("Please ensure all custom modules are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Wildlife Classification System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(StyleManager.create_custom_css(), unsafe_allow_html=True)


class WildlifeClassificationApp:
    """Main application class with enhanced deep learning visualizations"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.data_manager = DataManager()
        self.visualizer = ProfessionalVisualizer()

        # Initialize session state
        self.session_manager.initialize_session_state()

        # Load previous session if available
        if st.sidebar.button("📄 Load Previous Session"):
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
        """Display professional header"""
        st.markdown('<h1 class="main-header">🦁 African Wildlife Classification System</h1>',
                   unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Advanced ML & Deep Learning with Comprehensive Visualizations</h3>',
                   unsafe_allow_html=True)

        # Display validation strategy info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{Config.CV_FOLDS}-Fold CV</h3>
                <p>Traditional ML</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>70/15/15</h3>
                <p>Deep Learning</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            device_info = "GPU" if hasattr(Config, 'DEVICE') and Config.DEVICE.type == "cuda" else "CPU"
            st.markdown(f"""
            <div class="metric-container">
                <h3>{device_info}</h3>
                <p>Device</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            n_jobs = getattr(Config, 'N_JOBS', 'Auto')
            st.markdown(f"""
            <div class="metric-container">
                <h3>{n_jobs}</h3>
                <p>CPU Cores</p>
            </div>
            """, unsafe_allow_html=True)

    def display_sidebar_status(self):
        """Display comprehensive sidebar status"""
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

        st.sidebar.info(f"Traditional ML: {Config.CV_FOLDS}-fold CV")
        st.sidebar.info("Deep Learning: 70/15/15 split")

        # Enhanced features info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ✨ Enhanced Features")
        st.sidebar.success("🎨 Advanced Visualizations")
        st.sidebar.success("🧠 Model Interpretability")
        st.sidebar.success("📈 Real-time Training Analytics")
        st.sidebar.success("🎯 Ensemble Predictions")

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
        """Display comprehensive system overview"""
        st.header("🏠 System Overview")

        # Dataset check
        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()

        if not dataset_exists:
            st.error("Dataset not found! Please ensure the 'african-wildlife' directory exists with the required structure.")
            st.info("Expected structure: african-wildlife/{buffalo,elephant,rhino,zebra}/")
            return

        st.success(message)

        # Validation strategy explanation
        st.subheader("🔍 Mixed Validation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="info-message">
            <strong>Traditional ML: Cross-Validation</strong><br>
            • {Config.CV_FOLDS}-fold cross-validation for model selection<br>
            • Uses 100% of data for training<br>
            • No separate test set holdout<br>
            • Robust performance estimates<br>
            • Ideal for smaller datasets
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="success-message">
            <strong>Deep Learning: Train/Val/Test Split</strong><br>
            • Stratified 70/15/15 split<br>
            • Separate validation for early stopping<br>
            • Independent test set for final evaluation<br>
            • Prevents overfitting in complex models<br>
            • Standard practice for deep learning
            </div>
            """, unsafe_allow_html=True)

        # Enhanced features showcase
        st.subheader("✨ Enhanced Features")

        feature_cols = st.columns(4)

        with feature_cols[0]:
            st.markdown("""
            <div class="metric-container">
                <h4>🎨 Advanced Visualizations</h4>
                <p>Comprehensive training analytics, feature maps, and model insights</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[1]:
            st.markdown("""
            <div class="metric-container">
                <h4>🧠 Model Interpretability</h4>
                <p>Attention maps and prediction explanations</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[2]:
            st.markdown("""
            <div class="metric-container">
                <h4>📈 Real-time Analytics</h4>
                <p>Live training progress, overfitting detection</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[3]:
            st.markdown("""
            <div class="metric-container">
                <h4>🎯 Ensemble Methods</h4>
                <p>Multi-model predictions and analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # EDA section
        st.subheader("📊 Exploratory Data Analysis")

        if st.session_state.get('eda_completed', False):
            st.success("✅ EDA completed successfully!")

            if 'eda_results' in st.session_state:
                eda_results = st.session_state.eda_results
                dataset_info = eda_results.get('dataset_info', {})
                total_images = dataset_info.get('total_images', 0)
                quality_report = eda_results.get('quality_report', {})

                # Quick metrics overview
                col3, col4, col5, col6 = st.columns(4)
                with col3:
                    st.metric("Total Images", f"{total_images:,}")
                with col4:
                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    st.metric("Classes", len(classes))
                with col5:
                    if class_counts:
                        balance_ratio = max([info['count'] if isinstance(info, dict) else info
                                           for info in class_counts.values()]) / \
                                      min([info['count'] if isinstance(info, dict) else info
                                          for info in class_counts.values()])
                        st.metric("Balance Ratio", f"{balance_ratio:.1f}")
                    else:
                        st.metric("Balance Ratio", "N/A")
                with col6:
                    quality_score = quality_report.get('quality_score', 0)
                    st.metric("Quality Score", f"{quality_score}%")

                recommendations = eda_results.get('recommendations', [])
                if recommendations:
                    with st.expander("💡 Dataset Recommendations"):
                        for rec in recommendations:
                            st.write(f"• {rec}")

            if st.button("🔄 Re-run EDA"):
                st.session_state.eda_completed = False
                st.session_state.eda_results = None

                with st.spinner("Performing comprehensive EDA..."):
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA re-run completed!")
                    else:
                        st.error("EDA failed. Please check your dataset structure.")
        else:
            st.info("EDA not completed yet")
            if st.button("🚀 Start EDA", type="primary"):
                with st.spinner("Performing comprehensive EDA..."):
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA completed!")
                    else:
                        st.error("EDA failed. Please check your dataset structure.")

    def show_traditional_ml(self):
        """Display traditional ML training interface"""
        st.header("🔧 Traditional Machine Learning")

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        st.info(f"Using {Config.CV_FOLDS}-fold cross-validation for model evaluation")

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
                help="Compare multiple feature selection methods using cross-validation"
            )

        with col3:
            train_ensembles = st.checkbox(
                "Train Ensemble Methods",
                value=True,
                help="Train comprehensive ensemble models"
            )

        if st.session_state.get('ml_models_trained', False):
            st.success("✅ Traditional ML models already trained!")

            if st.button("🔄 Retrain Models"):
                st.session_state.ml_models_trained = False
                st.session_state.ml_results = None
                st.rerun()

            if 'ml_results' in st.session_state and st.session_state.ml_results:
                self.display_ml_results(st.session_state.ml_results)

            return

        if st.button("🚀 Start Traditional ML Training", type="primary"):
            self.train_traditional_ml(use_optimized, perform_feature_selection, train_ensembles)

    def train_traditional_ml(self, use_optimized, perform_feature_selection, train_ensembles):
        """Train traditional ML models"""

        start_time = time.time()

        try:
            # Feature extraction
            st.header("🔍 Feature Extraction")

            if 'feature_extractor' not in st.session_state or st.session_state.feature_extractor is None:
                feature_extractor = FeatureExtractor()

                with st.spinner("Extracting features for cross-validation..."):
                    X, y = feature_extractor.prepare_dataset()

                if X is None:
                    st.error("Feature extraction failed!")
                    return

                st.success(f"✅ Features extracted! Total samples: {X.shape[0]}")
                st.info(f"Original feature dimension: {X.shape[1]}")

                st.session_state.feature_extractor = feature_extractor
                st.session_state.X = X
                st.session_state.y = y
            else:
                feature_extractor = st.session_state.feature_extractor
                X = st.session_state.X
                y = st.session_state.y
                st.info("✅ Using cached features")

            # Feature optimization
            X_final = X

            if perform_feature_selection:
                st.header("🎯 Advanced Feature Selection")

                if 'feature_optimizer' not in st.session_state:
                    optimizer = FeatureOptimizer()
                    X_selected, best_selector = optimizer.compare_feature_selection_methods(X, y)
                    X_final, final_components = optimizer.apply_pca_analysis(X_selected, y)

                    feature_extractor.set_feature_pipeline(
                        feature_selector=best_selector,
                        pca_transformer=optimizer.pca_transformer
                    )

                    st.session_state.feature_optimizer = optimizer
                    st.session_state.X_final = X_final
                else:
                    X_final = st.session_state.X_final

            # Model training
            st.header("🤖 Model Training")

            trainer = MLModelTrainer()

            with st.spinner("Training individual models with cross-validation..."):
                individual_results = trainer.train_individual_models(X_final, y, use_optimized)

            if train_ensembles:
                st.subheader("🎯 Ensemble Methods")
                with st.spinner("Training ensemble methods..."):
                    ensemble_results = trainer.train_ensemble_methods(X_final, y, individual_results)
                    all_results = {**individual_results, **ensemble_results}
            else:
                all_results = individual_results

            st.session_state.ml_results = all_results
            st.session_state.ml_models_trained = True

            training_time = time.time() - start_time
            best_cv_acc = max([r['cv_mean'] for r in all_results.values() if r['model'] is not None])
            self.session_manager.add_training_event(
                'Traditional ML Training',
                f"{len(all_results)} models",
                best_cv_acc,
                training_time
            )

            self.display_ml_results(all_results)
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    def display_ml_results(self, all_results):
        """Display ML results"""

        st.header("📊 Training Results")

        trainer = MLModelTrainer()
        best_model_info = trainer.display_results(all_results)

        if best_model_info:
            st.subheader("📈 Performance Visualizations")
            self.visualizer.create_model_comparison_dashboard(all_results)

        # Performance summary
        summary_data = []
        for name, result in all_results.items():
            if result['model'] is not None and result['accuracy'] > 0:
                row = {
                    'Model': name,
                    'CV Mean': f"{result['cv_mean']:.4f}",
                    'CV Std': f"{result['cv_std']:.4f}",
                    'CV Accuracy (%)': f"{result['cv_mean']*100:.2f}%",
                    'Training Time': f"{result.get('training_time', 0):.2f}s",
                }
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('CV Mean', ascending=False)
            st.dataframe(df, use_container_width=True)

    def show_enhanced_deep_learning(self):
        """Enhanced deep learning training interface"""
        st.header("🧠 Enhanced Deep Learning Models")

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        st.info("🎨 Enhanced deep learning with comprehensive visualizations")

        # Configuration
        st.subheader("⚙️ Advanced Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            model_choice = st.selectbox(
                "Select Model Architecture",
                ["Custom CNN", "ResNet-18 Transfer"],
                help="Choose between custom CNN or transfer learning"
            )

        with col2:
            epochs = st.slider("Training Epochs", 5, 30, 15)
            show_augmentation = st.checkbox("Show Data Augmentation", value=True)

        with col3:
            enable_visualizations = st.checkbox("Enable Advanced Visualizations", value=True)
            save_checkpoints = st.checkbox("Save Model Checkpoints", value=False)

        # Check training status
        model_key = 'cnn_trained' if model_choice == "Custom CNN" else 'resnet_trained'
        result_key = 'cnn_results' if model_choice == "Custom CNN" else 'resnet_results'

        if st.session_state.get(model_key, False):
            st.success(f"✅ {model_choice} already trained!")

            if result_key in st.session_state and st.session_state[result_key]:
                self.display_enhanced_deep_learning_results(st.session_state[result_key], enable_visualizations)

            if st.button(f"🔄 Retrain {model_choice}"):
                st.session_state[model_key] = False
                st.session_state[result_key] = None
                st.rerun()

            return

        # Training button
        if st.button(f"🚀 Start Enhanced {model_choice} Training", type="primary"):
            self.train_enhanced_deep_learning_model(model_choice, epochs, enable_visualizations, show_augmentation)

    def train_enhanced_deep_learning_model(self, model_choice, epochs, enable_visualizations=True, show_augmentation=True):
        """Enhanced deep learning training"""

        start_time = time.time()

        try:
            trainer = DeepLearningTrainer()

            # Training with visualizations
            if model_choice == "Custom CNN":
                with st.spinner("Training Custom CNN with enhanced analytics..."):
                    results = trainer.train_custom_cnn(epochs, enable_visualizations)

                if results:
                    st.session_state.cnn_results = results
                    st.session_state.cnn_trained = True

                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Enhanced Deep Learning Training',
                        'Custom CNN',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_enhanced_deep_learning_results(results, enable_visualizations)

            else:  # ResNet-18
                with st.spinner("Training ResNet-18 with enhanced analytics..."):
                    results = trainer.train_resnet_transfer(epochs, enable_visualizations)

                if results:
                    st.session_state.resnet_results = results
                    st.session_state.resnet_trained = True

                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Enhanced Deep Learning Training',
                        'ResNet-18',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_enhanced_deep_learning_results(results, enable_visualizations)

            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Enhanced deep learning training failed: {str(e)}")
            st.exception(e)

    def display_enhanced_deep_learning_results(self, results, enable_visualizations=True):
        """Display enhanced deep learning results"""

        st.header("📊 Enhanced Deep Learning Results")

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Test Accuracy", f"{results['best_accuracy']:.4f}")

        with col2:
            val_accuracy = results.get('val_accuracy', 0)
            st.metric("Validation Accuracy", f"{val_accuracy:.4f}")

        with col3:
            st.metric("Test Accuracy %", f"{results['best_accuracy']*100:.2f}%")

        with col4:
            final_epoch = results.get('final_epoch', 0)
            st.metric("Final Epoch", str(final_epoch))

        # Comprehensive analysis with visualizations
        if enable_visualizations and 'history' in results:
            trainer = DeepLearningTrainer()

            st.subheader("📈 Comprehensive Analysis")

            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Training Dashboard", "Model Architecture", "Performance Analysis"])

            with tab1:
                # Training dashboard
                trainer.visualizer.create_training_dashboard(results['history'])

            with tab2:
                # Model architecture
                trainer.visualizer.create_model_architecture_visualization(
                    results['model'], results['model_name']
                )

                # Model complexity analysis
                if results.get('model'):
                    complexity = analyze_model_complexity(results['model'])

                    st.subheader("Model Complexity Analysis")
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("Parameters", f"{complexity['total_params']:,}")
                    with col6:
                        st.metric("Model Size (MB)", f"{complexity['model_size_mb']:.1f}")
                    with col7:
                        st.metric("Est. FLOPs", f"{complexity['estimated_flops']:,}")

            with tab3:
                # Performance analysis
                self.create_performance_analysis(results)

        # Performance summary
        st.subheader("📋 Performance Summary")

        summary_data = {
            'Metric': ['Test Accuracy', 'Validation Accuracy', 'Final Epoch', 'Model Size'],
            'Value': [
                f"{results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)",
                f"{val_accuracy:.4f} ({val_accuracy*100:.2f}%)",
                str(final_epoch),
                f"{sum(p.numel() for p in results['model'].parameters()):,} params" if results.get('model') else "N/A"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    def create_performance_analysis(self, results):
        """Create detailed performance analysis"""

        st.subheader("Model Performance Analysis")

        # Standard evaluation
        trainer = DeepLearningTrainer()
        evaluation_results = trainer.evaluate_model(results, results['model_name'])

        if evaluation_results:
            st.success("Model evaluation completed")

    def show_ensemble_predictions(self):
        """Ensemble predictions interface"""

        st.header("🎯 Ensemble Model Predictions")

        # Check available models
        available_models = {}

        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            available_models['Custom CNN'] = st.session_state.cnn_results['model']

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            available_models['ResNet-18'] = st.session_state.resnet_results['model']

        if len(available_models) < 2:
            st.info("Train at least 2 deep learning models to enable ensemble predictions")
            return

        st.success(f"Ensemble available with {len(available_models)} models: {', '.join(available_models.keys())}")

        # Create ensemble
        ensemble_model = create_model_ensemble(
            {k: {'model': v} for k, v in available_models.items()},
            Config.DEVICE
        )

        if ensemble_model is None:
            st.error("Failed to create ensemble model")
            return

        # Prediction interface
        uploaded_file = st.file_uploader("Upload image for ensemble prediction", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🎯 Make Ensemble Prediction"):
                self.make_ensemble_prediction(ensemble_model, image, list(available_models.keys()))

    def make_ensemble_prediction(self, ensemble_model, image, model_names):
        """Make ensemble prediction"""

        try:
            # Prepare image
            transform = transforms.Compose([
                transforms.Resize((Config.DEEP_IMG_SIZE, Config.DEEP_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

            # Make prediction
            ensemble_model.eval()
            with torch.no_grad():
                ensemble_output = ensemble_model(input_tensor)
                probabilities = ensemble_output.squeeze().cpu().numpy()
                prediction = probabilities.argmax()

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                predicted_class = Config.CLASSES[prediction]
                confidence = probabilities[prediction]

                st.metric("Predicted Class", predicted_class.title())
                st.metric("Ensemble Confidence", f"{confidence:.4f}")
                st.metric("Models Used", len(model_names))

                if confidence > 0.8:
                    st.success("High Confidence")
                elif confidence > 0.6:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")

            with col2:
                # Probability chart
                class_names = [cls.title() for cls in Config.CLASSES]
                prob_data = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probabilities
                })

                import plotly.express as px
                fig = px.bar(prob_data, x='Class', y='Probability',
                            title="Ensemble Prediction Probabilities",
                            color='Probability', color_continuous_scale='viridis')
                fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ensemble prediction failed: {str(e)}")

    def show_model_comparison(self):
        """Display comprehensive model comparison"""
        st.header("📊 Comprehensive Model Comparison")

        st.info("Traditional ML shows cross-validated performance. Deep learning shows test set performance.")

        # Collect all results
        all_results = {}

        # Traditional ML results
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results
            for name, result in ml_results.items():
                if result['model'] is not None and result['cv_mean'] > 0:
                    all_results[f"ML_{name}"] = {
                        'accuracy': result['cv_mean'],
                        'model_type': 'Traditional ML'
                    }

        # Deep learning results
        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            all_results["Deep_Custom_CNN"] = {
                'accuracy': cnn_result['best_accuracy'],
                'model_type': 'Deep Learning'
            }

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            all_results["Deep_ResNet18"] = {
                'accuracy': resnet_result['best_accuracy'],
                'model_type': 'Deep Learning'
            }

        if not all_results:
            st.warning("No trained models found. Please train some models first!")
            return

        # Create comparison dashboard
        self.visualizer.create_model_comparison_dashboard(all_results)

        # Ranking table
        st.subheader("Model Performance Ranking")

        ranking_data = []
        for name, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            clean_name = name.replace('ML_', '').replace('Deep_', '').replace('_', ' ')
            validation_method = "CV" if result['model_type'] == 'Traditional ML' else "Test"
            ranking_data.append({
                'Rank': len(ranking_data) + 1,
                'Model': clean_name,
                'Type': result['model_type'],
                'Validation': validation_method,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Accuracy (%)': f"{result['accuracy']*100:.2f}%"
            })

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True)

        # Performance insights
        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        ml_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Traditional ML'}
        dl_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Deep Learning'}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="success-message">
            <strong>Best Overall Performance</strong><br>
            Model: {best_model[0].replace('ML_', '').replace('Deep_', '').replace('_', ' ')}<br>
            Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)
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
                <div class="info-message">
                <strong>Method Comparison</strong><br>
                {advantage} performs better<br>
                Advantage: {diff:.2f} percentage points
                </div>
                """, unsafe_allow_html=True)

    def show_prediction_interface(self):
        """Display enhanced prediction interface"""
        st.header("Model Prediction Interface")

        # Get available models
        available_models = self._get_organized_models()

        if not available_models:
            st.warning("No trained models available. Please train some models first!")
            return

        # Display model categories
        st.subheader("Available Models")

        total_models = sum(len(models) for models in available_models.values())
        st.info(f"Total available models: {total_models}")

        # Model selection
        model_categories = list(available_models.keys())
        selected_category = st.selectbox("Select Model Category", model_categories)

        if selected_category in available_models and available_models[selected_category]:
            models_in_category = available_models[selected_category]

            if selected_category == "Top 5 Traditional ML":
                st.write("**Top 5 performing traditional ML models based on CV accuracy:**")
                for i, (model_name, cv_score) in enumerate(models_in_category, 1):
                    st.write(f"{i}. {model_name}: {cv_score:.4f}")
            elif selected_category == "Ensemble Methods":
                st.write("**Individual ensemble methods:**")
                for model_name, cv_score in models_in_category:
                    st.write(f"• {model_name}: {cv_score:.4f}")
            elif selected_category == "Deep Learning":
                st.write("**Trained deep learning models:**")
                for model_name, test_score in models_in_category:
                    st.write(f"• {model_name}: {test_score:.4f} (test accuracy)")

            model_options = [f"{name} (Acc: {score:.4f})" for name, score in models_in_category]
            model_names = [name for name, score in models_in_category]

            selected_display = st.selectbox("Select Specific Model", model_options)
            selected_model = model_names[model_options.index(selected_display)]
        else:
            st.warning(f"No models available in {selected_category} category")
            return

        # Input method selection
        st.subheader("Input Method")
        input_method = st.radio("Choose Input Method", ["Upload Image", "Select from Dataset"])

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Make Prediction"):
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
                    st.image(image, caption=f"Selected from {selected_class}", use_column_width=True)

                    if st.button("Make Prediction"):
                        self.make_prediction(image, selected_model, true_class=selected_class)

    def _get_organized_models(self):
        """Get available models organized by category"""
        organized_models = {}

        # Traditional ML models
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results

            traditional_models = []
            ensemble_models = []

            for name, result in ml_results.items():
                if result['model'] is not None and 'cv_mean' in result:
                    cv_score = result['cv_mean']

                    if any(ensemble_name in name.lower() for ensemble_name in
                           ['voting', 'bagging', 'gradient', 'ada', 'boost']):
                        ensemble_models.append((name, cv_score))
                    else:
                        traditional_models.append((name, cv_score))

            traditional_models.sort(key=lambda x: x[1], reverse=True)
            top_5_traditional = traditional_models[:5]

            if top_5_traditional:
                organized_models["Top 5 Traditional ML"] = top_5_traditional

            if ensemble_models:
                ensemble_models.sort(key=lambda x: x[1], reverse=True)
                organized_models["Ensemble Methods"] = ensemble_models

        # Deep learning models
        deep_learning_models = []

        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            test_score = cnn_result.get('best_accuracy', 0)
            deep_learning_models.append(("Custom_CNN", test_score))

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            test_score = resnet_result.get('best_accuracy', 0)
            deep_learning_models.append(("ResNet18", test_score))

        if deep_learning_models:
            deep_learning_models.sort(key=lambda x: x[1], reverse=True)
            organized_models["Deep Learning"] = deep_learning_models

        return organized_models

    def make_prediction(self, image, model_name, true_class=None):
        """Make prediction with comprehensive validation"""

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Result")

                # Handle traditional ML models
                if model_name in ['SVM', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes'] or \
                   any(ensemble_name in model_name for ensemble_name in
                       ['Voting', 'Bagging', 'Gradient', 'AdaBoost']):

                    if 'feature_extractor' not in st.session_state:
                        st.error("Feature extractor not available")
                        return

                    if 'ml_results' not in st.session_state:
                        st.error("ML results not available")
                        return

                    feature_extractor = st.session_state.feature_extractor
                    ml_results = st.session_state.ml_results

                    debug_mode = st.checkbox("Enable Debug Mode")

                    ml_model_name = f"ML_{model_name}"
                    prediction, probabilities, result_message = safe_make_prediction_with_debug(
                        image, ml_model_name, feature_extractor, ml_results, debug_mode
                    )

                    if prediction is None:
                        st.error(f"Prediction failed: {result_message}")
                        return

                elif model_name in ["Custom_CNN", "ResNet18"]:
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

                else:
                    st.error(f"Unknown model type: {model_name}")
                    return

                # Display results
                classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                predicted_class = classes[prediction]
                st.metric("Predicted Class", predicted_class.title())

                if true_class:
                    is_correct = predicted_class == true_class
                    st.metric("Result", "Correct" if is_correct else "Incorrect")

            with col2:
                if probabilities is not None:
                    st.subheader("Class Probabilities")

                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    prob_df = pd.DataFrame({
                        'Class': [cls.title() for cls in classes],
                        'Probability': probabilities
                    })

                    import plotly.express as px
                    fig = px.bar(prob_df, x='Class', y='Probability',
                                title="Prediction Confidence",
                                color='Probability', color_continuous_scale='viridis')
                    fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    max_prob = np.max(probabilities)
                    st.metric("Max Probability", f"{max_prob:.3f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    def run(self):
        """Main application runner"""

        # Display header
        self.display_header()

        # Display sidebar
        self.display_sidebar_status()

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Navigation")

        page = st.sidebar.selectbox(
            "Choose a section",
            [
                "System Overview",
                "Traditional ML",
                "Enhanced Deep Learning",
                "Model Comparison",
                "Prediction Interface",
                "Ensemble Predictions"
            ],
            key="navigation"
        )

        # Route to appropriate page
        if page == "System Overview":
            self.show_system_overview()
        elif page == "Traditional ML":
            self.show_traditional_ml()
        elif page == "Enhanced Deep Learning":
            self.show_enhanced_deep_learning()
        elif page == "Model Comparison":
            self.show_model_comparison()
        elif page == "Prediction Interface":
            self.show_prediction_interface()
        elif page == "Ensemble Predictions":
            self.show_ensemble_predictions()

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(f"Device: {Config.DEVICE}")


# Main execution
def main():
    """Main function to run the enhanced application"""

    try:
        app = WildlifeClassificationApp()
        app.run()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

        # Recovery options
        st.markdown("---")
        st.subheader("Recovery Options")

        if st.button("Restart Application"):
            st.rerun()

        if st.button("Clear All Data"):
            session_manager = SessionManager()
            session_manager.clear_session()
            st.success("All data cleared. Please refresh the page.")


if __name__ == "__main__":
    main()