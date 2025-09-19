"""
Feature extraction and optimization for traditional ML models
Fixed version with stable visualizations and proper state management
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed
import time
import hashlib
from config import Config


class FeatureExtractor:
    """Extract comprehensive features from images with cross-validation pipeline management"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None  # Will store the best feature selector
        self.pca_transformer = None   # Will store PCA if used
        self.feature_names = []
        self.extraction_stats = {}
        self.is_fitted = False
        self.pipeline_info = {}  # Track what transformations were applied

    def extract_single_image_features(self, image_path, extract_stats=False):
        """Extract comprehensive features from a single image"""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            img_resized = cv2.resize(img, Config.IMG_SIZE)
            img_eq = cv2.equalizeHist(img_resized)

            features = []
            feature_names = []

            # 1. HOG Features (Histogram of Oriented Gradients)
            hog_features, hog_image = hog(
                img_eq,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm="L2-Hys"
            )
            features.extend(hog_features)
            feature_names.extend([f'hog_{i}' for i in range(len(hog_features))])

            # 2. LBP Features (Local Binary Patterns)
            radius, n_points = 2, 16
            lbp = local_binary_pattern(img_eq, n_points, radius, method="uniform")
            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, n_points + 3),
                range=(0, n_points + 2)
            )
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)
            features.extend(lbp_hist)
            feature_names.extend([f'lbp_{i}' for i in range(len(lbp_hist))])

            # 3. Intensity Histogram Features
            hist = cv2.calcHist([img_eq], [0], None, [64], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            features.extend(hist)
            feature_names.extend([f'hist_{i}' for i in range(len(hist))])

            # 4. Statistical Features
            stats = [
                np.mean(img_eq),  # Mean intensity
                np.std(img_eq),  # Standard deviation
                np.var(img_eq),  # Variance
                np.min(img_eq),  # Minimum intensity
                np.max(img_eq),  # Maximum intensity
                np.median(img_eq),  # Median intensity
                np.percentile(img_eq, 25),  # 25th percentile
                np.percentile(img_eq, 75),  # 75th percentile
                cv2.Laplacian(img_eq, cv2.CV_64F).var()  # Laplacian variance (focus measure)
            ]
            features.extend(stats)
            feature_names.extend([
                'mean_intensity', 'std_intensity', 'var_intensity',
                'min_intensity', 'max_intensity', 'median_intensity',
                'q25_intensity', 'q75_intensity', 'laplacian_var'
            ])

            # 5. Texture Features (GLCM-based)
            sobelx = cv2.Sobel(img_eq, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_eq, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

            texture_features = [
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.mean(np.abs(sobelx)),
                np.mean(np.abs(sobely))
            ]
            features.extend(texture_features)
            feature_names.extend([
                'gradient_magnitude_mean', 'gradient_magnitude_std',
                'sobel_x_mean', 'sobel_y_mean'
            ])

            # Store feature names for first extraction
            if not self.feature_names:
                self.feature_names = feature_names

            features_array = np.array(features)

            # Store extraction statistics if requested
            if extract_stats:
                self.extraction_stats[image_path] = {
                    'feature_count': len(features_array),
                    'hog_features': len(hog_features),
                    'lbp_features': len(lbp_hist),
                    'hist_features': len(hist),
                    'stat_features': len(stats),
                    'texture_features': len(texture_features)
                }

            return features_array

        except Exception as e:
            if extract_stats:
                self.extraction_stats[image_path] = {'error': str(e)}
            return None

    def extract_features_batch(self, image_paths, show_progress=True):
        """Extract features from multiple images in parallel"""
        if show_progress:
            st.info(f"Extracting features from {len(image_paths)} images using {Config.N_JOBS} cores")

        # Extract features in parallel
        features_list = Parallel(n_jobs=Config.N_JOBS, verbose=0)(
            delayed(self.extract_single_image_features)(img_path, extract_stats=False)
            for img_path in image_paths
        )

        # Filter out None results
        valid_features = [f for f in features_list if f is not None]

        return valid_features

    def prepare_dataset(self, perform_quality_check=True):
        """Prepare complete dataset for cross-validation"""
        st.info("Preparing dataset for cross-validation...")

        # Validate configuration
        Config.validate_data_splits()

        X, y = [], []
        all_paths = []
        all_labels = []
        class_file_counts = {}

        # Collect all image paths and labels
        for label, cls in enumerate(Config.CLASSES):
            cls_path = os.path.join(Config.DATA_DIR, cls)
            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                         if os.path.splitext(f)[1].lower() in Config.VALID_EXTS]

                class_file_counts[cls] = len(files)

                for file in files:
                    img_path = os.path.join(cls_path, file)
                    all_paths.append(img_path)
                    all_labels.append(label)

        total_files = len(all_paths)
        st.info(f"Processing {total_files} images across {len(Config.CLASSES)} classes")

        # Display class distribution
        if perform_quality_check:
            st.subheader("Dataset Composition")
            col1, col2, col3 = st.columns(3)

            with col1:
                for cls, count in class_file_counts.items():
                    st.metric(f"{cls.title()}", count)

        # Process images in batches
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = Config.BATCH_SIZE
        for i in range(0, total_files, batch_size):
            batch_end = min(i + batch_size, total_files)
            batch_paths = all_paths[i:batch_end]
            batch_labels = all_labels[i:batch_end]

            status_text.text(f"Processing batch {i // batch_size + 1}/{(total_files - 1) // batch_size + 1}")

            batch_features = self.extract_features_batch(batch_paths, show_progress=False)

            # Align features with labels (handle failed extractions)
            valid_batch_labels = []
            for j, features in enumerate(batch_features):
                if features is not None:
                    X.append(features)
                    valid_batch_labels.append(batch_labels[j])

            y.extend(valid_batch_labels)
            progress_bar.progress(batch_end / total_files)

        progress_bar.empty()
        status_text.empty()

        if not X:
            st.error("No features could be extracted from the dataset!")
            return None, None

        X = np.array(X)
        y = np.array(y)

        st.success(f"Successfully extracted features from {len(X)} images")
        st.info(f"Feature vector dimension: {X.shape[1]}")

        # For cross-validation, we use all data for training
        # The scaler will be fitted on the full dataset
        self.scaler.fit(X)
        self.is_fitted = True

        # Store pipeline info for later reference
        self.pipeline_info = {
            'original_features': X.shape[1],
            'scaling_applied': True,
            'feature_selection_applied': False,
            'pca_applied': False,
            'total_samples': len(X)
        }

        st.info("Feature scaler fitted on complete dataset for cross-validation")

        return X, y

    def set_feature_pipeline(self, feature_selector=None, pca_transformer=None):
        """Set the complete feature transformation pipeline"""
        self.feature_selector = feature_selector
        self.pca_transformer = pca_transformer

        # Update pipeline info
        if feature_selector is not None:
            self.pipeline_info['feature_selection_applied'] = True
            if hasattr(feature_selector, 'k'):  # SelectKBest
                self.pipeline_info['selected_features'] = feature_selector.k
            elif hasattr(feature_selector, 'n_features_'):  # RFE or SelectFromModel
                self.pipeline_info['selected_features'] = feature_selector.n_features_

        if pca_transformer is not None:
            self.pipeline_info['pca_applied'] = True
            self.pipeline_info['pca_components'] = pca_transformer.n_components_

        st.info(f"Feature pipeline configured: {self.get_pipeline_summary()}")

    def get_pipeline_summary(self):
        """Get a summary of the applied transformations"""
        summary = []

        if self.pipeline_info.get('scaling_applied', False):
            summary.append("Scaling")

        if self.pipeline_info.get('feature_selection_applied', False):
            n_features = self.pipeline_info.get('selected_features', 'unknown')
            summary.append(f"Feature Selection ({n_features} features)")

        if self.pipeline_info.get('pca_applied', False):
            n_components = self.pipeline_info.get('pca_components', 'unknown')
            summary.append(f"PCA ({n_components} components)")

        return " → ".join(summary) if summary else "No transformations"

    def transform_features(self, features):
        """Apply the complete transformation pipeline to features"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first by calling prepare_dataset()")

        # Ensure features is 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Step 1: Scale features
        features_transformed = self.scaler.transform(features)

        # Step 2: Apply feature selection if available
        if self.feature_selector is not None:
            features_transformed = self.feature_selector.transform(features_transformed)

        # Step 3: Apply PCA if available
        if self.pca_transformer is not None:
            features_transformed = self.pca_transformer.transform(features_transformed)

        return features_transformed

    def get_expected_feature_count(self):
        """Get the expected number of features after all transformations"""
        if not self.is_fitted:
            return None

        expected_count = self.pipeline_info.get('original_features')

        if self.pipeline_info.get('feature_selection_applied', False):
            expected_count = self.pipeline_info.get('selected_features', expected_count)

        if self.pipeline_info.get('pca_applied', False):
            expected_count = self.pipeline_info.get('pca_components', expected_count)

        return expected_count


class FeatureOptimizer:
    """Advanced feature selection and optimization with cross-validation and fixed visualizations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.selection_methods = {}
        self.results = {}
        self.best_method = None
        self.best_selector = None
        self.pca_transformer = None

    def compare_feature_selection_methods(self, X, y):
        """Compare multiple feature selection methods using cross-validation with fixed visualizations"""

        st.subheader("Advanced Feature Selection Analysis")
        st.info("Comparing multiple feature selection methods using cross-validation...")

        # Scale features first
        X_scaled = self.scaler.fit_transform(X)

        selection_methods = {}

        # 1. Baseline (no selection)
        selection_methods['No Selection'] = {
            'X_transformed': X_scaled,
            'n_features': X_scaled.shape[1],
            'selector': None,
            'description': 'All original features'
        }

        # 2. Univariate Feature Selection
        st.text("Running Univariate F-score selection...")
        k_best = min(1000, X_scaled.shape[1])
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_f = f_selector.fit_transform(X_scaled, y)

        selection_methods['Univariate F-test'] = {
            'X_transformed': X_f,
            'n_features': X_f.shape[1],
            'selector': f_selector,
            'description': f'Top {k_best} features by F-score'
        }

        # 3. Recursive Feature Elimination
        st.text("Running Recursive Feature Elimination...")
        rf_estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS
        )
        n_features_rfe = min(800, X_scaled.shape[1])
        rfe_selector = RFE(
            estimator=rf_estimator,
            n_features_to_select=n_features_rfe,
            step=max(1, X_scaled.shape[1] // 20)
        )
        X_rfe = rfe_selector.fit_transform(X_scaled, y)

        selection_methods['RFE'] = {
            'X_transformed': X_rfe,
            'n_features': X_rfe.shape[1],
            'selector': rfe_selector,
            'description': f'RFE with Random Forest ({n_features_rfe} features)'
        }

        # 4. Tree-based importance
        st.text("Running Tree-based feature importance...")
        extra_trees = ExtraTreesClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS
        )
        extra_trees.fit(X_scaled, y)
        importance_selector = SelectFromModel(extra_trees, threshold='median')
        X_importance = importance_selector.fit_transform(X_scaled, y)

        selection_methods['Tree Importance'] = {
            'X_transformed': X_importance,
            'n_features': X_importance.shape[1],
            'selector': importance_selector,
            'description': 'Extra Trees feature importance'
        }

        # Performance evaluation using cross-validation
        st.text("Evaluating feature selection methods...")
        performance_results = {}

        progress_bar = st.progress(0)
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

        for i, (method, data) in enumerate(selection_methods.items()):
            X_method = data['X_transformed']
            start_time = time.time()

            # Use SVM for quick evaluation with CV
            svm_test = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
            cv_scores = cross_val_score(svm_test, X_method, y, cv=cv, scoring='accuracy',
                                        n_jobs=min(Config.N_JOBS, Config.CV_FOLDS))

            training_time = time.time() - start_time
            mean_accuracy = cv_scores.mean()

            performance_results[method] = {
                'accuracy': mean_accuracy,
                'std': cv_scores.std(),
                'time': training_time,
                'n_features': data['n_features'],
                'description': data['description']
            }

            progress_bar.progress((i + 1) / len(selection_methods))

        progress_bar.empty()

        # Display comprehensive results with fixed visualization
        self._display_feature_selection_results_fixed(performance_results, X_scaled.shape[1])

        # Select best method
        best_method_name = self._select_best_method(performance_results)
        best_method_data = selection_methods[best_method_name]

        # Store the best selector for pipeline configuration
        self.best_selector = best_method_data['selector']

        # Store results
        self.results['feature_selection'] = performance_results
        self.best_method = best_method_name
        self.selection_methods = selection_methods

        return best_method_data['X_transformed'], best_method_data['selector']

    def _display_feature_selection_results_fixed(self, performance_results, original_features):
        """Display comprehensive feature selection results with stable visualization"""

        # Create unique key based on data hash for consistency
        data_hash = hashlib.md5(str(performance_results).encode()).hexdigest()[:8]
        plot_key = f"feature_selection_{data_hash}"

        # Store in session state for persistence
        if plot_key not in st.session_state:
            st.session_state[plot_key] = {
                'performance_results': performance_results,
                'original_features': original_features,
                'timestamp': time.time()
            }

        # Create results table
        results_data = []
        for method, results in performance_results.items():
            n_features = results['n_features']
            reduction = (1 - n_features / original_features) * 100

            results_data.append({
                'Method': method,
                'Features': n_features,
                'Reduction (%)': f"{reduction:.1f}%",
                'CV Accuracy': f"{results['accuracy']:.4f}",
                'CV Std': f"{results.get('std', 0):.4f}",
                'Time (s)': f"{results['time']:.2f}",
                'Description': results['description']
            })

        st.subheader("Feature Selection Methods Comparison")
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        # Create stable visualization with container and refresh option
        viz_container = st.container()

        with viz_container:
            col1, col2 = st.columns([4, 1])

            with col2:
                st.markdown("**Controls:**")
                if st.button("Refresh Plot", key=f"refresh_{plot_key}"):
                    if plot_key in st.session_state:
                        del st.session_state[plot_key]
                    st.experimental_rerun()

                st.markdown("**Status:**")
                st.success("✅ Plot Ready")

            with col1:
                # Create the visualization
                fig = self._create_feature_selection_plot(results_df)
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_key}")

    def _create_feature_selection_plot(self, results_df):
        """Create feature selection comparison plot"""

        # Convert accuracy strings back to floats for plotting
        results_df['Accuracy_Float'] = results_df['CV Accuracy'].str.replace('CV Accuracy: ', '').astype(float)
        results_df['Features_Int'] = results_df['Features'].astype(int)

        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Feature Selection Performance', 'Features vs Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Bar chart of accuracy
        fig.add_trace(
            go.Bar(
                x=results_df['Method'],
                y=results_df['Accuracy_Float'],
                name='CV Accuracy',
                marker_color=px.colors.qualitative.Set3,
                text=[f'{acc:.3f}' for acc in results_df['Accuracy_Float']],
                textposition='outside'
            ),
            row=1, col=1
        )

        # Scatter plot of features vs accuracy
        fig.add_trace(
            go.Scatter(
                x=results_df['Features_Int'],
                y=results_df['Accuracy_Float'],
                mode='markers+text',
                text=results_df['Method'],
                textposition='top center',
                marker=dict(
                    size=12,
                    color=results_df['Accuracy_Float'],
                    colorscale='viridis',
                    showscale=True
                ),
                name='Performance'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=500,
            title_text="Feature Selection Analysis Dashboard",
            showlegend=False
        )

        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_yaxes(title_text="CV Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Number of Features", row=1, col=2)
        fig.update_yaxes(title_text="CV Accuracy", row=1, col=2)

        return fig

    def apply_pca_analysis(self, X_selected, y):
        """Apply PCA with comprehensive analysis using cross-validation and fixed visualization"""

        st.subheader("PCA Dimensionality Reduction Analysis")
        st.info("Analyzing PCA with different variance thresholds...")

        pca_results = {}
        pca_performance = {}

        # Test different variance thresholds
        for threshold in Config.PCA_VARIANCE_THRESHOLDS:
            pca = PCA(n_components=threshold, random_state=Config.RANDOM_STATE)
            X_pca = pca.fit_transform(X_selected)

            pca_results[threshold] = {
                'pca': pca,
                'X_transformed': X_pca,
                'n_components': pca.n_components_,
                'explained_variance': pca.explained_variance_ratio_.sum()
            }

            # Performance evaluation using cross-validation
            start_time = time.time()
            cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
            svm_pca = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
            cv_scores = cross_val_score(svm_pca, X_pca, y, cv=cv, scoring='accuracy')
            accuracy = cv_scores.mean()

            training_time = time.time() - start_time

            pca_performance[threshold] = {
                'accuracy': accuracy,
                'time': training_time,
                'components': pca.n_components_
            }

        # Display PCA analysis with fixed visualization
        self._display_pca_analysis_fixed(pca_results, pca_performance)

        # Choose best configuration based on cross-validation performance
        best_threshold = max(pca_performance.keys(),
                           key=lambda x: pca_performance[x]['accuracy'])
        best_config = pca_results[best_threshold]

        # Store the best PCA transformer
        self.pca_transformer = best_config['pca']

        st.success(f"Best PCA configuration: {best_threshold*100:.0f}% variance ({best_config['n_components']} components)")

        return best_config['X_transformed'], best_config['n_components']

    def _display_pca_analysis_fixed(self, pca_results, pca_performance):
        """Display PCA analysis with stable visualization"""

        # Create unique key for PCA visualization
        pca_hash = hashlib.md5(str(pca_performance).encode()).hexdigest()[:8]
        pca_key = f"pca_analysis_{pca_hash}"

        # Store in session state
        if pca_key not in st.session_state:
            st.session_state[pca_key] = {
                'pca_results': pca_results,
                'pca_performance': pca_performance,
                'timestamp': time.time()
            }

        # Create PCA dashboard with container
        pca_container = st.container()

        with pca_container:
            # Control panel
            col_control, col_viz = st.columns([1, 4])

            with col_control:
                st.markdown("**PCA Controls:**")
                if st.button("Refresh PCA Plot", key=f"refresh_pca_{pca_key}"):
                    if pca_key in st.session_state:
                        del st.session_state[pca_key]
                    st.experimental_rerun()

                st.markdown("**Analysis:**")
                best_threshold = max(pca_performance.keys(),
                                   key=lambda x: pca_performance[x]['accuracy'])
                st.metric("Best Threshold", f"{best_threshold*100:.0f}%")
                st.metric("Components", pca_results[best_threshold]['n_components'])
                st.metric("Accuracy", f"{pca_performance[best_threshold]['accuracy']:.4f}")

            with col_viz:
                # Create PCA visualization
                fig = self._create_pca_dashboard(pca_results, pca_performance)
                st.plotly_chart(fig, use_container_width=True, key=f"pca_plot_{pca_key}")

    def _create_pca_dashboard(self, pca_results, pca_performance):
        """Create comprehensive PCA analysis dashboard"""

        # Prepare data
        thresholds = list(pca_results.keys())
        components = [pca_results[t]['n_components'] for t in thresholds]
        explained_var = [pca_results[t]['explained_variance'] for t in thresholds]
        accuracies = [pca_performance[t]['accuracy'] for t in thresholds]
        times = [pca_performance[t]['time'] for t in thresholds]

        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Explained Variance by Components',
                'Performance vs Components',
                'Variance Threshold Comparison',
                'PCA Summary Statistics'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}]
            ]
        )

        # 1. Explained variance plot
        fig.add_trace(
            go.Scatter(
                x=components,
                y=explained_var,
                mode='lines+markers',
                name='Explained Variance',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )

        # 2. Performance vs components
        fig.add_trace(
            go.Scatter(
                x=components,
                y=accuracies,
                mode='lines+markers',
                name='CV Accuracy',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ),
            row=1, col=2
        )

        # 3. Variance threshold comparison
        fig.add_trace(
            go.Bar(
                x=[f"{t*100:.0f}%" for t in thresholds],
                y=components,
                name='Components Required',
                marker_color='green',
                text=components,
                textposition='outside'
            ),
            row=2, col=1
        )

        # 4. Summary table
        summary_data = {
            'Threshold': [f"{t*100:.0f}%" for t in thresholds],
            'Components': components,
            'Explained Var': [f"{v:.3f}" for v in explained_var],
            'CV Accuracy': [f"{a:.4f}" for a in accuracies],
            'Time (s)': [f"{t:.2f}" for t in times]
        }

        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(summary_data.keys()),
                    fill_color='lightblue',
                    align='center'
                ),
                cells=dict(
                    values=list(summary_data.values()),
                    fill_color='white',
                    align='center'
                )
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=700,
            title_text="Comprehensive PCA Analysis Dashboard",
            showlegend=False
        )

        # Update axis labels
        fig.update_xaxes(title_text="Number of Components", row=1, col=1)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)

        fig.update_xaxes(title_text="Number of Components", row=1, col=2)
        fig.update_yaxes(title_text="CV Accuracy", row=1, col=2)

        fig.update_xaxes(title_text="Variance Threshold", row=2, col=1)
        fig.update_yaxes(title_text="Components Required", row=2, col=1)

        return fig

    def _select_best_method(self, performance_results):
        """Select the best feature selection method based on cross-validation performance"""
        # Simply select method with highest CV accuracy
        best_method = max(performance_results.keys(),
                         key=lambda x: performance_results[x]['accuracy'])

        # Display recommendation
        st.subheader("Feature Selection Recommendation")

        best_results = performance_results[best_method]

        st.markdown(f"""
        **Recommended Method:** {best_method}

        - **CV Accuracy:** {best_results['accuracy']:.4f} ± {best_results.get('std', 0):.4f}
        - **Features:** {best_results['n_features']:,}
        - **Training Time:** {best_results['time']:.2f}s
        - **Description:** {best_results['description']}
        """)

        return best_method