"""
Feature extraction and optimization for traditional ML models
Updated with proper train/validation/test splitting
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed
import time
from config import Config


class FeatureExtractor:
    """Extract comprehensive features from images with proper data splitting"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = None
        self.feature_names = []
        self.extraction_stats = {}

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
        """Prepare complete dataset with proper train/validation/test splitting"""
        st.info("Preparing dataset with proper train/validation/test splits...")

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
            return None, None, None, None, None, None

        X = np.array(X)
        y = np.array(y)

        st.success(f"Successfully extracted features from {len(X)} images")
        st.info(f"Feature vector dimension: {X.shape[1]}")

        # Proper data splitting based on configuration
        if Config.USE_CROSS_VALIDATION:
            # Cross-validation approach: only need train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )

            # For cross-validation, we don't create a separate validation set
            X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None

            st.info(f"Using {Config.CV_FOLDS}-fold cross-validation")
            st.info(f"Training+Validation: {len(X_train)} samples, Test: {len(X_test)} samples")

        else:
            # Three-way split: train/validation/test
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )

            # Second split: separate training and validation from remaining data
            val_size_adjusted = Config.VALIDATION_SIZE / (Config.TRAIN_SIZE + Config.VALIDATION_SIZE)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=Config.RANDOM_STATE,
                stratify=y_temp
            )

            # Display split information
            if perform_quality_check:
                st.subheader("Dataset Split Information")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Training Samples", len(X_train))
                    st.metric("Training %", f"{(len(X_train) / len(X)) * 100:.1f}%")

                with col2:
                    st.metric("Validation Samples", len(X_val))
                    st.metric("Validation %", f"{(len(X_val) / len(X)) * 100:.1f}%")

                with col3:
                    st.metric("Test Samples", len(X_test))
                    st.metric("Test %", f"{(len(X_test) / len(X)) * 100:.1f}%")

        return X_train, X_val, X_test, y_train, y_val, y_test


class FeatureOptimizer:
    """Advanced feature selection and optimization with proper validation"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.selection_methods = {}
        self.results = {}
        self.best_method = None

    def compare_feature_selection_methods(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Compare multiple feature selection methods using validation set"""

        st.subheader("Advanced Feature Selection Analysis")
        st.info("Comparing multiple feature selection methods using validation set...")

        # Scale features first
        X_train_scaled = self.scaler.fit_transform(X_train)

        if Config.USE_CROSS_VALIDATION:
            # For cross-validation, we don't have a separate validation set
            X_val_scaled = None
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

        selection_methods = {}

        # 1. Baseline (no selection)
        selection_methods['No Selection'] = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'n_features': X_train_scaled.shape[1],
            'selector': None,
            'description': 'All original features'
        }

        # 2. Univariate Feature Selection
        st.text("Running Univariate F-score selection...")
        k_best = min(1000, X_train_scaled.shape[1])
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_f = f_selector.fit_transform(X_train_scaled, y_train)

        if X_val_scaled is not None:
            X_val_f = f_selector.transform(X_val_scaled)
        else:
            X_val_f = None
        X_test_f = f_selector.transform(X_test_scaled)

        selection_methods['Univariate F-test'] = {
            'X_train': X_train_f,
            'X_val': X_val_f,
            'X_test': X_test_f,
            'n_features': X_train_f.shape[1],
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
        n_features_rfe = min(800, X_train_scaled.shape[1])
        rfe_selector = RFE(
            estimator=rf_estimator,
            n_features_to_select=n_features_rfe,
            step=max(1, X_train_scaled.shape[1] // 20)
        )
        X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)

        if X_val_scaled is not None:
            X_val_rfe = rfe_selector.transform(X_val_scaled)
        else:
            X_val_rfe = None
        X_test_rfe = rfe_selector.transform(X_test_scaled)

        selection_methods['RFE'] = {
            'X_train': X_train_rfe,
            'X_val': X_val_rfe,
            'X_test': X_test_rfe,
            'n_features': X_train_rfe.shape[1],
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
        extra_trees.fit(X_train_scaled, y_train)
        importance_selector = SelectFromModel(extra_trees, threshold='median')
        X_train_importance = importance_selector.fit_transform(X_train_scaled, y_train)

        if X_val_scaled is not None:
            X_val_importance = importance_selector.transform(X_val_scaled)
        else:
            X_val_importance = None
        X_test_importance = importance_selector.transform(X_test_scaled)

        selection_methods['Tree Importance'] = {
            'X_train': X_train_importance,
            'X_val': X_val_importance,
            'X_test': X_test_importance,
            'n_features': X_train_importance.shape[1],
            'selector': importance_selector,
            'description': 'Extra Trees feature importance'
        }

        # Performance evaluation using validation set or cross-validation
        st.text("Evaluating feature selection methods...")
        performance_results = {}

        progress_bar = st.progress(0)

        if Config.USE_CROSS_VALIDATION:
            # Use cross-validation for evaluation
            cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

            for i, (method, data) in enumerate(selection_methods.items()):
                X_train_method = data['X_train']
                start_time = time.time()

                # Use SVM for quick evaluation with CV
                svm_test = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
                cv_scores = cross_val_score(svm_test, X_train_method, y_train, cv=cv, scoring='accuracy')

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
        else:
            # Use validation set for evaluation
            for i, (method, data) in enumerate(selection_methods.items()):
                X_train_method = data['X_train']
                X_val_method = data['X_val']

                start_time = time.time()

                # Use SVM for quick evaluation
                svm_test = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
                svm_test.fit(X_train_method, y_train)
                y_pred = svm_test.predict(X_val_method)

                training_time = time.time() - start_time
                accuracy = accuracy_score(y_val, y_pred)

                performance_results[method] = {
                    'accuracy': accuracy,
                    'time': training_time,
                    'n_features': data['n_features'],
                    'description': data['description']
                }

                progress_bar.progress((i + 1) / len(selection_methods))

        progress_bar.empty()

        # Display comprehensive results
        self._display_feature_selection_results(performance_results, X_train_scaled.shape[1])

        # Select best method
        best_method_name = self._select_best_method(performance_results)
        best_method_data = selection_methods[best_method_name]

        # Store results
        self.results['feature_selection'] = performance_results
        self.best_method = best_method_name
        self.selection_methods = selection_methods

        return best_method_data['X_train'], best_method_data['X_val'], best_method_data['X_test'], best_method_data['selector']

    def _display_feature_selection_results(self, performance_results, original_features):
        """Display comprehensive feature selection results"""

        # Create results table
        results_data = []
        for method, results in performance_results.items():
            n_features = results['n_features']
            reduction = (1 - n_features / original_features) * 100

            if Config.USE_CROSS_VALIDATION:
                results_data.append({
                    'Method': method,
                    'Features': n_features,
                    'Reduction (%)': f"{reduction:.1f}%",
                    'CV Accuracy': f"{results['accuracy']:.4f}",
                    'CV Std': f"{results.get('std', 0):.4f}",
                    'Time (s)': f"{results['time']:.2f}",
                    'Description': results['description']
                })
            else:
                results_data.append({
                    'Method': method,
                    'Features': n_features,
                    'Reduction (%)': f"{reduction:.1f}%",
                    'Val Accuracy': f"{results['accuracy']:.4f}",
                    'Time (s)': f"{results['time']:.2f}",
                    'Description': results['description']
                })

        st.subheader("Feature Selection Methods Comparison")
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

    def _select_best_method(self, performance_results):
        """Select the best feature selection method based on validation performance"""

        # Simply select method with highest validation accuracy
        best_method = max(performance_results.keys(),
                         key=lambda x: performance_results[x]['accuracy'])

        # Display recommendation
        st.subheader("Feature Selection Recommendation")

        best_results = performance_results[best_method]

        if Config.USE_CROSS_VALIDATION:
            st.markdown(f"""
            **Recommended Method:** {best_method}

            - **CV Accuracy:** {best_results['accuracy']:.4f} ± {best_results.get('std', 0):.4f}
            - **Features:** {best_results['n_features']:,}
            - **Training Time:** {best_results['time']:.2f}s
            - **Description:** {best_results['description']}
            """)
        else:
            st.markdown(f"""
            **Recommended Method:** {best_method}

            - **Validation Accuracy:** {best_results['accuracy']:.4f}
            - **Features:** {best_results['n_features']:,}
            - **Training Time:** {best_results['time']:.2f}s
            - **Description:** {best_results['description']}
            """)

        return best_method

    def apply_pca_analysis(self, X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test):
        """Apply PCA with comprehensive analysis using validation set"""

        st.subheader("PCA Dimensionality Reduction Analysis")
        st.info("Analyzing PCA with different variance thresholds...")

        pca_results = {}
        pca_performance = {}

        # Test different variance thresholds
        for threshold in Config.PCA_VARIANCE_THRESHOLDS:
            pca = PCA(n_components=threshold, random_state=Config.RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train_selected)

            if Config.USE_CROSS_VALIDATION:
                X_val_pca = None
            else:
                X_val_pca = pca.transform(X_val_selected)
            X_test_pca = pca.transform(X_test_selected)

            pca_results[threshold] = {
                'pca': pca,
                'X_train': X_train_pca,
                'X_val': X_val_pca,
                'X_test': X_test_pca,
                'n_components': pca.n_components_,
                'explained_variance': pca.explained_variance_ratio_.sum()
            }

            # Performance evaluation
            start_time = time.time()

            if Config.USE_CROSS_VALIDATION:
                # Use cross-validation
                cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
                svm_pca = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
                cv_scores = cross_val_score(svm_pca, X_train_pca, y_train, cv=cv, scoring='accuracy')
                accuracy = cv_scores.mean()
            else:
                # Use validation set
                svm_pca = SVC(kernel='rbf', C=1.0, random_state=Config.RANDOM_STATE)
                svm_pca.fit(X_train_pca, y_train)
                y_pred_pca = svm_pca.predict(X_val_pca)
                accuracy = accuracy_score(y_val, y_pred_pca)

            training_time = time.time() - start_time

            pca_performance[threshold] = {
                'accuracy': accuracy,
                'time': training_time,
                'components': pca.n_components_
            }

        # Choose best configuration based on validation performance
        best_threshold = max(pca_performance.keys(),
                           key=lambda x: pca_performance[x]['accuracy'])
        best_config = pca_results[best_threshold]

        st.success(f"Best PCA configuration: {best_threshold*100:.0f}% variance ({best_config['n_components']} components)")

        return best_config['X_train'], best_config['X_val'], best_config['X_test'], best_config['n_components']