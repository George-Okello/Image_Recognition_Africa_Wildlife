"""
Traditional Machine Learning Models Training and Evaluation
Updated to use only cross-validation without train/test splits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import copy
import time
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from config import Config


class MLModelTrainer:
    """ML model training with cross-validation only"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def get_model_configurations(self, use_optimized=False):
        """Get model configurations"""
        if use_optimized:
            return {
                'SVM': SVC(C=1.0, gamma='scale', kernel='rbf', probability=True,
                           random_state=Config.RANDOM_STATE),
                'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10,
                                                        min_samples_split=10,
                                                        random_state=Config.RANDOM_STATE,
                                                        n_jobs=Config.N_JOBS),
                'Logistic Regression': LogisticRegression(C=0.1, penalty='l1',
                                                          solver='liblinear',
                                                          max_iter=2000,
                                                          random_state=Config.RANDOM_STATE),
                'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance',
                                            metric='cosine', n_jobs=Config.N_JOBS),
                'Naive Bayes': GaussianNB(var_smoothing=1e-12)
            }
        else:
            return {
                'SVM': SVC(kernel="rbf", C=1.0, probability=True,
                           random_state=Config.RANDOM_STATE),
                'Random Forest': RandomForestClassifier(n_estimators=200,
                                                        random_state=Config.RANDOM_STATE,
                                                        n_jobs=Config.N_JOBS),
                'Logistic Regression': LogisticRegression(random_state=Config.RANDOM_STATE,
                                                          max_iter=1000),
                'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=Config.N_JOBS),
                'Naive Bayes': GaussianNB()
            }

    def train_individual_models(self, X, y, use_optimized=False):
        """Train individual ML models using cross-validation"""

        models_config = self.get_model_configurations(use_optimized)
        config_type = "Optimized" if use_optimized else "Default"

        st.info(f"Training models with {config_type} parameters using {Config.CV_FOLDS}-fold cross-validation")

        individual_results = {}
        progress_bar = st.progress(0)

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)

        for i, (name, model) in enumerate(models_config.items()):
            start_time = time.time()
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=cv,
                                            scoring=Config.CV_SCORING,
                                            n_jobs=min(Config.N_JOBS, Config.CV_FOLDS))

                # Train final model on full dataset for predictions
                final_model = copy.deepcopy(model)
                final_model.fit(X, y)

                # Get cross-validated predictions for confusion matrix
                from sklearn.model_selection import cross_val_predict
                cv_predictions = cross_val_predict(model, X, y, cv=cv)

                individual_results[name] = {
                    'model': final_model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores,
                    'accuracy': cv_scores.mean(),  # For compatibility
                    'predictions': cv_predictions,
                    'y_true': y,
                    'training_time': time.time() - start_time
                }

                st.success(
                    f"{name}: CV {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

            except Exception as e:
                st.error(f"Failed to train {name}: {str(e)}")
                individual_results[name] = {
                    'model': None, 'accuracy': 0.0, 'error': str(e)
                }

            progress_bar.progress((i + 1) / len(models_config))

        progress_bar.empty()
        return individual_results

    def train_ensemble_methods(self, X, y, individual_results):
        """Train ensemble methods using cross-validation"""

        st.subheader("Training Ensemble Methods")

        # Get valid models for ensembles
        valid_models = [(name.lower().replace(' ', '_'), result['model'])
                        for name, result in individual_results.items()
                        if result['model'] is not None and
                        name in ['SVM', 'Random Forest', 'Logistic Regression']]

        if len(valid_models) < 2:
            st.error("Need at least 2 valid models for ensemble methods")
            return {}

        ensemble_results = {}
        ensemble_methods = [
            ('Hard Voting', self._train_voting_ensemble, {'voting': 'hard'}),
            ('Soft Voting', self._train_voting_ensemble, {'voting': 'soft'}),
            ('Bagging', self._train_bagging_ensemble, {}),
            ('Gradient Boosting', self._train_gradient_boosting, {}),
            ('AdaBoost', self._train_adaboost, {})
        ]

        progress_bar = st.progress(0)

        for i, (name, train_func, params) in enumerate(ensemble_methods):
            try:
                result = train_func(valid_models, X, y, **params)
                ensemble_results[name] = result

                st.success(f"{name}: CV {result.get('cv_mean', 0):.4f} (±{result.get('cv_std', 0)*2:.4f})")

            except Exception as e:
                st.error(f"Failed to train {name}: {str(e)}")
                ensemble_results[name] = {'model': None, 'accuracy': 0.0, 'error': str(e)}

            progress_bar.progress((i + 1) / len(ensemble_methods))

        progress_bar.empty()
        return ensemble_results

    def _train_voting_ensemble(self, base_models, X, y, voting='hard'):
        """Train voting ensemble using cross-validation"""
        start_time = time.time()

        voting_clf = VotingClassifier(estimators=base_models, voting=voting,
                                      n_jobs=Config.N_JOBS)

        # Cross-validation
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(voting_clf, X, y, cv=cv,
                                    scoring=Config.CV_SCORING)

        # Train final model
        final_model = copy.deepcopy(voting_clf)
        final_model.fit(X, y)

        # Get cross-validated predictions
        from sklearn.model_selection import cross_val_predict
        cv_predictions = cross_val_predict(voting_clf, X, y, cv=cv)

        return {
            'model': final_model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': cv_scores.mean(),
            'predictions': cv_predictions,
            'y_true': y,
            'training_time': time.time() - start_time
        }

    def _train_bagging_ensemble(self, base_models, X, y):
        """Train bagging ensemble using cross-validation"""
        start_time = time.time()

        bagging_clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=Config.RANDOM_STATE),
            n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(bagging_clf, X, y, cv=cv,
                                    scoring=Config.CV_SCORING)

        # Train final model
        final_model = copy.deepcopy(bagging_clf)
        final_model.fit(X, y)

        # Get cross-validated predictions
        from sklearn.model_selection import cross_val_predict
        cv_predictions = cross_val_predict(bagging_clf, X, y, cv=cv)

        return {
            'model': final_model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': cv_scores.mean(),
            'predictions': cv_predictions,
            'y_true': y,
            'training_time': time.time() - start_time
        }

    def _train_gradient_boosting(self, base_models, X, y):
        """Train gradient boosting using cross-validation"""
        start_time = time.time()

        gb_clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=Config.RANDOM_STATE
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(gb_clf, X, y, cv=cv,
                                    scoring=Config.CV_SCORING)

        # Train final model
        final_model = copy.deepcopy(gb_clf)
        final_model.fit(X, y)

        # Get cross-validated predictions
        from sklearn.model_selection import cross_val_predict
        cv_predictions = cross_val_predict(gb_clf, X, y, cv=cv)

        return {
            'model': final_model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': cv_scores.mean(),
            'predictions': cv_predictions,
            'y_true': y,
            'training_time': time.time() - start_time
        }

    def _train_adaboost(self, base_models, X, y):
        """Train AdaBoost using cross-validation"""
        start_time = time.time()

        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=Config.RANDOM_STATE),
            n_estimators=100, learning_rate=1.0, random_state=Config.RANDOM_STATE
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(ada_clf, X, y, cv=cv,
                                    scoring=Config.CV_SCORING)

        # Train final model
        final_model = copy.deepcopy(ada_clf)
        final_model.fit(X, y)

        # Get cross-validated predictions
        from sklearn.model_selection import cross_val_predict
        cv_predictions = cross_val_predict(ada_clf, X, y, cv=cv)

        return {
            'model': final_model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': cv_scores.mean(),
            'predictions': cv_predictions,
            'y_true': y,
            'training_time': time.time() - start_time
        }

    def display_results(self, all_results):
        """Display comprehensive results"""

        # Filter valid results
        valid_results = {name: result for name, result in all_results.items()
                         if result['model'] is not None and result['accuracy'] > 0}

        if not valid_results:
            st.error("No valid models to display")
            return

        # Sort by CV accuracy
        sorted_results = sorted(valid_results.items(),
                                key=lambda x: x[1]['accuracy'], reverse=True)

        # Display ranking
        st.subheader("Model Performance Ranking")

        ranking_data = []
        for i, (name, result) in enumerate(sorted_results, 1):
            row_data = {
                'Rank': i,
                'Model': name,
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}",
                'CV Accuracy (%)': f"{result['cv_mean'] * 100:.2f}%",
                'Training Time': f"{result.get('training_time', 0):.2f}s"
            }
            ranking_data.append(row_data)

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True)

        # Best model
        best_name, best_result = sorted_results[0]
        st.success(f"Best Model: {best_name} (CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f})")

        # Display confusion matrix for best model
        self.display_confusion_matrix(best_result, best_name)

        return sorted_results[0]

    def display_confusion_matrix(self, result, model_name):
        """Display confusion matrix using cross-validated predictions"""

        y_true = result['y_true']
        y_pred = result['predictions']

        if y_pred is None:
            st.error("No predictions available")
            return

        st.subheader(f"Confusion Matrix - {model_name} (Cross-Validated)")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        class_names = [cls.title() for cls in Config.CLASSES]

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax1)
        ax1.set_title(f'Confusion Matrix - {model_name}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax2)
        ax2.set_title(f'Normalized Confusion Matrix - {model_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        plt.tight_layout()
        st.pyplot(fig)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names,
                                       output_dict=True)

        report_data = []
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                report_data.append({
                    'Class': class_name,
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1-score']:.3f}",
                    'Support': int(metrics['support'])
                })

        # Add weighted average
        if 'weighted avg' in report:
            weighted = report['weighted avg']
            report_data.append({
                'Class': 'Weighted Avg',
                'Precision': f"{weighted['precision']:.3f}",
                'Recall': f"{weighted['recall']:.3f}",
                'F1-Score': f"{weighted['f1-score']:.3f}",
                'Support': int(weighted['support'])
            })

        st.subheader("Classification Report (Cross-Validated)")
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)