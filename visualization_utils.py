"""
Professional visualization utilities for wildlife classification system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from config import Config

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ProfessionalVisualizer:
    """Create professional-grade visualizations for ML results"""

    def __init__(self):
        self.colors = Config.COLORS
        self.figure_config = Config.REPORT_CONFIG

    def create_model_comparison_dashboard(self, all_results):
        """Create comprehensive model comparison dashboard"""

        if not all_results:
            st.warning("No results available for visualization")
            return

        # Prepare data
        model_data = []
        for model_name, result in all_results.items():
            accuracy = result.get('accuracy', 0) if isinstance(result, dict) else result
            if accuracy > 0:
                model_data.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Type': self._get_model_type(model_name),
                    'Training_Time': result.get('training_time', 0) if isinstance(result, dict) else 0
                })

        if not model_data:
            st.warning("No valid model results for visualization")
            return

        df = pd.DataFrame(model_data)

        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'Accuracy by Model Type',
                            'Training Time vs Accuracy', 'Performance Rankings'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Model accuracy bar chart
        fig.add_trace(
            go.Bar(
                x=df['Model'],
                y=df['Accuracy'],
                name='Accuracy',
                marker_color=px.colors.qualitative.Set3,
                text=[f'{acc:.3f}' for acc in df['Accuracy']],
                textposition='outside'
            ),
            row=1, col=1
        )

        # 2. Box plot by model type
        for model_type in df['Type'].unique():
            type_data = df[df['Type'] == model_type]
            fig.add_trace(
                go.Box(
                    y=type_data['Accuracy'],
                    name=model_type,
                    boxpoints='all'
                ),
                row=1, col=2
            )

        # 3. Scatter plot: Training Time vs Accuracy
        fig.add_trace(
            go.Scatter(
                x=df['Training_Time'],
                y=df['Accuracy'],
                mode='markers+text',
                text=df['Model'],
                textposition='top center',
                marker=dict(size=10, opacity=0.7),
                name='Models'
            ),
            row=2, col=1
        )

        # 4. Ranking chart
        df_sorted = df.sort_values('Accuracy', ascending=True)
        fig.add_trace(
            go.Bar(
                y=df_sorted['Model'],
                x=df_sorted['Accuracy'],
                orientation='h',
                name='Ranking',
                marker_color=px.colors.sequential.Viridis
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Wildlife Classification Model Performance Dashboard",
            showlegend=False
        )

        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)

        fig.update_xaxes(title_text="Model Type", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)

        fig.update_xaxes(title_text="Training Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        fig.update_xaxes(title_text="Accuracy", row=2, col=2)
        fig.update_yaxes(title_text="Models", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def create_confusion_matrix_heatmap(self, y_true, y_pred, model_name, class_names=None):
        """Create professional confusion matrix heatmap"""

        from sklearn.metrics import confusion_matrix

        if class_names is None:
            class_names = [cls.title() for cls in Config.CLASSES]

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'Confusion Matrix - {model_name}')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title(f'Normalized Confusion Matrix - {model_name}')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        return fig

    def create_feature_importance_plot(self, importances, feature_names, top_n=20):
        """Create feature importance visualization"""

        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}'
                        for i in indices]

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(range(len(top_features)), top_importances, color='skyblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        return fig

    def create_training_progress_dashboard(self, training_results):
        """Create comprehensive training progress dashboard"""

        if not training_results:
            st.warning("No training results available")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves',
                            'Learning Progress', 'Final Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        epochs = range(1, len(training_results['train_losses']) + 1)

        # Loss curves
        fig.add_trace(
            go.Scatter(x=list(epochs), y=training_results['train_losses'],
                       mode='lines', name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=training_results['val_losses'],
                       mode='lines', name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )

        # Accuracy curves
        fig.add_trace(
            go.Scatter(x=list(epochs), y=[acc * 100 for acc in training_results['train_accs']],
                       mode='lines', name='Training Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=[acc * 100 for acc in training_results['val_accs']],
                       mode='lines', name='Validation Accuracy', line=dict(color='orange')),
            row=1, col=2
        )

        # Learning progress (loss vs accuracy)
        fig.add_trace(
            go.Scatter(x=training_results['train_losses'],
                       y=[acc * 100 for acc in training_results['train_accs']],
                       mode='markers', name='Training Progress',
                       marker=dict(color='blue', size=6)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=training_results['val_losses'],
                       y=[acc * 100 for acc in training_results['val_accs']],
                       mode='markers', name='Validation Progress',
                       marker=dict(color='red', size=6)),
            row=2, col=1
        )

        # Final performance metrics
        metrics = ['Best Test Acc', 'Final Epoch', 'Total Epochs']
        values = [training_results['best_accuracy'] * 100,
                  training_results['final_epoch'],
                  len(training_results['train_losses'])]

        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Final Metrics',
                   marker_color=['green', 'blue', 'orange']),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Training Progress Dashboard - {training_results.get('model_name', 'Model')}",
            showlegend=True
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)

        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

        fig.update_xaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)

        fig.update_xaxes(title_text="Metrics", row=2, col=2)
        fig.update_yaxes(title_text="Values", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def create_class_performance_radar(self, class_report, class_names=None):
        """Create radar chart for per-class performance"""

        if class_names is None:
            class_names = [cls.title() for cls in Config.CLASSES]

        # Extract metrics for each class
        metrics = ['precision', 'recall', 'f1-score']

        fig = go.Figure()

        for metric in metrics:
            values = [class_report[class_name][metric] for class_name in class_names
                      if class_name in class_report]
            values += [values[0]]  # Close the radar chart

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=class_names + [class_names[0]],
                fill='toself',
                name=metric.title()
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Per-Class Performance Radar Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_dataset_analysis_dashboard(self, dataset_info):
        """Create comprehensive dataset analysis dashboard"""

        class_distribution = dataset_info.get('class_distribution', {})

        if not class_distribution:
            st.warning("No dataset information available for visualization")
            return

        # Prepare class data
        class_data = []
        total_count = 0
        for cls, info in class_distribution.items():
            count = info.get('count', 0) if isinstance(info, dict) else info
            class_data.append({
                'Class': cls.title(),
                'Count': count
            })
            total_count += count

        # Add percentage calculation
        for item in class_data:
            item['Percentage'] = (item['Count'] / total_count) * 100 if total_count > 0 else 0

        df = pd.DataFrame(class_data)

        # Create dashboard with separate charts
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for counts
            fig_bar = px.bar(
                df,
                x='Class',
                y='Count',
                title='Class Distribution (Count)',
                color='Count',
                color_continuous_scale='viridis',
                text='Count'
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Pie chart for percentages
            fig_pie = px.pie(
                df,
                values='Count',
                names='Class',
                title='Class Distribution (%)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Balance analysis chart
        st.subheader("Data Balance Analysis")
        mean_count = df['Count'].mean()

        # Create balance analysis chart
        fig_balance = px.bar(
            df,
            x='Class',
            y='Count',
            title='Class Balance Analysis',
            color=['Below Average' if count < mean_count * 0.8 else 'Above Average'
                   for count in df['Count']],
            color_discrete_map={'Below Average': 'red', 'Above Average': 'green'}
        )

        # Add mean line as annotation
        fig_balance.add_hline(
            y=mean_count,
            line_dash="dash",
            annotation_text=f"Mean: {mean_count:.0f}",
            annotation_position="top right"
        )

        st.plotly_chart(fig_balance, use_container_width=True)

        # Summary statistics
        st.subheader("Dataset Summary")
        col3, col4, col5, col6 = st.columns(4)

        with col3:
            st.metric("Total Images", f"{total_count:,}")
        with col4:
            st.metric("Classes", len(df))
        with col5:
            st.metric("Max per Class", f"{df['Count'].max():,}")
        with col6:
            st.metric("Min per Class", f"{df['Count'].min():,}")

        # Balance ratio calculation
        if df['Count'].min() > 0:
            balance_ratio = df['Count'].max() / df['Count'].min()
            st.metric("Imbalance Ratio", f"{balance_ratio:.2f}")

    def _get_model_type(self, model_name):
        """Determine model type based on name"""
        if any(ml in model_name for ml in ['SVM', 'Random', 'Logistic', 'KNN', 'Naive']):
            return 'Traditional ML'
        elif any(ensemble in model_name for ensemble in ['Voting', 'Bagging', 'Boosting', 'AdaBoost', 'Gradient']):
            return 'Ensemble'
        elif any(deep in model_name for deep in ['CNN', 'ResNet', 'Deep']):
            return 'Deep Learning'
        else:
            return 'Other'

    def create_performance_report(self, all_results, save_path=None):
        """Generate comprehensive performance report"""

        if not all_results:
            st.warning("No results available for report generation")
            return

        # Create comprehensive report figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Prepare data
        model_names = []
        accuracies = []
        model_types = []
        training_times = []

        for name, result in all_results.items():
            accuracy = result.get('accuracy', 0) if isinstance(result, dict) else result
            if accuracy > 0:
                model_names.append(name)
                accuracies.append(accuracy)
                model_types.append(self._get_model_type(name))
                training_times.append(result.get('training_time', 0) if isinstance(result, dict) else 0)

        if not model_names:
            st.warning("No valid results for report")
            return

        # 1. Overall performance ranking
        ax1 = fig.add_subplot(gs[0, :2])
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_accs = [accuracies[i] for i in sorted_indices]

        bars = ax1.barh(range(len(sorted_names)), sorted_accs,
                        color=plt.cm.viridis(np.linspace(0, 1, len(sorted_names))))
        ax1.set_yticks(range(len(sorted_names)))
        ax1.set_yticklabels(sorted_names)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Performance Ranking', fontsize=14, fontweight='bold')

        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{acc:.3f}', va='center', fontsize=10)

        # 2. Performance by model type
        ax2 = fig.add_subplot(gs[0, 2:])
        type_performance = {}
        for name, acc, mtype in zip(model_names, accuracies, model_types):
            if mtype not in type_performance:
                type_performance[mtype] = []
            type_performance[mtype].append(acc)

        bp = ax2.boxplot([type_performance[t] for t in type_performance.keys()],
                         labels=list(type_performance.keys()), patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)

        ax2.set_ylabel('Accuracy')
        ax2.set_title('Performance by Model Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Training time vs accuracy
        ax3 = fig.add_subplot(gs[1, :2])
        scatter = ax3.scatter(training_times, accuracies,
                              c=[plt.cm.tab10(i % 10) for i in range(len(model_names))],
                              s=100, alpha=0.7)

        for i, name in enumerate(model_names):
            if len(name) > 15:  # Truncate long names
                display_name = name[:12] + "..."
            else:
                display_name = name
            ax3.annotate(display_name, (training_times[i], accuracies[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8)

        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Training Efficiency Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Performance statistics
        ax4 = fig.add_subplot(gs[1, 2:])
        best_model = model_names[np.argmax(accuracies)]
        if len(best_model) > 20:
            best_model = best_model[:17] + "..."

        stats_data = {
            'Best Performance': f"{max(accuracies):.4f}",
            'Average Performance': f"{np.mean(accuracies):.4f}",
            'Performance Std': f"{np.std(accuracies):.4f}",
            'Total Models': len(model_names),
            'Best Model': best_model,
            'Fastest Training': f"{min(training_times):.2f}s"
        }

        ax4.axis('off')
        table_data = [[k, v] for k, v in stats_data.items()]
        table = ax4.table(cellText=table_data,
                          colLabels=['Metric', 'Value'],
                          cellLoc='left',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Performance Statistics', fontsize=14, fontweight='bold')

        # Add overall title
        fig.suptitle('African Wildlife Classification - Comprehensive Performance Report',
                     fontsize=18, fontweight='bold', y=0.95)

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            st.success(f"Report saved to {save_path}")

        st.pyplot(fig)
        return fig