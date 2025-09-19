"""
Deep Learning Models for Wildlife Classification with Integrated Visualizations
Fixed version with proper train/val/test split and corrected syntax
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import copy
import time
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                           precision_recall_fscore_support, roc_curve, auc)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from collections import Counter
from config import Config


class DeepLearningVisualizer:
    """Integrated visualization class for deep learning models"""

    def __init__(self):
        self.colors = Config.COLORS if hasattr(Config, 'COLORS') else {
            'primary': '#2E8B57',
            'secondary': '#667eea',
            'success': '#28a745'
        }

    def create_training_dashboard(self, training_history, key_prefix=""):
        """Create comprehensive training progress dashboard"""

        if not training_history:
            st.warning("No training history available")
            return

        # Extract data
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        train_accs = training_history.get('train_accs', [])
        val_accs = training_history.get('val_accs', [])

        if not train_losses or not val_losses:
            st.warning("Incomplete training history data")
            return

        epochs = range(1, len(train_losses) + 1)

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Training & Validation Loss', 'Training & Validation Accuracy', 'Learning Rate Schedule',
                'Overfitting Analysis', 'Convergence Analysis', 'Training Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]
            ]
        )

        # 1. Loss curves
        fig.add_trace(
            go.Scatter(x=list(epochs), y=train_losses, mode='lines+markers',
                      name='Training Loss', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=val_losses, mode='lines+markers',
                      name='Validation Loss', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Add best epoch marker
        best_epoch = np.argmin(val_losses) + 1
        fig.add_vline(x=best_epoch, line_dash="dash", line_color="green",
                     annotation_text=f"Best: {best_epoch}", row=1, col=1)

        # 2. Accuracy curves
        train_acc_pct = [acc * 100 for acc in train_accs]
        val_acc_pct = [acc * 100 for acc in val_accs]

        fig.add_trace(
            go.Scatter(x=list(epochs), y=train_acc_pct, mode='lines+markers',
                      name='Training Accuracy', line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=val_acc_pct, mode='lines+markers',
                      name='Validation Accuracy', line=dict(color='orange', width=2)),
            row=1, col=2
        )

        # 3. Learning rate (if available)
        if 'learning_rates' in training_history:
            learning_rates = training_history['learning_rates']
            fig.add_trace(
                go.Scatter(x=list(epochs), y=learning_rates, mode='lines',
                          name='Learning Rate', line=dict(color='purple', width=2)),
                row=1, col=3
            )

        # 4. Overfitting analysis
        train_val_gap = [t - v for t, v in zip(train_acc_pct, val_acc_pct)]

        fig.add_trace(
            go.Scatter(x=list(epochs), y=train_val_gap, mode='lines+markers',
                      name='Train-Val Gap', line=dict(color='red', width=2)),
            row=2, col=1
        )

        fig.add_hline(y=10, line_dash="dash", line_color="orange",
                     annotation_text="Warning (10%)", row=2, col=1)

        # 5. Convergence analysis
        window_size = max(1, len(train_losses) // 10)
        if window_size > 1:
            train_loss_smooth = pd.Series(train_losses).rolling(window=window_size).mean()
            val_loss_smooth = pd.Series(val_losses).rolling(window=window_size).mean()

            fig.add_trace(
                go.Scatter(x=list(epochs), y=train_loss_smooth, mode='lines',
                          name='Smooth Train Loss', line=dict(color='lightblue', width=3)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=val_loss_smooth, mode='lines',
                          name='Smooth Val Loss', line=dict(color='lightcoral', width=3)),
                row=2, col=2
            )

        # 6. Training summary table
        metrics_summary = {
            'Metric': [
                'Best Val Accuracy', 'Final Train Accuracy', 'Best Epoch',
                'Total Epochs', 'Min Val Loss', 'Final Gap'
            ],
            'Value': [
                f"{max(val_accs):.4f}",
                f"{train_accs[-1]:.4f}",
                str(best_epoch),
                str(len(epochs)),
                f"{min(val_losses):.4f}",
                f"{train_val_gap[-1]:.2f}%"
            ]
        }

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=[metrics_summary['Metric'], metrics_summary['Value']],
                          fill_color='white', align='left')
            ),
            row=2, col=3
        )

        fig.update_layout(
            height=800,
            title_text="Deep Learning Training Dashboard",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}training_dashboard")

    def create_model_architecture_visualization(self, model, model_name, key_prefix=""):
        """Visualize model architecture and parameters"""

        st.subheader(f"Model Architecture Analysis - {model_name}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col3:
            st.metric("Model Size (MB)", f"{total_params * 4 / (1024**2):.2f}")

        # Layer analysis
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_info.append({
                        'Layer': name,
                        'Type': type(module).__name__,
                        'Parameters': params
                    })

        if layer_info:
            # Create parameter distribution chart
            layer_df = pd.DataFrame(layer_info)
            top_layers = layer_df.nlargest(10, 'Parameters')

            fig = px.bar(
                top_layers,
                x='Parameters',
                y='Layer',
                title='Top 10 Layers by Parameter Count',
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}model_architecture_params")

    def create_prediction_analysis_dashboard(self, model, test_loader, class_names, key_prefix=""):
        """Create comprehensive prediction analysis"""

        st.subheader("Prediction Analysis Dashboard")

        device = next(model.parameters()).device
        model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []
        prediction_confidences = []

        # Collect predictions
        progress_bar = st.progress(0)
        total_batches = len(test_loader)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                confidences = probabilities.max(1)[0].cpu().numpy()
                prediction_confidences.extend(confidences)

                progress_bar.progress((batch_idx + 1) / total_batches)

        progress_bar.empty()

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        prediction_confidences = np.array(prediction_confidences)

        # Create dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Confusion Matrix', 'Confidence Distribution', 'Per-Class Performance',
                'ROC Curves', 'Precision-Recall', 'Confidence vs Accuracy'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "histogram"}, {"type": "bar"}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "scatter"}]
            ]
        )

        # 1. Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig.add_trace(
            go.Heatmap(
                z=cm_normalized,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=1, col=1
        )

        # 2. Confidence Distribution
        fig.add_trace(
            go.Histogram(x=prediction_confidences, nbinsx=30, name='Confidence'),
            row=1, col=2
        )

        # 3. Per-Class Performance
        report = classification_report(all_labels, all_predictions,
                                     target_names=class_names, output_dict=True)

        precision_scores = [report[cls]['precision'] for cls in class_names]
        recall_scores = [report[cls]['recall'] for cls in class_names]
        f1_scores = [report[cls]['f1-score'] for cls in class_names]

        fig.add_trace(
            go.Bar(x=class_names, y=precision_scores, name='Precision', marker_color='blue'),
            row=1, col=3
        )
        fig.add_trace(
            go.Bar(x=class_names, y=recall_scores, name='Recall', marker_color='red'),
            row=1, col=3
        )
        fig.add_trace(
            go.Bar(x=class_names, y=f1_scores, name='F1-Score', marker_color='green'),
            row=1, col=3
        )

        # 4. ROC Curves
        y_bin = label_binarize(all_labels, classes=range(len(class_names)))

        for i, class_name in enumerate(class_names):
            if i < len(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probabilities[:, i])
                roc_auc = auc(fpr, tpr)

                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'{class_name} (AUC={roc_auc:.3f})'),
                    row=2, col=1
                )

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      showlegend=False),
            row=2, col=1
        )

        # 5. Precision-Recall Curves
        for i, class_name in enumerate(class_names):
            if i < len(class_names):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probabilities[:, i])
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, mode='lines', name=f'{class_name}'),
                    row=2, col=2
                )

        # 6. Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []

        for i in range(len(confidence_bins) - 1):
            mask = (prediction_confidences >= confidence_bins[i]) & (prediction_confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(all_predictions[mask] == all_labels[mask])
                bin_confidence = np.mean(prediction_confidences[mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)

        fig.add_trace(
            go.Scatter(x=bin_confidences, y=bin_accuracies, mode='markers+lines',
                      name='Calibration', marker=dict(size=10)),
            row=2, col=3
        )

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='red'),
                      name='Perfect Calibration'),
            row=2, col=3
        )

        fig.update_layout(
            height=800,
            title_text="Comprehensive Prediction Analysis",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}prediction_analysis_dashboard")


def visualize_data_augmentation_effects(original_images, augmented_images, class_names):
    """Visualize data augmentation effects"""

    st.subheader("Data Augmentation Examples")

    for i, (orig, aug) in enumerate(zip(original_images[:3], augmented_images[:3])):
        st.write(f"**Sample {i+1}:**")

        col1, col2 = st.columns(2)

        with col1:
            if isinstance(orig, torch.Tensor):
                orig = orig.numpy()
            if orig.shape[0] == 3:
                orig = np.transpose(orig, (1, 2, 0))

            st.image(orig, caption="Original", use_column_width=True, clamp=True)

        with col2:
            if isinstance(aug, torch.Tensor):
                aug = aug.numpy()
            if aug.shape[0] == 3:
                aug = np.transpose(aug, (1, 2, 0))

            st.image(aug, caption="Augmented", use_column_width=True, clamp=True)


def analyze_model_complexity(model):
    """Analyze model computational complexity"""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate FLOPs (simplified)
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = module.out_channels
            flops += kernel_flops * output_elements
        elif isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features

    model_size_mb = total_params * 4 / (1024**2)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'estimated_flops': flops
    }


def create_model_ensemble(models_dict, device):
    """Create ensemble from multiple trained models"""

    class ModelEnsemble(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)

        def forward(self, x):
            outputs = []
            for model in self.models:
                with torch.no_grad():
                    output = model(x)
                    outputs.append(F.softmax(output, dim=1))

            ensemble_output = torch.stack(outputs).mean(0)
            return ensemble_output

    models = [results['model'] for results in models_dict.values()
              if results['model'] is not None]

    if len(models) < 2:
        st.warning("Need at least 2 models for ensemble")
        return None

    ensemble = ModelEnsemble(models).to(device)
    return ensemble


class WildlifeDataset(Dataset):
    """Custom PyTorch Dataset for Wildlife Images"""

    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if os.path.splitext(filename)[1].lower() in Config.VALID_EXTS:
                        self.samples.append((os.path.join(class_dir, filename), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            if self.transform:
                black_img = Image.new('RGB', (224, 224), color='black')
                return self.transform(black_img), label
            else:
                return torch.zeros(3, 224, 224), label


class ResidualBlock(nn.Module):
    """Enhanced Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_block(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WildlifeCNN(nn.Module):
    """Enhanced CNN Architecture"""

    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(WildlifeCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(64, 64, stride=1, dropout_rate=0.1)
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout_rate=0.15)
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout_rate=0.2)
        self.layer4 = ResidualBlock(256, 512, stride=2, dropout_rate=0.25)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        attention_weights = self.attention(x)
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)
        x = x * attention_weights

        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DeepLearningTrainer:
    """Deep learning training pipeline with integrated visualizations"""

    def __init__(self):
        self.device = Config.DEVICE
        self.training_history = {}
        self.visualizer = DeepLearningVisualizer()

    def get_data_transforms(self, show_augmentation_examples=False):
        """Get data transforms with optional visualization"""

        train_transform = transforms.Compose([
            transforms.Resize((Config.DEEP_IMG_SIZE, Config.DEEP_IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
        ])

        val_transform = transforms.Compose([
            transforms.Resize((Config.DEEP_IMG_SIZE, Config.DEEP_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Show augmentation examples
        if show_augmentation_examples:
            self._show_augmentation_examples(train_transform, val_transform)

        return train_transform, val_transform

    def _show_augmentation_examples(self, train_transform, val_transform):
        """Show data augmentation examples"""

        st.subheader("Data Augmentation Examples")

        sample_images = []
        for cls in Config.CLASSES[:2]:
            cls_path = os.path.join(Config.DATA_DIR, cls)
            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                        if os.path.splitext(f)[1].lower() in Config.VALID_EXTS]
                if files:
                    img_path = os.path.join(cls_path, files[0])
                    img = Image.open(img_path).convert('RGB')
                    sample_images.append(img)

        if sample_images and len(sample_images) >= 2:
            original_images = [val_transform(img) for img in sample_images]
            augmented_images = [train_transform(img) for img in sample_images]

            visualize_data_augmentation_effects(original_images, augmented_images, Config.CLASSES)

    def create_data_loaders(self, train_batch_size=None, val_batch_size=None):
        """Create data loaders with train/val/test split"""

        batch_size = train_batch_size or Config.DEEP_BATCH_SIZE
        val_batch_size = val_batch_size or batch_size

        train_transform, val_transform = self.get_data_transforms()

        temp_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES)

        if len(temp_dataset) == 0:
            st.error("No images found in dataset!")
            return None, None, None

        indices = list(range(len(temp_dataset)))
        labels = [temp_dataset[i][1] for i in indices]

        try:
            # First split: train+val (85%) vs test (15%)
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=0.15,
                random_state=Config.RANDOM_STATE,
                stratify=labels
            )

            # Second split: train (70% of total) vs val (15% of total)
            train_val_labels = [labels[i] for i in train_val_idx]
            val_size_adjusted = 0.15 / (0.70 + 0.15)  # 15% out of 85%

            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                random_state=Config.RANDOM_STATE,
                stratify=train_val_labels
            )

            # Create datasets
            train_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, train_transform)
            val_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, val_transform)
            test_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, val_transform)

            # Create subsets
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)
            test_subset = Subset(test_dataset, test_idx)

            import platform
            num_workers = 0 if platform.system() == 'Windows' else 2

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True,
                persistent_workers=False
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=False
            )

            test_loader = DataLoader(
                test_subset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=False
            )

            st.info(f"Data split: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
            return train_loader, val_loader, test_loader

        except ValueError as e:
            st.error(f"Error splitting data: {e}")
            return None, None, None

    def train_with_validation(self, model_class, model_name, epochs=None, show_visualizations=True):
        """Train model with comprehensive visualizations"""

        epochs = epochs or Config.CNN_EPOCHS
        st.info(f"Training {model_name} with enhanced visualizations")

        # Create unique key prefix based on model name
        key_prefix = model_name.lower().replace(' ', '_').replace('-', '_') + "_"

        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        if train_loader is None:
            return None

        # Create model
        model = model_class().to(self.device)

        # Show model architecture
        if show_visualizations:
            self.visualizer.create_model_architecture_visualization(model, model_name, key_prefix)

        # Train the model
        training_history, best_val_acc = self._train_single_fold(
            model, train_loader, val_loader, epochs, "Training"
        )

        # Create training visualizations
        if show_visualizations:
            st.header("Training Analysis")
            self.visualizer.create_training_dashboard(training_history, key_prefix)

        # Final evaluation
        test_accuracy = self._evaluate_on_test_set(model, test_loader)

        # Prediction analysis
        if show_visualizations and test_loader:
            st.header("Model Performance Analysis")
            class_names = [cls.title() for cls in Config.CLASSES]
            self.visualizer.create_prediction_analysis_dashboard(model, test_loader, class_names, key_prefix)

        st.success(f"Training completed - Val: {best_val_acc:.4f}, Test: {test_accuracy:.4f}")

        return {
            'model': model,
            'best_accuracy': test_accuracy,
            'val_accuracy': best_val_acc,
            'train_losses': training_history['train_losses'],
            'train_accs': training_history['train_accs'],
            'val_losses': training_history['val_losses'],
            'val_accs': training_history['val_accs'],
            'final_epoch': training_history['final_epoch'],
            'model_name': model_name,
            'history': training_history
        }

    def _train_single_fold(self, model, train_loader, val_loader, epochs, fold_name):
        """Train with comprehensive monitoring"""

        criterion = FocalLoss(gamma=2, alpha=1)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )

        max_grad_norm = 1.0
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        learning_rates = []

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        st.info(f"Training {fold_name} with enhanced monitoring")

        epoch_progress_bar = st.progress(0)
        batch_progress_bar = st.progress(0)
        status_text = st.empty()

        with st.expander("Training Progress History", expanded=True):
            epoch_history_container = st.empty()

        epoch_history = []

        for epoch in range(1, epochs + 1):
            epoch_progress_bar.progress(epoch / epochs)

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            status_text.text(f"Epoch {epoch}/{epochs} - Training...")

            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_progress = (batch_idx + 1) / len(train_loader)
                batch_progress_bar.progress(batch_progress)

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                train_loss += loss.item()
                pred = outputs.argmax(1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)

            train_acc = train_correct / train_total
            train_loss_avg = train_loss / len(train_loader)
            train_losses.append(train_loss_avg)
            train_accs.append(train_acc)

            # Validation phase
            status_text.text(f"Epoch {epoch}/{epochs} - Validating...")
            batch_progress_bar.progress(0)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_batch_idx, (images, labels) in enumerate(val_loader):
                    val_batch_progress = (val_batch_idx + 1) / len(val_loader)
                    batch_progress_bar.progress(val_batch_progress)

                    images, labels = images.to(self.device), labels.to(self.device)

                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    pred = outputs.argmax(1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss_avg = val_loss / len(val_loader)
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc)

            # Store learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Early stopping
            improved = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

            # Update epoch history
            train_val_gap = train_acc - val_acc
            epoch_info = {
                'epoch': epoch,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'lr': current_lr,
                'gap': train_val_gap,
                'patience': patience_counter,
                'best_val': best_val_acc,
                'improved': '✅' if improved else '⏸️'
            }
            epoch_history.append(epoch_info)

            # Display epoch history
            history_text = "**Recent Training History:**\n\n"
            for ep_info in epoch_history[-5:]:  # Show last 5 epochs
                history_text += f"**Epoch {ep_info['epoch']}/{epochs}** {ep_info['improved']}\n"
                history_text += f"• Train Acc: {ep_info['train_acc']:.4f} | Val Acc: {ep_info['val_acc']:.4f}\n"
                history_text += f"• Train Loss: {ep_info['train_loss']:.4f} | Val Loss: {ep_info['val_loss']:.4f}\n"
                history_text += f"• LR: {ep_info['lr']:.2e} | Gap: {ep_info['gap']:.4f}\n\n"

            epoch_history_container.markdown(history_text)

            # Early stopping check
            if patience_counter >= Config.PATIENCE:
                st.info(f"Early stopping: No improvement for {Config.PATIENCE} epochs")
                break

            if current_lr < 1e-7:
                st.info(f"Early stopping: Learning rate too low ({current_lr:.2e})")
                break

        # Clean up progress bars
        epoch_progress_bar.empty()
        batch_progress_bar.empty()
        status_text.empty()

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            st.success(f"Best model restored: Val Acc {best_val_acc:.4f}")

        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'learning_rates': learning_rates,
            'final_epoch': epoch,
            'epoch_history': epoch_history
        }, best_val_acc

    def _evaluate_on_test_set(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                pred = outputs.argmax(1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total if test_total > 0 else 0
        st.info(f"Final test set accuracy: {test_accuracy:.4f}")
        return test_accuracy

    def train_custom_cnn(self, epochs=None, show_visualizations=True):
        """Train custom CNN with visualizations"""
        return self.train_with_validation(
            lambda: WildlifeCNN(num_classes=len(Config.CLASSES)),
            'Custom CNN',
            epochs,
            show_visualizations
        )

    def train_resnet_transfer(self, epochs=None, show_visualizations=True):
        """Train ResNet-18 with transfer learning and visualizations"""

        def create_resnet():
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

            # Freeze early layers
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze last layers
            for param in model.layer4.parameters():
                param.requires_grad = True

            # Replace classifier
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, len(Config.CLASSES))
            )

            return model

        return self.train_with_validation(
            create_resnet,
            'ResNet-18',
            epochs,
            show_visualizations
        )

    def evaluate_model(self, result, model_name):
        """Comprehensive model evaluation"""

        st.subheader(f"Model Evaluation - {model_name}")

        # Create test data loader
        _, _, test_loader = self.create_data_loaders()
        if test_loader is None:
            return

        model = result['model']
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                predictions = outputs.argmax(1).cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report

        accuracy = accuracy_score(all_labels, all_predictions)
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=[cls.title() for cls in Config.CLASSES],
            output_dict=True
        )

        # Display results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Test Accuracy", f"{result['best_accuracy']:.4f}")
        with col2:
            st.metric("Test Accuracy %", f"{result['best_accuracy'] * 100:.2f}%")
        with col3:
            weighted_f1 = class_report['weighted avg']['f1-score']
            st.metric("Weighted F1-Score", f"{weighted_f1:.4f}")
        with col4:
            macro_f1 = class_report['macro avg']['f1-score']
            st.metric("Macro F1-Score", f"{macro_f1:.4f}")

        # Enhanced confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
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

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'class_report': class_report
        }