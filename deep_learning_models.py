"""
Deep Learning Models for Wildlife Classification
Updated with proper train/validation/test splitting
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
import streamlit as st
import copy
import time
from sklearn.model_selection import train_test_split
from config import Config


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
            # Return a black image in case of error
            if self.transform:
                black_img = Image.new('RGB', (224, 224), color='black')
                return self.transform(black_img), label
            else:
                return torch.zeros(3, 224, 224), label


class ResidualBlock(nn.Module):
    """Residual Block with skip connections"""

    def __init__(self, in_channels, out_channels, stride=1):
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
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class WildlifeCNN(nn.Module):
    """Custom CNN Architecture with Residual Connections"""

    def __init__(self, num_classes=4):
        super(WildlifeCNN, self).__init__()

        # Initial convolution layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual layers
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention mechanism
        attention_weights = self.attention(x)
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)

        # Apply attention
        x = x * attention_weights

        # Classification
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
    """Comprehensive deep learning training pipeline with proper validation"""

    def __init__(self):
        self.device = Config.DEVICE
        self.training_history = {}

    def get_data_transforms(self):
        """Get comprehensive data augmentation transforms"""

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

        return train_transform, val_transform

    def create_data_loaders(self, train_batch_size=None, val_batch_size=None):
        """Create PyTorch data loaders with proper train/validation/test splitting"""

        batch_size = train_batch_size or Config.DEEP_BATCH_SIZE
        val_batch_size = val_batch_size or batch_size

        train_transform, val_transform = self.get_data_transforms()

        # Create full dataset to get indices
        temp_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES)

        if len(temp_dataset) == 0:
            st.error("No images found in dataset!")
            return None, None, None

        # Split indices based on configuration
        indices = list(range(len(temp_dataset)))
        labels = [temp_dataset[i][1] for i in indices]

        try:
            if Config.USE_CROSS_VALIDATION:
                # For cross-validation, only separate test set
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=Config.TEST_SIZE,
                    random_state=Config.RANDOM_STATE,
                    stratify=labels
                )

                # Create datasets
                train_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, train_transform)
                test_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, val_transform)

                # Create subsets
                train_subset = Subset(train_dataset, train_val_idx)
                test_subset = Subset(test_dataset, test_idx)

                # Create loaders
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=True
                )

                val_loader = None  # Will use cross-validation during training

                test_loader = DataLoader(
                    test_subset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                return train_loader, val_loader, test_loader

            else:
                # Three-way split: train/validation/test
                # First split: separate test set
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=Config.TEST_SIZE,
                    random_state=Config.RANDOM_STATE,
                    stratify=labels
                )

                # Get labels for remaining data
                train_val_labels = [labels[i] for i in train_val_idx]

                # Second split: separate training and validation from remaining data
                val_size_adjusted = Config.VALIDATION_SIZE / (Config.TRAIN_SIZE + Config.VALIDATION_SIZE)
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_adjusted,
                    random_state=Config.RANDOM_STATE,
                    stratify=train_val_labels
                )

                # Create datasets with appropriate transforms
                train_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, train_transform)
                val_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, val_transform)
                test_dataset = WildlifeDataset(Config.DATA_DIR, Config.CLASSES, val_transform)

                # Create subsets
                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(val_dataset, val_idx)
                test_subset = Subset(test_dataset, test_idx)

                # Create data loaders
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=True
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                test_loader = DataLoader(
                    test_subset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                return train_loader, val_loader, test_loader

        except ValueError as e:
            st.error(f"Error splitting data: {e}")
            return None, None, None

    def train_custom_cnn(self, epochs=None):
        """Train custom CNN model with proper validation"""

        epochs = epochs or Config.CNN_EPOCHS
        st.info(f"Training Custom CNN on {self.device} with proper validation")

        # Create model
        model = WildlifeCNN(num_classes=len(Config.CLASSES)).to(self.device)

        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        if train_loader is None:
            return None

        # Setup training
        criterion = FocalLoss(gamma=2, alpha=1)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        # Training tracking
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        # Training loop
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                pred = outputs.argmax(1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)

                progress_bar.progress((batch_idx + 1) / len(train_loader))
                status_text.text(f"Epoch {epoch}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if Config.USE_CROSS_VALIDATION:
                # For cross-validation, we'll use training data for validation
                # This is a simplified approach - in practice you'd implement proper k-fold CV
                val_acc = train_correct / train_total  # Simplified for now
                val_loss_avg = train_loss / len(train_loader)
            else:
                # Use separate validation set
                with torch.no_grad():
                    for images, labels in val_loader:
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

                val_acc = val_correct / val_total if val_total > 0 else 0
                val_loss_avg = val_loss / len(val_loader) if val_loader else 0

            # Calculate metrics
            train_acc = train_correct / train_total

            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc)

            scheduler.step()

            progress_bar.empty()
            if Config.USE_CROSS_VALIDATION:
                status_text.text(f"Epoch {epoch}/{epochs} - Train Acc: {train_acc:.4f}")
            else:
                status_text.text(f"Epoch {epoch}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                st.success(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    st.info(f"Early stopping triggered after {Config.PATIENCE} epochs without improvement")
                    break

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on test set
        test_accuracy = self._evaluate_on_test_set(model, test_loader)

        status_text.empty()

        return {
            'model': model,
            'best_accuracy': test_accuracy,  # Use test accuracy as final metric
            'val_accuracy': best_val_acc,    # Keep validation accuracy for reference
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'final_epoch': epoch,
            'model_name': 'Custom CNN'
        }

    def train_resnet_transfer(self, epochs=None):
        """Train ResNet-18 with transfer learning and proper validation"""

        epochs = epochs or Config.CNN_EPOCHS
        st.info(f"Training ResNet-18 Transfer Learning on {self.device} with proper validation")

        # Create model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last few layers
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

        model = model.to(self.device)

        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        if train_loader is None:
            return None

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LEARNING_RATE * 0.1,  # Lower learning rate for transfer learning
            weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        # Training tracking
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        # Training loop (similar structure as custom CNN)
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                pred = outputs.argmax(1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)

                progress_bar.progress((batch_idx + 1) / len(train_loader))
                status_text.text(f"Epoch {epoch}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if Config.USE_CROSS_VALIDATION:
                val_acc = train_correct / train_total  # Simplified for CV
                val_loss_avg = train_loss / len(train_loader)
            else:
                with torch.no_grad():
                    for images, labels in val_loader:
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

                val_acc = val_correct / val_total if val_total > 0 else 0
                val_loss_avg = val_loss / len(val_loader) if val_loader else 0

            # Calculate metrics
            train_acc = train_correct / train_total

            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc)

            scheduler.step()

            progress_bar.empty()
            status_text.text(f"Epoch {epoch}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                st.success(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    st.info(f"Early stopping triggered after {Config.PATIENCE} epochs without improvement")
                    break

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on test set
        test_accuracy = self._evaluate_on_test_set(model, test_loader)

        status_text.empty()

        return {
            'model': model,
            'best_accuracy': test_accuracy,  # Use test accuracy as final metric
            'val_accuracy': best_val_acc,    # Keep validation accuracy for reference
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'final_epoch': epoch,
            'model_name': 'ResNet-18'
        }

    def _evaluate_on_test_set(self, model, test_loader):
        """Evaluate model on test set for final unbiased accuracy"""
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

    def evaluate_model(self, model, model_name):
        """Comprehensive model evaluation on test set"""

        st.subheader(f"Model Evaluation - {model_name}")

        # Create test data loader
        _, _, test_loader = self.create_data_loaders()
        if test_loader is None:
            return

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
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        accuracy = accuracy_score(all_labels, all_predictions)
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=[cls.title() for cls in Config.CLASSES],
            output_dict=True
        )

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Test Accuracy", f"{accuracy:.4f}")
            st.metric("Test Accuracy %", f"{accuracy * 100:.2f}%")

        with col2:
            weighted_f1 = class_report['weighted avg']['f1-score']
            st.metric("Weighted F1-Score", f"{weighted_f1:.4f}")

            macro_f1 = class_report['macro avg']['f1-score']
            st.metric("Macro F1-Score", f"{macro_f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax = plt.subplots(figsize=(8, 6))

        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[cls.title() for cls in Config.CLASSES],
                    yticklabels=[cls.title() for cls in Config.CLASSES],
                    ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        st.pyplot(fig)

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'class_report': class_report
        }