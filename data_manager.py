"""
Data management and exploratory data analysis functionality
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
from collections import Counter
import streamlit as st
from config import Config


class DataManager:
    """Comprehensive data management and analysis"""

    def __init__(self):
        self.dataset_info = {}
        self.eda_results = {}

    def check_dataset_existence(self):
        """Check if dataset exists and return basic information"""
        if not os.path.exists(Config.DATA_DIR):
            return False, f"Dataset directory '{Config.DATA_DIR}' not found", {}

        class_info = {}
        total_images = 0

        for cls in Config.CLASSES:
            cls_path = os.path.join(Config.DATA_DIR, cls)
            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                         if os.path.splitext(f)[1].lower() in Config.VALID_EXTS]
                class_info[cls] = {
                    'count': len(files),
                    'files': files[:10]  # Store first 10 for samples
                }
                total_images += len(files)
            else:
                class_info[cls] = {'count': 0, 'files': []}

        self.dataset_info = {
            'total_images': total_images,
            'class_distribution': class_info,
            'classes': Config.CLASSES,
            'valid_extensions': Config.VALID_EXTS
        }

        return True, f"Dataset found with {total_images} images", class_info

    def perform_comprehensive_eda(self):
        """Perform comprehensive exploratory data analysis"""
        if not self.dataset_info:
            exists, message, _ = self.check_dataset_existence()
            if not exists:
                return None

        st.subheader("📊 Comprehensive Exploratory Data Analysis")

        # 1. Dataset Overview
        self._analyze_dataset_structure()

        # 2. Class Distribution Analysis
        self._analyze_class_distribution()

        # 3. Image Properties Analysis
        image_stats = self._analyze_image_properties()

        # 4. Sample Images Analysis
        self._display_sample_images()

        # 5. Data Quality Assessment
        quality_report = self._assess_data_quality()

        # Compile EDA results
        self.eda_results = {
            'dataset_info': self.dataset_info,
            'image_stats': image_stats,
            'quality_report': quality_report,
            'recommendations': self._generate_recommendations()
        }

        # Mark EDA as completed
        st.session_state.eda_completed = True
        st.session_state.eda_results = self.eda_results

        return self.eda_results

    def _analyze_dataset_structure(self):
        """Analyze basic dataset structure"""
        col1, col2, col3, col4 = st.columns(4)

        total_images = self.dataset_info['total_images']
        num_classes = len(Config.CLASSES)
        avg_per_class = total_images / num_classes if num_classes > 0 else 0

        class_counts = [info['count'] for info in self.dataset_info['class_distribution'].values()]
        class_imbalance = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')

        with col1:
            st.metric("Total Images", f"{total_images:,}")
        with col2:
            st.metric("Classes", num_classes)
        with col3:
            st.metric("Avg per Class", f"{avg_per_class:.0f}")
        with col4:
            imbalance_color = "red" if class_imbalance > 2 else "green"
            st.metric("Class Imbalance Ratio", f"{class_imbalance:.1f}")

    def _analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        st.subheader("Class Distribution Analysis")

        class_data = []
        for cls, info in self.dataset_info['class_distribution'].items():
            class_data.append({
                'Class': cls.title(),
                'Count': info['count'],
                'Percentage': (info['count'] / self.dataset_info['total_images']) * 100
            })

        df = pd.DataFrame(class_data)

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart
            fig_bar = px.bar(
                df, x='Class', y='Count',
                title='Images per Class',
                color='Count',
                color_continuous_scale='viridis',
                text='Count'
            )
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Pie chart
            fig_pie = px.pie(
                df, values='Count', names='Class',
                title='Class Distribution (%)',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Distribution table
        st.dataframe(df, use_container_width=True)

        # Class balance analysis
        std_dev = df['Count'].std()
        mean_count = df['Count'].mean()
        cv = (std_dev / mean_count) * 100  # Coefficient of variation

        if cv < 10:
            balance_status = "Well Balanced"
            balance_color = "green"
        elif cv < 25:
            balance_status = "Moderately Imbalanced"
            balance_color = "orange"
        else:
            balance_status = "Highly Imbalanced"
            balance_color = "red"

        st.markdown(f"""
        <div style="background-color: {balance_color}; padding: 10px; border-radius: 5px; color: white; margin: 10px 0;">
        <strong>Class Balance Status: {balance_status}</strong><br>
        Coefficient of Variation: {cv:.1f}%
        </div>
        """, unsafe_allow_html=True)

    def _analyze_image_properties(self):
        """Analyze image properties across the dataset"""
        st.subheader("Image Properties Analysis")

        image_stats = {
            'dimensions': [],
            'file_sizes': [],
            'aspect_ratios': [],
            'color_profiles': []
        }

        # Sample images for analysis
        sample_count = 0
        max_samples = 100  # Analyze first 100 images for speed

        progress_bar = st.progress(0)

        for cls, info in self.dataset_info['class_distribution'].items():
            cls_path = os.path.join(Config.DATA_DIR, cls)
            files_to_analyze = info['files'][:min(20, len(info['files']))]  # 20 per class max

            for i, filename in enumerate(files_to_analyze):
                if sample_count >= max_samples:
                    break

                try:
                    img_path = os.path.join(cls_path, filename)

                    # File size
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    image_stats['file_sizes'].append(file_size)

                    # Image properties
                    img = Image.open(img_path)
                    width, height = img.size

                    image_stats['dimensions'].append((width, height))
                    image_stats['aspect_ratios'].append(width / height)

                    # Color profile
                    if img.mode == 'RGB':
                        image_stats['color_profiles'].append('RGB')
                    elif img.mode == 'L':
                        image_stats['color_profiles'].append('Grayscale')
                    else:
                        image_stats['color_profiles'].append('Other')

                    sample_count += 1
                    progress_bar.progress(sample_count / max_samples)

                except Exception as e:
                    continue

            if sample_count >= max_samples:
                break

        progress_bar.empty()

        # Visualize image statistics
        if image_stats['dimensions']:
            col1, col2 = st.columns(2)

            with col1:
                # Dimensions scatter plot
                widths, heights = zip(*image_stats['dimensions'])
                fig_dims = px.scatter(
                    x=widths, y=heights,
                    title='Image Dimensions Distribution',
                    labels={'x': 'Width (pixels)', 'y': 'Height (pixels)'},
                    opacity=0.6
                )
                st.plotly_chart(fig_dims, use_container_width=True)

            with col2:
                # File size histogram
                fig_size = px.histogram(
                    x=image_stats['file_sizes'],
                    title='File Size Distribution',
                    labels={'x': 'File Size (KB)', 'y': 'Count'},
                    nbins=20
                )
                st.plotly_chart(fig_size, use_container_width=True)

            # Statistics summary
            stats_df = pd.DataFrame({
                'Metric': ['Width (px)', 'Height (px)', 'File Size (KB)', 'Aspect Ratio'],
                'Mean': [
                    np.mean(widths),
                    np.mean(heights),
                    np.mean(image_stats['file_sizes']),
                    np.mean(image_stats['aspect_ratios'])
                ],
                'Std Dev': [
                    np.std(widths),
                    np.std(heights),
                    np.std(image_stats['file_sizes']),
                    np.std(image_stats['aspect_ratios'])
                ],
                'Min': [
                    min(widths),
                    min(heights),
                    min(image_stats['file_sizes']),
                    min(image_stats['aspect_ratios'])
                ],
                'Max': [
                    max(widths),
                    max(heights),
                    max(image_stats['file_sizes']),
                    max(image_stats['aspect_ratios'])
                ]
            })

            st.subheader("Image Statistics Summary")
            st.dataframe(stats_df.round(2), use_container_width=True)

        return image_stats

    def _display_sample_images(self):
        """Display sample images from each class"""
        st.subheader("Sample Images by Class")

        for cls in Config.CLASSES:
            st.write(f"**{cls.title()}**")
            cls_path = os.path.join(Config.DATA_DIR, cls)

            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                         if os.path.splitext(f)[1].lower() in Config.VALID_EXTS]

                if files:
                    # Display first 4 images
                    cols = st.columns(4)
                    for i, filename in enumerate(files[:4]):
                        try:
                            img_path = os.path.join(cls_path, filename)
                            img = Image.open(img_path)

                            with cols[i]:
                                st.image(img, caption=f"Sample {i + 1}", use_container_width=True)
                        except Exception as e:
                            with cols[i]:
                                st.error(f"Error loading image: {str(e)}")

    def _assess_data_quality(self):
        """Assess overall data quality"""
        st.subheader("Data Quality Assessment")

        quality_issues = []
        recommendations = []

        # Check class balance
        class_counts = [info['count'] for info in self.dataset_info['class_distribution'].values()]
        min_count, max_count = min(class_counts), max(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 3:
            quality_issues.append("Significant class imbalance detected")
            recommendations.append("Consider data augmentation for underrepresented classes")

        # Check minimum samples per class
        if min_count < 50:
            quality_issues.append("Some classes have very few samples")
            recommendations.append("Collect more data or consider data augmentation")

        # Check total dataset size
        if self.dataset_info['total_images'] < 500:
            quality_issues.append("Small dataset size")
            recommendations.append("Consider data augmentation or transfer learning")

        # Display quality assessment
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Quality Issues")
            if quality_issues:
                for issue in quality_issues:
                    st.markdown(f"❌ {issue}")
            else:
                st.markdown("✅ No major quality issues detected")

        with col2:
            st.markdown("#### Recommendations")
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"💡 {rec}")
            else:
                st.markdown("✅ Dataset appears ready for training")

        return {
            'issues': quality_issues,
            'recommendations': recommendations,
            'quality_score': max(0, 100 - len(quality_issues) * 20)
        }

    def _generate_recommendations(self):
        """Generate training recommendations based on EDA"""
        recommendations = []

        class_counts = [info['count'] for info in self.dataset_info['class_distribution'].values()]
        total_images = sum(class_counts)

        # Data augmentation recommendations
        if min(class_counts) < max(class_counts) * 0.7:
            recommendations.append("Apply data augmentation to balance classes")

        # Training approach recommendations
        if total_images < 1000:
            recommendations.append("Use transfer learning due to limited data")

        if total_images > 5000:
            recommendations.append("Consider training from scratch with custom architectures")

        # Preprocessing recommendations
        recommendations.append("Apply standard image preprocessing (resize, normalize)")
        recommendations.append("Consider advanced augmentation techniques")

        return recommendations

    def get_dataset_summary(self):
        """Get a comprehensive dataset summary"""
        if not self.dataset_info:
            return None

        return {
            'total_images': self.dataset_info['total_images'],
            'classes': len(Config.CLASSES),
            'class_distribution': self.dataset_info['class_distribution'],
            'balance_ratio': max([info['count'] for info in self.dataset_info['class_distribution'].values()]) /
                             min([info['count'] for info in self.dataset_info['class_distribution'].values()]),
            'eda_completed': st.session_state.get('eda_completed', False)
        }