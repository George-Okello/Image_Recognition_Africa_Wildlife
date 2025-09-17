"""
Data management and exploratory data analysis functionality
Updated with complete EDA visualizations including sample images
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

        # 3. Sample Images Analysis
        self._display_sample_images()

        # 4. Image Properties Analysis
        image_stats = self._analyze_image_properties()

        # 5. Data Quality Assessment
        quality_report = self._assess_data_quality()

        # 6. Advanced Visualizations
        self._create_advanced_visualizations()

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
        st.subheader("📈 Dataset Structure Overview")

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
            imbalance_color = "🔴" if class_imbalance > 2 else "🟢"
            st.metric("Class Imbalance Ratio", f"{imbalance_color} {class_imbalance:.1f}")

    def _analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        st.subheader("📊 Class Distribution Analysis")

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
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Pie chart
            fig_pie = px.pie(
                df, values='Count', names='Class',
                title='Class Distribution (%)',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Distribution table
        st.dataframe(df, use_container_width=True)

        # Class balance analysis
        std_dev = df['Count'].std()
        mean_count = df['Count'].mean()
        cv = (std_dev / mean_count) * 100  # Coefficient of variation

        if cv < 10:
            balance_status = "Well Balanced"
            balance_color = "🟢"
        elif cv < 25:
            balance_status = "Moderately Imbalanced"
            balance_color = "🟡"
        else:
            balance_status = "Highly Imbalanced"
            balance_color = "🔴"

        st.info(f"**Class Balance Status:** {balance_color} {balance_status} (CV: {cv:.1f}%)")

    def _display_sample_images(self):
        """Display sample images from each class with enhanced layout"""
        st.subheader("🖼️ Sample Images by Class")

        for cls in Config.CLASSES:
            st.write(f"**{cls.title()} Samples**")
            cls_path = os.path.join(Config.DATA_DIR, cls)

            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                         if os.path.splitext(f)[1].lower() in Config.VALID_EXTS]

                if files:
                    # Display first 4 images in a grid
                    cols = st.columns(4)
                    for i, filename in enumerate(files[:4]):
                        try:
                            img_path = os.path.join(cls_path, filename)
                            img = Image.open(img_path)

                            # Resize for consistent display
                            img.thumbnail((200, 200), Image.Resampling.LANCZOS)

                            with cols[i]:
                                st.image(img, caption=f"Sample {i + 1}", use_column_width=True)
                                # Show image dimensions
                                original_img = Image.open(img_path)
                                st.caption(f"Size: {original_img.width}x{original_img.height}")
                        except Exception as e:
                            with cols[i]:
                                st.error(f"Error loading image: {str(e)}")
                else:
                    st.warning(f"No valid images found in {cls} directory")
            else:
                st.error(f"Directory not found: {cls_path}")

    def _analyze_image_properties(self):
        """Analyze image properties across the dataset"""
        st.subheader("📏 Image Properties Analysis")

        image_stats = {
            'dimensions': [],
            'file_sizes': [],
            'aspect_ratios': [],
            'color_profiles': [],
            'file_formats': []
        }

        # Sample images for analysis
        sample_count = 0
        max_samples = 100  # Analyze first 100 images for speed

        progress_bar = st.progress(0)
        status_text = st.empty()

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
                    image_stats['file_formats'].append(img.format or 'Unknown')

                    # Color profile
                    if img.mode == 'RGB':
                        image_stats['color_profiles'].append('RGB')
                    elif img.mode == 'L':
                        image_stats['color_profiles'].append('Grayscale')
                    elif img.mode == 'RGBA':
                        image_stats['color_profiles'].append('RGBA')
                    else:
                        image_stats['color_profiles'].append('Other')

                    sample_count += 1
                    status_text.text(f"Analyzing images: {sample_count}/{max_samples}")
                    progress_bar.progress(sample_count / max_samples)

                except Exception as e:
                    continue

            if sample_count >= max_samples:
                break

        progress_bar.empty()
        status_text.empty()

        # Visualize image statistics
        if image_stats['dimensions']:
            self._create_image_statistics_visualizations(image_stats)
        else:
            st.warning("No images could be analyzed for properties")

        return image_stats

    def _create_image_statistics_visualizations(self, image_stats):
        """Create comprehensive image statistics visualizations"""

        # Extract data
        widths, heights = zip(*image_stats['dimensions'])

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Image Dimensions', 'File Size Distribution', 'Aspect Ratio Distribution',
                          'Color Profile Distribution', 'File Format Distribution', 'Statistics Summary'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "pie"}, {"type": "table"}]]
        )

        # 1. Dimensions scatter plot
        fig.add_trace(
            go.Scatter(
                x=widths, y=heights,
                mode='markers',
                name='Dimensions',
                opacity=0.6,
                marker=dict(color='blue', size=4)
            ),
            row=1, col=1
        )

        # 2. File size histogram
        fig.add_trace(
            go.Histogram(
                x=image_stats['file_sizes'],
                name='File Size (KB)',
                nbinsx=20,
                marker_color='green'
            ),
            row=1, col=2
        )

        # 3. Aspect ratio histogram
        fig.add_trace(
            go.Histogram(
                x=image_stats['aspect_ratios'],
                name='Aspect Ratio',
                nbinsx=20,
                marker_color='orange'
            ),
            row=1, col=3
        )

        # 4. Color profile pie chart
        color_counts = Counter(image_stats['color_profiles'])
        fig.add_trace(
            go.Pie(
                labels=list(color_counts.keys()),
                values=list(color_counts.values()),
                name="Color Profiles"
            ),
            row=2, col=1
        )

        # 5. File format pie chart
        format_counts = Counter(image_stats['file_formats'])
        fig.add_trace(
            go.Pie(
                labels=list(format_counts.keys()),
                values=list(format_counts.values()),
                name="File Formats"
            ),
            row=2, col=2
        )

        # 6. Statistics summary table
        stats_data = {
            'Metric': ['Width (px)', 'Height (px)', 'File Size (KB)', 'Aspect Ratio'],
            'Mean': [
                f"{np.mean(widths):.1f}",
                f"{np.mean(heights):.1f}",
                f"{np.mean(image_stats['file_sizes']):.1f}",
                f"{np.mean(image_stats['aspect_ratios']):.2f}"
            ],
            'Std Dev': [
                f"{np.std(widths):.1f}",
                f"{np.std(heights):.1f}",
                f"{np.std(image_stats['file_sizes']):.1f}",
                f"{np.std(image_stats['aspect_ratios']):.2f}"
            ],
            'Min': [
                f"{min(widths)}",
                f"{min(heights)}",
                f"{min(image_stats['file_sizes']):.1f}",
                f"{min(image_stats['aspect_ratios']):.2f}"
            ],
            'Max': [
                f"{max(widths)}",
                f"{max(heights)}",
                f"{max(image_stats['file_sizes']):.1f}",
                f"{max(image_stats['aspect_ratios']):.2f}"
            ]
        }

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Mean', 'Std Dev', 'Min', 'Max']),
                cells=dict(values=[stats_data['Metric'], stats_data['Mean'],
                                 stats_data['Std Dev'], stats_data['Min'], stats_data['Max']])
            ),
            row=2, col=3
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Comprehensive Image Analysis Dashboard"
        )

        # Update subplot titles and axis labels
        fig.update_xaxes(title_text="Width (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Height (pixels)", row=1, col=1)
        fig.update_xaxes(title_text="File Size (KB)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Aspect Ratio", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)

        st.plotly_chart(fig, use_container_width=True)

        # Additional statistics in expandable section
        with st.expander("📊 Detailed Statistics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Images Analyzed", len(image_stats['dimensions']))
                st.metric("Average Width", f"{np.mean(widths):.0f} px")
                st.metric("Average Height", f"{np.mean(heights):.0f} px")

            with col2:
                st.metric("Average File Size", f"{np.mean(image_stats['file_sizes']):.1f} KB")
                st.metric("Most Common Format", max(format_counts, key=format_counts.get))
                st.metric("Most Common Color Mode", max(color_counts, key=color_counts.get))

            with col3:
                st.metric("Min Dimensions", f"{min(widths)}x{min(heights)}")
                st.metric("Max Dimensions", f"{max(widths)}x{max(heights)}")
                st.metric("Aspect Ratio Range", f"{min(image_stats['aspect_ratios']):.2f} - {max(image_stats['aspect_ratios']):.2f}")

    def _assess_data_quality(self):
        """Assess overall data quality"""
        st.subheader("🔍 Data Quality Assessment")

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
                    st.markdown(f"⚠️ {issue}")
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

    def _create_advanced_visualizations(self):
        """Create advanced visualizations for deeper insights"""
        st.subheader("📈 Advanced Dataset Insights")

        # Class balance visualization
        col1, col2 = st.columns(2)

        with col1:
            # Class balance radar chart
            class_counts = [info['count'] for info in self.dataset_info['class_distribution'].values()]
            class_names = [cls.title() for cls in Config.CLASSES]

            # Normalize counts for radar chart
            max_count = max(class_counts)
            normalized_counts = [count / max_count for count in class_counts]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_counts + [normalized_counts[0]],  # Close the shape
                theta=class_names + [class_names[0]],
                fill='toself',
                name='Class Balance'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Class Balance Radar Chart",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Dataset readiness score
            total_images = self.dataset_info['total_images']
            num_classes = len(Config.CLASSES)
            balance_score = 1 - (max(class_counts) - min(class_counts)) / max(class_counts)
            size_score = min(total_images / 1000, 1.0)  # Normalize to 1000 images

            readiness_score = (balance_score * 0.4 + size_score * 0.6) * 100

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = readiness_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Dataset Readiness Score"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Dataset Summary")
        summary_cols = st.columns(4)

        with summary_cols[0]:
            st.metric("Readiness Score", f"{readiness_score:.1f}%")
        with summary_cols[1]:
            st.metric("Balance Score", f"{balance_score:.2f}")
        with summary_cols[2]:
            st.metric("Size Score", f"{size_score:.2f}")
        with summary_cols[3]:
            quality_rating = "Excellent" if readiness_score >= 90 else "Good" if readiness_score >= 70 else "Fair" if readiness_score >= 50 else "Poor"
            st.metric("Overall Rating", quality_rating)

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

        if total_images < 2000:
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