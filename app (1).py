import streamlit as st
import cv2
import numpy as np
from skimage import exposure
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import random

# Set page configuration
st.set_page_config(
    page_title="Fetal Heart Image Enhancement",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def calibrate_thresholds(image_folder, sample_size=50):
    """Auto-calibrate thresholds based on dataset samples"""
    if not os.path.exists(image_folder):
        st.error(f"Dataset folder not found: {image_folder}")
        return None
    
    files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        st.error(f"No image files found in {image_folder}")
        return None
    
    sample_files = random.sample(files, min(sample_size, len(files)))
    metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f in enumerate(sample_files):
        status_text.text(f"Calibrating... {i+1}/{len(sample_files)}")
        progress_bar.progress((i + 1) / len(sample_files))
        
        img = cv2.imread(f)
        if img is None:
            continue
        b, c, e = compute_quality_metrics(img)
        metrics.append((b, c, e))
    
    progress_bar.empty()
    status_text.empty()
    
    if not metrics:
        st.error("No valid images found for calibration")
        return None
    
    metrics = np.array(metrics)
    brightness, contrast, entropy = metrics[:,0], metrics[:,1], metrics[:,2]
    
    thresholds = {
        "brightness": (np.percentile(brightness, 20), np.percentile(brightness, 50), np.percentile(brightness, 80)),
        "contrast":   (np.percentile(contrast, 20), np.percentile(contrast, 50), np.percentile(contrast, 80)),
        "entropy":    (np.percentile(entropy, 20), np.percentile(entropy, 50), np.percentile(entropy, 80)),
    }
    
    return thresholds

def compute_quality_metrics(image):
    """Compute quality metrics for BGR image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    entropy = shannon_entropy(gray)
    return brightness, contrast, entropy

def classify_quality(brightness, contrast, entropy, thresholds):
    """Classify image quality based on calibrated thresholds"""
    b20, b50, b80 = thresholds["brightness"]
    c20, c50, c80 = thresholds["contrast"]
    e20, e50, e80 = thresholds["entropy"]
    
    if brightness < b20 or contrast < c20 or entropy < e20:
        return "poor"
    elif b20 <= brightness <= b50 and c20 <= contrast <= c50 and e20 <= entropy <= e50:
        return "moderate"
    elif b50 < brightness <= b80 and c50 < contrast <= c80 and e50 < entropy <= e80:
        return "good"
    else:
        return "excellent"

def apply_clahe(image):
    """Apply CLAHE enhancement to BGR image"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_histogram_stretch_clahe(image):
    """Apply histogram stretching + CLAHE to BGR image"""
    p2, p98 = np.percentile(image, (2, 98))
    stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
    stretched = np.uint8(stretched * 255) if stretched.max() <= 1 else np.uint8(stretched)
    
    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adaptive_enhancement(image, thresholds):
    """Apply adaptive enhancement based on calibrated thresholds"""
    brightness, contrast, entropy = compute_quality_metrics(image)
    quality = classify_quality(brightness, contrast, entropy, thresholds)
    
    enhancement_info = {
        'brightness': brightness,
        'contrast': contrast,
        'entropy': entropy,
        'quality': quality
    }
    
    if quality in ["good", "excellent"]:
        enhancement_type = "No Enhancement"
        enhanced_image = image
        enhancement_info['action'] = "No enhancement needed - image quality is sufficient"
    elif quality == "poor":
        enhancement_type = "CLAHE"
        enhanced_image = apply_histogram_stretch_clahe(image)
        enhancement_info['action'] = "Applied  CLAHE for enhancement"
    else:  # moderate
        enhancement_type = "CLAHE Only"
        enhanced_image = apply_clahe(image)
        enhancement_info['action'] = "Applied CLAHE enhancement"
    
    return enhanced_image, enhancement_type, enhancement_info

def display_thresholds(thresholds):
    """Display calibrated thresholds in a nice format"""
    st.subheader("📊 Auto-Calibrated Quality Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Brightness Ranges:**")
        st.write(f"- Poor: < {thresholds['brightness'][0]:.1f}")
        st.write(f"- Moderate: {thresholds['brightness'][0]:.1f} - {thresholds['brightness'][1]:.1f}")
        st.write(f"- Good: {thresholds['brightness'][1]:.1f} - {thresholds['brightness'][2]:.1f}")
        st.write(f"- Excellent: > {thresholds['brightness'][2]:.1f}")
    
    with col2:
        st.write("**Contrast Ranges:**")
        st.write(f"- Poor: < {thresholds['contrast'][0]:.1f}")
        st.write(f"- Moderate: {thresholds['contrast'][0]:.1f} - {thresholds['contrast'][1]:.1f}")
        st.write(f"- Good: {thresholds['contrast'][1]:.1f} - {thresholds['contrast'][2]:.1f}")
        st.write(f"- Excellent: > {thresholds['contrast'][2]:.1f}")
    
    with col3:
        st.write("**Entropy Ranges:**")
        st.write(f"- Poor: < {thresholds['entropy'][0]:.2f}")
        st.write(f"- Moderate: {thresholds['entropy'][0]:.2f} - {thresholds['entropy'][1]:.2f}")
        st.write(f"- Good: {thresholds['entropy'][1]:.2f} - {thresholds['entropy'][2]:.2f}")
        st.write(f"- Excellent: > {thresholds['entropy'][2]:.2f}")

def create_threshold_visualization(thresholds):
    """Create visualization of calibrated thresholds"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['brightness', 'contrast', 'entropy']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        ranges = thresholds[metric]
        
        # Create bar chart showing ranges
        categories = ['Poor\n(<20%)', 'Moderate\n(20-50%)', 'Good\n(50-80%)', 'Excellent\n(>80%)']
        values = [ranges[0], ranges[1], ranges[2], ranges[2]*1.2]
        
        # Create bars with individual alpha values
        alphas = [0.3, 0.5, 0.7, 0.9]
        bars = []
        for j, (cat, val, alpha_val) in enumerate(zip(categories, values, alphas)):
            bar = ax.bar(cat, val, color=color, alpha=alpha_val)
            bars.append(bar)
        
        ax.set_title(f'{metric.capitalize()} Thresholds', fontweight='bold')
        ax.set_ylabel('Value')
        
        # Add threshold lines
        for j, thresh in enumerate(ranges):
            ax.axhline(y=thresh, color='red', linestyle='--', alpha=0.7)
            ax.text(0.1, thresh + thresh*0.02, f'{thresh:.1f}', color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("💓 Adaptive Fetal Heart Ultrasound Enhancement")
    st.markdown("*Automatically calibrated enhancement based on your dataset*")
    st.markdown("---")
    
    # Sidebar for dataset configuration
    st.sidebar.header("Dataset Configuration")
    
    dataset_folder = st.sidebar.text_input(
        "Dataset Folder Path", 
        value="Focus_data/training/images",
        help="Path to your training images folder for auto-calibration"
    )
    
    sample_size = st.sidebar.slider(
        "Calibration Sample Size", 
        min_value=10, 
        max_value=200, 
        value=50,
        help="Number of images to use for threshold calibration"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Enhancement Strategy:**")
    st.sidebar.markdown("- **Poor Quality**: CLAHE")
    st.sidebar.markdown("- **Moderate Quality**: CLAHE only")
    st.sidebar.markdown("- **Good/Excellent**: No enhancement")
    
    # Calibration section
    st.header("🔧 Dataset Calibration")
    
    if st.button("🎯 Auto-Calibrate Thresholds"):
        st.info("Calibrating thresholds based on your dataset...")
        thresholds = calibrate_thresholds(dataset_folder, sample_size)
        
        if thresholds is not None:
            st.session_state.thresholds = thresholds
            st.success(f"✅ Calibration complete! Analyzed {sample_size} images from your dataset.")
            
            # Display calibrated thresholds
            display_thresholds(thresholds)
            
            # Show threshold visualization
            st.subheader("📈 Threshold Visualization")
            threshold_fig = create_threshold_visualization(thresholds)
            st.pyplot(threshold_fig)
        else:
            st.error("❌ Calibration failed. Please check your dataset path.")
    
    # Check if thresholds are available
    if 'thresholds' not in st.session_state:
        st.warning("⚠️ Please calibrate thresholds first using your dataset.")
        return
    
    # File upload section
    st.header("📤 Upload and Enhance Image")
    uploaded_file = st.file_uploader(
        "Choose an ultrasound image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a fetal heart ultrasound image for adaptive enhancement"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process image
            image_pil = Image.open(uploaded_file)
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            original_rgb = np.array(image_pil)
            original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
            
            st.success(f"✅ Image loaded: {original_rgb.shape[0]}×{original_rgb.shape[1]} pixels")
            
            # Show original image and analysis
            st.header("📊 Image Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(original_rgb, caption="Original Image", use_column_width=True)
            
            with col2:
                # Get metrics and classification
                brightness, contrast, entropy = compute_quality_metrics(original_bgr)
                quality = classify_quality(brightness, contrast, entropy, st.session_state.thresholds)
                
                st.metric("Brightness", f"{brightness:.2f}")
                st.metric("Contrast", f"{contrast:.2f}")
                st.metric("Entropy", f"{entropy:.2f}")
                
                # Quality classification with color coding
                quality_colors = {
                    "poor": "🔴",
                    "moderate": "🟡", 
                    "good": "🟢",
                    "excellent": "🌟"
                }
                st.markdown(f"**Quality:** {quality_colors[quality]} {quality.upper()}")
            
            # Enhancement
            st.markdown("---")
            if st.button("🚀 Apply Adaptive Enhancement", type="primary"):
                with st.spinner("Applying adaptive enhancement..."):
                    enhanced_bgr, enhancement_type, enhancement_info = adaptive_enhancement(
                        original_bgr, st.session_state.thresholds
                    )
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                
                st.success(f"✅ {enhancement_info['action']}")
                
                # Display results
                st.header("📈 Enhancement Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(original_rgb, use_column_width=True)
                    st.write(f"**Quality:** {quality.upper()}")
                    st.write(f"**Brightness:** {brightness:.2f}")
                    st.write(f"**Contrast:** {contrast:.2f}")
                    st.write(f"**Entropy:** {entropy:.2f}")
                
                with col2:
                    st.subheader("Enhanced")
                    st.image(enhanced_rgb, use_column_width=True)
                    
                    # Enhanced metrics
                    enh_brightness, enh_contrast, enh_entropy = compute_quality_metrics(enhanced_bgr)
                    enh_quality = classify_quality(enh_brightness, enh_contrast, enh_entropy, st.session_state.thresholds)
                    
                    st.write(f"**Quality:** {enh_quality.upper()}")
                    st.write(f"**Brightness:** {enh_brightness:.2f}")
                    st.write(f"**Contrast:** {enh_contrast:.2f}")
                    st.write(f"**Entropy:** {enh_entropy:.2f}")
                    st.write(f"**Enhancement:** {enhancement_type}")
                
                # Improvement metrics
                st.header("📊 Improvement Summary")
                col1, col2, col3 = st.columns(3)
                
                brightness_change = enh_brightness - brightness
                contrast_change = enh_contrast - contrast  
                entropy_change = enh_entropy - entropy
                
                with col1:
                    st.metric("Brightness", f"{enh_brightness:.2f}", f"{brightness_change:+.2f}")
                
                with col2:
                    st.metric("Contrast", f"{enh_contrast:.2f}", f"{contrast_change:+.2f}")
                
                with col3:
                    st.metric("Entropy", f"{enh_entropy:.2f}", f"{entropy_change:+.2f}")
                
                # Download section
                st.header("💾 Download Enhanced Image")
                enhanced_pil = Image.fromarray(enhanced_rgb)
                img_buffer = io.BytesIO()
                enhanced_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=img_buffer,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime="image/png"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

if __name__ == "__main__":
    main()