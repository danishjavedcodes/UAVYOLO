# HMAY-TSF Model Architecture

## Overview
**Hybrid Multi-scale Attention + Temporal-Spatial Fusion (HMAY-TSF)** is a baseline model architecture designed for UAV traffic domain object detection with enhanced feature refinement capabilities.

## Architecture Components

### 1. **Hybrid Feature Refinement Module (HFRM)**
- **Purpose**: Multi-scale feature alignment and attention-based refinement
- **Components**:
  - **Adaptive Channel Attention (ACA)**: Emphasizes important feature channels
  - **Adaptive Spatial Attention (ASA)**: Focuses on spatial regions of interest
  - **Multi-scale Pooling Block**: SPP-like pooling for scale alignment (1x1 + 3x3)
  - **Learnable Amplification**: Adaptive feature enhancement factor

### 2. **Base Detection Model**
- **Backbone**: ResNet-50 with pre-trained weights
- **Neck**: Feature Pyramid Network (FPN) with multi-scale refinement
- **Head**: ROI heads with custom box predictor
- **Integration**: HFRM integrated into FPN as `multi_scale_refinement`

### 3. **Adaptive Anchor Initialization**
- **Purpose**: Optimized for UAV small-object detection
- **Configuration**: Reduced anchor sizes (16, 32, 64, 128, 256)
- **Aspect Ratios**: (0.5, 1.0, 2.0) for better object coverage

### 4. **Super-Resolution Augmentation (SRDA) Pipeline**
- **Purpose**: External preprocessing for high-resolution simulation
- **Features**: Configurable upscale factors and target sizes
- **Integration**: Placeholder for advanced SR techniques (ESRGAN, SRCNN, etc.)

## Usage Examples

### Basic Model Creation
```python
from models import get_hmaytsf_baseline_model

# Create model with custom number of classes
model = get_hmaytsf_baseline_model(num_classes=10)
```

### SRDA Pipeline Integration
```python
from models import create_srda_pipeline

# Create SRDA pipeline with 2x upscaling
srda = create_srda_pipeline(upscale_factor=2.0)

# Apply to single image
augmented_image = srda.apply_srda(input_image)

# Apply to batch
augmented_batch = srda.apply_batch_srda(input_batch)
```

### Direct Module Usage
```python
from models import HybridFeatureRefinementModule

# Create HFRM with custom dimensions
hfrm = HybridFeatureRefinementModule(in_channels=256, out_channels=256)
refined_features = hfrm(input_features)
```

## Key Features

### ✨ **Multi-Scale Alignment**
- SPP-like pooling block for better scale handling
- Residual connections for feature preservation
- Adaptive pooling sizes (1x1, 3x3)

### 🎯 **Attention Mechanisms**
- **ACA**: Channel-wise feature emphasis
- **ASA**: Spatial region focus
- Combined attention for occlusion robustness

### 🚁 **UAV Optimization**
- Smaller anchor sizes for small objects
- Multi-scale feature processing
- Enhanced feature representation

### 🔧 **Modular Design**
- Easy to extend and modify
- Configurable components
- Clean separation of concerns

## Model Flow

```
Input Image → ResNet-50 → FPN → HFRM → ROI Heads → Output
                ↓           ↓      ↓
            Backbone   Multi-scale  Attention +
                      Refinement   SPP-like Pooling
```

## Performance Considerations

- **Memory**: SPP-like pooling adds minimal overhead
- **Speed**: Attention mechanisms are lightweight
- **Scalability**: Modular design allows easy optimization
- **Training**: Compatible with standard detection training pipelines

## Future Enhancements

1. **Advanced SR Techniques**: Integrate state-of-the-art super-resolution models
2. **Temporal Fusion**: Add temporal consistency for video sequences
3. **Dynamic Attention**: Adaptive attention based on object density
4. **Multi-task Learning**: Joint detection and segmentation

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Python >= 3.7

## Citation

If you use this architecture in your research, please cite:
```
@misc{hmaytsf2024,
  title={HMAY-TSF: Hybrid Multi-scale Attention + Temporal-Spatial Fusion for UAV Traffic Detection},
  author={Your Name},
  year={2024}
}
```
