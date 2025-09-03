from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .nr_module import HybridFeatureRefinementModule

def get_hmaytsf_baseline_model(num_classes):
    """
    HMAY-TSF Baseline Model with Adaptive Anchor Initialization
    Optimized for UAV traffic domain with small-object detection capabilities
    """
    # Initialize base model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Adaptive Anchor Initialization for UAV small-object detection
    # Reduce anchor sizes to better handle small objects in aerial imagery
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))  # Reduced from default sizes
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    # Apply custom anchor configuration
    if hasattr(model.rpn, 'anchor_generator'):
        model.rpn.anchor_generator.sizes = anchor_sizes
        model.rpn.anchor_generator.aspect_ratios = aspect_ratios
    
    # Get feature dimensions for HFRM
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Initialize Hybrid Feature Refinement Module (HFRM)
    hfrm_module = HybridFeatureRefinementModule(in_features, in_features)
    
    # Update box predictor for custom classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Integrate HFRM into FPN as multi-scale refinement
    model.backbone.fpn.add_module("multi_scale_refinement", hfrm_module)
    
    return model