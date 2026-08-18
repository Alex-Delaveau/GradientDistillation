import sys, torch, clip
sys.path.append("/lustre/fswork/projects/rech/rbw/ucw75ke/GradientDistillation")  # racine projet
from src.models.moco_vision_tansformer import VisionTransformerMoCoV3

torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
clip.load("ViT-B/32")
VisionTransformerMoCoV3.from_pretrained("nyu-visionx/moco-v3-vit-b", num_classes=0)
print("caches remplis")
