import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image

def generate_figure_1_workflow():
    """Generates a block diagram for the Neuro-Symbolic Workflow"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Define blocks
    blocks = [
        ("Image\nInput", (0, 0.4), '#d5f5e3'),
        ("CLIP Vision\nEncoder", (0.2, 0.6), '#f9d0c4'),
        ("Multi-Task\nTrait Net", (0.4, 0.6), '#f9d0c4'),
        ("Zero-Shot\nText Scorer", (0.2, 0.2), '#f9d0c4'),
        ("Log-Space\nEnsemble", (0.6, 0.4), '#d4e6f1'),
        ("SPARQL\nReasoner", (0.8, 0.4), '#d4e6f1')
    ]
    
    # Draw blocks
    for text, (x, y), color in blocks:
        rect = patches.FancyBboxPatch((x, y), 0.15, 0.2, boxstyle="round,pad=0.02", 
                                      edgecolor='black', facecolor=color, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.075, y + 0.1, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        ((0.15, 0.5), (0.2, 0.7)),   # Input -> Vision
        ((0.15, 0.5), (0.2, 0.3)),   # Input -> Text
        ((0.35, 0.7), (0.4, 0.7)),   # Vision -> Trait Net
        ((0.55, 0.7), (0.6, 0.55)),  # Trait Net -> Ensemble
        ((0.35, 0.3), (0.6, 0.45)),  # Text -> Ensemble
        ((0.75, 0.5), (0.8, 0.5))    # Ensemble -> Reasoner
    ]
    
    for (sx, sy), (ex, ey) in arrows:
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color='black', lw=2))
                    
    plt.title("Fig. 1. Neuro-Symbolic Architecture Workflow", y=1.05, fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_1_workflow.png', dpi=300, bbox_inches='tight')
    print("Generated figure_1_workflow.png")

def generate_figure_2_dataset_sample():
    """Finds images specifically from the dataset folders and stitches them into a grid."""
    target_folders = [
        r"F:\Production\hyperid-ke\mixed-breed",
        r"F:\Production\hyperid-ke\data\oxford-iiit-pet\images"
    ]
    
    image_paths = []
    for folder in target_folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, f))
                        if len(image_paths) == 4:
                            break
                if len(image_paths) == 4:
                    break
        if len(image_paths) == 4:
            break
            
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax in axes:
        ax.axis('off')
        
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).resize((224, 224))
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(img_path)[:10])
        except:
            pass
            
    if not image_paths:
        axes[0].text(0.5, 0.5, 'Insert Dataset\nImages Here', ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig('figure_2_dataset_sample.png', dpi=300)
    print("Generated figure_2_dataset_sample.png")



if __name__ == '__main__':
    generate_figure_1_workflow()
    generate_figure_2_dataset_sample()
