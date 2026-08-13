import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_dataset_distribution():
    breeds = [
        "Beagle", "Boxer", "Chihuahua", "Pug", "Samoyed", "Shiba Inu", 
        "Great Pyrenees", "GSP", "Staffordshire", "Yorkshire", 
        "Pomeranian", "Basset Hound", "Saint Bernard"
    ]
    counts = [100] * 13 # 100 images per breed from your oxford trainval subset

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(breeds, counts, color='#4c72b0')
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Distribution (13-Breed Oxford Subset)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('figure_3_dataset_distribution.png', dpi=300)
    print("Saved figure_3_dataset_distribution.png")

def plot_real_loss_curve():
    import json
    import os
    loss_log_path = os.path.join('trained_model_fact_checked', 'training_loss_log.json')
    if not os.path.exists(loss_log_path):
        print(f"Loss log not found at {loss_log_path}. Please run training first.")
        return

    with open(loss_log_path, 'r') as f:
        data = json.load(f)
        
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label='Train Loss', color='#4c72b0', linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#dd8452', linewidth=2)
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss (Multi-Task Trait Net)', fontsize=14)
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figure_5_loss_curve.png', dpi=300)
    print("Saved figure_5_loss_curve.png (from REAL data)")

def plot_trait_accuracy():
    import pickle
    import os
    metadata_path = os.path.join('trained_model_fact_checked', 'training_metadata.pkl')
    if not os.path.exists(metadata_path):
        print(f"Metadata not found at {metadata_path}. Please run training first.")
        return
        
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        
    acc_dict = metadata['validation_trait_accuracy']
    traits = list(acc_dict.keys())
    acc = [acc_dict[t] * 100 for t in traits]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(traits, acc, color='#55a868')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy per Physical Trait', fontsize=14)
    ax.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
                    
    plt.tight_layout()
    plt.savefig('figure_6_trait_accuracy.png', dpi=300)
    print("Saved figure_6_trait_accuracy.png (from REAL data)")

def plot_ablation_study():
    # Ablation study on OOD Shepsky Size Class prediction
    models = ['Neural Model Only', 'CLIP Zero-Shot Only', 'Hybrid Ensemble (Ours)']
    large_size_prob = [9.58, 45.82, 61.98] 
    small_size_prob = [90.21, 10.5, 20.93] 
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, large_size_prob, width, label='Probability of LargeSize (Correct)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, small_size_prob, width, label='Probability of SmallSize (Shortcut Error)', color='#c44e52')
    
    ax.set_ylabel('Prediction Confidence (%)', fontsize=12)
    ax.set_title('Ablation Study: Overcoming Neural Shortcut Bias (Shepsky Case)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    plt.tight_layout()
    plt.savefig('figure_8_ablation_study.png', dpi=300)
    print("Saved figure_8_ablation_study.png")

if __name__ == '__main__':
    plot_dataset_distribution()
    plot_real_loss_curve()
    plot_trait_accuracy()
    plot_ablation_study()
