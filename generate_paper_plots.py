import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison():
    # Data from Table I
    models = ['Zero-Shot CLIP', 'ResNet-18 CNN', 'Grounded System (Ours)']
    top1 = [38.92, 94.62, 91.92]
    top3 = [45.38, 99.62, 99.92]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color='#4c72b0')
    rects2 = ax.bar(x + width/2, top3, width, label='Top-3 Accuracy', color='#dd8452')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Closed-Set Performance Comparison on Oxford-IIIT Pet (13-breed subset)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)
    
    # Add labels on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    print("Saved plot to performance_comparison.png")

if __name__ == '__main__':
    # Use an IEEE-friendly style
    plt.style.use('seaborn-v0_8-whitegrid')
    plot_performance_comparison()
