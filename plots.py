import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_classification_report(y_true, y_pred, title="Classification Report"):
    print(f"📊 {title}")
    print(classification_report(y_true, y_pred))

def plot_label_distribution(y, class_names=None, title="Label Distribution"):
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if class_names and len(class_names) == len(unique_classes):
        labels = class_names
    else:
        labels = [str(cls) for cls in unique_classes]
    
    sns.barplot(x=labels, y=counts, palette='magma')
    plt.xlabel("Class Labels")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

