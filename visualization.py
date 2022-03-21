# 3rd party libraries
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# standard libraries
from os import PathLike
from typing import Union

from numpy.core.numeric import indices

# local libraries
from dataset_utils import CLASSES, TRAIN_SET, IMG_SHAPE, create_dataset, info_folder

NODROP_NONORM_PERF = "data/model_performance/alexnet_nodrop_nonorm.csv"
DROP_PERF = "data/model_performance/alexnet_drop.csv"
DROPNORM_PERF = "data/model_performance/alexnet_dropnorm.csv"

def plot_samples_img(images: np.ndarray, 
                    labels: np.ndarray, 
                    classes: list[str],
                    indices: list[int], 
                    destination: Union[str, bytes, PathLike],
                    plot_title: str="",
                    show: bool=False) -> None:
    
    """Plot 10 images, one x class, in a 5x2 grid

    Args:
        images (np.ndarray): images as numpy arrays
        labels (np.ndarray): classes as numpy arrays
        classes (list[str]): classes
        destination (Union[str, bytes, PathLike]): path
        plot_title (str, optional): title of plot. Defaults to "".
        show (bool, optional): show plot in window. Defaults to False.
    """
    # creating grid
    fig, axs = plt.subplots(nrows=5, ncols=2)
    fig.suptitle(plot_title)
    fig.tight_layout()
    for ax, i in zip(axs.flat, indices):
        ax.imshow(images[i-1])
        ax.set_title(classes[labels[i-1, 0]])
        ax.axis("off")
    
    fig.savefig(destination) 
    print(f'Image saved in "{destination}"')  
    
    if show:
        plt.show()  
        
def plot_class_frequency(class_frequency: list[int], 
                         classes:list[str], 
                         destination:Union[str, bytes, PathLike],
                         show: bool=False) -> None:
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, class_frequency, align='center', alpha=0.5, color="purple", edgecolor="black")
    plt.yticks(y_pos, classes)
    plt.xlabel('Frequency')
    plt.title('Sample x Class')
    plt.tight_layout()
    
    plt.savefig(destination)
    print(f'Image saved in "{destination}"')
    
    if show:
        plt.show()  
        
def plot_model_performance(performance: pd.DataFrame, 
                     destination: Union[str, bytes, PathLike], 
                     show: bool=False) -> None:  
    # plot performance
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    
    fig.suptitle("Performance")
    # plot loss function
    ax1.set_title("Loss function")
    ax1.plot(performance["loss"], color="#c51f5d", label="Train Loss")
    ax1.plot(performance["val_loss"], color="#7bb3ff", label="Val Loss")
    ax1.legend()
    
    # plot accuracy
    ax2.set_title("Accuracy")
    ax2.plot(performance["accuracy"], color="#c51f5d", label="Train Accuracy")
    ax2.plot(performance["val_accuracy"], color="#7bb3ff", label="Val Accuracy")
    ax2.legend()
    
    fig.savefig(destination)  
                
    if show:
        plt.show()
        
def main():
    
    info_trainset = info_folder(folder=TRAIN_SET, classes=CLASSES[:10])
    x, y = create_dataset(folder=TRAIN_SET, classes=CLASSES[:10], img_shape=IMG_SHAPE)
    
    indices = np.cumsum(info_trainset[1])
    plot_samples_img(images=x, 
                    labels=y, 
                    classes=CLASSES, 
                    indices= indices,
                    plot_title="Immagini d'esempio",
                    destination="data/viz/img_es.png",
                    show=True) 
    
    plot_class_frequency(class_frequency=info_trainset[1], 
                         classes=CLASSES[:10], 
                         destination="data/viz/frequency.png", 
                         show=True)


    # plot models performance
    models_perf = {"nodrop_nonorm": pd.read_csv(NODROP_NONORM_PERF), 
                 "drop": pd.read_csv(DROP_PERF), 
                 "dropnorm": pd.read_csv(DROPNORM_PERF)}
    
    for name, model_perf in models_perf.items():
        plot_model_performance(performance=model_perf, 
                         destination=f"data/viz/alexnet_{name}_perf.png",
                         show=True)       
    
if __name__ == "__main__":
    main()