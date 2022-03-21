# 3rd party libraries 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# standard libraries
from pathlib import Path
import itertools
from os import PathLike
from typing import Union

TRAIN_SET = Path("data/butterflies/train")
VALID_SET = Path("data/butterflies/valid")
TEST_SET = Path("data/butterflies/test")
IMG_SAMPLE = Path("data/butterflies/train/adonis/01.jpg")

# iterdir() return subdir in arbitrary order
CLASSES = sorted([path.name for path in TRAIN_SET.iterdir()])
IMG_SHAPE = (224, 224, 3)

def img_to_np(filename: Union[str, bytes, PathLike]) -> np.ndarray:
    with Image.open(filename) as img:
        arr_img = np.asarray(img)
    return arr_img

def info_folder(folder: Path, classes: list[str]) -> tuple[dict, list]:
    
    grouped_images = {name: list(folder.glob(f"{name}/*")) for name in classes}
    
    # get number of samples x class
    freq_class = [len(grouped_images[k]) for k in grouped_images]
    
    return grouped_images, freq_class
    
def create_dataset(folder: Path, classes: list[str], img_shape: tuple[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return feature matrix and target vector

    Args:
        folder (Path): folder's path that contain images
        classes (list[str]): classes
        img_shape (tuple[int]): size of image

    Returns:
        tuple[np.ndarray, np.ndarray]: feature matrix, target matrix
    """
    grouped_images, n_images = info_folder(folder=folder, classes=classes)
    
    # calculate cumulative sum for labels
    index = np.concatenate(([0], np.cumsum(n_images)))
    
    y = np.empty(shape=(index[-1],1), dtype=int)
    x = np.empty(shape=(index[-1], *img_shape), dtype=int) 
    
    all_images = itertools.chain.from_iterable(grouped_images.values()) 
      
    for i, img_path in enumerate(all_images):   
            x[i] = img_to_np(filename=img_path)
       
    for i in range(1, len(index)):
        y[index[i-1]:index[i]] = i-1

    return x, y

def main():
    
    img = img_to_np(filename=IMG_SAMPLE)
    grouped_images, n_images = info_folder(folder=TRAIN_SET, classes=CLASSES)
    
    print(n_images)
    x_train, y_train = create_dataset(folder=TRAIN_SET, classes=CLASSES[:10], img_shape=img.shape)
    x_valid, y_valid = create_dataset(folder=VALID_SET, classes=CLASSES[:10], img_shape=img.shape)
    x_test, y_test = create_dataset(folder=TEST_SET, classes=CLASSES[:10], img_shape=img.shape)
    
    print(img.shape)
    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)
    
     
        
if __name__ == "__main__":
    main()
    

