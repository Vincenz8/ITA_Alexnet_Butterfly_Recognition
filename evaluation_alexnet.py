# 3rd party libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# local libraries
from dataset_utils import TEST_SET, CLASSES, IMG_SHAPE, create_dataset

def main():
    
    x_test, y_test = create_dataset(folder=TEST_SET, classes=CLASSES[:10], img_shape=IMG_SHAPE)
    y_test = to_categorical(y=y_test)

    alexnet = load_model("data/model/alexnet_dropnorm.h5")
    
    print("EVALUATION")
    alexnet.evaluate(x=x_test, y=y_test, batch_size=32)
    
if __name__ == "__main__":
    main()