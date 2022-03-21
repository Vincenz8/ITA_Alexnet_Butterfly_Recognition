# 3rd party libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pandas as pd

# local libraries
from dataset_utils import TRAIN_SET, VALID_SET, CLASSES, IMG_SHAPE
from dataset_utils import create_dataset
from layer_alexnet import get_layers_dropnorm, get_layers_dropout, get_layers_nodrop

def main():
    
    classes = CLASSES[:10]
    n_classes = len(classes)
    
    x_train, y_train = create_dataset(folder=TRAIN_SET, classes=classes, img_shape=IMG_SHAPE)
    x_valid, y_valid = create_dataset(folder=VALID_SET, classes=classes, img_shape=IMG_SHAPE)
    
    # preprocessing
    y_train = to_categorical(y=y_train)
    y_valid = to_categorical(y=y_valid)
    
    # init layers of alexnet
    alexnet_variants = {"nodrop_nonorm": get_layers_nodrop(IMG_SHAPE, n_classes),
                        "drop": get_layers_dropout(IMG_SHAPE, n_classes),
                        "dropnorm": get_layers_dropnorm(IMG_SHAPE, n_classes)}
    
    for key, alexnet_layers in alexnet_variants.items():
        
        alexnet = Sequential(layers=alexnet_layers, name=key)
        alexnet.summary()
        
        alexnet.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

        history = alexnet.fit(x=x_train, y=y_train, batch_size=32, epochs=1, validation_data=(x_valid, y_valid))
        
        alexnet_performance = pd.DataFrame(history.history)
        alexnet_performance.to_csv(f"data/model_performance/alexnet_{key}.csv",index=False)
        alexnet.save(f"data/model/alexnet_{key}.h5")
        
    
if __name__ == "__main__":
   main()