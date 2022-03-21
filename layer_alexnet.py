# 3rd party libraries
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.engine.base_layer import Layer

def get_layers_dropnorm(input_shape: tuple[int], n_classes: int) ->list[Layer]:
    
    layers = [InputLayer(input_shape=input_shape),
              BatchNormalization(),
              Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
              BatchNormalization(),
              MaxPooling2D(pool_size=3, strides=2),
              Conv2D(filters=256, kernel_size=5, padding="same",activation="relu"),
              BatchNormalization(),
              MaxPooling2D(pool_size=3, strides=2),
              Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
              Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
              Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
              BatchNormalization(),
              Flatten(),
              Dense(units=4096, activation="relu"),
              Dropout(rate=0.5, seed=1),
              Dense(units=4096, activation="relu"),
              Dropout(rate=0.5, seed=1),
              Dense(units=n_classes, activation="softmax")
              ]
    
    return layers

def get_layers_nodrop(input_shape: tuple[int], n_classes: int) ->list[Layer]:
    
    layers = [InputLayer(input_shape=input_shape),
                Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
                MaxPooling2D(pool_size=3, strides=2),
                Conv2D(filters=256, kernel_size=5, padding="same",activation="relu"),
                MaxPooling2D(pool_size=3, strides=2),
                Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
                Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
                Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
                Flatten(),
                Dense(units=4096, activation="relu"),
                Dense(units=4096, activation="relu"),
                Dense(units=n_classes, activation="softmax")
                ]
     
    return layers  

def get_layers_dropout(input_shape: tuple[int], n_classes: int) ->list[Layer]:
    layers = [InputLayer(input_shape=input_shape),
              Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
              MaxPooling2D(pool_size=3, strides=2),
              Conv2D(filters=256, kernel_size=5, padding="same",activation="relu"),
              MaxPooling2D(pool_size=3, strides=2),
              Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
              Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
              Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
              Flatten(),
              Dense(units=4096, activation="relu"),
              Dropout(rate=0.5, seed=1),
              Dense(units=4096, activation="relu"),
              Dropout(rate=0.5, seed=1),
              Dense(units=n_classes, activation="softmax")
              ]
    return layers     