
# Select models
from keras.applications import InceptionV3, ResNet152V2, Xception, VGG16, VGG19, DenseNet121, MobileNet, DenseNet201, NASNetLarge, InceptionResNetV2
from keras import models, layers, Model

import keras, sys
import tensorflow as tf
from keras.models import Sequential, Model
from keras import models, layers
from keras.layers import Dropout, Dense, Input
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Dropout
from keras.layers import Conv2D, GlobalAveragePooling2D

"""# Select models

## DenseNet201
"""

def tl_DenseNet201(input_shape):
    """
    Initializes a DenseNet201 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained DenseNet201 model with frozen layers.
    """
    conv_base = DenseNet201(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_DenseNet201(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.BatchNormalization())
  model.add(layers.Flatten())

  model.add(layers.Dense(2048, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))  #5
  model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
  return model

"""## Mobilenet"""

def tl_Mobilenet(input_shape):
    """
    Initializes a MobileNet model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained MobileNet model with frozen layers.
    """
    conv_base = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_Mobilenet(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.BatchNormalization())
  model.add(layers.Flatten())
  
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.6))
  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  return model

"""## InceptionV3"""

def tl_InceptionV3(input_shape):
    """
    Initializes a InceptionV3 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained InceptionV3 model with frozen layers.
    """
    conv_base = InceptionV3(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_InceptionV3(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the InceptionV3 convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained InceptionV3 convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.2))

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  return model

"""## InceptionResNetV2"""

def tl_InceptionResNetV2(input_shape):
    """
    Initializes an InceptionResNetV2 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained InceptionResNetV2 model with frozen layers.
    """
    
    conv_base = InceptionResNetV2(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_InceptionResNetV2(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.2))

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  return model

"""## ResNet152V2"""

def tl_ResNet152V2(input_shape):
    """
    Initializes a ResNet152V2 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained ResNet152V2 model with frozen layers.
    """
    conv_base = ResNet152V2(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_ResNet152V2(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the ResNet152V2 convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained ResNet152V2 convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.BatchNormalization())
  model.add(layers.Flatten())

  model.add(layers.Dense(2048, activation='relu'))
  model.add(layers.Dropout(0.8))
  model.add(layers.BatchNormalization())


  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))  #5
  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  #optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])
  return model

"""## Xception"""

def tl_Xception(input_shape):
    """
    Initializes a Xception model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained Xception model with frozen layers.
    """
    conv_base = Xception(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_Xception(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the Xception convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained Xception convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.GlobalAveragePooling2D())

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))  #5
  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  #optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])
  return model

"""## VGG16"""

def tl_VGG16(input_shape):
    conv_base = VGG16(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_VGG16(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the VGG16 convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained VGG16 convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.Flatten())
  model.add(layers.Dense(2048, activation='relu'))
  model.add(layers.Dropout(0.4))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.4))

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))  #5
  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  #optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])
  return model

"""## VGG19"""

def tl_VGG19(input_shape):
    """
    Initializes a VGG19 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.Model: Pre-trained VGG19 model with frozen layers.
    """
    conv_base = VGG16(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)

    for layer in conv_base.layers:
      layer.trainable = False

    return conv_base

def new_VGG19(conv_base, CATEGORIES):
  """
  Builds a complete model by adding fully connected layers to the VGG19 convolutional base.

  Args:
      conv_base (keras.Model): Pre-trained VGG19 convolutional base.
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  model = models.Sequential()
  model.add(conv_base)

  model.add(layers.Flatten())
  model.add(layers.Dense(2048, activation='relu'))
  model.add(layers.Dropout(0.4))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.4))

  model.add(layers.Dense(len(CATEGORIES), activation='softmax'))  #5
  model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  #optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])
  return model

def sel_cnn0(_model, input_shape, CATEGORIES):
  """
  Selects a pre-trained model, adds fully connected layers and returns a compiled model.

  Args:
      _model (str): Name of the pre-trained model to use.
      input_shape (tuple): Shape of the input images (height, width, channels).
      CATEGORIES (list): List of classification categories.

  Returns:
      keras.Model: Compiled complete model.
  """
  if _model == 'InceptionResNetV2':
    conv_base=tl_InceptionResNetV2(input_shape)
    model = new_InceptionResNetV2(conv_base, CATEGORIES)

  if _model == 'DenseNet201':
    conv_base=tl_DenseNet201(input_shape)
    model = new_DenseNet201(conv_base, CATEGORIES)

  if _model == 'Mobilenet':
    conv_base=tl_Mobilenet(input_shape)
    model = new_Mobilenet(conv_base, CATEGORIES)

  if _model == 'InceptionV3':
    conv_base=tl_InceptionV3(input_shape)
    model = new_InceptionV3(conv_base, CATEGORIES)

  if _model == 'ResNet152V2':
    conv_base=tl_ResNet152V2(input_shape)
    model = new_ResNet152V2(conv_base, CATEGORIES)
  if _model == 'Xception':
      conv_base=tl_Xception(input_shape)
      model = new_Xception(conv_base, CATEGORIES)
  if _model == 'VGG16':
      conv_base=tl_VGG16(input_shape)
      model = new_VGG16(conv_base, CATEGORIES)
  if _model == 'VGG19':
      conv_base=tl_VGG19(input_shape)
      model = new_VGG19(conv_base, CATEGORIES)
  model.summary()
  return model


def sel_cnn(_model, input_shape, CATEGORIES):
    """
    Dynamically select and initialize a pre-trained CNN model with custom classification layers.

    Args:
        _model (str): Name of the model to use ('InceptionResNetV2', 'DenseNet201', 'Mobilenet', etc.).
        input_shape (tuple): Shape of the input images.
        CATEGORIES (list): List of class labels for classification.

    Returns:
        keras.Sequential: Fully assembled and compiled model.
    """
    # Dictionary mapping model names to their respective functions
    model_mapping = {
        'InceptionResNetV2': (tl_InceptionResNetV2, new_InceptionResNetV2),
        'DenseNet201': (tl_DenseNet201, new_DenseNet201),
        'Mobilenet': (tl_Mobilenet, new_Mobilenet),
        'InceptionV3': (tl_InceptionV3, new_InceptionV3),
        'ResNet152V2': (tl_ResNet152V2, new_ResNet152V2),
        'Xception': (tl_Xception, new_Xception),
        'VGG16': (tl_VGG16, new_VGG16),
        'VGG19': (tl_VGG19, new_VGG19),
    }

    # Check if the specified model is supported
    if _model not in model_mapping:
        raise ValueError(f"Unsupported model: {_model}")

    # Retrieve the functions for the specified model
    tl_function, new_function = model_mapping[_model]

    # Initialize the base model and create the complete model
    conv_base = tl_function(input_shape)
    model = new_function(conv_base, CATEGORIES)

    # Print model summary
    model.summary()
    return model