from keras import Model
import keras, sys
from keras.layers import Dense
<<<<<<< HEAD
from keras import layers
from keras.applications import ResNet50, MobileNet, DenseNet201, InceptionV3
from keras.applications import ResNet152V2, Xception, VGG16, VGG19

sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
from models import utils

def print_layer(conv_model, layers_params, verbose=0):
    """
    Prints and saves information about the layers of a convolutional model.

    Parameters
    ----------
    conv_model : keras.Model
        The convolutional model whose layer details will be printed and optionally saved.
    layers_params : dict
        Dictionary containing parameters for layer information storage, including:
            - 'save_dir' : str
                Directory path where the layer information will be saved.
            - 'id_test' : str
                Unique identifier for the test or model instance.
            - 'model' : str
                Model name to include in the saved file.

    Returns
    -------
    None
        This function prints layer details and, if a directory is specified, saves it to a CSV file.
    
    Notes
    -----
    If 'save_dir' is provided, this function creates a directory structure in 'save_dir/models/' 
    and saves the CSV file named '<id_test>_<model>_layers.csv'. The file includes the 
    trainable status and name of each layer in the model.
    """
    # Define save directory based on parameters
    save_dir = layers_params['save_dir']
    # Create a model name using test ID and model name from parameters
    nm_model = f"{layers_params['id_test']}_{layers_params['model']}"
    
    if save_dir:
        # Specify the full path to the model save directory
        #save_dir = f"{save_dir}/{nm_model}/"
        if verbose>0:
            print('save_dir ', save_dir)
        # Create necessary folders in the specified save directory
        utils.create_folders(save_dir, flag=0)
        
        # Define CSV path for saving layer information
        _csv_layers = save_dir + '/' + nm_model + '_layers.csv'
        
        # Add initial test ID and model name to the CSV
        utils.add_row_csv(_csv_layers, [['id_test', layers_params['id_test']]])
        utils.add_row_csv(_csv_layers, [['model', layers_params['model']]])
        utils.add_row_csv(_csv_layers, [['trainable', 'name']])

    # Initialize an array to store layer details
    layers_arr = []
    # Iterate through each layer in the model
    for i, layer in enumerate(conv_model.layers):
        # Print the layer index, trainable status, and name
        if verbose>0:
            print("{0} {1}:\t{2}".format(i, layer.trainable, layer.name))
        layers_arr.append([layer.trainable, layer.name])
    
    # Save layer details to CSV if save directory was specified
    if save_dir:
        utils.add_row_csv(_csv_layers, layers_arr)


def hyper_model(config_model, verbose=0):
    """
    Builds and configures a fine-tuned model based on a pre-trained base model.

    Parameters
    ----------
    config_model : dict
        Configuration dictionary with the following keys:
            - 'model' : str
                Name of the pre-trained model to be used (e.g., "VGG16").
            - 'id_test' : str
                Identifier for the test or specific model instance.
            - 'num_classes' : int
                Number of output classes for the classification task.
            - 'last_activation' : str
                Activation function for the final dense layer (e.g., "softmax").
            - 'freeze' : int
                Number of layers to freeze in the base model for transfer learning.
            - 'save_dir' : str
                Path to the directory where layer information will be saved.
            - 'optimizer' : str
                Name of the optimizer to use (e.g., "Adam", "SGD", "RMSprop", "Adagrad").
            - 'learning_rate' : float
                Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Compiled Keras model with fine-tuning and the specified configuration.

    Notes
    -----
    This function loads a pre-trained model (with 'imagenet' weights), freezes a certain number
    of layers as specified, adds a custom dense layer for classification, and optionally unfreezes
    some layers for further fine-tuning. The optimizer and learning rate are also set according 
    to the provided configuration.

    Documentation Style
    -------------------
    - Function documentation follows the **NumPy style** for readability and structured presentation.
    - Code is refactored according to **PEP8** coding standards for Python, focusing on readability,
      modularity, and clear comments.
    """
    
    # Initialize the specified pre-trained model
    model = eval(config_model['model'])
    id_test = config_model['id_test']
    
    # Load the base model with 'imagenet' weights and no top layer
    base_model = model(include_top=True, weights='imagenet')

    # Freeze all layers in the base model initially
    for layer in base_model.layers:
        layer.trainable = False
        if verbose>0:
            base_model.summary()

    # Connect a custom output layer to the base model
    conv_output = base_model.layers[-2].output  # Use the second last layer as output
    output = layers.Dense(config_model['num_classes'], name='predictions', 
                          activation=config_model['last_activation'])(conv_output)
    fine_model = Model(inputs=base_model.input, outputs=output)

    # Unfreeze layers based on the 'freeze' index for fine-tuning
    freeze = config_model['freeze']
    for layer in base_model.layers[freeze:]:
        layer.trainable = True

    # Prepare parameters for saving layer information
    layers_params = {'id_test': id_test, 'save_dir': config_model['save_dir'], 'model': config_model['model']}
    
    # Print and save layer details
    if verbose>0:
        print_layer(fine_model, layers_params, 1)

    # Set the optimizer based on configuration
    optimizer_name = config_model['optimizer']
    learning_rate = config_model['learning_rate']
    if optimizer_name == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if verbose>0:
        print(opt)

    # Compile the model with categorical crossentropy loss and accuracy metric
    fine_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    if verbose>0:
        fine_model.summary()
=======

sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
from models import utils_lib

def print_layer(conv_model, layers_params, verbose):
    """
    Prints the layers of the convolutional model and logs the details to a CSV file.

    Args:
        conv_model (keras.Model): The convolutional model to print.
        layers_params (dict): Parameters for the layers, including file paths for logging.
        verbose (int): Level of verbosity for logging. If > 0, prints details to the console.
    """
    save_dir = layers_params['save_dir']
    nm_model = layers_params['model']
    
    if save_dir:
        _csv_layers = f"{save_dir}{nm_model}_layers.csv"
        utils_lib.add_row_csv(_csv_layers, [['id_test', layers_params['id_test']]])
        utils_lib.add_row_csv(_csv_layers, [['freeze', layers_params['freeze'], 'depth', layers_params['depth'], 'percentil', layers_params['percentil']]])
        utils_lib.add_row_csv(_csv_layers, [['model', layers_params['model']]])
        utils_lib.add_row_csv(_csv_layers, [['trainable', 'name']])

    layers_arr = []
    for i, layer in enumerate(conv_model.layers):
        if verbose > 0:
            print(f"{i} {layer.trainable}:\t{layer.name}")
        layers_arr.append([i, layer.trainable, layer.name])
    
    if save_dir:
        utils_lib.add_row_csv(_csv_layers, layers_arr)

def hyper_model(rows, num_classes, verbose=0):
    """
    Builds a hypermodel based on a specified pre-trained architecture.

    Args:
        rows (dict): A dictionary containing model configuration parameters.
        num_classes (int): The number of classes for classification.
        verbose (int): Level of verbosity for logging. If > 0, prints details to the console.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    model = eval(rows['model'])
    save_dir = rows['root']
    learning_rate = rows['learning_rate']
    img_size = rows['img_size']
    input_shape = (img_size, img_size, 3)

    # Pre-trained model setup
    base_model = model(include_top=True, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None)

    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Exclude the last layer and set up the output layer
    conv = base_model.layers[-2].output
    output = Dense(num_classes, name='predictions', activation=rows['last_activation'])(conv)
    fine_model = Model(inputs=base_model.input, outputs=output)

    # Freeze layers
    freeze = rows['freeze']
    if verbose > 0:
        print("\n[INFO] Building architecture with freezing...")
    
    depth = len(base_model.layers)
    percentil = round(freeze / depth * 100, 2)
    if verbose > 0:
        print(f"{rows['model']} - depth: {depth}, freeze: {freeze}, freeze percentil: {percentil}%")

    if freeze != depth:
        for layer in base_model.layers[freeze:]:
            layer.trainable = True

    layers_params = {'id_test': rows['id_test'], 'save_dir': save_dir, 'model': rows['model'], 
                     'freeze': freeze, 'depth': depth, 'percentil': percentil}
    
    print_layer(base_model, layers_params, verbose)

    # Optimizer setup
    optimizer_options = {
        'Adam': keras.optimizers.Adam(learning_rate=learning_rate),
        'RMSprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
        'Adagrad': keras.optimizers.Adagrad(learning_rate=learning_rate),
        'SGD': keras.optimizers.SGD(learning_rate=learning_rate)
    }
    
    opt = optimizer_options.get(rows['optimizer'])
    if opt is None:
        raise ValueError(f"Optimizer '{rows['optimizer']}' is not supported.")

    if verbose > 0:
        print(f"Using optimizer: {opt}")
    
    # Compile model
    fine_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    if verbose > 0:
        fine_model.summary()
    
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
    return fine_model
