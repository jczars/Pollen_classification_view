from keras import Model
import keras, sys
from keras.layers import Dense

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
    
    return fine_model
