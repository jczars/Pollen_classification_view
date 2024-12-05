"""
Module for training Keras models with additional functionalities, 
including callbacks and custom training loops.
"""

import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Removed unused imports
from keras.utils import custom_object_scope

class LRSchedulerMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Obtém o valor atual da learning rate
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        logs['lr'] = lr  # Adiciona o LR aos logs de cada época
        print(f"Epoch {epoch + 1}: Learning Rate is {lr:.6e}")


def create_callbacks0():
    """
    Creates a list of callbacks for training, including early stopping.

    Returns:
        list: A list of Keras callbacks.

    Notes
    -----
    The callbacks are used to monitor the training process and
    to take actions when the model is not improving. The
    callbacks are: EarlyStopping and ReduceLROnPlateau.

    EarlyStopping stops the training process when the model
    is not improving. The patience parameter defines the number
    of epochs to wait before stopping the training process.

    ReduceLROnPlateau reduces the learning rate when the model
    is not improving. The patience parameter defines the number
    of epochs to wait before reducing the learning rate.
    """
    rlp = ReduceLROnPlateau(monitor='val_loss',
                   factor=0.1,
                   min_delta=1e-5,
                   patience=5,
                   verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    return [early_stopping, rlp]

def create_callbacks():
    """
    Creates a list of callbacks for training, including early stopping and learning rate scheduler monitor.

    Returns:
        list: A list of Keras callbacks.
    """
    # Reduzir a taxa de aprendizado quando a validação não melhorar
    rlp = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            min_delta=1e-5,
                            patience=5,
                            verbose=1)
    
    # Parar o treinamento caso não haja melhoria após 'patience' épocas
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    
    # Callback para monitorar a Learning Rate
    lr_monitor = LRSchedulerMonitor()

    return [early_stopping, rlp, lr_monitor]


def run_train0(train_data, val_data, model_fine, train_config):
    """
    Trains a Keras model, monitoring execution time and identifying the best epoch, with an option to display logs.

    Parameters:
        train_data: Training dataset.
        val_data: Validation dataset.
        model_fine: Model to be trained.
        train_config (dict): Dictionary containing training 
         configurations (batch_size, epochs, verbosity).

    Returns:
        history: Training history.
        start_time: Start time of the training.
        end_time: End time of the training.
        duration: Duration of the training.
        best_epoch: Best epoch based on validation loss.
    """
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    verbosity = train_config.get('verbosity', 1)

    # Record start time
    start_time = datetime.datetime.now().replace(microsecond=0)
    if verbosity > 0:
        print(f'Batch size: {batch_size}\nTraining start time: {start_time}')

    # Configure callbacks and train on GPU
    with tf.device('/device:GPU:0'):
        print('\n', start_time)
        callbacks_list = create_callbacks()
        history = model_fine.fit(
            train_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=val_data
        )

    # End time and duration of training
    end_time = datetime.datetime.now().replace(microsecond=0)
    duration = end_time - start_time
    if verbosity > 0:
        print(f'Training duration: {duration}')

    # Identify the best epoch based on the lowest validation loss
    val_loss = history.history.get('val_loss', [])
    best_epoch = val_loss.index(min(val_loss)) + 1 if val_loss else None
    if verbosity > 0 and best_epoch:
        print(f'Best epoch: {best_epoch} with validation loss: {min(val_loss):.4f}')

    return {
        'history':history, 
        'start_time': start_time, 
        'end_time': end_time, 
        'duration':duration, 
        'best_epoch': best_epoch}

def run_train(train_data, val_data, model_fine, train_config):
    """
    Trains a Keras model, monitoring execution time and identifying the best epoch, with an option to display logs.

    Parameters:
        train_data: Training dataset.
        val_data: Validation dataset.
        model_fine: Model to be trained.
        train_config (dict): Dictionary containing training 
         configurations (batch_size, epochs, verbosity).

    Returns:
        history: Training history.
        start_time: Start time of the training.
        end_time: End time of the training.
        duration: Duration of the training.
        best_epoch: Best epoch based on validation loss.
    """
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    verbosity = train_config.get('verbosity', 1)

    # Record start time
    start_time = datetime.datetime.now().replace(microsecond=0)
    if verbosity > 0:
        print(f'Batch size: {batch_size}\nTraining start time: {start_time}')

    # Configure callbacks and train on GPU
    with tf.device('/device:GPU:0'):
        print('\n', start_time)
        callbacks_list = create_callbacks()
        history = model_fine.fit(
            train_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=val_data
        )

    # End time and duration of training
    end_time = datetime.datetime.now().replace(microsecond=0)
    duration = end_time - start_time
    if verbosity > 0:
        print(f'Training duration: {duration}')

    # Identify the best epoch based on the lowest validation loss
    val_loss = history.history.get('val_loss', [])
    best_epoch = val_loss.index(min(val_loss)) + 1 if val_loss else None
    if verbosity > 0 and best_epoch:
        print(f'Best epoch: {best_epoch} with validation loss: {min(val_loss):.4f}')

    num_eapoch=len(history.history['loss'])

    # Capture the learning rate at the best epoch
    # Get the learning rate from the history (for ReduceLROnPlateau)
    lr_values = history.history.get('lr', [])
    best_lr = lr_values[best_epoch - 1] if best_epoch and lr_values else None

    if verbosity > 0 and best_lr:
        print(f'Learning rate at best epoch: {best_lr}')

    return {
        'history': history, 
        'start_time': start_time, 
        'end_time': end_time, 
        'duration': duration, 
        'best_epoch': best_epoch,
        'best_lr': best_lr,
        'num_eapoch': num_eapoch
    }


def load_model(path_model, verbose=0):
    """
    Loads a Keras model from the specified path.

    Parameters:
        path_model (str): Path to the saved model.
        verbose (int): Verbosity level for model summary.

    Returns:
        model: The loaded Keras model.
    """
    model_rec = tf.keras.models.load_model(path_model)
    
    if verbose > 0:
        model_rec.summary()

    return model_rec

def load_model_vit(model_path, verbose=0):
    """
    Loads a Vision Transformer model from the specified path, using a custom optimizer if necessary.

    Parameters:
        model_path (str): Path to the saved model.
        verbose (int): Verbosity level for model summary.

    Returns:
        model: The loaded Vision Transformer model.
    """
    # Load the model with the registered optimizer in the custom object scope
    with custom_object_scope({'Addons>RectifiedAdam': tfa.optimizers.RectifiedAdam}):
        model = tf.keras.models.load_model(model_path)
    if verbose > 0:
        model.summary() 
    return model

if __name__ == "__main__":
    help(create_callbacks)
    help(run_train)
    help(load_model_vit)