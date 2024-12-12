#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:19:09 2024

@author: jczars
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import copy
import cv2
import os


def load_model(path_model):
    model_rec=tf.keras.models.load_model(path_model)
    model_rec.summary()

    return model_rec

def load_img_gen(img_path, target_size, verbose=0):
    # Carregar imagem com tamanho alvo
    img = load_img(img_path, target_size=target_size)
    
    # Converter a imagem em um array numpy
    img_array = img_to_array(img)
    
    # Adicionar uma dimensão extra para simular um lote (batch)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Criar o ImageDataGenerator para pré-processamento
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Aplicar o ImageDataGenerator na imagem carregada
    img_iterator = datagen.flow(img_array)
    
    # Acessar a imagem processada
    processed_image = img_iterator.next()
    
    # Visualizar a imagem processada (opcional)
    if verbose==1:
        plt.imshow(processed_image[0])
        plt.show()

    return processed_image[0]

def predict_un_img(img_test, model_tl, CATEGORIES):
    # Adicionar uma dimensão extra para simular um lote
    img_test = np.expand_dims(img_test, axis=0)  
    # Fazer a previsão
    y_test = model_tl.predict(img_test)
    
    # Obter o índice da classe com a maior probabilidade
    predicted_class = np.argmax(y_test, axis=1)
    predict_label=CATEGORIES[predicted_class[0]]
    # Obten a probabilidade
    prob = np.max(y_test)
    # Exibir a classe prevista
    print(f" index: {predicted_class}, Classe prevista: {predict_label}, Probabilidade: {prob} ")
    return predict_label

def load_img_batch(folder_path, target_size):
    images_batch=[]
    filenames_batch=[]
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # Carregar imagem com tamanho alvo
        img = load_img(img_path, target_size=target_size)
        
        # Converter a imagem em um array numpy
        img_array = img_to_array(img)
        
        # Adicionar uma dimensão extra para simular um lote (batch)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Criar o ImageDataGenerator para pré-processamento
        datagen = ImageDataGenerator(rescale=1./255)
        
        # Aplicar o ImageDataGenerator na imagem carregada
        img_iterator = datagen.flow(img_array)
        
        # Acessar a imagem processada
        processed_image = img_iterator.next()
        images_batch.append(processed_image[0])
        filenames_batch.append(filename)

    return np.array(images_batch), filenames_batch

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Adicionar uma dimensão extra para simular um lote
    img_array = np.expand_dims(img_array, axis=0) 
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy() 

def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    #conv_layer = model.get_layer(layer_name)
    #heatmap_model = Model([model.inputs], [conv_layer.output, model.output])
    heatmap_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name is not None:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap

def ScoreCam(model, img_array, layer_name, max_N=-1):
    # Adicionar uma dimensão extra para simular um lote
    img_array = np.expand_dims(img_array, axis=0) 
    
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsample to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

# Função para sobrepor o mapa de calor na imagem original
def superimpose_heatmap_on_image_0(img_src, heatmap, alpha=0.4, beta=1):
    img=copy.deepcopy(img_src)
    img = np.uint8(255 * img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap= cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img * beta
    # scale 0 to 255  
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8) 
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    
    return superimposed_img

def superimpose_heatmap_on_image(img_src, heatmap, true_label, pred_label, alpha=0.4, beta=1, verbose=0):
    img=copy.deepcopy(img_src)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap= cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img * beta
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8) # scale 0 to 255  
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    if verbose==1:
        fig, axs = plt.subplots(ncols=3, figsize=(9, 4))
    
        axs[0].imshow(img_src)
        axs[0].set_title("original image")
        axs[0].axis("off")
        
        axs[1].imshow(heatmap)
        axs[1].set_title("heatmap")
        axs[1].axis("off")
        
        axs[2].imshow(superimposed_img)
        axs[2].set_title("superimposed image")
        axs[2].axis("off")
        
        title =  "True label: " + true_label + " - Predicted label: " + pred_label
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    return superimposed_img
    
    
def act_max(model, image, layer_name, filter_numbers, verbose=0):
    img_array = np.expand_dims(image, axis=0)
    
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)
    
    # Gerar as ativações
    activations = intermediate_layer_model.predict(img_array)
    
    #Visualização de Filtros Específicos
    filter_activation = activations[0, :, :, filter_numbers]

    filter_activation = activations[0, :, :, filter_numbers]

    # As ativações têm a forma (1, 14, 14, 512) para a camada 'block5_conv3'
    # Vamos visualizar o filtro com a máxima ativação
    max_activation = np.max(activations, axis=-1)[0]
    
    if verbose==1:
        print(layer_name)
        fig, axs = plt.subplots(ncols=3, figsize=(9, 4))
    
        axs[0].imshow(image)
        axs[0].set_title("original image")
        axs[0].axis("off")
        
        axs[1].imshow(filter_activation, cmap='viridis')
        axs[1].set_title("Filtros Específicos")
        axs[1].axis("off")
    
        # Exibir a imagem da ativação
        axs[2].imshow(max_activation, cmap='viridis')
        axs[2].set_title("imagem da ativação")
        axs[2].axis("off")
        
        #title =  "True label: " + true_label + " - Predicted label: " + pred_label
        #plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    return max_activation
    
    
def get_occlusion_map(model,image,window_size=20,step_size=5, verbose=0):
    img_array = np.expand_dims(image, axis=0)
    
    # Define the occlusion window size and step size
    window_size = window_size
    step_size = step_size

    # Create a blank image to store the results
    importance_map = np.zeros((img_array.shape[1], img_array.shape[2]))

    # Iterate over the image and occlude a portion of it at a time
    for i in range(0, img_array.shape[1], step_size):
        for j in range(0, img_array.shape[2], step_size):
            # Occlude the image
            occluded_image = img_array.copy()
            occluded_image[:, i:i+window_size, j:j+window_size, :] = 0

            # Get the model's output on the occluded image
            occluded_prediction = model.predict(occluded_image)

            # Get the model's output on the unoccluded image
            unoccluded_prediction = model.predict(img_array)

            # Calculate the difference between the two outputs
            #diff = occluded_prediction - unoccluded_prediction
            # Considera o rótulo alvo
            target_label = np.argmax(unoccluded_prediction[0])  # Rótulo de maior confiança
            diff = occluded_prediction[0][target_label] - unoccluded_prediction[0][target_label]

            # Store the difference in the importance map
            importance_map[i:i+window_size, j:j+window_size] = diff
    if verbose==1:     
        plt.imshow(importance_map, cmap='jet')
        plt.title("Occlusion Map")
        plt.show()
    return importance_map

def display_cam_grid(images, filename, classes, model, conv_layer_name, predict, n_cols =6, alpha=0.4):
    fig, axs = plt.subplots(ncols=n_cols, nrows=5, figsize=(25, 9))
    
    for i in range(n_cols):        
        img = images[i]
    
        # original
        axs[0, i].imshow(img)
        axs[0, i].set_title(filename[i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        if i == 0:
            axs[0, i].set_ylabel("True_label\n"+classes[i], rotation=0, ha="right")
        
        # Grad-CAM
        heatmap=make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam =superimpose_heatmap_on_image_0(img, heatmap, alpha=alpha)
        
        axs[1, i].imshow(img_grad_cam)
        axs[1, i].set_title("pred: " + predict[i])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        if i == 0:
            axs[1, i].set_ylabel("Grad-CAM", rotation=0, ha="right")
        
        # Grad-CAM++
        heatmap_pp = grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = superimpose_heatmap_on_image_0(img, heatmap_pp, alpha)
        axs[2, i].imshow(img_grad_cam_plus_plus)
        #axs[2, i].set_title("pred: " + predict[i])
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        if i == 0:
            axs[2, i].set_ylabel("Grad-CAM++", rotation=0, ha="right")
    
        # Score-CAM
        heatmap_psc = ScoreCam(model, img, conv_layer_name)
        img_score_cam = superimpose_heatmap_on_image_0(img, heatmap_psc, alpha)
        axs[3, i].imshow(img_score_cam)
        #axs[3, i].set_title("pred: " + predict[i])
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        if i == 0:
            axs[3, i].set_ylabel("Score-CAM", rotation=0, ha="right")
    
        # max activation
        act = act_max(model, img, conv_layer_name, filter_numbers=10)
    
        axs[4, i].imshow(act)
        #axs[4, i].set_title("pred: " + predict[i])
        axs[4, i].set_xticks([])
        axs[4, i].set_yticks([])
        if i == 0:
            axs[4, i].set_ylabel("Max Activation", rotation=0, ha="right")
    
    plt.show()
    
def get_image_cams(images, filename, classes, model, conv_layer_name, predict, batch_size=6):
    total_batches = len(images) // batch_size + (len(images) % batch_size > 0)

    for i in range(total_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        
        batch_images = images[start_index:end_index]
        print(len(batch_images))
        batch_classes = classes[start_index:end_index]
        batch_predicts = predict[start_index:end_index]

        display_cam_grid(batch_images, batch_classes, filename, model, conv_layer_name, batch_predicts, n_cols = len(batch_images), alpha=0.4)

def display_grid(img_array, classes, n_cols =6):
    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(25, 9))
    axs = axs.flatten()  # Para facilitar a indexação como uma lista unidimensional

    for i in range(len(img_array)):        
        img = img_array[i]
        axs[i].imshow(img)
        axs[i].set_title(classes[i])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Oculta eixos não utilizados
    for j in range(len(img_array), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def get_image_batches(images, labels, batch_size=6):
    total_batches = len(images) // batch_size + (len(images) % batch_size > 0)

    for i in range(total_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        
        batch_images = images[start_index:end_index]
        print(len(batch_images))
        batch_labels = labels[start_index:end_index]

        display_grid(batch_images, batch_labels, n_cols = len(batch_images))
        