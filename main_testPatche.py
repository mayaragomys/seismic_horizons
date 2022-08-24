# for tensorflow
import tensorflow as tf
from tensorflow import keras

# other usefull library
import numpy as np
import math
import pandas as pd
import os
from os.path import join as pjoin
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import *
from loss import *
from resUnet import *
from dcUnet import *
from config import *
from dataReader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def reshapeImg(patches, shape, size_patche):
    qtd_linha = math.ceil(shape[0]/config.ksize)
    qtd_coluna = math.ceil(shape[1]/config.ksize)
    linha = qtd_linha*config.ksize
    coluna = qtd_coluna*config.ksize

    img = np.empty([linha, coluna], dtype=float)
    cont = 0
    for l in range(qtd_linha):
        for c in range(qtd_coluna):
            lin = l*size_patche
            col = c*size_patche
            img[lin:(lin+size_patche),col:(col+size_patche)] = patches[cont][:,:,0]
            cont += 1

    img = tf.expand_dims(img, -1)
    return img


def run_test(config, idx):    
    ''' Executa o treinamento '''    
    model = config.getModel()
    model.summary()
    model.load_weights(config.folder_modeling_path + "/Train/" + str(idx) + "/logs/Unet.h5")

    splits = ['test1', 'test2']
    list_pred = []
    list_y = []
    for sdx, split in enumerate(splits):
        createFolder(config.folder_modeling_path + '/Test/' + split + '/'+ str(idx) +'/')           
        #for split in splits:
        base_teste = DataGenPatche(config, split, config.dataset_path, batch_size=1, image_size=config.image_width) 
        print(split + ': ', len(base_teste))
        
        for (images, masks, directions, numbers, indexs, shape) in base_teste:
            results_patches = []
            print(np.array(images).shape, np.array(masks).shape, directions, numbers, shape)
            for p in range(len(images)):
                
                img = images[p]
                img = np.expand_dims(img, axis=0)
                result = model.predict(img)
                results_patches.append(result[0])

                list_pred.append(result[0][:,:,0])
                list_y.append(masks[p][:,:,0].copy())

            reconstructed_pred = reshapeImg(results_patches, shape, config.ksize)
            reconstructed_label = reshapeImg(masks, shape, config.ksize)
            reconstructed_img = reshapeImg(images, shape, config.ksize)
            
            name = directions + str(numbers)
            path = config.folder_modeling_path + '/Test/' + split + '/'+ str(idx) +'/'
            print("name:", path, name)
            print(reconstructed_img.shape, reconstructed_label.shape, reconstructed_pred.shape)
            save_result(reconstructed_img, reconstructed_label, reconstructed_pred, name, path)
                    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data", np.array(list_pred).shape)
    results = calculate_metrics(list_y, list_pred)
    print("test loss, test acc:", results)
    write_dict_csv(results, config.folder_modeling_path + '/Test/' + str(idx) +'1_resultado_geral_' + config.name_modelagem + '.csv') 
    

if __name__ == '__main__':
    
    ##Carrega as configurações
    logs_paths = [  "resUnet/1_resUnet_focal_tversky_patche_128/"
                    ]
                        
    for log_path in logs_paths:
        folder_modeling_path = "/segmentation/" + log_path
        config_dict = read_config_dict(folder_modeling_path)
        
        config = Config()
        config.folder_modeling_path = config_dict["folder_modeling_path"]
        config.image_width = int(config_dict["image_width"])
        config.image_height = int(config_dict["image_height"])
        config.num_channels = int(config_dict["num_channels"])
        config.name_model = config_dict["name_model"]
        config.dataset_path = "/data/mayaragomes/dissertacao/"
        config.name_modelagem = config_dict["name_modelagem"]
        config.name_loss = config_dict["name_loss"]    
        config.ksize = int(config_dict["ksize"])
        config.strides = int(config_dict["strides"])
        config.input_size = (config.image_height, config.image_width, config.num_channels)
        config.batch_size = 1

        if config_dict["augmentation"] == 'None':
            config.augmentation = None
        else:
            config.augmentation = True
        if config_dict["patche"] == 'False':
            config.patche = False
        else:
            config.patche = True

        run_test(config, idx=0)
