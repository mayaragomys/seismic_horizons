# for tensorflow
import tensorflow as tf
from tensorflow import keras

# other usefull library
import numpy as np
import pandas as pd
import os
from os.path import join as pjoin
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from loss import *
from resUnet import *
from dcUnet import *
from config import *
from dataReader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#if __name__ == '__main__':
def run_test(config, idx=0):
    # Carrega o modelo com os pesos
    model = config.getModel()
    model.summary()
    model.load_weights(config.folder_modeling_path + "/Train/" + str(idx) + "/logs/Unet.h5")

    #splits = ['test1', 'test2']
    splits = ['test1', 'test2']

    list_pred = []
    list_y = []
    for sdx, split in enumerate(splits):                
        #for split in splits:
        base_teste = DataGen(config, split, config.dataset_path, batch_size=1, image_size=config.image_width) 
        print(split + ': ', len(base_teste))
        createFolder(config.folder_modeling_path + '/Test/' + split + '/'+ str(idx) +'/')

        for (images, masks, direction, number) in base_teste:
            images = images[0]
            masks = masks[0]
            
            #print('img: ', direction, number, np.array(images).shape, np.array(masks).shape)
            img = np.expand_dims(images, axis=0)
            result = model.predict(img) 

            name = direction + str(number)
            path = config.folder_modeling_path + '/Test/' + split + '/'+ str(idx) +'/'

            save_result(img, masks, result[0].copy(), name, path)
            list_pred.append(result[0][:,:,0])
            list_y.append(masks[:,:,0])
                    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data", np.array(list_pred).shape)
    results = calculate_metrics(list_y, list_pred)
    print("test loss, test acc:", results)
    write_dict_csv(results, config.folder_modeling_path + '/Test/' + str(idx) +'1_resultado_geral_' + config.name_modelagem + '.csv') 
    

if __name__ == '__main__':
    
    ##Carrega as configurações
    logs_paths = ["ResUnet/teste_ResUnet_focal_tversky_aumentation/"]
                    
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
        #config.dilatation = int(config_dict["dilatation"])

        if config_dict["augmentation"] == 'None':
            config.augmentation = None
        else:
            config.augmentation = True
        if config_dict["patche"] == 'False':
            config.patche = False
        else:
            config.patche = True

        run_test(config, idx=0)
