import numpy as np
import os
from os.path import join as pjoin
from config import Config
from utils import *
from dataReader import *
from loss import *
from resUnet import *
from dcUnet import *
from main_train import *

# other usefull library
import numpy as np
import os
from utils import *
from dataReader import *
from loss import *
from resUnet import *
from dcUnet import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

print('GPUs: ', len(gpus))


def runTrain(config, model, dataset, validation, idx):
    config.name_modelagem_path = config.folder_modeling_path+ '/Train/' + str(idx)
    createFolder(config.name_modelagem_path)

    adam = keras.optimizers.Adam(0.001)
    # callbacks
    callback_learning_rate = config.get_learning_rate_reduce()
    callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=config.folder_modeling_path+ '/Train/' + str(idx)+ '/logs', histogram_freq=1, write_graph=True,
                write_images=False, update_freq='epoch'
            ),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto',
                                    baseline=None, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(config.folder_modeling_path + '/Train/' + str(idx)+ '/logs/Unet.h5', 
                                    monitor='jacard', mode='max', save_weights_only=True),
            callback_learning_rate,
        ]

    #Compilação 
    
    print("Compile")
    function_loss = config.getLoss()
    model.compile(optimizer=adam, loss=function_loss, metrics=[dice_coef, jacard, 'accuracy', tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),])
    model.summary()

    print("Inicializando treinamento..........")
    #treinamento (samples_x, samples_y)
    model.fit(dataset, epochs=config.num_epochs, validation_data=validation, callbacks=callbacks, workers=8)


def run_main(config):
    print('inicio')

    '''# CRIAR OS DIREToRIOS PARA SALVAR OS DADOS DO TREINAMENTO E TESTE   '''             
    createFolderModeling(config)

    ''' 2# Carrega a base'''
    print("2 Carregando base..........")
    dataset = DataGenerator(config.dataset_path+"train/images", config.dataset_path+"train/masks", config.batch_size, config.num_classes, config)
    validation = DataGenerator(config.dataset_path+"val/images",config. dataset_path+"val/masks", config.batch_size, config.num_classes, config)
    print(len(dataset), len(validation))

    ''' 3# Carrega o modelo da rede'''
    print("3 Carregando Modelo..........") 
    model = config.getModel()
    print(model.summary())   

    print(config.folder_modeling_path)
    dict_train = config.__dict__
    write_dict_csv(dict_train, config.folder_modeling_path + '/config_treino.txt') 

    ''' 4# treinamento '''
    print("TREINAMENTO .....................")
    
    runTrain(config, model, dataset, validation, idx=0)

if __name__ == "__main__":

  config = Config()
  run_main(config=config)

    

            