import numpy as np
from utils import *
from config import Config
from dataReader import *
from augmentation import *


def saveDataSet(config):
    ''' Gera e salva a base de acordo com as configurações'''
    #Divide a base de treinamento em train e val
    if config.split:
        split_train_val(config.dataset_path)

    data_aug = None
    if config.augmentation:
        # data augmentation Horizontally Flip
        data_aug = Compose([HorizontallyFlip()])
    
    #Carrega a base de treino e validacao    
    train_gen = None
    valid_gen = None
    name = ''
    if config.patche:
        name = str(config.ksize) + '_dataSetPatche/'
        if config.augmentation:
            name = str(config.ksize) + '_dataSetPatche_aug/'
        train_gen = DataGenPatche(config, 'train', config.dataset_path, batch_size=config.batch_size, image_size=config.image_width, augmentations=data_aug)
        valid_gen = DataGenPatche(config, 'val', config.dataset_path, batch_size=config.batch_size, image_size=config.image_width)
    else:
        name = 'dataSet/'
        if config.augmentation:
            name = 'dataSet_aug/'
        train_gen = DataGen(config, 'train', config.dataset_path, batch_size=config.batch_size, image_size=config.image_width, augmentations=data_aug)
        valid_gen = DataGen(config, 'val', config.dataset_path, batch_size=config.batch_size, image_size=config.image_width)

    #salvar dados do treinamento e validacao    
    print("LIST...")
    X_train, Y_train, _, _ = train_gen.loadDataSet()
    X_val, Y_val, _, _ = valid_gen.loadDataSet()
    path = config.dataset_path + 'data/' + name
    createFolder(path)

    save_base(X_train, Y_train, path, PHASE="train")
    save_base(X_val, Y_val, path, PHASE="val")
    print(name, " - Treino size: ", len(Y_train))


if __name__ == '__main__':
    ''''''        
    # definindo se data augementation e verdadeiro ou falso
    aug = True
    # Se patches defina True senão None
    patche = True
    # Se patches defina um valor para k senão None
    k = 255
    
    #Carrega as configuracoes
    config = Config()
    config.dataset_path = "/dissertacao/"
    config.augmentation = aug
    config.image_width = k
    config.image_height = k
    config.input_size = (k, k, 1)
    config.ksize = k
    config.strides = k
    config.patche = patche

    print("INICIANDO...")
    saveDataSet(config)
