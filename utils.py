import numpy as np
import os
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from metrics import *
import cv2


def createFolder(path):
    ''' Cria uma pasta se ela não existir
        Args:
            path (string) -> caminho da pasta a ser criada
    '''
    pasta = path
    if os.path.isdir(pasta): # vemos de este diretorio ja existe
        print ('Ja existe uma pasta com esse nome!')
    else:
        os.mkdir(pasta) # aqui criamos a pasta caso nao exista
        print ('Pasta criada com sucesso!')


def split_train_val(path_drive, per_val=0.1):
    """
    Separa a base em treino e validacao
    :parametro per_val: representar a proporcao do conjunto de dados a ser incluido na divisao de validacao (entre 0,0 e 1,0)
    """
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = np.load(pjoin((path_drive + 'data'), 'train', 'train_labels.npy'))
    
    i_list = list(range(labels.shape[0]))
    i_list = ['i_'+str(inline) for inline in i_list]
    
    x_list = list(range(labels.shape[1]))
    x_list = ['x_'+str(crossline) for crossline in x_list]
    
    list_train_val = i_list + x_list
    
    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    #write to files to disK:
    file_object = open(
        pjoin((path_drive + 'data'), 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(
        pjoin((path_drive + 'data'), 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin((path_drive + 'data'), 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
    

def write_dict_csv(dct, path):
    ''' Cria uma pasta se ela não existir
        Args:
            dct (Object) -> dicionário
            path (string) -> caminho para salvar o dicionário
    '''
    fo = open(path, "w")
    for k, v in dct.items():
        fo.write(str(k) + ':'+ str(v) + ',\n')

    fo.close()


def backGround_none(img):
    img_aux = img.copy()
    img_aux[img_aux==0]=None
    return img_aux

def label_reverse(img):
    copy = np.zeros((img.shape[0], img.shape[1]))
    copy[img==0] = 1
    return copy

def save_compare_images(titulo, img, gt_none, gt_reverse, y_pred_reverse, path):

    cmap = ListedColormap(["red"])
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(img[:,:,0], cmap="gray")
    plt.title('Amplitude')
    plt.subplot(1, 4, 2)
    plt.imshow(img[:,:,0], 'gray', interpolation='none')
    plt.imshow(gt_none, cmap, interpolation='none', alpha=0.9)
    plt.subplot(1, 4, 3)
    plt.imshow(gt_reverse*255, cmap="gray")
    plt.title('Ground Truth')    
    plt.subplot(1, 4, 4)
    plt.imshow(y_pred_reverse*255, cmap="gray")
    plt.title('Binary')   
    plt.suptitle(titulo)
    plt.savefig(path + "_.png", format="png")
    plt.close()

def save_result(img, y, pred, name, path):
    ''' Cria uma pasta se ela não existir
        Args:
            img (ndarray) -> Seção
            y (ndarray) -> Ground Truth
            pred (ndarray) -> predição
            name (string) -> nome da imagem
            path (string) -> caminho para salvar a imagem com os resultados
    '''
    image_size_c = y.shape[1]  
    image_size_l = y.shape[0]     

    gt = np.reshape(y, (image_size_c, image_size_l))
    gt = np.array(gt, dtype=np.float32)
    y_pred = np.reshape(pred, (image_size_c, image_size_l))
    
    # arredondando os valores dos predicts
    y_pred = np.round(y_pred, 0)
    # Calcula métricas
    jacard = calculate_IoU(gt, y_pred)
    dice = calculate_dice(gt, y_pred)

    # Ground Truth com fundo None
    gt_none = backGround_none(gt)
    # Troca as labels
    gt_reverse = label_reverse(gt)
    y_pred_reverse = label_reverse(y_pred)

    # Salva os resultados das imagens
    titulo =  name + '-- Dice:' + str(dice) + ', ioU :' + str(jacard)
    save_compare_images(titulo, img, gt_none, gt_reverse, y_pred_reverse, path + name)


 
def save_base(images, masks, path, PHASE="train"):
    ''' Salva as imagens no diretorio de referencia no formato .npy
        Args:
            images (ndarray) -> imagens
            masks (ndarray) -> imagens do Ground Truth
            path (string) -> caminho para salvar a imagem com os resultados
            PHASE (string) -> se base de treino ou validação.  [train, val)
    '''
    # Salva as imagens no diretorio de referencia no formato .npy
    image_path = path + PHASE + "/images"
    mask_path = path + PHASE + "/masks"
    createFolder(path + PHASE)
    createFolder(image_path)
    createFolder(mask_path)
    idx = 0
    for img, mask in zip(images, masks):
        np.save(image_path + "/" + str(idx) + ".npy", img)
        np.save(mask_path + "/" + str(idx) + ".npy", mask)
        idx += 1


def createFolderModeling(config):
    '''# Criando pastas para salvar os dados gerados de teste'''  
    # caminho da modelagem de acordo com as configuracoes
    # Rede + loss 
    path = config.prefixo + config.name_model+'_'+config.name_loss
    # Rede + loss + aumentation (se data aumentation)
    if config.augmentation:
        path += '_aumentation' 
    # Rede + loss + aumentation (se data aumentation) + patche + ksize
    if config.patche:
        path += '_patche_' + str(config.ksize)

    config.name_modelagem = path

    # Criando Pasta da modelagem 
    config.folder_modeling_path += config.name_model + '/'
    createFolder(config.folder_modeling_path)
    config.folder_modeling_path += path
    createFolder(config.folder_modeling_path)
    
    createFolder(config.folder_modeling_path + '/logs')
    createFolder(config.folder_modeling_path + '/Train')
    createFolder(config.folder_modeling_path + '/Train/logs')
    createFolder(config.folder_modeling_path + '/Test')
    createFolder(config.folder_modeling_path + '/Test/test1')
    createFolder(config.folder_modeling_path + '/Test/test2')


def read_config_dict(log_path):
    # Carrega o arquivo e adiciona as informações no dicionário 
    dict_path = os.path.join(log_path, "config_treino.txt")
    dicionario = {}
    with open(dict_path, "r") as f:
        for line in f:
            (key, value) = line.replace("'", "").replace(", \n", "").split(":")
            aux_value = value.split(",")
            dicionario[key] = aux_value[0]
    return dicionario

def dilatation(labels, num_kernel):
    kernel = np.ones((num_kernel,num_kernel), np.uint8) 

    imagens_dilatation = []
    for img in labels:
        #img = np.reshape(img, (256, 256))
        img = img.astype('uint8')
        img_dilation = cv2.dilate(img, kernel, iterations=1) 
        imagens_dilatation.append(img_dilation)

    return imagens_dilatation
