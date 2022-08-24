import numpy as np
import os
from os.path import join as pjoin
import tensorflow as tf
from tensorflow import keras
import collections
import cv2
from utils import *
from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DataGen(keras.utils.Sequence):
    def __init__(self, config, split, path, batch_size=8, image_size=256, augmentations=None):
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.split = split
        self.mean = 0.000941 # average of the training data
        self.sections = collections.defaultdict(list)
        self.config = config
        self.augmentations = augmentations

        if 'test' not in self.split: 
          # Normal train/val mode
          self.seismic = np.load(pjoin((path + 'data'),'train','train_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'train','train_labels.npy'))
        elif 'test1' in self.split:
          self.seismic = np.load(pjoin((path + 'data'),'test_once','test1_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'test_once','test1_labels.npy'))
        elif 'test2' in self.split:
          self.seismic = np.load(pjoin((path + 'data'),'test_once','test2_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'test_once','test2_labels.npy'))
        else:
          raise ValueError('Unknown split.')

        if 'test' not in self.split:
          # We are in train/val mode. Most likely the test splits are not saved yet, 
          # so don't attempt to load them.  
          for split in ['train', 'val', 'train_val']:
            # reading the file names for 'train', 'val', 'trainval'""
            path_save = pjoin((path + 'data'), 'splits', 'section_' + split + '.txt')
            file_list = tuple(open(path_save, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        elif 'test' in split:
          # We are in test mode. Only read the given split. The other one might not 
          # be available. 
          path_save = pjoin((path + 'data'), 'splits', 'section_' + split + '.txt')
          file_list = tuple(open(path_save,'r'))
          file_list = [id_.rstrip() for id_ in file_list]
          self.sections[split] = file_list
        else:
          raise ValueError('Unknown split.')

        self.ids = np.arange(len(self.sections[self.split]))
                
    def __load__(self, index):
        '''Fluxo: carregamento, e pre-processamento
        return: 
            image (array) -> seção
            mask (array) -> mascaras binarias
            direction (array) -> direção, se inline (i) ou crossline (x)
            number (array) -> número da seção
        '''
        #Carrega as seções
        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')
        #Se inline
        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:]
        #Se não se crossline
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:]

        ## Transform
        im, lbl = self.transform(im, lbl)

        image = []
        mask = []            
        #Se data augmentation
        # Pre-processamento da seção sem Data augementation: Resize e binarização da label
        img, mas = self.pre_process_resize(im.copy(), lbl.copy())
        image.append(img)
        mask.append(mas)

        aug = None
        if self.augmentations is not None:
            # Data augementation
            im_aug, lbl_aug = self.augmentations(img[:,:,0].copy(), mas[:,:,0].copy())
            im_aug = np.expand_dims(im_aug, axis=-1) 
            lbl_aug = np.expand_dims(lbl_aug, axis=-1)   
            image.append(im_aug)
            mask.append(lbl_aug)  
            
        return image, mask, direction, number

    def pre_process_resize(self, im, lbl):
        '''Pre-processamento da seção: Resize e binarização da label
            im -> seção
            lbl -> labels da seção
        return: 
            im -> seção
            lbl -> labels da seção
        '''
        ##resize
        image = cv2.resize(im, (self.image_size,self.image_size))
        image = np.expand_dims(image, axis=-1)        
        mask = cv2.resize(lbl, (self.image_size,self.image_size), interpolation=cv2.INTER_NEAREST)
        #Binariza
        mask = self.binary(mask)
        mask = np.expand_dims(mask, axis=-1)

        return np.array(image), np.array(mask) 
            
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

    def __getitem__(self, index):
        img, mask, direction, number = self.__load__(index)

        return img, mask, direction, number

    def loadDataSet(self):
        files = self.ids
        
        image = []
        mask  = []    
        directions = []
        numbers = []    
        for id_name in files:
              
            _img, _mask, direction, number = self.__load__(id_name)

            for idx, elem in enumerate(_img):
                image.append(elem)
                mask.append(_mask[idx])
                directions.append(direction)
                numbers.append(number)

        return np.array(image), np.array(mask), directions, numbers


    def transform(self, img, lbl):
      """
      Transforma os dados na posição correta
      :parâmetro img: dados da imagem  
      :parâmetro lbl: dados das labels 
      return: img(imagem), lbl(label)
      """ 
      img -= self.mean

      # to be in the BxCxHxW: 
      img, lbl = img.T, lbl.T 
                  
      return np.array(img), np.array(lbl)   
    
    def binary(self, img):
        '''Pre-processamento da seção: Resize e binarização da label
            im -> labels da seção
        return: 
            binary -> labels da seção binarizadas
        '''
        binary = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
        for i in range(0, img.shape[0]-1):
            for j in range(0, img.shape[1]-1):
                # Compara com o pixel abaixo 
                if (i < img.shape[0]) and (img[i][j] != img[i+1][j]):
                    binary[i][j] = 1
                # Compara com o pixel do lado esquerdo
                elif (j != 0) and (img[i][j] != img[i][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel acima 
                elif (i != 0) and (img[i][j] != img[i-1][j]):
                    binary[i][j] = 1
                # Compara com o pixel do lado direito
                elif (j < img.shape[1]) and (img[i][j] != img[i][j+1]):
                    binary[i][j] = 1
                # Compara com o pixel abaixo na diagonal esquerda
                elif (i < img.shape[0] and j != 0) and (img[i][j] != img[i+1][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel acima na diagonal esquerda
                elif (i != 0 and j != 0) and (img[i][j] != img[i-1][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel abaixo na diagonal direita
                elif (i < img.shape[0] and j < img.shape[1]) and (img[i][j] != img[i+1][j+1]):
                    binary[i][j] = 1
                # Compara com o pixel acima na diagonal direita
                elif (i != 0 and j < img.shape[1]) and (img[i][j] != img[i-1][j+1]):
                    binary[i][j] = 1

        return binary

class DataGenPatche(keras.utils.Sequence):
    def __init__(self, config, split, path, batch_size=8, image_size=256, augmentations=None):
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.split = split
        self.mean = 0.000941 # average of the training data
        self.sections = collections.defaultdict(list)
        self.config = config
        self.augmentations = augmentations

        if 'test' not in self.split: 
          # Normal train/val mode
          self.seismic = np.load(pjoin((path + 'data'),'train','train_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'train','train_labels.npy'))
        elif 'test1' in self.split:
          self.seismic = np.load(pjoin((path + 'data'),'test_once','test1_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'test_once','test1_labels.npy'))
        elif 'test2' in self.split:
          self.seismic = np.load(pjoin((path + 'data'),'test_once','test2_seismic.npy'))
          self.labels = np.load(pjoin((path + 'data'),'test_once','test2_labels.npy'))
        else:
          raise ValueError('Unknown split.')

        if 'test' not in self.split:
          # We are in train/val mode. Most likely the test splits are not saved yet, 
          # so don't attempt to load them.  
          for split in ['train', 'val', 'train_val']:
            # reading the file names for 'train', 'val', 'trainval'""
            path_save = pjoin((path + 'data'), 'splits', 'section_' + split + '.txt')
            file_list = tuple(open(path_save, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        elif 'test' in split:
          # We are in test mode. Only read the given split. The other one might not 
          # be available. 
          path_save = pjoin((path + 'data'), 'splits', 'section_' + split + '.txt')
          file_list = tuple(open(path_save,'r'))
          file_list = [id_.rstrip() for id_ in file_list]
          self.sections[split] = file_list
        else:
          raise ValueError('Unknown split.')

        self.ids = np.arange(len(self.sections[self.split]))

    def __getitem__(self, index):
        img, mask, direction, number, idx, shape = self.__load__(index)

        return img, mask, direction, number, idx, shape
                
    def __load__(self, index):
        '''Fluxo: carregamento, e pre-processamento
        return: 
            image (array) -> seção
            mask (array) -> mascaras binarias
            direction (array) -> direção, se inline (i) ou crossline (x)
            number (array) -> número da seção
        '''
        #Carrega as seções
        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')
        #Se inline
        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:]
        #Se não se crossline
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:]

        ## Transform
        im, lbl = self.transform(im, lbl)

        image = []
        mask = []
        indexs = []    
        cont = 0        

        # Pre-processamento da seção sem Data augementation: Resize e binarização da label
        shape = im.shape
        img, mas = self.pre_process(im.copy(), lbl.copy())
        for i, m in zip(img, mas):
            image.append(i)
            mask.append(m)
            indexs.append(cont)
            cont += 1

        if self.augmentations is not None:
            # Data augementation
            im_aug, lbl_aug = self.augmentations(im.copy(), lbl.copy())
            img, mas = self.pre_process(im_aug, lbl_aug)  
            for i, m in zip(img, mas):
                image.append(i)
                mask.append(m)
                indexs.append(cont)
                cont += 1 
            
        return image, mask, direction, number, indexs, shape

    def pre_process(self, im, lbl):
        '''Pre-processamento da seção: binarização da label e divisão dos patches
            im -> seção
            lbl -> labels da seção
        return: 
            im -> seção
            lbl -> labels da seção
        '''        
        ksize_rows = self.config.ksize
        ksize_cols = self.config.ksize
        strides_rows = self.config.strides
        strides_cols = self.config.strides

        # The size of sliding window
        ksizes = [1, ksize_rows, ksize_cols, 1] 

        # How far the centers of 2 consecutive patches are in the image
        strides = [1, strides_rows, strides_cols, 1]

        # The document is unclear. However, an intuitive example posted on StackOverflow illustrate its behaviour clearly. 
        # http://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
        rates = [1, 1, 1, 1] # sample pixel consecutively

        # padding algorithm to used
        padding = self.config.padding # or 'SAME' 'VALID'

        ##expande para 1 canal
        image = np.expand_dims(im, axis=-1) 
        image = tf.expand_dims(image, 0)       
        #Binariza
        mask = self.binary(lbl)
        mask = np.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, 0)

        # extração de patches
        images_patches = tf.image.extract_patches(images=image, sizes=ksizes, strides=strides, rates=rates, padding=padding)
        masks_patches = tf.image.extract_patches(images=mask, sizes=ksizes, strides=strides, rates=rates, padding=padding)

        images = []
        masks = []

        for imgs, label in zip(images_patches, masks_patches):
            linha = imgs.shape[0]
            coluna = imgs.shape[1]
            for r in range(linha):
                for c in range(coluna):
                    images.append(tf.reshape(imgs[r,c],shape=(self.config.ksize,self.config.ksize,1)).numpy().astype("float64"))
                    masks.append(tf.reshape(label[r,c],shape=(self.config.ksize,self.config.ksize,1)).numpy().astype("int8"))

        return np.array(images), np.array(masks) 
            
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

    def loadDataSet(self):
        files = self.ids
        
        image = []
        mask  = []    
        directions = []
        numbers = []    
        for id_name in files:
              
            _img, _mask, direction, number, indexs, shape = self.__load__(id_name)

            for idx, elem in enumerate(_img):
                image.append(elem)
                mask.append(_mask[idx])
                directions.append(direction)
                numbers.append(number)

        return np.array(image), np.array(mask), directions, numbers


    def transform(self, img, lbl):
      """
      Transforma os dados na posição correta
      :parâmetro img: dados da imagem  
      :parâmetro lbl: dados das labels 
      return: img(imagem), lbl(label)
      """ 
      img -= self.mean

      # to be in the BxCxHxW: 
      img, lbl = img.T, lbl.T 
                  
      return np.array(img), np.array(lbl)   
    
    def binary(self, img):
        '''Pre-processamento da seção: binarização da label, dimensão 2
            im -> labels da seção
        return: 
            binary -> labels da seção binarizadas
        '''
        binary = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
        for i in range(0, img.shape[0]-1):
            for j in range(0, img.shape[1]-1):
                # Compara com o pixel abaixo 
                if (i < img.shape[0]) and (img[i][j] != img[i+1][j]):
                    binary[i][j] = 1
                # Compara com o pixel do lado esquerdo
                elif (j != 0) and (img[i][j] != img[i][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel acima 
                elif (i != 0) and (img[i][j] != img[i-1][j]):
                    binary[i][j] = 1
                # Compara com o pixel do lado direito
                elif (j < img.shape[1]) and (img[i][j] != img[i][j+1]):
                    binary[i][j] = 1
                # Compara com o pixel abaixo na diagonal esquerda
                elif (i < img.shape[0] and j != 0) and (img[i][j] != img[i+1][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel acima na diagonal esquerda
                elif (i != 0 and j != 0) and (img[i][j] != img[i-1][j-1]):
                    binary[i][j] = 1
                # Compara com o pixel abaixo na diagonal direita
                elif (i < img.shape[0] and j < img.shape[1]) and (img[i][j] != img[i+1][j+1]):
                    binary[i][j] = 1
                # Compara com o pixel acima na diagonal direita
                elif (i != 0 and j < img.shape[1]) and (img[i][j] != img[i-1][j+1]):
                    binary[i][j] = 1

        return binary

    
class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_path, label_path, batch_size, num_class, config):
        self.config=config
        self.num_class=num_class
        self.batch_size=batch_size
        imgdir=os.listdir(img_path)
        labeldir=os.listdir(label_path)
        assert len(imgdir)==len(labeldir),"the count of img and label is not equality"
        self.imgdir=[]
        self.labeldir=[]
        for l in imgdir:
            self.imgdir.append(img_path+"/"+l)
        for l in imgdir:
            self.labeldir.append(label_path+"/"+l)
        self.image_id=np.arange(len(self.imgdir))

    def __len__(self):
        return int(np.ceil(len(self.imgdir)/float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.image_id)

    def __getitem__(self,idx):
        id=self.image_id[idx*self.batch_size:(idx+1)*self.batch_size]
        images=[]
        labels=[]
        for i in id:
            img= np.load(self.imgdir[i])
            img = np.asarray(img, "f")
            label = np.load(self.labeldir[i])
            label = np.array(label, dtype=int)
            if self.config.dilatation:
                label = dilatation(label, self.config.dilatation)
            label=np.asarray(label,'f')

            images.append(img)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        
        return images,labels