from loss import *
from resUnet import *
from dcUnet import *
from tensorflow import keras


class Config:
    def __init__(self):
        self.prefixo = 'teste_'
        self.image_width = 256
        self.image_height = 256
        self.input_size = (256, 256, 1)
        self.batch_size = 8
        self.num_classes = 1
        self.num_channels = 1
        self.num_epochs = 300
        self.resize = False
        self.augmentation = True
        self.neighborhood = 8
        self.split = False   
        self.EarlyStopping = True     
        self.name_model = "ResUnet"
        self.dataset_path = "/data/dataSet_aug/"
        self.folder_modeling_path = "/segmentation/"
        self.name_modelagem = ''
        self.name_loss = "focal_tversky"    
        self.inline = True
        self.crossline = True
        self.ksize = 256
        self.strides = 256
        self.padding = 'SAME'
        self.patche = False
        self.baselist = None       
        self.learning_rate=0.001
        self.learning_rate_callback = ""
        self.dilatation = 2


    def getLoss(self, w=None):
        if self.name_loss == "custom_loss":
            function_loss = custom_loss(focal_tversky, iou_loss)
        elif self.name_loss == "generalized_dice_loss":
            function_loss = generalized_dice_loss
        elif self.name_loss == "focal_tversky":
            function_loss = focal_tversky
        elif self.name_loss == "iou_loss":
            function_loss = iou_loss
        elif self.name_loss == "cross_entropy_w":
            function_loss = iou_loss
        else:
            function_loss = keras.losses.BinaryCrossentropy()

        return function_loss
    
    def getModel(self):
        input_size = (self.image_height, self.image_width, self.num_channels)
        self.input_size = input_size
        if self.name_model == "dcunet":
            print('modelo DC-Unet')
            model = DCUNet(self.image_height, self.image_width, self.num_channels) 
        else:
            #Se modelo da ResUNet
            print('modelo ResUnet')
            model = ResUNet(self.input_size)

        return model

    def get_learning_rate_reduce(self, sample_count=None):
        
        callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                                        factor=0.1, 
                                                        patience=10, 
                                                        min_lr=0.0001, 
                                                        verbose=1)

        return callback


