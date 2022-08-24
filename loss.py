import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

# other usefull library
import numpy as np
from sklearn.utils.extmath import cartesian
import math
from typing import Callable

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred,  tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def jacard(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred,  tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = keras.backend.sum ( y_true_f * y_pred_f)
    union = keras.backend.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def qtd_TP(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred,  tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = keras.backend.sum ( y_true_f * y_pred_f)
    count_TP = tf.reduce_sum(y_true_f)

    return intersection/count_TP

def iou_loss(y_true,y_pred):
    return 1 - jacard(y_true, y_pred)

def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred,  tf.float32)
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = keras.backend.sum(y_true_pos * (1-y_pred_pos))
    false_pos = keras.backend.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return keras.backend.pow((1-pt_1), gamma)

#Generalized Dice loss auxiliar
def generalized_dice(y_true, y_pred):

    """
    Generalized Dice Score
    https://arxiv.org/pdf/1707.03237

    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    #y_true = tf.keras.backend.flatten(y_true)
    #y_pred = tf.keras.backend.flatten(y_pred)
    y_true    = keras.backend.reshape(y_true,shape=(-1,4))
    y_pred    = keras.backend.reshape(y_pred,shape=(-1,4))
    sum_p     = keras.backend.sum(y_pred, -2)
    sum_r     = keras.backend.sum(y_true, -2)
    sum_pr    = keras.backend.sum(y_true * y_pred, -2)
    weights   = keras.backend.pow(keras.backend.square(sum_r) + keras.backend.epsilon(), -1)
    generalized_dice = (2 * keras.backend.sum(weights * sum_pr)) / (keras.backend.sum(weights * (sum_r + sum_p)))

    return generalized_dice

#Generalized Dice loss
def generalized_dice_loss(y_true, y_pred):   
    return 1-generalized_dice(y_true, y_pred)



# Funcao de soma da entropia desbalanceada com o dice   
def custom_loss(functionA, functionB, multA=1, multB=1):

    def sum_loss(y_true, y_pred):
        
        """
        The final loss function consists of the summation of two losses "GDL" and "CE"
        with a regularization term.
        """
        
        return multA * functionA(y_true, y_pred) + multB * functionB(y_true, y_pred)
    return sum_loss

