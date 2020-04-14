from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.backend as K
from keras.layers import *
from keras.models import Sequential, Model
from keras import applications
import os
import cv2
import glob
from PIL import Image
from utils import get_augmented
from keras.models import load_model
from metrics import iou, iou_thresholded, jaccard_coef, dice_coef

img_width, img_height = 1024, 1024

batch_size = 1

tiles_samples = 0
labels_samples = 0

nb_epoch = 90
# initial_image_dir = '/home/drone/adithya/unet/images'
initial_image_dir = '/home/ndnbase/adithya/unet/images'
tiles_dir = initial_image_dir + '/hr_tiles'
labels_dir = initial_image_dir + '/hr_labels'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

masks = glob.glob(labels_dir + "/*.png")
orgs = glob.glob(tiles_dir + "/*.png")
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    print("Tile: %s\nLabel: %s" % (image, mask))
    img = cv2.resize(cv2.imread(image), (1024, 1024))
    img = img / 255.0
    imgs_list.append(img)
    im = cv2.resize(cv2.imread(mask, 0), (1024, 1024))
    im = np.expand_dims(np.asarray(im, dtype=np.float32), axis=-1)
    mask = np.zeros((1024, 1024, 1))
    im = np.maximum(mask, im)
    im = im / 255.0
    masks_list.append(im)

x = np.asarray(imgs_list)
y = np.asarray(masks_list)

train_gen = get_augmented(
    x,
    y,
    batch_size=1,
    data_gen_args=dict(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant'))

#------------------------------------------------------------------------------------------
val_tiles_dir = initial_image_dir + '/hr_tiles_val'
val_labels_dir = initial_image_dir + '/hr_labels_val'
masks_val = glob.glob(val_labels_dir + "/*.png")
orgs_val = glob.glob(val_tiles_dir + "/*.png")
imgs_list_val = []
masks_list_val = []
for image, mask in zip(orgs_val, masks_val):
    print("Tile: %s\nLabel: %s" % (image, mask))
    img = cv2.resize(cv2.imread(image), (1024, 1024))
    img = img / 255.0
    imgs_list_val.append(img)
    im = cv2.resize(cv2.imread(mask, 0), (1024, 1024))
    im = np.expand_dims(np.asarray(im, dtype=np.float32), axis=-1)
    mask = np.zeros((1024, 1024, 1))
    im = np.maximum(mask, im)
    im = im / 255.0
    masks_list_val.append(im)

x_val = np.asarray(imgs_list_val)
y_val = np.asarray(masks_list_val)

val_gen = get_augmented(
    x_val,
    y_val,
    batch_size=2,
    data_gen_args=dict(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant'))

#------------------------------------------------------------------------------------------

def FCN8(nClasses=1, input_height=1024, input_width=1024,VGG_Weights_path="./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512
    
    vgg  = Model(  img_input , pool5  )
    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('sigmoid'))(o)
    
    model = Model(img_input, o)

    return model


model = FCN8()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc", iou, iou_thresholded, jaccard_coef, dice_coef])
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'fcn_%d_val_all.h5'%nb_epoch,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1),
    keras.callbacks.TensorBoard(log_dir='./logs_fcn', histogram_freq=0, batch_size=2, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
]

hist1 = model.fit_generator(
    train_gen,
    steps_per_epoch=78,
    validation_data=val_gen,
    validation_steps=20,
    callbacks=callbacks,
    epochs=nb_epoch)

# model = load_model("unet_100.h5")
filenames = [
    "93823-60786-17-sr", "93822-60787-17-sr", "93822-60786-17-sr",
    "93822-60768-17-sr", "93824-60786-17-sr"
]
for filename in filenames:
    image = tiles_dir + "/" + filename + ".png"
    im = cv2.resize(cv2.imread(image), (1024, 1024))
    im = im / 255.0
    validation_image = np.zeros((1, 1024, 1024, 3))
    validation_image[0, :, :, :] = im
    predicted_image = model.predict(validation_image)
    mask = np.zeros((1024, 1024, 1))
    im = np.maximum(mask, predicted_image[0].astype(np.float32) * 255)
    print(filename)
    cv2.imwrite(labels_dir + "/" + filename + "_out.png", im)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for key in ['loss', 'val_loss']:
    ax.plot(hist1.history[key], label=key)
ax.set_xlabel("Epochs")
ax.set_ylabel('Loss')
plt.legend()
plt.savefig("fcn_100_loss.png")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
for key in ['val_acc', 'val_iou','val_iou_thresholded','val_jaccard_coef','val_dice_coef']:
    ax1.plot(hist1.history[key], label=key)
ax1.set_xlabel("Epochs")
ax1.set_ylabel('Metrics')
plt.legend()
plt.savefig("fcn_100_metrics.png")

