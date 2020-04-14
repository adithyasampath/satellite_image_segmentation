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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
from keras import applications
import os
import cv2
import glob
from PIL import Image
from utils import get_augmented
from keras.models import load_model
from metrics import iou, iou_thresholded, jaccard_coef, dice_coef
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

img_width, img_height = 1024, 1024

batch_size = 1

tiles_samples = 0
labels_samples = 0

nb_epoch = 50

# initial_image_dir = '/home/drone/adithya/unet/images'
initial_image_dir = '/home/ndnbase/adithya/unet/images'
tiles_dir = initial_image_dir + '/hr_tiles'
labels_dir = initial_image_dir + '/hr_labels'

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


def segnet():
    # encoder
    img_input = Input(shape=(1024, 1024, 3))
    x = img_input
    # Encoder
    x = Convolution2D(32, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = Convolution2D(256, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(32, (3, 3), border_mode="same")(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(1, 1, 1, border_mode="valid")(x)
    x = Activation("sigmoid")(x)
    model = Model(img_input, x)
    return model


model = segnet()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc", iou, iou_thresholded, jaccard_coef, dice_coef])
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'segnet_%d_val_all_1.h5' % nb_epoch,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)
]

hist1 = model.fit_generator(
    train_gen,
    steps_per_epoch=78,
    validation_data=val_gen,
    validation_steps=20,
    callbacks=callbacks,
    epochs=nb_epoch),
    keras.callbacks.TensorBoard(log_dir='./logs_segnet', histogram_freq=0, batch_size=2, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

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
plt.savefig("segnet_100_loss_1.png")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
for key in [
        'val_acc', 'val_iou', 'val_iou_thresholded', 'val_jaccard_coef',
        'val_dice_coef'
]:
    ax1.plot(hist1.history[key], label=key)
ax1.set_xlabel("Epochs")
ax1.set_ylabel('Metrics')
plt.legend()
plt.savefig("segnet_100_metrics_1.png")
