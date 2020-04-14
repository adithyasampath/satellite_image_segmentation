import keras
# from keras.datasets.fashion_mnist import load_data
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
# from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.utils import shuffle
from skimage.io import imread, imsave
import numpy as np
from skimage.transform import resize
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

target_size = 96

def build_model():
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_layer = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')
    for layer in base_layer.layers:
        layer.trainable = True
    op = Dense(256, activation='relu')(base_layer.output)
    op = Dropout(0.25)(op)
    output_tensor = Dense(10, activation='softmax')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    return model


def preprocess_image(x):
    # Resize the image to have the shape of (96,96)
    x = resize(
        x, (target_size, target_size), mode='constant', anti_aliasing=False)

    # convert to 3 channel (RGB)
    x = np.stack((x, ) * 3, axis=-1)

    # Make sure it is a float32, here is why
    # https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow
    return x.astype(np.float32)


def save_test_images(images):
    """
    images : numpy arrays
    """
    image_folder = os.path.join(APP_ROOT, "images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    for img, num in zip(images, range(len(images))):
        imsave(os.path.join(image_folder, "img_%03d.jpg" % num), img)


def load_generator_data(x, y, batch_size=64):
    num_samples = x_train.shape[0]
    while 1:
        try:
            shuffle(x)
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]
                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            print(err)


def train():
    model = build_model()
    train_generator = load_generator_data(
        norm_x_train, encode_y_train, batch_size=64)
    model.fit_generator(
        generator=train_generator, steps_per_epoch=100, verbose=1, epochs=2)
    print("\nTraining complete!\n Begining validation...\n")
    test_generator = load_generator_data(
        norm_x_test, encode_y_test, batch_size=64)
    model.evaluate_generator(generator=test_generator, steps=150, verbose=1)
    model_path = os.path.join(APP_ROOT, "weights", "resnet50.h5")
    model.save(model_path)
    print("Model saved!")



(x_train, y_train), (x_test, y_test) = load_data()
print("X_train: ", x_train.shape)
print("Y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("Y_test: ", y_test.shape)

norm_x_train = x_train.astype('float32') / 255
norm_x_test = x_test.astype('float32') / 255
encode_y_train = to_categorical(y_train, num_classes=10)
encode_y_test = to_categorical(y_test, num_classes=10)

if __name__ == "__main__":
    save_test_images(x_train[:10])
    train()
   
    