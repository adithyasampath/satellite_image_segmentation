from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2, '../mapbox_india_data/data_first_150/', 'unet_tiles_train', 'unet_labels_train', data_gen_args, image_color_mode="rgb",
                        save_to_dir=None)  # using mask in grayscale

model = unet('unet_membrane.hdf5')
# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
testGene = testGenerator("../mapbox_india_data/data_first_150/tiles/test", as_gray=False)
results = model.predict_generator(testGene, 25, verbose=1)
saveResult("../mapbox_india_data/data_first_150/results", results)