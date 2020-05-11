######################################### create augmented dataset start #########################################


from numpy import expand_dims
from cv2 import resize
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from os import listdir


###################### Training Data Augmentation ######################

newSize = 150

for Normal_img in listdir('train/NORMAL/'):
    if not Normal_img == '.DS_Store':
        img = load_img('train/NORMAL/' + Normal_img)
        # convert to numpy array
        data = resize(img_to_array(img), (newSize, newSize))
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        # datagen = ImageDataGenerator(rotation_range=45, horizontal_flip='True')
        datagen = ImageDataGenerator(rotation_range=30, zoom_range=(.65, 1.))
        # prepare iterator
        it = datagen.flow(samples, batch_size=32, save_prefix='norm',save_to_dir='aug_testing/augments_dir/Normal')
        
        for i in range(600): next(it)

 

for Pneum_img in listdir('train/PNEUMONIA/'):
    if not Pneum_img == '.DS_Store':
        img = load_img('train/PNEUMONIA/' + Pneum_img)
        # convert to numpy array
        data = resize(img_to_array(img), (newSize, newSize))
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        # datagen = ImageDataGenerator(rotation_range=45, horizontal_flip='True')
        datagen = ImageDataGenerator(rotation_range=30, zoom_range=(.65, 1.))
        # prepare iterator
        it = datagen.flow(samples, batch_size=32, save_prefix='pneum', save_to_dir='aug_testing/augments_dir/Pneumonia')
        
        for i in range(200): next(it)


######################################### create augmented dataset end #########################################

