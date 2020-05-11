from sklearn.utils import shuffle
from cv2 import imread, cvtColor, resize, filter2D, medianBlur
from cv2 import COLOR_BGR2GRAY
from os import listdir
import numpy as np

# Image size to input to the model
newSize = 150

################################################## Data Preperation Start ##################################################
##############
#### Data Preperation includes the following steps:
#    1- Read the image
#    2- Convert into grayscale
#    3- Sharpen the image
#    4- Normalize pixel values
#    5- Appending the processed image to its coressponding category's array (train, validation, test)
#    6- Append the coressponding dependant variable value (in case of training and validation)

# Training data preperations
train_loc = 'aug_testing/augments_dir/Merged/'

xtrain = []
ytrain = []

for image_name in listdir(train_loc):
    img = imread(train_loc + image_name)

    img = cvtColor(img, COLOR_BGR2GRAY)

    kernel = np.array([[-1/9,-1/9,-1/9], [-1/9,1,-1/9], [-1/9,-1/9,-1/9]])
    img = filter2D(img, -1, kernel)
    img = medianBlur(img, 3)

    img = img/255.

    img = np.expand_dims(img, 2)

    xtrain.append(img)
    ytrain.append(0 if 'norm' in image_name else 1)

xtrain, ytrain = shuffle(xtrain, ytrain)
xtrain, ytrain = np.asarray(xtrain), np.asarray(ytrain)

# Validation data preperations
validation_loc = 'val/Merged/'

xval = []
yval = []

for image_name in listdir(validation_loc):
    img = imread(validation_loc + image_name)
    img = resize(cvtColor(img, COLOR_BGR2GRAY), (newSize, newSize))
    
    kernel = np.array([[-1/9,-1/9,-1/9], [-1/9,1,-1/9], [-1/9,-1/9,-1/9]])
    img = filter2D(img, -1, kernel)
    img = medianBlur(img, 3)
    
    img = img/255.
    img = np.expand_dims(img, 2)
    xval.append(img)
    yval.append(0 if 'IM' in image_name else 1)

xval, yval = shuffle(xval, yval)
xval, yval = np.asarray(xval), np.asarray(yval)


# Testing data preperations
test_loc = 'test_merged/'


xtest = []
ytest = []
   
    
for image_name in listdir(test_loc):
    img = imread(test_loc + image_name)
    img = resize(cvtColor(img, COLOR_BGR2GRAY), (newSize, newSize))
    
    kernel = np.array([[-1/9,-1/9,-1/9], [-1/9,1,-1/9], [-1/9,-1/9,-1/9]])
    img = filter2D(img, -1, kernel)
    img = medianBlur(img, 3)
    
    img = img / 255.
    img = np.expand_dims(img, 2)
    xtest.append(img)
    ytest.append(0 if 'IM' in image_name else 1)

xtest, ytest = shuffle(xtest, ytest)
xtest, ytest = np.asarray(xtest), np.asarray(ytest)


######################################### data preperation end #########################################

######################################### model creation start #########################################


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score

# Define early stopping callback
es = EarlyStopping(monitor='val_acc', restore_best_weights=True, patience=5)


model = Sequential()

# Fs: 64, 128
model.add(Conv2D(
        128,
        (3, 3),
        activation='relu',
        input_shape=(150, 150, 1)
    ))

model.add(MaxPooling2D())

model.add(BatchNormalization())

model.add(Conv2D(
        64,
        (3, 3),
        activation='relu'
    ))

model.add(MaxPooling2D())

model.add(BatchNormalization())

model.add(Conv2D(
        32,
        (3, 3),
        activation='relu'
    ))

model.add(MaxPooling2D())

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(
        256,
        activation='relu',
    ))

# Drops 0.3, 0.2/ 0.2,0.1
model.add(Dropout(rate=0.2))

model.add(Dense(
        128,
        activation='relu',
    ))

model.add(Dropout(rate=0.1))

model.add(Dense(
        1,
        activation='sigmoid'
    ))

model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )

# BS: 32, 120, 64
model.fit(
        xtrain,
        ytrain,
        batch_size=64,
        epochs=15,
        validation_split=0.05,
        callbacks=[es]
        ,validation_data=(xval, yval)
    )


predictions = model.predict(xtest)

predictions = 1*(predictions >= .65)

cm = confusion_matrix(ytest, predictions)

acc = accuracy_score(ytest, predictions)

