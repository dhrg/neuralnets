# CNN with Keras/Theano/scikit-learn for image classification (dogs vs cats). You will need to load dataset images manually,
# download these from https://www.kaggle.com/c/dogs-vs-cats/data. Overall 25000 images.
import theano
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn import datasets
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2, os, random

K.set_image_dim_ordering('th')
TRAIN_DIR = 'C:/code/projects/ML/NNs/dogsvscats/dataset/train/'
TEST_DIR = 'C:/code/projects/ML/NNs/dogsvscats/dataset/test/'

ROWS = 42
COLS = 42
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i] # 12500 dogs
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i] # 12500 cats
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)] # 12501 test images

# create balanced reduced dataset, delete if full dataset
train_images = train_dogs[:250] + train_cats[:250]
random.shuffle(train_images)
test_images =  test_images[:200]

# lets load images 
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

X_train = prep_data(train_images)
X_test = prep_data(test_images)

# normalize data 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))

# labels should be binary

Y_train = []
for i in train_images:
    if 'dog' in i[-19:]:
        Y_train.append(1)
    else:
        Y_train.append(0)
Y_train = np.array(Y_train)
        
sn.countplot(Y_train)
sn.plt.title('Cats and Dogs')

def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure()
    plt.imshow(pair)
    plt.show()
    
for idx in range(0,5):
    show_cats_and_dogs(idx)

# defining neural net   

model = Sequential()

model.add(Convolution2D(64, 3, 3, input_shape=X_train.shape[1:], activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

nb_epoch = 20    # some deep nets are trained for 100-1000 epochs
batch_size = 10

# callback for loss logging per epoch, optional
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
history = LossHistory()

# this might take a while, especially on cpu (better get gpu)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.25, verbose=1, shuffle=True, callbacks=[history])

# predict probabilities that is 'dog'
predictions = model.predict(X_test, verbose=1)


