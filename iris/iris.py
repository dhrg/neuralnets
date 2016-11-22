# dense NN with keras/theano for sample dataset, iris. You can also load model and weights from files and 
# make predictions that way. It's shown down how to do it.  
#import theano
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn import datasets
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
import itertools
import matplotlib.pyplot as plt
#import cv2
#import os
 	
# some custom function to print nice confusion matrix later on
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# fix random seed, reproducibility 
seed = 1
np.random.seed(seed)

# generate dummy radom training data or get dataset. We will get iris dataset.
'''
data_dim = 3
data_size = 1000
nb_classes = 2

X = np.random.random((data_size, data_dim))
Yr = np.random.randint(nb_classes, size=(data_size, ))
# purpeosuly converting to strings as usually labels of data are strings and 
# classes should be in integers, or if they are strings they should be first converted to integers
Ys = np.char.mod('%d', Yr)  #making our integer classes synthetically strings
'''

iris = sn.load_dataset("iris")
sn.pairplot(iris, hue='species')
X = iris.values[:,0:4]
Y = iris.values[:,4]

# train test split
X_train, X_test, Ys_train, Ys_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

# converting string labels to integers
encoder = LabelEncoder()
encoder.fit(Y)
list(encoder.classes_)
Yi_train = encoder.transform(Ys_train)
Yi_test = encoder.transform(Ys_test)

# convert integers now to dummy variables (i.e. one-hot encoded)
Y_train = np_utils.to_categorical(Yi_train)
Y_test = np_utils.to_categorical(Yi_test)

model = Sequential()

# create network architecture with regularization
model.add(Dense(40, input_shape=(4,), init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(3, init='uniform'))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,  
              metrics=["accuracy"])
 
# training data
# check also http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/ for using train_test_split and
# validationa_data within model.fit
model.fit(X_train, Y_train, nb_epoch=200, batch_size=1)

# accuracy on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print("test %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#serialize model and weights 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
 
# some time later in the future...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# lets do some confusion matrix and more insights, evaluate loaded model on test data, accuracy
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print "loaded model %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

# confusion matrix and some insights
predictions = loaded_model.predict_classes(X_test, verbose=0)
print(predictions)
pred = encoder.inverse_transform(predictions)
print(pred)

target_names = list(encoder.classes_)
print(classification_report(Yi_test, predictions, target_names=target_names))

# compute confusion matrix
cnf_matrix = confusion_matrix(Yi_test, predictions)
np.set_printoptions(precision=2)

# plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# apply custom prediction on your flower
prediction = loaded_model.predict_classes(np.array([[4, 3, 1, 0.1]]), verbose=0)
yourflower = encoder.inverse_transform(prediction)
print "Your flower is %s." % yourflower[0]

# final words, if you deploing on AWS or Azure ML likelly you would need to check https://myjourneyasadatascientist.com/tag/ec2/ or 
# https://gallery.cortanaintelligence.com//Experiment/Theano-Keras-1. However there would be other ways too. 