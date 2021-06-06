import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

### Image preprocessing pipeline
def preprocess(image, color='RGB'):
    # by default use input image color space as is.
    img = image
    # convert to YUV color space
    if color == 'BGR':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color == 'RGB':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # trim image to only see section with road
    cropped_img = img[45:115,:,:]
    # rescale to nvidea model input size
    rescaled = cv2.resize(cropped_img,(200, 66), interpolation = cv2.INTER_LINEAR)
    return rescaled

### Build neural network model    
def PilotNet():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5-1.0,input_shape=(66,200,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Convolution2D(64,3,3, W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Convolution2D(64,3,3, W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Dense(50, W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Dense(10, W_regularizer = l2(1e-6)))
    model.add(Activation('elu'))
    model.add(Dense(1, W_regularizer = l2(1e-6)))
    model.add(Activation('tanh'))

    adam = Adam(lr = 0.0001)
    model.compile(optimizer= adam, loss='mse')
    
    return model

### Load training data
def load_data(path, images, measurements, augment = True):
    with open(path + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # read in images from center, left and right cameras
            filename = row[0]
            local_path = path + "IMG/" + filename
            # add images to data set
            images.append((local_path,False))
            if augment == True:
                images.append((local_path,True))
            steering_center = float(row[1])

            # add steering angles to data set
            if augment == True:
                measurements.append(steering_center)
                measurements.append(-steering_center)
            else:
                measurements.append(steering_center)

    return images, measurements

def generator(samples, measurements, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples, measurements = shuffle(samples, measurements)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = measurements[offset:offset+batch_size]

            images = []
            angles = []
            for i in range(len(batch_samples)):
                name = batch_samples[i][0]
                image = cv2.imread(name)
                angle = batch_labels[i]
                if batch_samples[i][1] == True:
                    image = cv2.flip(image, 1)
                # trim image to only see section with road
                images.append(preprocess(image, color='BGR'))
                angles.append(angle)
                if(i==0):
                    cv2.imwrite("sample.jpg",images[0])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

### Load data, train and save convolutional neural network
if __name__ == '__main__':
    samples = []
    measurements = []

    correction = 0.2 # this is a parameter to tune
    samples, measurements = load_data("./data/MiniMonaco/", samples, measurements, augment = False)

    print("Track1 Samples: ", len(samples))
    samples, measurements = shuffle(samples, measurements)
    train_samples, validation_samples, train_labels, validation_labels = train_test_split(samples, measurements, test_size=0.2)

    model = PilotNet()
    model.summary()

    ### Save un trained model. Do not save if model.h5 is going to be loaded from previous train.
    model.save('model_init.h5')
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, train_labels, batch_size=32)
    validation_generator = generator(validation_samples, validation_labels, batch_size=32)

    
    # Train for 10 epoch to gauge how model is performing
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

    ### Save trained model
    model.save('model.h5')
    
    import matplotlib.pyplot as plt

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./data/model_training_5epoch.png')
    plt.show()
    """

    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

    ### Save trained model
    model.save('model.h5')

    
    ### Training for jungle track
    model.load_weights('model_init.h5')

    initial_fc_weights = []
    for layer in model.layers[-8:]:
        initial_fc_weights.append(layer.get_weights())

    # Training from previously saved model
    model.load_weights('model.h5')

    # Reduce dropout to 0.1
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0.1

    ### Load data from jungle track
    samples2 = []
    measurements2 = []

    correction2 = 0.5 # this is a parameter to tune
    samples2, measurements2 = load_data("./data/Track2-Rightlane/", samples2, measurements2, correction = correction2, augment = False)
    samples2, measurements2 = load_data("./data/Track2-Recovery/", samples2, measurements2, correction = correction2, augment = False)
    samples2, measurements2 = load_data("./data/Track2-Counter-clock/", samples2, measurements2, correction = correction2, augment = False)

    print("Track2 Samples: ", len(samples2))
    samples2, measurements2 = shuffle(samples2, measurements2)
    train_samples2, validation_samples2, train_labels2, validation_labels2 = train_test_split(samples2, measurements2, test_size=0.2)

    train_generator2 = generator(train_samples2, train_labels2, batch_size=32)
    validation_generator2 = generator(validation_samples2, validation_labels2, batch_size=32)
    model.fit_generator(train_generator2, samples_per_epoch= len(train_samples2), validation_data=validation_generator2, nb_val_samples=len(validation_samples2), nb_epoch=2)

    ### Freeze initial 3 convolution layers.
    for layer in model.layers[:6]:
        layer.trainable = False

    model.summary()

    ### Reset weights of fully connected layers
    i= 0
    for layer in model.layers[-8:]:
        layer.set_weights(initial_fc_weights[i])
        i=i+1

    ### Train on both tracks together
    all_samples = np.append(samples,samples2,axis=0)
    all_measurements = np.append(measurements,measurements2)

    print(len(all_samples))

    all_samples,all_measurements = shuffle(all_samples,all_measurements)
    train_samples_all, validation_samples_all, train_labels_all, validation_labels_all = train_test_split(all_samples,all_measurements, test_size=0.2)

    train_generator_all = generator(train_samples_all,train_labels_all, batch_size=32)
    validation_generator_all = generator(validation_samples_all,validation_labels_all, batch_size=32)
    model.fit_generator(train_generator_all, samples_per_epoch= len(train_samples_all), validation_data=validation_generator_all, nb_val_samples=len(validation_samples_all), nb_epoch=5)

    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

    for layer in model.layers[:6]:
        layer.trainable = True

    model.summary()

    model.save('model2.h5')
    """