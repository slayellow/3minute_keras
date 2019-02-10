from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sfile
import plot
from keras import datasets
import keras

class CNN(Model):
    def __init__(model, nb_classes, in_shape=None):
        super().__init__()
        model.nb_classes = nb_classes
        model.in_shape = in_shape
        model.build_model()
        super().__init__(model.x, model.y)
        model.compile()

    def build_model(model):
        nb_classes = model.nb_classes
        in_shape = model.in_shape

        x = Input(in_shape)
        h = Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=in_shape)(x)
        h = Conv2D(32, kernel_size=(3,3), activation='relu')(h)
        h = MaxPooling2D(pool_size=(2,2))(h)
        h = Dropout(0.25)(h)
        h = Flatten()(h)
        z_cl = h

        h = Dense(128, activation='relu')(h)
        h = Dropout(0.5)(h)
        z_fl = h

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)
        model.x, model.y = x, y

    def compile(model):
        Model.compile(model, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

class DataSet():
    def __init__(self, X, y, nb_classes, scaling=True, test_size=0.2, random_state=0):
        self.X = X
        self.add_channels()
        X = self.X

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        if scaling:
            scaler = MinMaxScaler()
            n = X_train.shape[0]
            X_train = scaler.fit_transform(X_train.reshape(n, -1)).reshape(X_train.shape)
            n = X_test.shape[0]
            X_test = scaler.transform(X_test.reshape(n, -1)).reshape(X_test.shape)
            self.scaler = scaler

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        self.X_train, self.X_test = X_train, X_test
        self.Y_train, self.Y_test = Y_train, Y_test
        self.y_train, self.y_test = y_train, y_test

    def add_channels(self):
        X = self.X
        if len(X.shape) == 3:
            N, img_rows, img_cols = X.shape
            if K.image_dim_ordering() == 'th':
                X = X.reshape(X.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                X = X.reshape(X.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)
        else:
            input_shape = X.shape[1:]

        self.X = X
        self.input_shape = input_shape

class Machine():
    def __init__(self, X, y, nb_classes=2, fig=True):
        self.nb_classes = nb_classes
        self.set_data(X,y)
        self.set_model()
        self.fig = fig

    def set_data(self, X, y):
        nb_classes = self.nb_classes
        self.data = DataSet(X, y, nb_classes)

    def set_model(self):
        nb_classes = self.nb_classes
        data = self.data
        self.model = (CNN(nb_classes=nb_classes, in_shape=data.input_shape))

    def fit(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model

        history = model.fit(data.X_train, data.Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, validation_data=(data.X_test, data.Y_test))
        return history

    def run(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model
        fig = self.fig
        history = self.fit(nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        score = model.evaluate(data.X_test, data.Y_test, verbose=0)
        print('Confusion Matrix')
        Y_test_pred = model.predict(data.X_test, verbose=0)
        y_test_pred = np.argmax(Y_test_pred, axis=1)
        print(metrics.confusion_matrix(data.y_test, y_test_pred))

        print('Test Socre: ',score[0])
        print('Test Accuracy: ',score[1])

        suffix = sfile.unique_filename('datatime')
        foldname = 'output_' + suffix
        os.makedirs(foldname)
        plot.save_history_history('history_history.npy',history.history, fold=foldname)
        model.save_weights(os.path.join(foldname, 'dl_model.h5'))
        print('Output result are save in ', foldname)

        if fig:
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plot.plot_acc(history)
            plt.subplot(1,2,2)
            plot.plot_loss(history)
            plt.show()
        self.history = history
        return foldname

assert keras.backend.image_data_format() == 'channels_last'

class RunMachine(Machine):
    def __init__(self):
        (X, y), (x_test, y_test) = datasets.cifar10.load_data()
        super().__init__(X, y, nb_classes=10)

def main():
    m = RunMachine()
    m.run()

if __name__ == '__main__':
    main()