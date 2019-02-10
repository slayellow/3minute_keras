import numpy as np
from keras import datasets
from keras.utils import np_utils
from keras import layers, models

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W*H*C)
    X_test = X_test.reshape(-1, W*H*C)

    X_train = X_train / 255
    X_test = X_test / 255

    return (X_train, Y_train), (X_test, Y_test)

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_1, Pd_1, Nout):
        super().__init__()

        self.add(layers.Dense(Nh_1[0], activation='relu', input_shape=(Nin, ), name='Hidden_1'))
        self.add(layers.Dropout(Pd_1[0]))
        self.add(layers.Dense(Nh_1[1], activation='relu', name='Hidden_2'))
        self.add(layers.Dropout(Pd_1[1]))
        self.add(layers.Dense(Nout, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def main():
    Nh_1 = [256, 128]
    Pd_1 = [0.2, 0.2]
    Nout = 10

    (X_train, Y_train), (X_test, Y_test) = Data_func()
    model = DNN(X_train.shape[1], Nh_1, Pd_1, Nout)
    history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2)

    performance_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy -> ', performance_test)

if __name__ == '__main__':
    main()