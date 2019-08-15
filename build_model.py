import tensorflow as tf
import numpy as np


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from read_txt_data import read_data


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint



from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.optimizers import Adam
def get_data():
    data, label = read_data()
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size= 0.96)
    scaler = preprocessing.StandardScaler().fit(x_train)
    scaler_2 = preprocessing.StandardScaler().fit(y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = scaler_2.transform(y_train)
    y_test = scaler_2.transform(y_test)

    #y_train = np.array(y_train)
    #y_test = np.array(y_test)

    pca = PCA(n_components=36)

    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("pca.explained_variance_ratio_:{}".format(pca.explained_variance_ratio_))
    print("pca_explained_variance_:{}".format(pca.explained_variance_))
    return x_train, x_test, y_train, y_test









def build_model(input_dim):

    main_input = Input(shape=(input_dim, ))
    den1 = Dense(256, activation="relu")(main_input)
    den1_dropout = Dropout(0.6)(den1)
    den2 = Dense(128, activation="relu")(den1_dropout)

    den2_dropout = Dropout(0.5)(den2)
    main_output = Dense(4, activation="tanh")(den2_dropout)
    model = Model(inputs=main_input, output=main_output)
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mean_absolute_percentage_error", optimizer=adam, metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train, x_test, y_test, epoch, batch_size):
    #callback = ModelCheckpoint(filepath=None, save_best_only=True, save_weights_only=True)
    #early_stop = EarlyStopping(monitor='val_mse', patience=5, verbose=1)


    #reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.2, patience=5, min_lr=0.00001)
    #callback_list = [ early_stop, reduce_lr]
    model.fit(x_train, y_train,batch_size=batch_size,shuffle=False,epochs=epoch, validation_data=(x_test, y_test))
    output = model.predict(x_test, batch_size=1)
    #error = sum(abs(y_test - output) / y_test) / y_test.shape[0]
    error = np.zeros((y_test.shape[1], ), dtype=np.float32)
    for i in range(y_train.shape[1]):
        error[i] = sum(abs(y_test[:, i] - output[:, i]) / y_test[:, i]) / y_test.shape[0]
    return error, output












if __name__ == "__main__":
    batch_size = 10
    epoch = 2000

    x_train, x_test, y_train, y_test = get_data()
    print("x_train.shape:{}".format(x_train.shape))
    print("y_train.shape:{}".format(y_train.shape))
    print("x_test.shape:{}".format(x_test.shape))
    print("y_test.shape:{}".format(y_test.shape))
    model = build_model(x_train.shape[1])

    error, output = train_model(model, x_train, y_train, x_test, y_test,epoch, batch_size)
    
    print("error:{}".format(error))