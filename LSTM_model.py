from regress_model import regress_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

class LSTM_model(regress_model):

    def __init__(self, n_features = 1):
        # сеть с одним входным, выходным слоем и скрытым LSTM слоем на 50 нейронов
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(1, n_features)))
        # одномерный ряд, поэтому число функций = одной для одной переменной
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary() #структура модели
        

    def predict(self, x, y, scaler, seq):
        prediction = self.model.predict(x)
        # invert predictions
        prediction = scaler.inverse_transform(prediction)
        pred_y = scaler.inverse_transform([y])
        # проверим качество предсказания через RMSE
        score = sqrt(mean_squared_error(pred_y[0], prediction[:,0]))
        print('Score: %.2f RMSE' % (score))
        return prediction


    def train(self, x,y, epochs = 10):
        self.model.fit(x, y, epochs=epochs, batch_size=1, verbose=2)


    def create_plot(self, dataset, prediction, seq, scaler):
        # shift train predictions for plotting
        plot = np.empty_like(dataset)
        plot[:, :] = np.nan
        plot[seq:len(prediction)+seq, :] = prediction
        # plot baseline and predictions
        real_plot = scaler.inverse_transform(dataset)
        return plot, real_plot