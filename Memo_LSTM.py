from regress_model import regress_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from LSTM_model import LSTM_model

class Memory_LSTM_model(LSTM_model):

    batch_size = 1
    model = Sequential()

    def __init__(self, n_features = 1):
        # сеть с одним входным, выходным слоем и скрытым LSTM слоем на 4 нейронов
        self.model.add(LSTM(50, batch_input_shape=(self.batch_size, n_features, 1), stateful=True))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary() #структура модели

    def predict(self, x, y, scaler, seq):
        prediction = self.model.predict(x, batch_size=self.batch_size)
        # invert predictions
        prediction = scaler.inverse_transform(prediction)
        pred_y = scaler.inverse_transform([y])
        # проверим качество предсказания через RMSE
        score = sqrt(mean_squared_error(pred_y[0], prediction[:,0]))
        print('Score: %.2f RMSE' % (score))
        return prediction

    def train(self, x,y, epochs = 10, repeat = 100):
        for i in range(repeat):
            self.model.fit(x, y, epochs=epochs, batch_size=self.batch_size, verbose=2, shuffle=False)
            self.model.reset_states()
    