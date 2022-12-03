import nasdaqdatalink
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class Data:
    raw_data = pd.DataFrame()
    x,y,test_x,test_y = [],[],[],[]
    NORMILIZE = None

    def __init__(self, csv_file=None):
        # Грузим дату либо считываем данные с файла
        if csv_file is None:
            self.raw_data = nasdaqdatalink.get("NSE/OIL.1", start_date="2001-12-31", end_date="2021-12-31", returns="pandas")
            # сохранить данные в файл (чтобы при тесттировании не привлекать nasdaqdatalink)
            # self.raw_data.to_csv('mydata.csv') 
        else:
            self.raw_data = pd.read_csv(csv_file, usecols=[1], delimiter=',')
            # self.raw_data = pd.read_csv(csv_file, delimiter=',')
        self.scaler = MinMaxScaler(feature_range=(0, 1))


    def prepare_data(self, seq = 1, target_model = 'lstm'):
    # --готовим дату для обучения
    # Результат - две выборки обуч и тест, а также класс выборок - для обучения и для проверки теста

        if 'Date' in self.raw_data.columns:
            t = list()
            for d in self.raw_data.Date.values:
                date = pd.to_datetime(d, format="%Y-%m-%d")
                date = date.to_pydatetime().timestamp()
                t.append(date)
            self.raw_data.Date = pd.Series(t)

        print(self.raw_data.head())

        # Нормализуем данные. Столбцы с фиктивными переменными не нормализуем
        dataset = self.raw_data.iloc[len(self.raw_data)+seq-550:]
        dataset = dataset.values.astype('float32')
        if self.NORMILIZE is None:
            dataset = self.scaler.fit_transform(dataset)

        # Делим данные на тест и обуч выборки
        # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state = 0)
        # self.train_data = self.raw_data.iloc[len(self.raw_data)-550:]
        # self.test_data = train_data.iloc[len(train_data)-50:]
        self.train_dataset = dataset
        self.test_dataset = dataset[len(dataset)+seq-50:]
        
        # делим данные на блоки по seq знач
        x, self.y = self.split_data(self.train_dataset, seq)
        test_x, self.test_y = self.split_data(self.test_dataset, seq)
        if target_model == 'lstm':
            self.x = self.reshape_for_lstm(x)
            self.test_x = self.reshape_for_lstm(test_x)
        elif target_model == 'memo_lstm':
            self.x = self.reshape_for_memo_lstm(x)
            self.test_x = self.reshape_for_memo_lstm(test_x)

        # self.x = np.reshape(x, (x.shape[0], x.shape[1], 1 )) #меняем размерность массива на (length, 1, 1)
        # self.test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1 ))

    

    def split_data(self, dataset, look_back=1):
        # раздлеить массив на блоки по look_back значений
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)


    def reshape_for_lstm(self, x):
        #меняем размерность массива чтобы подошел для lstm на (length, 1, 1)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1])) 
        return x

    def reshape_for_memo_lstm(self, x):
        #меняем размерность массива на (length, column, 1)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1 ))
        return x