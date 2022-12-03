# python3.exe '.\__main__.py' 2 4 10 1 mydata.csv
from Data import Data
from Memo_LSTM import Memory_LSTM_model
from LSTM_model import LSTM_model

import matplotlib.pyplot as plt
import sys

test_data = None
# Аргумент командной строки - номер модели
try:
    neuro = (int)(sys.argv[1])
except IndexError:
    # по умолчанию запускаем первую модель
    neuro = 1
try:
    seq_length = (int)(sys.argv[2])
except IndexError:
    # длина последовательности
    seq_length = 4
try:
    epochs = (int)(sys.argv[3])
except IndexError:
    # кол-во эпох при обучении
    epochs = 100
try:
    repeat = (int)(sys.argv[4])
except IndexError:
    # кол-во повторений обучения нейросети. Актуально только для 2 модели
    repeat = 100
try:
    # путь к файлу с данными
    data_path = sys.argv[5]
except IndexError:
    data_path = None


myDataObject = Data(data_path)

if neuro == 1:
    myDataObject.prepare_data(seq_length)
    lstm = LSTM_model(seq_length)
    lstm.train(myDataObject.x, myDataObject.y, epochs)
    prediction = lstm.predict(myDataObject.x, myDataObject.y, myDataObject.scaler,  seq_length)
    test_plot, test_data = lstm.create_plot(myDataObject.train_dataset, prediction, seq_length, myDataObject.scaler)

    prediction = lstm.predict(myDataObject.test_x, myDataObject.test_y, myDataObject.scaler, seq_length)
    train_plot, train_data = lstm.create_plot(myDataObject.test_dataset, prediction, seq_length, myDataObject.scaler)
elif neuro == 2:
    myDataObject.prepare_data(seq_length,'memo_lstm')
    lstm_2 = Memory_LSTM_model(seq_length)
    lstm_2.train(myDataObject.x, myDataObject.y, epochs, repeat)
    prediction = lstm_2.predict(myDataObject.x, myDataObject.y, myDataObject.scaler,  seq_length)
    test_plot, test_data = lstm_2.create_plot(myDataObject.train_dataset, prediction, seq_length, myDataObject.scaler)

    prediction = lstm_2.predict(myDataObject.test_x, myDataObject.test_y, myDataObject.scaler, seq_length)
    train_plot, train_data = lstm_2.create_plot(myDataObject.test_dataset, prediction, seq_length, myDataObject.scaler)

if test_data is not None:
    plt.plot(test_data, color='grey')
    plt.plot(test_plot, color='darkgrey')
    plt.plot(train_data, color='darkgreen')
    plt.plot(train_plot, color='green')
    plt.savefig('./myVol/filename.svg') # для отображения изображения в linux без граф интерфейса или докере
    plt.show()