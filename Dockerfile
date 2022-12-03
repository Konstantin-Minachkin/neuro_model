FROM python:3.10

RUN apt-get update

# по-правильному через другого user-а сделать не получилось
#   тогда к volume у программы нет доступа
# RUN adduser --home /home/myuser myuser
# USER myuser
VOLUME /myVol

RUN pip install --user -U pip
RUN pip install --user nasdaq-data-link
RUN pip install --user -U scikit-learn
RUN pip install --user -U matplotlib
RUN pip install --user -U keras
RUN pip install --user -U tensorflow

ADD regress_model.py /
ADD LSTM_model.py /
ADD Memo_LSTM.py /
ADD Data.py /
ADD __main__.py /
ADD mydata.csv /

# ENV PATH="/home/myuser/.local/bin:${PATH}"
# COPY --chown=myuser:myuser . .