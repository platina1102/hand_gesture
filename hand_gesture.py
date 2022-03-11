import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob2
import tqdm
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow import keras

def deep_model_fit(epoch, fit_data_x, fit_data_y):
    model = keras.Sequential()
    #배치정규화:보통 전체 데이터의 일부분인 배치라는 단위로 분할해서 학습하는데 그 배치 단위별로 정규화하는것
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    #가중치 감소:학습중 가중치가 큰것에 대해서 일종의 패널티를 부과하여 과적합 위험성을 줄인다
    #Dropout:신경망 모델이 복잡해질 때 가중치 감소만으로는 어려운데 Dropout에서는 뉴런의 연결을 임의로 삭제한다
    #즉, 훈련 중 임의의 뉴런을 골라 삭제하여 신호를 전달하지 않도록 한다(테스트시에는 모든 뉴런 사용)
    model.add(BatchNormalization())
    model.add(Dropout(0.10))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.10))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.10))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))
    model.add(Dense(4, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    model.fit(fit_data_x, fit_data_y, epochs=epoch, callbacks=[checkpoint_cb, early_stopping_cb])

    return model


train_data = pd.read_csv('train.csv')
print(train_data.info())
data_x = train_data.drop(['id', 'target'], axis=1)
data_y = train_data['target']

test_data = pd.read_csv('test.csv')
test_data = test_data.drop(['id'], axis=1)


model = deep_model_fit(1000, data_x, data_y)
model = keras.models.load_model('best-model.h5')

# test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
# print("Accuracy : ", np.round(test_acc,5))

result = np.argmax(model.predict(test_data), axis=1)
result = pd.DataFrame(list(result))
print(result)
result.to_csv('result.csv')