import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
normal_df = pd.read_csv(r'E:\aHieu\YOLO_pose_sleep\NORMAL.txt')
sleep_df = pd.read_csv(r'E:\aHieu\YOLO_pose_sleep\SLEEP.txt')
X = []
y = []
no_of_timesteps = 10


dataset = sleep_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)
dataset = normal_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
# Mã hóa one-hot cho nhãn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model1.h5")


