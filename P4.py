import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import blockchain 
df = pd.read_csv('iot_lab_dataset.csv')

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model_lstm = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(1, X_train.shape[2]), return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

blockchain.connect()

lstm_loss, lstm_acc = model_lstm.evaluate(X_test, y_test)

print("LSTM Model Loss:", lstm_loss)
print("LSTM Model Accuracy:", lstm_acc)

