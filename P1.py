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

model_cnn = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_cnn.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

model_dnn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_dnn.fit(X_train_dnn, y_train_dnn, epochs=10, validation_data=(X_test_dnn, y_test_dnn))


blockchain.connect() 
cnn_loss, cnn_acc = model_cnn.evaluate(X_test, y_test)
dnn_loss, dnn_acc = model_dnn.evaluate(X_test, y_test)

print("CNN Model Loss:", cnn_loss)
print("CNN Model Accuracy:", cnn_acc)
print("DNN Model Loss:", dnn_loss)
print("DNN Model Accuracy:", dnn_acc)

