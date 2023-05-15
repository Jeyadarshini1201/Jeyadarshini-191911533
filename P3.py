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

model_dnn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

cnn_out = model_cnn.layers[-2].output
dnn_out = model_dnn.layers[-2].output
concatenated = keras.layers.concatenate([cnn_out, dnn_out])
output_layer = keras.layers.Dense(10, activation='softmax')(concatenated)
model_hybrid = keras.models.Model(inputs=[model_cnn.input, model_dnn.input], outputs=output_layer)

model_hybrid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_hybrid.fit([X_train.reshape(-1,28,28,1), X_train], y_train, epochs=10, validation_data=([X_test.reshape(-1,28,28,1), X_test], y_test))

blockchain.connect() 

hybrid_loss, hybrid_acc = model_hybrid.evaluate([X_test.reshape(-1,28,28,1), X_test], y_test)

print("Hybrid Model Loss:", hybrid_loss)
print("Hybrid Model Accuracy:", hybrid_acc)

