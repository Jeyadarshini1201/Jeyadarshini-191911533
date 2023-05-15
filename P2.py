import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import blockchain 

df = pd.read_csv('iot_lab_dataset.csv')

# Preprocess the dataset for training and testing
X = df.iloc[:, :-1].values  # Extract features (all columns except the last one)
y = df.iloc[:, -1].values  # Extract labels (the last column)
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

model_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


model_drl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Define the loss function for the deep reinforcement learning model
def custom_loss(y_true, y_pred, actions):
    return -tf.reduce_mean(tf.math.log(tf.reduce_sum(y_true * y_pred * actions, axis=1)))

model_drl.compile(optimizer='adam', loss=custom_loss)

for epoch in range(10):
    state = get_current_state()
    
    action = get_action(state)
    
    next_state, reward = take_action(state, action)
    
    model_drl.train_on_batch(next_state, reward, actions)

cnn_loss, cnn_acc = model_cnn.evaluate(X_test, y_test)
drl_loss = model_drl.evaluate(X_test, y_test)

print("CNN Model Loss:", cnn_loss)
print("CNN Model Accuracy:", cnn_acc)
print("DRL Model Loss:", drl_loss)

