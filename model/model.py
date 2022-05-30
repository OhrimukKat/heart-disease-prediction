import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

cardio = pd.read_csv('new_cardio.csv', sep=',')
cardio = cardio.drop(columns = ['rang', 'len', 'Unnamed: 0'], axis = 1)
y = cardio['cardio']

X = cardio.drop(columns=['cardio'], axis=1)

train, test, target, target_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)

def build_ann(optimizer='adam'):
# Initializing the ANN
    ann = Sequential()
    
    # Adding the input layer and the first hidden layer of the ANN with dropout
    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='sigmoid', input_shape=(len(train.columns),)))
    
    # Add other layers, it is not necessary to pass the shape because there is a layer before
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='softmax'))
    ann.add(Dropout(rate=0.5))
    
        # Add other layers, it is not necessary to pass the shape because there is a layer before
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='softmax'))
    ann.add(Dropout(rate=0.5))
    
      # Add other layers, it is not necessary to pass the shape because there is a layer before
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='softmax'))
    ann.add(Dropout(rate=0.5))
    
    # Adding the output layer
    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    # Compiling the ANN
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return ann

opt = keras.optimizers.Adam(lr=0.001)
ann = build_ann(opt)
# Training the ANN
history = ann.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_val, y_val))

pickle.dump(ann, open('model.pkl','wb'))