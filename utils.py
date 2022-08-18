import boto3
import yaml

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.preprocessing.image_preprocessing import Rescaling
from keras.models import Sequential
import tensorflow as tf

def load_credentials(path):
    with open(path) as file:
        credentials = yaml.load(file)
    
    return credentials
    
def get_bucket(credentials):
    session = boto3.Session(
        aws_access_key_id=credentials['S3']['aws_access_key_id'], 
        aws_secret_access_key=credentials['S3']['aws_secret_access_key'], 
        aws_session_token=credentials['S3']['aws_session_token']
    )
    return session.resource('s3').Bucket(credentials['S3']['BUCKET_NAME'])

def plot_confusion_matrix(y_test, predictions, label_dict):
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    confusion_df = pd.DataFrame(conf_matrix)
    confusion_df.rename(index=label_dict, columns=label_dict, inplace=True)

    plt.figure(figsize=(14,10))
    sns.heatmap(confusion_df, annot=True, cmap='cividis')

    plt.ylabel('True values', fontdict={'fontsize':20}, labelpad=20)
    plt.xlabel('Predicted values', fontdict={'fontsize':20}, labelpad=20)
    plt.show()

def fn_train_test_split(X, y, train_size, seed):
    '''
    Params:
    X, y, seed
    
    Returns
    X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed, stratify=y)

    X_train_dim = keras.backend.expand_dims(X_train, axis=-1) # or axis=3
    X_test_dim = keras.backend.expand_dims(X_test, axis=-1) # or axis=3

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def fn_cnn_model(input_shape, num_classes):
    cnn = Sequential([
        Rescaling(1./255, input_shape=input_shape),

        Conv2D(filters=16, 
               kernel_size=(3,3), 
               activation='relu'),

        MaxPooling2D(pool_size=(2,4)),

        Conv2D(filters=32,
               kernel_size=(3,3), 
               activation='relu'),  
               
        MaxPooling2D(pool_size=(2,4)),
        
        Flatten(),

        Dense(64, activation='relu'),
        Dropout(.25),
        Dense(num_classes, activation='softmax')
    ])

    cnn.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    return cnn
    