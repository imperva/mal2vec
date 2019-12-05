import re
import os
import gensim
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from keras_tqdm import TQDMNotebookCallback
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm
tqdm.pandas()


def prepare_dataset(sentences_df, sentences_filepath, dataset_name, vector_size=30, epochs=50, min_sentence_count=200):
    print('Step 1/2: Extracting embedding using Doc2Vec (VECTOR_SIZE: {}; EPOCHS: {})'.format(vector_size, epochs))
    sentences = gensim.models.doc2vec.TaggedLineDocument(sentences_filepath)

    doc2vec_model = gensim.models.Doc2Vec(sentences, 
                                          epochs=epochs, 
                                          vector_size=vector_size, 
                                          workers=multiprocessing.cpu_count(), 
                                          min_count=1
                                         )

    output_folder = os.path.join(os.getcwd(), 'models', dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model_filepath = os.path.join(output_folder, 'doc2vec_model.h5')
    doc2vec_model.save(model_filepath)
    

    print('Step 2/2: Preparing dataset')
    # -----------------------------------
    sentences_df = sentences_df.groupby('label').filter(lambda x: len(x) > min_sentence_count)
    min_row_count = sentences_df.groupby('label').size().min()
    sentences_df = sentences_df.groupby('label').apply(lambda x: x.sample(n=min_row_count)).reset_index(drop=True)

    X = np.array([doc2vec_model.infer_vector(sentence.split(' ')) for sentence in sentences_df['sentence']])

    encoder = LabelEncoder().fit(sentences_df['label'])
    encoded_Y = encoder.transform(sentences_df['label'])
    n_classes = len(encoder.classes_)

    y = to_categorical(encoded_Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    y_test = y_test.argmax(axis=1)

    sentences_df.groupby('label').size()
    
    return X_train, X_test, y_train, y_test, encoder.classes_

def train_classifier(X_train, X_test, y_train, y_test, classes, dataset_name):
    output_folder = os.path.join(os.getcwd(), 'models', dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_filepath = os.path.join(output_folder, 'classification_model.h5')

    print('Training Model')
    # -------------------------------
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.998, amsgrad=True)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(X_train, 
                        y_train, 
                        callbacks=[
                            TQDMNotebookCallback(leave_outer=True),
                            EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto'), 
                            TerminateOnNaN(), 
                            ModelCheckpoint(model_filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
                        ],
                        validation_split=0.2, 
                        batch_size=32, 
                        epochs=500, 
                        verbose=0)

    model.load_weights(model_filepath)
    
    print('Evaluating model')
    # ---------------------------------
    y_pred = model.predict(X_test).argmax(axis=1)
    
    report = classification_report(y_test, y_pred, target_names=classes)
    df_cm = confusion_matrix(y_test, y_pred)

    return history, report, df_cm














