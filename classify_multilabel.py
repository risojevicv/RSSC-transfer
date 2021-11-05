#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Multilabel classification based on extracted features

Created on Tue Aug 27 08:49:25 2019

@author: vlado
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
batch_size = 100

parser = argparse.ArgumentParser()
parser.add_argument('features',
                    help='Path to the features file.')
args = parser.parse_args()
head, tail = os.path.split(args.features)
name, _ = os.path.splitext(tail)

npzfile = np.load(args.features)
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_test']
y_test = npzfile['y_test']
nr_classes = y_train.shape[1]
nr_training = len(x_train)

ss = StandardScaler()
ss.fit_transform(x_train)
ss.transform(x_test)

clf = Sequential()
clf.add(Dense(nr_classes, activation='sigmoid', input_shape=(2048,)))

opt = Adam(lr=1e-3)
clf.compile(optimizer=opt, 
            loss='binary_crossentropy', 
            metrics=[tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()])

h = clf.fit(x=x_train, 
            y=y_train,
            batch_size=batch_size,
            epochs=100,
            validation_data=(x_test, y_test),
            verbose=1)

y_pred = clf.predict(x_test, 
                    batch_size=batch_size,
                    verbose=1)
print(classification_report(y_test, y_pred>0.5))


