#!/usr/bin/env python
# coding: utf-8

"""

Extract features using a pre-trained model.

Created on Thu Jul  4 23:37:36 2019

@author: vlado
"""
import os
import pickle
import random
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def preprocess(image, preproc):
    if preproc:
      image = tf.cast(image, tf.float32)
      means = tf.constant(np.reshape([123.68, 116.779, 103.939], (1, 1, 3)),
                          dtype=tf.float32)
      image = tf.math.subtract(image, means)
    else:
      image = tf.image.convert_image_dtype(image, tf.float32)

    return image

class LoadPreprocessImageVal():
  def __init__(self, image_path,
               load_size=(256, 256), 
               dim=(224, 224), 
               preproc=False):
    self.image_path = image_path
    self.load_size = load_size
    self.dim = dim
    self.preproc = preproc

  def __call__(self, record):
    image = tf.io.read_file(record['filename'])
    image = tf.image.decode_jpeg(image)
    image = preprocess(image, self.preproc)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, self.load_size)
    image = tf.image.central_crop(image, self.dim[0]/self.load_size[0])

    return image, record['class']

batch_size = 100

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True, help='Path to the images')
parser.add_argument('-d', '--data', dest='data', required=True, help='Dataset name')
parser.add_argument('-f', '--features', dest='features', required=True, help='Path to the features file')
parser.add_argument('-m', '--model', dest='model', required=True,
                    help='imagenet/path_to_the_pre-trained_model.')
args = parser.parse_args()

dataset = args.data
preproc = 'imagenet' in args.model

with open(f'data_splits/{dataset}-split.pkl', 'rb') as f:
    data_partition = pickle.load(f)

data_partition['train']['filename'] = [os.path.join(args.path, fname) for fname in data_partition['train']['filename']]
data_partition['test']['filename'] = [os.path.join(args.path, fname) for fname in data_partition['test']['filename']]

strategy = tf.distribute.MirroredStrategy()

train_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['train'])
train_ds = train_list_ds.map(LoadPreprocessImageVal(args.path,
                                                    load_size=(292, 292),
                                                    dim=(256, 256),
                                                    preproc=preproc),
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['test'])
test_ds = test_list_ds.map(LoadPreprocessImageVal(args.path,
                                                  load_size=(292, 292),
                                                  dim=(256, 256),
                                                  preproc=preproc),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

with strategy.scope():
  if args.model == 'imagenet':
    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          pooling='avg')
    fex = Model(base_model.input, base_model.output)
  else:
    base_model = tf.keras.models.load_model(args.model)
    fex = Model(base_model.input, base_model.layers[-2].output)

x_train = fex.predict(train_ds, verbose=1)
y_train = np.array(data_partition['train']['class'])
x_test = fex.predict(test_ds, verbose=1)
y_test = np.array(data_partition['test']['class'])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.savez(args.features, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
