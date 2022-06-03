#!/usr/bin/env python
# coding: utf-8

"""

Convnet training/fine-tuning.

Created on Thu Jul  4 23:37:36 2019

@author: vlado
"""
import os
import pickle
import random
import sklearn
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


class LRScheduleSteps(Callback):
  def __init__(self, opt, lr, w, p, factor=10.):
    super(LRScheduleSteps, self).__init__()
    self.optimizer = opt
    self.lr = lr
    self.w = w
    self.p = p
    self.factor = factor
    self.history = {}

    self.step = 0

  def on_train_begin(self, logs={}):
      logs = logs or {}

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}

    if self.step < self.w:
      lr = self.lr / self.w * self.step
    else:
      lr = K.get_value(self.optimizer.lr)
      for step in self.p:
        if step == self.step:
          lr = lr / self.factor

    self.history.setdefault('lr', []).append(K.get_value(self.optimizer.lr))
    self.history.setdefault('iterations', []).append(self.step)

    for k, v in logs.items():
        self.history.setdefault(k, []).append(v)

    K.set_value(self.optimizer.lr, lr)
    self.step += 1


def preprocess(image, preproc):
    if preproc:
      image = tf.cast(image, tf.float32)
      means = tf.constant(np.reshape([123.68, 116.779, 103.939], (1, 1, 3)),
                          dtype=tf.float32)
      image = tf.math.subtract(image, means)
    else:
      image = tf.image.convert_image_dtype(image, tf.float32)

    return image

class LoadPreprocessImage():
  def __init__(self, image_path, 
               load_size=(256, 256), 
               dim=(224, 224),
               crop_size=(32, 256),
               preproc=False):
    self.image_path = image_path
    self.load_size = load_size
    self.dim = dim
    self.crop_size = crop_size
    self.preproc = preproc

  def __call__(self, record):
    image = tf.io.read_file(record['filename'])
    image = tf.image.decode_jpeg(image)
    image = preprocess(image, self.preproc)
    image = tf.image.resize(image, self.load_size)
    image = tf.image.random_crop(image, self.dim+(3,))
    n = tf.random.uniform((), maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=n)
    if tf.random.uniform(()) > 0.5:
      image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
      image = tf.image.flip_up_down(image)

    return image, record['label']


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

    return image, record['label']

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True, help='Path to the images')
parser.add_argument('-d', '--data-split', dest='data', required=True, help='Dataset split')
parser.add_argument('-n', '--name', dest='name', help='Model name')
parser.add_argument('-m', '--model', dest='model',
                    help='None/imagenet/path_to_the_pre-trained_model.')
parser.add_argument('--lr', dest='max_lr', type=float, required=True, help='Maximal learning rate.')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                    required=False, default=100,
                    help='Batch size.')

args = parser.parse_args()

dataset = args.data
batch_size = args.batch_size
preproc = args.model is not None and 'imagenet' in args.model

with open(f'data_splits/{dataset}-split.pkl', 'rb') as f:
    data_partition = pickle.load(f)
with open(f'data_splits/{dataset}-le.pkl', 'rb') as f:
    le = pickle.load(f)

data_partition['train']['filename'] = [os.path.join(args.path, fname) for fname in data_partition['train']['filename']]
data_partition['test']['filename'] = [os.path.join(args.path, fname) for fname in data_partition['test']['filename']]

strategy = tf.distribute.MirroredStrategy()

nr_labels = len(data_partition['train']['label'][0])
nr_training = len(data_partition['train']['filename'])
steps_per_epoch = nr_training // batch_size

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

train_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['train'])
train_ds = train_list_ds.shuffle(nr_training)
train_ds = train_ds.map(LoadPreprocessImage(args.path,
                                            load_size=(292, 292),
                                            dim=(256, 256),
                                            crop_size=(32, 256),
                                            preproc=preproc),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
train_ds = train_ds.with_options(options)

test_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['test'])
test_ds = test_list_ds.map(LoadPreprocessImageVal(args.path,
                                                  load_size=(292, 292),
                                                  dim=(256, 256),
                                                  preproc=preproc),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.with_options(options)

with strategy.scope():
    if args.model is None or args.model == 'imagenet':
        base_model = ResNet50(include_top=False,
                              weights=args.model,
                              input_shape=(256, 256, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    else:
        base_model = tf.keras.models.load_model(args.model)
        x = base_model.layers[-1].input

    x = Dense(nr_labels, activation='sigmoid')(x)
    clf = Model(base_model.input, x)

    opt = optimizers.Adam(learning_rate=1e-5)
    clf.compile(opt,
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall()])

lrsched = LRScheduleSteps(opt, args.max_lr, 5*steps_per_epoch,
                          [50*steps_per_epoch, 70*steps_per_epoch, 90*steps_per_epoch], factor=5.)

checkpoint = ModelCheckpoint('models/checkpoint.h5',
                             save_freq=10*batch_size,
                             save_best_only=False)

h = clf.fit(train_ds,
            epochs=100,
            callbacks=[checkpoint, lrsched],
            validation_data=test_ds)

filename = 'models/rssc_resnet50'
if args.model == 'imagenet':
    filename += '_imagenet'
filename += '_{}_multilabel'.format(dataset)
if args.model is not None:
    filename += '_ft'
if args.name is not None:
    filename += '_{}'.format(args.name)
filename += '.h5'

clf.save(filename)

y_pred = clf.predict(test_ds,
                     batch_size=batch_size,
                     verbose=1)

print(classification_report(data_partition['test']['label'], y_pred>0.5))


