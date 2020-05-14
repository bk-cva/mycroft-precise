# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import attr
from os.path import isfile
from typing import *

from precise.functions import load_keras, false_pos, false_neg, weighted_log_loss, set_loss_bias
from precise.params import inject_params, pr

if TYPE_CHECKING:
    from keras.models import Sequential


@attr.s()
class ModelParams:
    """
    Attributes:
        recurrent_units:
        dropout:
        extra_metrics: Whether to include false positive and false negative metrics
        skip_acc: Whether to skip accuracy calculation while training
    """
    recurrent_units = attr.ib(20)  # type: int
    dropout = attr.ib(0.2)  # type: float
    extra_metrics = attr.ib(False)  # type: bool
    skip_acc = attr.ib(False)  # type: bool
    loss_bias = attr.ib(0.7)  # type: float
    freeze_till = attr.ib(0)  # type: bool


def load_precise_model(model_name: str) -> Any:
    """Loads a Keras model from file, handling custom loss function"""
    if not model_name.endswith('.net'):
        print('Warning: Unknown model type, ', model_name)

    inject_params(model_name)
    return load_keras().models.load_model(model_name)


def create_model(model_name: Optional[str], params: ModelParams) -> 'Sequential':
    """
    Load or create a precise model

    Args:
        model_name: Name of model
        params: Parameters used to create the model

    Returns:
        model: Loaded Keras model
    """
    if model_name and isfile(model_name):
        print('Loading from ' + model_name + '...')
        model = load_precise_model(model_name)
    else:
        from keras.layers.core import Dense
        from keras.layers.recurrent import GRU
        from keras.models import Sequential
        from keras.layers import Conv1D, Conv2D, MaxPooling1D, \
                                MaxPooling2D, Reshape, TimeDistributed, \
                                Dropout, LeakyReLU, Flatten


        input_shape = (pr.n_features, pr.feature_size)

        # model = Sequential()
        # model.add(GRU(
        #     32, activation='relu',
        #     input_shape=(pr.n_features, pr.feature_size), dropout=params.dropout, name='net'))
        # model.add(Dense(1, activation='sigmoid'))

        # model = Sequential()
        # model.add(GRU(
        #     32, activation='relu',
        #     input_shape=(pr.n_features, pr.feature_size), dropout=params.dropout, name='net',
        #     return_sequences=True))
        # model.add(GRU(
        #     20, activation='relu', dropout=params.dropout))
        # model.add(Dense(1, activation='sigmoid'))

        ## 2xCNN + GRU model
        # model = Sequential()
        # model.add(Reshape((pr.n_features, pr.feature_size, 1), input_shape=input_shape))
        # model.add(Conv2D(64, (8,20), padding='same', activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(MaxPooling2D(2, 2, 'same'))
        # model.add(Conv2D(64, (4,10), padding='same', activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Flatten())
        # # model.add(GRU(128,activation='relu', dropout=0.2))
        # model.add(Dense(16, activation='relu'))
        # # model.add(Dense(32))
        # model.add(Dense(1, activation='sigmoid'))

        ## CRNN
        model = Sequential()
        model.add(Reshape((pr.n_features, pr.feature_size, 1), input_shape=input_shape))
        model.add(Conv2D(48, kernel_size=(10, 4), strides=2, padding='valid', activation='relu'))
        model.add(Dropout(params.dropout))
        model.add(MaxPooling2D(2, 2, 'same'))
        model.add(Reshape((17, 3*48)))
        model.add(GRU(32, activation='relu', dropout=params.dropout, \
                    return_sequences=True))
        model.add(GRU(32,activation='relu', dropout=params.dropout))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(params.dropout))
        model.add(Dense(1, activation='sigmoid'))


    from keras.optimizers import Adam
    load_keras()
    metrics = ['accuracy'] + params.extra_metrics * [false_pos, false_neg]
    set_loss_bias(params.loss_bias)
    for i in model.layers[:params.freeze_till]:
        i.trainable = False
    optimizer = Adam(lr=0.0005)
    model.compile(optimizer, weighted_log_loss, metrics=(not params.skip_acc) * metrics)
    return model
