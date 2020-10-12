
import pickle as pickle
import time
import os
import sys
import numpy as np 

from music21 import instrument, note, stream, chord, duration

from NN import composer_params
from NN import composer_RNN
from NN import composer_utils

params = composer_params.set_params()

sets, lookups = composer_utils.load_lookups(params)

model, attn_model = composer_utils.load_NN_with_weights(sets, lookups, params, composer_RNN.RNN_Attention(params).construct_network())


