
import pickle as pickle
import time
import os
import sys
import numpy as np 

from music21 import instrument, note, stream, chord, duration

from NN import composer_params
from NN import composer_RNN
from NN import composer_utils

def RNN_Gen_Midi():

    print('Preparing for Midi Generation\n')
    params = composer_params.set_params()
    print('Generating....\n')
    sets, lookups = composer_utils.load_lookups(params)

    params['n_notes'] = sets[1]
    params['n_durations'] = sets[3]

    model, attn_model = composer_utils.load_NN_with_weights(sets, lookups, params, composer_RNN.RNN_Attention(params))

    notes, durations, sequence_length = composer_utils.get_seq_info(params)

    preds = composer_utils.generate_notes_from_input(params,
                                                    lookups[0], lookups[1],
                                                    lookups[2], lookups[3],
                                                    notes, durations, 
                                                    model, attn_model,
                                                    sequence_length)

    composer_utils.convert_pred_to_midi(preds, params)
    print('Midi Generated and Saved')
    return None


if __name__ == "__main__":
    
    RNN_Gen_Midi()







