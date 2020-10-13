import os
import pickle


def set_params():

    params = {}
    # setting file path parameters
    params['Section'] = 'generative_comps'
    params['run_id'] = '0007'
    params['music_name'] = 'melody'
    # setting directory path parameters
    params['run_path'] = 'run/' + params['Section'] + '/'
    params['run_folder'] = params['run_path'] + '_'.join([params['run_id'], params['music_name']])
    params['store_folder'] = os.path.join(params['run_folder'], 'storage')
    params['data_folder'] = os.path.join('data', params['music_name'])
    params['weights_folder'] = os.path.join(params['run_folder'], 'weights')
    params['output_folder'] = os.path.join(params['run_folder'], 'output')
    # data parameters
    params['intervals'] = range(1)
    params['sequence_length'] = 32
    params['max_sequence_len'] = 32
    # model parameters
    params['embed_size'] = 100
    params['rnn_units'] = 256
    params['use_attention'] = True
    params['EPOCHS'] = 200000
    params['BATCH_SIZE'] = 32
    params['VAL_SPLIT'] = 0.2
    params['model_name'] = 'RNN_Composer.h5'
    # mode
    params['mode'] = 'build'
    # loading model
    params['notes_temp'] = 0.5
    params['duration_temp'] = 0.5
    params['max_addit_notes'] = 125
    # prediction params
    params['notes_temp'] = 0.5
    params['duration_temp'] = 0.5
    params['max_extra_notes'] = 50
      
    return params






      