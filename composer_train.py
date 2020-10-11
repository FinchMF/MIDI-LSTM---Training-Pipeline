from NN import composer_params
from NN import composer_RNN
from NN import composer_utils

import build

def train_RNN():
        
    params = composer_params.set_params()
    print('[i] Paramters Instantiated ')
    build.build_directories(params['run_folder'])
    print('[i] Sub Directories checked and/or built...')
    notes, durations = composer_utils.note_extraction(params)
    print('[+] Notes and Durations Extracted')

    sets, lookups = composer_utils.make_lookup_tables(notes, durations, params)
    print('[+] Lookup Set\n')

    params['n_notes'] = sets[1]
    params['n_durations'] = sets[3]

    net_in, net_out = composer_utils.preprocess_seq(notes, durations, lookups, sets, params['sequence_length'])

    print('[i] Inpute Details:')

    print(f'Pitch Input: \n \
            {net_in[0][0]}')
    print(f'Duratoin Input: \n \
            {net_in[1][0]}') 
    print(f'Pitch Output: \n \
            {net_out[0][0]}')
    print(f'Duration Output: \n \
            {net_out[1][0]}')


    composer_RNN.RNN_Attention(params).train_and_save(net_in, net_out)

    return None


if __name__ == "__main__":

    train_RNN()


