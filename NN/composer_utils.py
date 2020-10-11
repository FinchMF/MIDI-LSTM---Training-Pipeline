import os 
import numpy as np
import pickle
import glob

from music21 import corpus, converter, chord, note

from keras.utils import np_utils



def fetch_midi_list(data):

    if data == 'chorales':
        file_list = ['bwv' + str(x['bwv']) for x in corpus.chorales.ChoraleList().byBWV.values()]
        parser = corpus
    else:
        file_list = glob.glob(os.path.join(data, '*.mid'))
        parser = converter

    return file_list, parser

def fetch_set(ele):
    
    ele_names = sorted(set(ele))
    n_ele = int(len(ele_names))

    return (ele_names, n_ele)

def gen_lookup_table(ele_names):

    ele_to_int = dict((ele, num) for num, ele in enumerate(ele_names))
    int_to_ele = dict((num, ele) for num, ele in enumerate(ele_names))

    return (ele_to_int, int_to_ele)


def preprocess_seq(notes, durations, lookups, distincts, seq_len):

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, durations_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    for i in range(len(notes) - seq_len):
        notes_seq_in = notes[i:i + seq_len]
        notes_seq_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_seq_in])
        notes_network_output.append([note_to_int[notes_seq_out]])

        durations_seq_in = durations[i:i + seq_len]
        durations_seq_out   = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_seq_in])
        durations_network_output.append(duration_to_int[durations_seq_out])

    n_patterns = len(notes_network_input)

    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))

    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)

    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)


def sample_with_temp(preds, temperature):

    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return np.random.choice(len(preds), p=preds)



def set_distincts(notes, durations, params):

    note_names, n_notes = fetch_set(notes)
    durations_names, n_durations = fetch_set(durations)

    sets = [note_names, n_notes, durations_names, n_durations]

    with open(os.path.join(params['store_folder'], 'sets'), 'wb') as f:
        pickle.dump(sets, f)

    return sets

def set_lookups(sets, params):

    note_to_int, int_to_note = gen_lookup_table(sets[0])
    duration_to_int, int_to_duration = gen_lookup_table(sets[2])

    lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

    with open(os.path.join(params['store_folder'], 'lookups'), 'wb') as f:
        pickle.dump(lookups, f)

    return lookups

def make_lookup_tables(notes, durations, params):

    sets = set_distincts(notes, durations, params)
    lookups = set_lookups(sets, params)

    return (sets, lookups)


def note_extraction(params):

    if params['mode'] == 'build':

        music_list, parser = fetch_midi_list(params['data_folder'])
        print(len(music_list), 'files in total')

        notes = []
        durations = []

        for idx, file in enumerate(music_list):
            print(idx+1, f'Parsing {file}')
            original_score = parser.parse(file).chordify()

            for intvl in params['intervals']:

                score = original_score.transpose(intvl)

                notes.extend(['START'] * params['sequence_length'])
                durations.extend([0] * params['sequence_length'])

                for element in score.flat:

                    if isinstance(element, note.Note):

                        if element.isRest:
                            notes.append(str(element.name))
                            durations.append(str(element.duration.quarterLength))
                        else:
                            notes.append(str(element.nameWithOctave))
                            durations.append(str(element.duration.quarterLength))

                    if isinstance(element, chord.Chord):

                        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                        durations.append(element.duration.quarterLength)

        with open(os.path.join(params['store_folder'], 'notes'), 'wb') as f:
            pickle.dump(notes, f)

        with open(os.path.join(params['store_folder'], 'durations'), 'wb') as f:
            pickle.dump(durations, f)
    else:
        with open(os.path.join(params['store_folder'], 'notes'), 'rb') as f:
            pickle.dump(notes, f)
        with open(os.path.join(params['store_folder'], 'durations'), 'rb') as f:
            pickle.dump(durations, f)


    return notes, durations


    def gen_RNN_input(notes, durations, sets, lookups, seq_len):

        net_in, net_out = preprocess_seq(notes, durations, sets, lookups, seq_len)

        return net_in, net_out







