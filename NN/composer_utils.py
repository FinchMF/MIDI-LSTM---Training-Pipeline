import os 
import time
import numpy as np
import pickle
import glob

from music21 import corpus, converter, chord, note, stream, instrument, duration

import keras.models as mods
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



def load_lookups(params):

    with open(os.path.join(params['store_folder'], 'sets'), 'rb') as f:
        sets = pickle.load(f)

    with open(os.path.join(params['store_folder'], 'lookups'), 'rb') as f:
        lookups = pickle.load(f)

    return sets, lookups


def load_NN_with_weights(sets, lookups, params, model_class):

    weights = os.path.join(params['run_folder'], 'weights')
    weight_matrix = 'weights.h5'

    model, attn_model = model_class(params).construct_network()

    trained_weights = os.path.join(weights, weight_matrix)

    model.load_weights(trained_weights)
    model.summary()

    return model, attn_model


def get_seq_info(params, notes, durations):

    if notes and durations == False:

        notes = ['START']
        durations = [0]

    else:
        notes = notes
        durations = durations

    if params['sequence_length'] is not None:
        notes = ['START'] * (params['sequence_length'] - len(notes)) + notes
        durations = [0] * (params['sequence_length'] - len(durations)) + durations

    sequence_length = len(notes)

    return notes, durations, sequence_length


def generate_notes_from_input(params, 
                              note_to_int, int_to_note, 
                              duration_to_int, int_to_duration, 
                              notes, durations, 
                              model, attn_model, 
                              sequence_length):

    prediction_output = []
    notes_input_seq = []
    durations_input_seq = []

    overall_preds = []

    for n, d in zip(notes, durations):
        note_int = note_to_int[n]
        duration_int = duration_to_int[d]

        notes_input_seq.append(note_int)
        durations_input_seq.append(duration_int)

        prediction_output.append([n, d])

        if n != 'START':
            midi_note = note.Note(n)

            new_note = np.zeros(128)
            new_note[midi_note.pitch.midi] = 1
            overall_preds.append(new_note)

    
    att_matrix = np.zeros(shape = (params['max_extra_notes'] + sequence_length, params['max_extra_notes']))

    for n_idx in range(params['max_extra_notes']):

        prediction_input = [

            np.array([notes_input_seq],
            np.array([durations_input_seq]))
            
        ]

        n_pred, d_pred = model.predict(prediction_input, verbose=0)
        
        if params['use_attention']:
            attn_pred = attn_model.predict(prediction_input, verbose=0)
            att_matrix[(n_idx - len(attn_pred)+sequence_length):(n_idx+sequence_length), n_idx] = attn_pred
            
            new_note = np.zeros(22)

            for idx, n_i in enumerate(n_pred[0]):
                try:
                    note_name = int_to_note[idx]
                    midi_note = note.Note(note_name)
                    new_note =[midi_note.pitch.midi] = n_i

                except:
                    pass

            overall_preds.append(new_note)

            i1 = sample_with_temp(n_pred[0], params['notes_temp'])
            i2 = sample_with_temp(d_pred[0], params['duration_temp'])

            n_result = int_to_note[i1]
            d_result = int_to_duration[i2]

            prediction_output.append([n_result, d_result])
            notes_input_seq.append(i1)
            durations_input_seq.append(i2)

            if len(notes_input_seq) > params['max_sequence_len']:

                notes_input_seq = notes_input_seq[1:]
                durations_input_seq = durations_input_seq[1:]

            if n_result == 'START':
                break

    overall_preds = np.transpose(np.array(overall_preds))

    return prediction_output


def convert_pred_to_midi(prediction_output, params):

    midi_stream = stream.Stream()

    for pattern in prediction_output:
        n_pattern, d_pattern = pattern

        if ('.' in n_pattern):

            notes_in_chord = n_pattern.split('.')
            chord_notes = []

            for curr_note in notes_in_chord:

                new_note = note.Note(curr_note)
                new_note.duration = duration.Duration(d_pattern)
                new_note.storedInstrument = instrument.Violoncello()
                chord_notes.append(new_note)
            
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)

        elif n_pattern == 'rest':

            new_note = note.Rest()
            new_note.duration = duration.Duration(d_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)

        elif n_pattern != 'START':

            new_note = note.Note(n_pattern)
            new_note.duration = duration.Duration(d_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)


    midi_stream = midi_stream.chordify()
    timestr = time.strftime('%Y%m%d-%H%M%S')
    out = params['output_folder']
    fp = os.path.join(out, f'output-{timestr}.mid')
    midi_stream.write('midi', fp=fp)

    print(f'Midi Generated \n \
            Saved at: {fp}')

    return None



    


