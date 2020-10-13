
import os

from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from keras.layers import Multiply, Lambda, Softmax
import keras.backend as K   
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils import np_utils



class RNN_Attention:

    def __init__(self, params):

        self.n_notes = params['n_notes']
        self.n_durations = params['n_durations']
        self.embed_size = params['embed_size']
        self.rnn_units = params['rnn_units']
        self.use_attention = params['use_attention']

        self.weights_folder = params['weights_folder']

        self.notes_in = Input(shape = (None,))
        self.durations_in = Input(shape = (None,))

        self.note_embed = Embedding(self.n_notes, self.embed_size)(self.notes_in)
        self.duration_embed = Embedding(self.n_durations, self.embed_size)(self.durations_in)

        self.lstm = LSTM(self.rnn_units, return_sequences=True, activation='relu')
        self.lstm_ = LSTM(self.rnn_units, return_sequences=True, activation='tanh')
        self.lstm_no_seq = LSTM(self.rnn_units)
        self.drop_2 = Dropout(0.2)
        self.fcl = Dense(1, activation='tanh')
        self.alpha = Activation('softmax')

        self.notes_out = Dense(self.n_notes, activation='softmax', name='pitch')
        self.durations_out = Dense(self.n_durations, activation='softmax', name='duration')

        self.opt = RMSprop(lr = 0.001)

        self.model, self.attn_model = self.construct_network()
        self.model_name = params['model_name']


        self.chckpnt_progress = ModelCheckpoint(os.path.join(self.weights_folder, 'weights-improvement-{epoch:02d}-{loss:.4f}-increase.h5'),
                                                        monitor='loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='min')
        self.chckpnt_final = ModelCheckpoint(os.path.join(self.weights_folder, 'weights.h5'),
                                                    monitor='loss',
                                                    verbose=0,
                                                    save_best_only=True,
                                                    mode='min')

        self.es = EarlyStopping(monitor='loss', restore_best_weights=True, patience=20)
        self.callback_list = [self.chckpnt_progress, self.chckpnt_final, self.es]

        self.epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.validation_split = params['VAL_SPLIT']

    
    def construct_network(self):    

        embed_layer = Concatenate()([self.note_embed, self.duration_embed])
        lstm_1 = self.lstm(embed_layer)
        x = self.drop_2(lstm_1)

        if self.use_attention:

            lstm_2 = self.lstm_(x)
            x = self.drop_2(lstm_2)
            
            fcl = self.fcl(x)
            fcl = Reshape([-1])(fcl)
            alpha = self.alpha(fcl)

            repeat_alpha = Permute([2, 1])(RepeatVector(self.rnn_units)(alpha))

            x = Multiply()([x, repeat_alpha])
            x = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(self.rnn_units,))(x)
        
        else: 
            lstm_2 = self.lstm_no_seq(x)
            x = self.drop_2(lstm_2)

        notes_out = self.notes_out(x)
        durations_out = self.durations_out(x)

        model = Model([self.notes_in, self.durations_in], [notes_out, durations_out])

        if self.use_attention:
            attn_model = Model([self.notes_in, self.durations_in], alpha)

        else:
            attn_model = None

        model.compile(loss=['categorical_crossentropy', 
                            'categorical_crossentropy'],
                      optimizer=self.opt,
                      metrics=['acc'])
        model.summary()

        return model, attn_model


    def train_and_save(self, net_in, net_out):

        net = self.model
        net.save_weights(os.path.join(self.weights_folder, 'weights.h5'))
        net.fit(net_in, net_out,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks= self.callback_list,
                shuffle=True)

        net.save(self.model_name)
        print('Model Saved')

        return None
        



            




