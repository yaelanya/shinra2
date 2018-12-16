import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.layers import Input, Embedding, Bidirectional, TimeDistributed, LSTM, Dense, concatenate, Dropout
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from seqeval.metrics import f1_score

# Fix ramdom seed.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

class MyTokenizer(object):
    def __init__(self):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.vocab_word = {self.PAD: 0, self.UNK: 1}
        self.vocab_char = {self.PAD: 0, self.UNK: 1}
        self.vocab_tag = {self.PAD: 0}
        
    def fit(self, sentences, tags, row_sentences=None):
        self._fit_word(sentences)
        
        if row_sentences:
            self._fit_char(row_sentences)
        else:
            self._fit_char(sentences)
        
        self._fit_tag(tags)
        
        self.vocab_word_size = len(self.vocab_word)
        self.vocab_char_size = len(self.vocab_char)
        self.vocab_tag_size = len(self.vocab_tag)
    
    def transform(self, sentences, tags, row_sentences=None):
        word_seq = self._trainsform_word(sentences)
        
        if row_sentences:
            char_seq = self._trainsform_char(row_sentences)
        else:
            char_seq = self._trainsform_char(sentences)
        
        tag_seq = self._transform_tag(tags)
        
        return word_seq, char_seq, tag_seq
    
    def inverse_transform_tag(self, tag_id_seq):
        seq = []
        inv_vocab_tag = {v: k for k, v in self.vocab_tag.items()}
        for tag_ids in tag_id_seq:
            tags = [inv_vocab_tag[tag_id] for tag_id in tag_ids]
            seq.append(tags)

        return seq
    
    def padding(self, word_seq, char_seq, tag_seq):
        return self._padding_word(word_seq), self._padding_char(char_seq), self._padding_tag(tag_seq)
        
    def _padding_word(self, word_seq):
        return pad_sequences(word_seq, padding='post')
    
    def _padding_char(self, char_seq):
        char_max = max([len(max(char_seq_in_sent, key=len)) for char_seq_in_sent in char_seq])
        pad_seq = [pad_sequences(char_seq_in_sent, maxlen=char_max, padding='post') for char_seq_in_sent in char_seq]
        
        # 文の長さも揃える
        return pad_sequences(pad_seq, padding='post')
    
    def _padding_tag(self, tag_seq):
        return pad_sequences(tag_seq, padding='post')
        
    
    def _fit_word(self, sentences):
        for s in sentences:
            for w in s:
                if w in self.vocab_word:
                    continue
                self.vocab_word[w] = len(self.vocab_word)
                
    def _fit_char(self, sentences):
        for s in sentences:
            for w in s:
                for c in w:
                    if c in self.vocab_char:
                        continue
                    self.vocab_char[c] = len(self.vocab_char)
                    
    def _fit_tag(self, tag_seq):
        for tags in tag_seq:
            for tag in tags:
                if tag in self.vocab_tag:
                    continue
                self.vocab_tag[tag] = len(self.vocab_tag)
                
    def _trainsform_word(self, sentences):
        seq = []
        for s in sentences:
            word_ids = [self.vocab_word.get(w, self.vocab_word[self.UNK]) for w in s]
            seq.append(word_ids)
            
        return seq
    
    def _trainsform_char(self, sentences):
        seq = []
        for s in sentences:
            char_seq = []
            for w in s:
                char_ids = [self.vocab_char.get(c, self.vocab_char[self.UNK]) for c in w]
                char_seq.append(char_ids)
            seq.append(char_seq)
            
        return seq
    
    def _transform_tag(self, tag_seq):
        seq = []
        for tags in tag_seq:
            tag_ids = [self.vocab_tag[tag] for tag in tags]
            seq.append(tag_ids)

        return seq
    

def batch_iter(data, labels, batch_size, tokenizer, shuffle=True):
    num_batches_per_epoch = int((len(data[0]) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data[0])
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = [np.array(data[0])[shuffle_indices], np.array(data[1])[shuffle_indices]]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = [shuffled_data[0][start_index: end_index], shuffled_data[1][start_index: end_index]]
                y = shuffled_labels[start_index: end_index]
                
                X[0], X[1], y = tokenizer.padding(X[0], X[1], y)
                
                yield X, y

    return num_batches_per_epoch, data_generator()


# load data
train_df = pd.read_pickle("../data/Production_train_repl_compound.pkl")
test_df = pd.read_pickle("../data/Production_test_repl_compound.pkl")

tokenizer = MyTokenizer()
tokenizer.fit(sentences=train_df.repl_words.tolist(), row_sentences=train_df.words.tolist(), tags=train_df.tag.tolist())

param = {
    'char_vocab_size': tokenizer.vocab_char_size
    , 'word_vocab_size':tokenizer.vocab_word_size
    , 'tag_size': tokenizer.vocab_tag_size
    , 'char_emb_dim': 25
    , 'word_emb_dim': 100
    , 'char_lstm_units': 25
    , 'word_lstm_units': 100
    , 'dropout_rate': 0.5
    , 'activation': 'tanh'
    , 'optimizer': 'adam'
}


# Model
char_input = Input(shape=(None, None))
word_input = Input(shape=(None,))

char_emb = Embedding(input_dim=param['char_vocab_size']
                     , output_dim=param['char_emb_dim']
                     , mask_zero=True)(char_input)
char_emb = TimeDistributed(Bidirectional(LSTM(units=param['char_lstm_units'], activation=param['activation'])))(char_emb)

word_emb = Embedding(input_dim=param['word_vocab_size']
                     , output_dim=param['word_emb_dim']
                     , mask_zero=True)(word_input)

feats = concatenate([char_emb, word_emb])

feats = Dropout(param['dropout_rate'])(feats)

feats = Bidirectional(LSTM(units=param['word_lstm_units'], return_sequences=True, activation=param['activation']))(feats)

feats = Dense(param['tag_size'])(feats)

crf = CRF(param['tag_size'])
pred = crf(feats)

model = Model(inputs=[word_input, char_input], outputs=[pred])

sgd = SGD(lr=0.001)
adam = Adam()

model.compile(loss=crf.loss_function, optimizer=adam)


x_word_train, x_char_train, y_train = \
    tokenizer.transform(sentences=train_df.repl_words.tolist(), row_sentences=train_df.words.tolist(), tags=train_df.tag.tolist())
y_train = np.array([np.identity(tokenizer.vocab_tag_size)[tags] for tags in y_train])

x_word_test, x_char_test, y_test = \
    tokenizer.transform(sentences=test_df.repl_words.tolist(), row_sentences=test_df.words.tolist(), tags=test_df.tag.tolist())
y_test = np.array([np.identity(tokenizer.vocab_tag_size)[tags] for tags in y_test])

batch_size = 128
train_steps, train_batches = batch_iter([x_word_train, x_char_train], y_train, batch_size, tokenizer)
valid_steps, valid_batches = batch_iter([x_word_test, x_char_test], y_test, batch_size, tokenizer)

model.fit_generator(train_batches, train_steps
                    ,epochs=100
                    , validation_data=valid_batches
                    , validation_steps=valid_steps
                   )

model.save("../model/BiLSTM_CRF_NER_20181214.h5")
