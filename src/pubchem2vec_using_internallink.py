import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Dot, Dense
from keras.preprocessing.sequence import skipgrams, make_sampling_table

EMBED_DIM = 0
LINK_SIZE = 0
tokenizer = None

class LinkTokenizer(object):
    def __init__(self):
        self.num_links = 0
        self.link_index = {}
        
    def fit(self, links: list):
        _links = np.array(list(set(links)))
        _links.sort()
        self.link_index = {link: i for i, link in enumerate(_links, 1)}
        self.num_links = len(self.link_index)
        
    def link_to_index(self, links: list):
        return np.array([self.link_index.get(link) if self.link_index.get(link) else 0 for link in links])

def main(**argv):
    global  EMBED_DIM, LINK_SIZE, tokenizer
    
    print("Loading data...")
    internal_link_df = pd.read_csv(argv['data_path'])
    
    tokenizer = LinkTokenizer()
    tokenizer.fit(np.append(internal_link_df.entry.values, internal_link_df.linked.values))
    
    internal_link_df['entry'] = tokenizer.link_to_index(internal_link_df.entry.values)
    internal_link_df['linked'] = tokenizer.link_to_index(internal_link_df.linked.values)

    LINK_SIZE = tokenizer.num_links + 1
    EMBED_DIM = argv['embedding_dim']

    model = build()
    print("Start training...")
    train(model, internal_link_df, argv['epochs'], argv['negative_samples'])
    print("Save model...")
    save(model, argv['output_path'])

def train(model, data, epochs=1, negative_samples=1.0):
    all_links = np.array(list(tokenizer.link_index.values()))

    for i in enumerate(range(epochs), 1):
        loss = 0.0
        for entry, link in tqdm(data.groupby('linked')):
            backlinks = link.entry.values
            entry, link, label = neg(entry, backlinks, all_links, negative_samples)
            loss += model.train_on_batch([entry, link], label)
            break

        print("Epoch {i}/{epochs}\tloss: {loss}".format(**locals()))

def build():
    input_target_entry = Input(shape=(1,), dtype='int32', name='input1')
    input_linked_entry = Input(shape=(1,), dtype='int32', name='input2')

    embedding = Embedding(LINK_SIZE, EMBED_DIM, name='embedding1')
    target_entry = embedding(input_target_entry)
    linked_entry = embedding(input_linked_entry)

    dot = Dot(axes=2, name='dot1')([target_entry, linked_entry])
    dot = Reshape((1,), name='reshape1')(dot)
    output = Dense(1, activation='sigmoid', name='dense1')(dot)

    model = Model(inputs=[input_target_entry, input_linked_entry], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def neg(entry, backlink, all_links, negative_samples=1.0):
    neg_samples = np.random.choice(np.setdiff1d(all_links, backlink), size=int(len(backlink) * negative_samples))
    e = [entry] * (len(backlink) + len(neg_samples))
    links = np.append(backlink, neg_samples)
    label = np.array([1] * len(backlink) + [0] * len(neg_samples))
    
    return e, links, label

def save(model, output_path):
    with open(output_path ,'w') as f:
        size = LINK_SIZE - 1
        embedding_dim = EMBED_DIM
        f.write("{size} {embedding_dim}\n".format(**locals()))
        vectors = model.get_weights()[0]
        for cid, i in tokenizer.link_index.items():
            vector = ' '.join(map(str, list(vectors[i, :])))
            f.write("{cid} {vector}\n".format(**locals()))

if __name__ == '__main__':
    argv = sys.argv
    argd = {}
    argd['embedding_dim'] = int(argv[1])
    argd['epochs'] = int(argv[2])
    argd['negative_samples'] = float(argv[3])
    argd['data_path'] = str(argv[4])
    argd['output_path'] = str(argv[5])

    main(**argd)