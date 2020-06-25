from flask import Flask, render_template, url_for
import json, random
import numpy as np
from textblob import TextBlob

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

app = Flask(__name__)

DIR = 'E:/Models/Inspyrobot/RNN'
def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model

def generate_text(start_seed, temperature=1.0):

    input_eval = [char_to_ind[s] for s in start_seed]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = list(start_seed)
    model.reset_states()

    while text_generated[-1] != '~':
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(ind_to_char[predicted_id])

    return str(TextBlob(''.join(text_generated[:-1])).correct())

vocab_size = 38
embed_dim = 64
rnn_neurons = 1024

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights(f'{DIR}/model.h5')
model.build(tf.TensorShape([1, None]))

char_to_ind = json.load(open(f'{DIR}/vocab_mapping.json'))
ind_to_char = np.array(json.load(open(f'{DIR}/vocab.json')))
words = json.load(open(f'{DIR}/words.json'))

@app.route('/')
def home():
    return render_template('inspyrobot.html')

@app.route('/inspyre/',methods=['POST'])
def inspyre():
    message = generate_text(random.choice(words), temperature=random.random())
    return render_template('inspyrobot.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
