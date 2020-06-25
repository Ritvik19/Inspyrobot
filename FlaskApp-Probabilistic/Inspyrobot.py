from flask import Flask, render_template, url_for
import json, random
from nltk import ngrams

app = Flask(__name__)

DIR = 'E:/Models/Inspyrobot/Probabilistic'
n = 4
q = lambda x : list(ngrams(x.lower().split(), n, pad_left=True, pad_right=False,  left_pad_symbol=''))[-1]
with open(f'{DIR}/model.json', 'r') as f:
    data = json.load(f)
    dic = json.loads(data)
    k = dic.keys()
    v = dic.values()
    k1 = [eval(i) for i in k]
    model =  dict(zip(*[k1,v]))
with open(f'{DIR}/vocab.json', 'r') as f:
    vocab = json.loads(json.load(f))

@app.route('/')
def home():
    return render_template('inspyrobot.html')

@app.route('/inspyre/',methods=['POST'])
def inspyre():
    text = str(random.choice(vocab))
    sentence_finished = False
    while not sentence_finished:
        r = random.random()
        accumulator = .0
        if model[q(text)].keys() is None or (len(model[q(text)].keys()) == 1 and list(model[q(text)].keys())[0] == ''):
            break
        for word in model[q(text)].keys():
            accumulator = model[q(text)][word]
            if word != 'endquote' and accumulator >= r:
                text += ' '+word
                break
            if word == 'endquote':
                sentence_finished = True
    message = ' '.join([t for t in text.split() if t]).capitalize()
    return render_template('inspyrobot.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
