#!/usr/bin/python
# -*- coding: utf-8 -*
'''
api.py: DeepBot Api Server
'''

import json
from flask import Flask, render_template, request, jsonify
from model import seq2seq
from data_process import sentence_to_char_index
import tensorflow as tf
import json


from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__, static_url_path='')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'some-secret-string'

db = SQLAlchemy(app)


class Message(db.Model):
  __table_name__ = 'message'
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(100))
  message = db.Column(db.String(1000))

  def __init__(self, message, name, **kwargs):
      self.message = message
      self.name = name

db.create_all()


def add_message(text, name):
    message = Message(message=text, name=name)
    db.session.add(message)
    db.session.commit()





PATH = "models"

# load vocab, reverse_vocab, vocab_size
with open('vocab.json', 'r') as fp:
    vocab = json.load(fp)
reverse_vocab = dict()
for key, value in vocab.items():
    reverse_vocab[value] = key
vocab_size = len(vocab)

# open session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# make model instance
model = seq2seq(sess, encoder_vocab_size=vocab_size, decoder_vocab_size=vocab_size, max_step=50)

# load trained model
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(PATH))



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/chatting', methods=['GET', 'POST'])
def chatting():
    message = request.json['message']
    add_message(message, 'User')

    speak = sentence_to_char_index([message], vocab, False)
    result = model.inference([speak])
    for sentence in result:
        response = ''
        for index in sentence:
            if index == 0:
                break
            response += reverse_vocab[index]

    add_message(response, 'DeepBot')
    return jsonify({'message': response})


if __name__ == '__main__':
    app.run(debug=True)
