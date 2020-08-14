import re
import tensorflow as tf
import tensorflow_hub as hub
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras
from collections import Counter
import os
import sys


#reload(sys)
#sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def clean_str(s):
	"""Clean sentence"""
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	s = s.strip().lower()
	#s = nltk.sent_tokenize(s)
	return s

def load_data_and_labels(filename):
	"""Load sentences and labels"""
	df = pd.read_csv(filename, compression='zip', dtype={'description': object})
	selected = ['assigned_to', 'description']
	#selected = ['assigned_to', 'description', 'summary']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drop non selected columns
	df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
	df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
	#df['merge'] = df.summary + ' ' + df.description

	# Map the actual labels to one hot labels
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))
	
	#x_raw = df['merge'].apply(lambda x: clean_str(x)).tolist()
	x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels



""""""
""" def load_data_and_labels(filename):
    #Load sentences and labels
    df = pd.read_csv(filename, compression='zip', dtype={'description': object})
    selected = ['assigned_to', 'description']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1) # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
    df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe

	# Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    #x = clean_str(df[selected[1]])
    x_raw = df[selected[1]].apply(lambda x: clean_str(x))
    y=df[selected[0]]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_enc=encode(le,y)

    #x_enc = df[selected[1]
    
    #x_raw = np.asarray(x_enc[:])
    y_raw = np.asarray(y_enc[:])
    
    #y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    return x_raw, y_raw, df, labels, y 

def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec) """

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	#Iterate the data batch by batch
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
"""
def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors """

def load_embedding_vectors_elmo(sentences, vector_size):
	url = "https://tfhub.dev/google/elmo/2"
	embed = hub.Module(url)
	embeddings = embed(sentences, signature="default", as_dict=True)["default"]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.tables_initializer())
		x = sess.run(embeddings)
	print(x.shape)
	#pca = PCA(n_components=300) #reduce down to 300 dim from 1024
	#embeddings = pca.fit_transform(x)
	return embeddings
