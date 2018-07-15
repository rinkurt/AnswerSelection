# -*- coding:utf-8 -*-

# -*- coding:utf-8 -*-

import gensim
import doc2vec
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier





def load_dict(filename):
	dict = defaultdict(list)
	with open(filename, 'r', encoding='UTF-8') as wordvec:
		for line in wordvec:
			lst = line.split(" ")
			vec = [float(x) for x in lst[1:-1]]
			dict[lst[0]] = vec
	return dict


def load_stop_words(filename):
	lst = []
	#lst = [",", ".", ":", ";", "?", "%", "*", "^", "-", "/", "[", "]", "，", "。", "、", "？", "！", "：",
	#		   "；", "（", "）", "(", ")", "《", "》", "【", "】", "“", "”", "★", "「", "」", ">", "=", "°", "'"]
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file:
			lst.append(line)
	return lst


def sum_vector(model, words, stop_words):
	dim = model.vector_size
	vec = np.zeros(dim)
	for w in words:
		if w not in stop_words:
			try:
				v = model[w]
				vec += np.array(v)
			except:
				pass

	return vec


def loadData(filename):
	stop_words = load_stop_words("stop_words.txt")
	model = Word2Vec.load("word2vec.model")

	trains = []
	labels = []

	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			line.rstrip("\n")
			list = line.split("\t")
			question = doc2vec.chinese_split(list[0])
			answer = doc2vec.chinese_split(list[1])
			qvec = sum_vector(model, question, stop_words)
			avec = sum_vector(model, answer, stop_words)
			vector = np.hstack((qvec, avec))
			trains.append(vector)
			labels.append(int(list[2]))

	# print(trains, labels)
	return trains, labels


def train():
	trains, labels = loadData("training.data")
	rf = GradientBoostingClassifier(n_estimators=35)
	#rf = RandomForestClassifier()
	rf.fit(trains, labels)
	joblib.dump(rf, "model.data")
	return rf

if __name__ == '__main__':
	train()


