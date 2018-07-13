# -*- coding:utf-8 -*-

import gensim
import doc2vec
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def loadData(filename):
	model = gensim.models.doc2vec.Doc2Vec.load("trained.doc2vec")
	trains = []
	labels = []

	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			line.rstrip("\n")
			list = line.split("\t")
			question = model.infer_vector(doc2vec.chinese_split(list[0]))
			answer = model.infer_vector(doc2vec.chinese_split(list[1]))
			vector = np.array(question) - np.array(answer)
			trains.append(vector)
			labels.append(int(list[2]))

	# print(trains, labels)
	return trains, labels

def train():
	trains, labels = loadData("training.data")
	rf = GradientBoostingClassifier(n_estimators=40)
	# rf = RandomForestClassifier()
	rf.fit(trains, labels)
	joblib.dump(rf, "model.data")
	return rf

if __name__ == '__main__':
	train()
