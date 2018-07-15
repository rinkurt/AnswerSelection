# -*- coding:utf-8 -*-

import gensim
import doc2vec
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def loadData(filename):
	qmodel = gensim.models.doc2vec.Doc2Vec.load("questions.doc2vec")
	amodel = gensim.models.doc2vec.Doc2Vec.load("answers.doc2vec")
	trains = []
	labels = []

	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			line.rstrip("\n")
			list = line.split("\t")
			vquestions = qmodel.infer_vector(doc2vec.chinese_split(list[0]))
			vanswers = amodel.infer_vector(doc2vec.chinese_split(list[1]))
			vector = np.hstack((vquestions, vanswers))
			trains.append(vector)
			labels.append(int(list[2]))

	# print(trains, labels)
	return trains, labels

def train():
	trains, labels = loadData("training.data")
	rf = LogisticRegression()
	# rf = GradientBoostingClassifier(n_estimators=20)
	# rf = RandomForestClassifier()
	rf.fit(trains, labels)
	joblib.dump(rf, "model.data")
	return rf

if __name__ == '__main__':
	train()
