# -*- coding:utf-8 -*-

import gensim
import doc2vec
import trainbyword
from sklearn.externals import joblib


def test(filename, model):
	f = open("score.txt", "w")
	# rf = joblib.load("rf.m")
	X, y = trainbyword.loadData(filename)
	predict = model.predict_proba(X)
	for i in predict:
		print(i[1], file=f)

	f.close()


if __name__ == '__main__':
	test("develop.data", joblib.load("model.data"))