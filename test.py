# -*- coding:utf-8 -*-

import gensim
import doc2vec
import train
from sklearn.externals import joblib

def sim():
	x_train = list(doc2vec.read_corpus("training.data"))

	model = gensim.models.doc2vec.Doc2Vec.load("trained.doc2vec")
	text = ['小行星', '123', '是谁', '发现', '的', '？']
	inferred = model.infer_vector(text)
	print(inferred)
	sims = model.docvecs.most_similar([inferred], topn=10)

	for count, sim in sims:
		sentence = x_train[count]
		words = ''
		for word in sentence[0]:
			words = words + word + ' '
		print(words, sim, len(sentence[0]))


def test(filename, model):
	f = open("score.txt", "w")
	# rf = joblib.load("rf.m")
	X, y = train.loadData(filename)
	predict = model.predict_proba(X)
	for i in predict:
		print(i[1], file=f)

	f.close()


if __name__ == '__main__':
	test("develop.data", joblib.load("model.data"))