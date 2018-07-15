# -*- coding:utf-8 -*-

# 训练分类器

import numpy as np
import word_vec
from gensim.models import Word2Vec
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

'''
def load_dict(filename):
	dict = defaultdict(list)
	with open(filename, 'r', encoding='UTF-8') as wordvec:
		for line in wordvec:
			lst = line.split(" ")
			vec = [float(x) for x in lst[1:-1]]
			dict[lst[0]] = vec
	return dict
'''

# 读取停止词
def load_stop_words(filename):
	lst = []
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file:
			lst.append(line)
	return lst


# 词向量简单加和作为句子向量
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


# 文件读为句向量
def loadData(filename):
	stop_words = load_stop_words("stop_words.txt")
	model = Word2Vec.load("word_vec.model")

	trains = []
	labels = []

	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			line.rstrip("\n")
			list = line.split("\t")
			question = word_vec.cn_split(list[0])
			answer = word_vec.cn_split(list[1])
			qvec = sum_vector(model, question, stop_words)
			avec = sum_vector(model, answer, stop_words)
			# 问句答句向量并列作为特征
			vector = np.hstack((qvec, avec))
			trains.append(vector)
			labels.append(int(list[2]))

	return trains, labels


# 训练分类器
def train():
	trains, labels = loadData("training.data")
	rf = GradientBoostingClassifier(n_estimators=35)
	rf.fit(trains, labels)
	joblib.dump(rf, "class.model")
	return rf


if __name__ == '__main__':
	train()


