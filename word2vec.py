# -*- coding:utf-8 -*-

from gensim.models import Word2Vec
from collections import Counter
import jieba

def chinese_split(s):
	return [x for x in jieba.cut(s)]

def read_file(filename):
	sentences = []
	question = ""
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			list = line.split("\t")
			if list[0] != question:
				question = list[0]
				sentences.append(chinese_split(question))
			sentences.append(chinese_split(list[1]))
	return sentences


def train(a):
	sentences = read_file("training.data")
	model = Word2Vec(size=a, window=3, min_count=2)
	model.build_vocab(sentences)
	model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
	model.save("word2vec.model")

if __name__ == '__main__':
	train(45)
