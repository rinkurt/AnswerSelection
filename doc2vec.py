# -*- coding:utf-8 -*-

# 使用 gensim 的 doc2vec 将语料训练为句向量

import gensim
import jieba

from gensim.models.doc2vec import TaggedDocument


def chinese_split(s):
	return [x for x in jieba.cut(s)]


def read_corpus(filename):
	question = ""
	count = 0
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			line.rstrip("\n")
			list = line.split("\t")
			if list[0] != question:
				question = list[0]
				count += 1
				yield TaggedDocument(words=chinese_split(question), tags=['SENT_%d' % count])
			count += 1
			yield TaggedDocument(words=chinese_split(list[1]), tags=['SENT_%d' % count])


def train():
	train_corpus = list(read_corpus("training.data"))
	model = gensim.models.doc2vec.Doc2Vec(vector_size=20, epochs=10)
	model.build_vocab(train_corpus)

	model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

	model.save("trained.doc2vec")


if __name__ == "__main__":
	train()
