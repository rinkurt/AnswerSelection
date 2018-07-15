# -*- coding:utf-8 -*-

# 使用 gensim 的 doc2vec 将语料训练为句向量

import gensim
import jieba

from gensim.models.doc2vec import TaggedDocument


def chinese_split(s):
	return [x for x in jieba.cut(s)]


def read_question(filename):
	question = ""
	count = 0
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			list = line.split("\t")
			if list[0] != question:
				question = list[0]
				count += 1
				yield TaggedDocument(words=chinese_split(question), tags=['Q_%d' % count])


def read_answer(filename):
	count = 0
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			list = line.split("\t")
			count += 1
			yield TaggedDocument(words=chinese_split(list[1]), tags=['A_%d' % count])


def train():
	questions = list(read_question("training.data"))
	answers = list(read_answer("training.data"))
	qmodel = gensim.models.doc2vec.Doc2Vec(vector_size=40, epochs=10, window=3, min_count=2)
	amodel = gensim.models.doc2vec.Doc2Vec(vector_size=40, epochs=10, window=3, min_count=2)
	qmodel.build_vocab(questions)
	amodel.build_vocab(answers)
	qmodel.train(questions, total_examples=qmodel.corpus_count, epochs=qmodel.epochs)
	amodel.train(answers, total_examples=amodel.corpus_count, epochs=amodel.epochs)
	qmodel.save("questions.doc2vec")
	amodel.save("answers.doc2vec")


if __name__ == "__main__":
	train()
