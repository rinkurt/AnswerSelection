# -*- coding:utf-8 -*-

# 使用 gensim 中的 Word2vec 生成词向量

from gensim.models import Word2Vec
import jieba

# 使用 jieba 分词
def cn_split(s):
	return [x for x in jieba.cut(s)]


# 将语料按句读为词表
def read_file(filename):
	sentences = []
	question = ""
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file.readlines():
			list = line.split("\t")
			if list[0] != question:
				question = list[0]
				sentences.append(cn_split(question))
			sentences.append(cn_split(list[1]))
	return sentences


# 训练词向量
def train():
	sentences = read_file("training.data")
	model = Word2Vec(size=45, window=3, min_count=5)
	model.build_vocab(sentences)
	model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
	model.save("word_vec.model")

if __name__ == '__main__':
	train()
