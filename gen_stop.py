# -*- coding:utf-8 -*-

from collections import Counter
import word2vec

def stop_words(sentences):
	f = open("stop_words.txt", "w", encoding="UTF-8")
	c = Counter()
	for s in sentences:
		c.update(s)
	for x in c.most_common(200):
		print(x[0], file=f)

if __name__ == '__main__':
	stop_words(word2vec.read_file("training.data"))

