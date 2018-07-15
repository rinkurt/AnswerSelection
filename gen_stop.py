# -*- coding:utf-8 -*-

# 生成停止词

from collections import Counter
import word_vec


# 选取出现频率高的词作为停止词
def stop_words(data):
	sentences = word_vec.read_file(data)
	f = open("stop_words.txt", "w", encoding="UTF-8")
	c = Counter()
	for s in sentences:
		c.update(s)
	for x in c.most_common(150):
		print(x[0], file=f)

if __name__ == '__main__':
	stop_words("training.data")

