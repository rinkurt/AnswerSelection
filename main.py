# -*- coding:utf-8 -*-

import word2vec
import trainbyword
import testword
import evaluation

if __name__ == '__main__':

	word2vec.train(45)
	tr = trainbyword.train()
	testword.test("develop.data", tr)
	evaluation.evaluate("develop.data", "score.txt", "output.txt")
	print("finished")

