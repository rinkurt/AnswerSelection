# -*- coding:utf-8 -*-

import gen_stop
import word_vec
import train
import test
import evaluation

if __name__ == '__main__':
	gen_stop.stop_words()
	word_vec.train()
	tr = train.train()
	test.test("develop.data", tr)
	evaluation.evaluate("develop.data", "score.txt", "output.txt")
