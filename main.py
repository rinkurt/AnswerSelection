# -*- coding:utf-8 -*-

import train
import test
import evaluation

if __name__ == '__main__':
	tr = train.train()
	test.test("develop.data", tr)
	evaluation.evaluate("develop.data", "score.txt", "output.txt")
