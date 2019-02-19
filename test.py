# -*- coding:utf-8 -*-

# 测试

import train
import sys
from sklearn.externals import joblib


def test(filename, model):
	f = open("pred_score.txt", "w")
	X = train.loadDataWOLabel(filename)
	predict = model.predict_proba(X) * 10
	for i in predict:
		print(i[1], file=f)
	f.close()


if __name__ == '__main__':
	test(sys.argv[1], joblib.load("class.model"))