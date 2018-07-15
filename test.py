# -*- coding:utf-8 -*-

# 测试

import train
from sklearn.externals import joblib


def test(filename, model):
	f = open("score.txt", "w")
	X, y = train.loadData(filename)
	predict = model.predict_proba(X)
	for i in predict:
		print(i[1], file=f)
	f.close()


if __name__ == '__main__':
	test("develop.data", joblib.load("class.model"))