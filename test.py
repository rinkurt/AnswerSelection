# -*- coding:utf-8 -*-

# 测试

import train
import sys
from sklearn.externals import joblib


def test(filename, model):
	f = open("score.txt", "w")
	X, y = train.loadData(filename)
	predict = model.predict_proba(X)
	for i in predict:
		print(i[1], file=f)
	f.close()


if __name__ == '__main__':
	data = "develop.data"
	model_file = "class.model"
	if (len(sys.argv) >= 2):
		data = sys.argv[1]
	if (len(sys.argv) == 3):
		model_file = sys.argv[2]
	test(data, joblib.load(model_file))