# -*- coding:utf-8 -*-

import random


def xiacai(filename):
	f = open("score.txt", "w")
	count = -1
	for count, line in enumerate(open(filename, 'rU', encoding="UTF-8")):
		pass
	count += 1
	for i in range(count):
		print(random.random(), file=f)

if __name__ == '__main__':
	xiacai("develop.data")
