import random
import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
import argparse
import os
import sys
import math

def main(args):
	dim=args.d
	top_k=args.r
	num_unlabelled_data=args.num_unlabelled_data
	num_train_data=args.num_train_data
	num_test_data=args.num_test_data

	principal_components = ortho_group.rvs(dim=dim-1)
	sigmas=np.zeros(dim-1)
	means=np.zeros(dim-1)
	for i in range(dim-1):
		if i < top_k:
			while sigmas[i] < 1:
				sigmas[i]=10*random.random()
			means[i]=random.randint(0,20)
		else:
			sigmas[i]=0.1*random.random()
			means[i]=random.randint(0,20)

	sigmas=np.sort(sigmas)[::-1]
	print(means)
	print(sigmas)
	alpha = np.random.uniform(-1, 1, size=(top_k))
	alpha = alpha/np.linalg.norm(alpha)
		
	f=open(args.directory+"/unlabelled.tsv","w+")
	for i in range(num_unlabelled_data):
		sum_x1=0
		data=np.zeros(dim-1)
		for j in range(dim-1):
			coeff= np.random.normal(means[j],sigmas[j])
			if j < top_k:
				sum_x1+=coeff*coeff
			data+=coeff*principal_components[j]
		f.write(str(0)+ " " +" ".join(map(str, data)) + "\t" + str(sum_x1) + "\t" + str(0) + "\n")

	f=open(args.directory+"/train.tsv","w+")
	for i in range(num_train_data):
		sum_x1=0
		sum_y=0
		data=np.zeros(dim-1)
		for j in range(dim-1):
			coeff= np.random.normal(means[j],sigmas[j])
			if j < top_k:
				sum_x1+=coeff*coeff
				sum_y+=coeff*coeff*alpha[j]
			data+=coeff*principal_components[j]
		sum_y+=np.random.normal(0,0.01)
		f.write(str(sum_x1)+ " " +" ".join(map(str, data)) + "\t" + str(sum_y) + "\n")

	f=open(args.directory+"/test.tsv","w+")
	for i in range(num_test_data):
		sum_x1=0
		sum_y=0
		data=np.zeros(dim-1)
		for j in range(dim-1):
			coeff= np.random.normal(means[j],sigmas[j])
			if j < top_k:
				sum_x1+=coeff*coeff
				sum_y+=coeff*coeff*alpha[j]
			data+=coeff*principal_components[j]
		sum_y+=np.random.normal(0,0.01)
		f.write(str(sum_x1)+ " " +" ".join(map(str, data)) + "\t" + str(sum_y) + "\n")
    
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
	argparser.add_argument("--num_unlabelled_data", type=int, default=10000)
	argparser.add_argument("--num_train_data", type=int, default=10000)
	argparser.add_argument("--num_test_data", type=int, default=1000)
	argparser.add_argument("--directory", type=str, default="d_100_r_30")
	argparser.add_argument("--r", type=int, default=30)
	argparser.add_argument("--d", type=int, default=100)
	args = argparser.parse_args()
	main(args)