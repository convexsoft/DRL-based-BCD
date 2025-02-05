import numpy as np
import pandas as pd
import csv
import random


def brutal_search(G,w,p_bar,sigma):
    diff = 0.01
    max_obj = 10e-8
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    for p_0 in np.arange(0, p_bar[0][0], diff):
        for p_1 in np.arange(0, p_bar[0][0]-p_0, diff):
            for p_2 in np.arange(0, p_bar[0][0]-p_0-p_1, diff):
                p = np.array([[p_0],[p_1],[p_2]])

                sinr = (1 / (np.dot(F, p) + v)) * p # 2*1,m=1
                f_func = np.log(1 + sinr)
                obj = w.T.dot(f_func)[0][0]
                # print("obj:", obj)

                if obj>=max_obj:
                    max_obj = obj
                    # print("max_obj:", max_obj)
                    # print("pinter:", p)
                    p_star = p
    print("p_star:", p_star)
    sinr_star = (1 / (np.dot(F, p_star) + v)) * p_star
    # gamma_star = np.log(sinr_star)
    return sinr_star

