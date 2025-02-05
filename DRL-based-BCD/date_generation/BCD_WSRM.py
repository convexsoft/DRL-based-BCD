import numpy as np
import csv
import random
import math


np.random.seed(42)
def step(o, a, label):
    L = len(label)
    G = o[0:L ** 2].reshape(L, L)
    w = o[L ** 2:L ** 2 + L].reshape(L, 1)
    sigma = o[L ** 2 + L:L ** 2 + 2 * L].reshape(L, 1)
    p_bar = o[L ** 2 + 2 * L]
    gamma_star = label
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.dot(v, np.ones((1, L)))
    # b = w[:,0]*y
    b = w[:, 0] * a
    til_gamma = iteration_for_subproblem(B, b)
    gamma = np.exp(til_gamma)

    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    reward = - (np.abs(obj_updated-obj_star))**2
    # reward = - np.linalg.norm(gamma - gamma_star, 1)
    # print("a:", a)
    # print("gamma:", gamma)
    # print("gamma_star:", gamma_star)
    # print("reward:", reward)
    o2 = np.append(o[:-L], gamma)
    d = False
    if reward >= -1e-3:
        d = True
    return o2, reward, d


def iteration_for_subproblem(B, b):
    z = np.random.rand(len(b))
    tol = 10e-9
    err = 1
    while err>tol:
        z_temp = z
        z = b/(B.T.dot(b/(B.dot(z))))
        err = np.linalg.norm(z_temp-z,1)

    res = B.dot(z)
    til_gamma = np.log(z/res)
    return til_gamma

def bcd_for_wsrm(G,w, sigma,p_bar,  y_init):
    L =len(y_init)
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.dot(v, np.ones((1, L)))
    y = y_init
    err = 1
    tol = 1e-3
    obj_before = 0
    step = 0

    # for i in range(13):
    while err > tol:
        # print("step:", step)
        b = w[:, 0] * y
        til_gamma = iteration_for_subproblem(B, b)
        gamma = np.exp(til_gamma)
        # gamma1.append(gamma[0])
        # gamma2.append(gamma[1])
        # gamma3.append(gamma[2])
        y = 1 / (1 /gamma + 1)

        obj_updated = w.T.dot(np.log(1 + gamma))[0]
        err = obj_updated - obj_before
        obj_before = obj_updated
        step += 1
    # print("gamma1:", gamma1)
    # print("gamma2:", gamma2)
    # print("gamma3:", gamma3)
    return step, gamma


def one_data_check():
    o = np.array([1.18, 0.05, 0.59, 0.61, 9.67, 0.56, 0.89, 0.99, 3.29, 0.66, 0.45, 0.46, 0.05, 0.05, 0.05, 2.79])
    label = np.array([27.804597701149422, 5.502345251826221, 0.0])

    G = o[0:9].reshape(3, 3)
    w = o[9:12].reshape(3, 1)
    sigma = o[12:15].reshape(3, 1)
    p_bar = o[15]
    gamma_star = label
    y_init = np.random.rand(3) * 10
    step, gamma = bcd_for_wsrm(G, w, sigma, p_bar, y_init)
    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    sumrate_acc = obj_updated/obj_star
    print("y_init:", y_init)
    print("step:", step)
    print("label:", label)
    print("gamma:", gamma)
    print("sumrate_acc:", sumrate_acc)



if __name__ == '__main__':
    one_data_check()

