from td3_sr import TD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import pandas as pd

def env_o_init(sr_data_single,L):
    G = sr_data_single[0:L ** 2].reshape(L, L)
    w = sr_data_single[L ** 2:L ** 2 + L].reshape(L, 1)
    sigma = sr_data_single[L ** 2 + L:L ** 2 + 2 * L].reshape(L, 1)
    p_bar = sr_data_single[L ** 2 + 2 * L]
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    notil_gamma = np.random.rand(L) * 1

    p = np.linalg.inv(np.eye(len(notil_gamma)) - np.diag(notil_gamma) @ F) @ np.diag(notil_gamma) @ v
    Fpv = F.dot(p) + v
    res = np.log(1 + notil_gamma)
    rate = np.diag(w[:, 0]) @ np.log(1 + notil_gamma)

    o_init = np.hstack((sr_data_single, p.reshape(L), Fpv.reshape(L), rate.reshape(L), notil_gamma))
    return o_init

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
    b = w[:, 0] * a
    til_gamma = iteration_for_subproblem(B, b)
    notil_gamma = np.exp(til_gamma)

    p = np.linalg.inv(np.eye(len(notil_gamma)) - np.diag(notil_gamma) @ F) @ np.diag(notil_gamma) @ v
    Fpv = F.dot(p) + v
    res = np.log(1 + notil_gamma)
    rate = np.diag(w[:, 0])@np.log(1 + notil_gamma)


    obj_updated = w.T.dot(np.log(1 + notil_gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    reward = - (np.abs(obj_updated-obj_star))**2
    obj_err = abs(np.abs(obj_updated - obj_star)) / obj_star
    o2 = np.hstack((o[:L ** 2 + 2 * L+1],p.reshape(L), Fpv.reshape(L), rate.reshape(L), notil_gamma))

    d = False
    if reward >= -1e-3:
        d = True
    return o2, reward, d, obj_err

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


def main(obs_dim,nn_dim,act_dim,epoch_num,MAX_EPISODE=100,MAX_STEP=5000,update_every=50,batch_size=10,start_update=40,
         replay_size=int(1e5), gamma=0.9, pi_lr=1e-6,q_lr=1e-6, policy_delay=5):
    sr_data = list()
    sr_target = list()
    with open("res/data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sr_data.append(list(map(float, line[:-act_dim])))
            sr_target.append(list(map(float, line[-act_dim:])))

    td3 = TD3(obs_dim, act_dim,nn_dim,replay_size=replay_size, gamma=gamma, pi_lr=pi_lr, q_lr=q_lr,
                  act_noise=0.05, target_noise=0.05, noise_clip=0.5, policy_delay=policy_delay)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        filename = f"res/rewardv3_v2_ME{MAX_EPISODE}_SU{start_update}_Ga{gamma}_UE{update_every}_BS{batch_size}_lr{pi_lr}_MS{MAX_STEP}_Epoch{epoch}" + '.pdf'

        rewardList = []
        err_list = []
        stop_step_list = []
        outlier_list = []
        obj_err_list = []

        for episode in range(MAX_EPISODE):
            j_converge_list = [MAX_STEP]
            # o = env.reset()
            o_init = env_o_init(np.array(sr_data[episode]),act_dim)
            # sr_data_a = sr_data[epoch]
            # print("o_init:",o_init)
            label = np.array(sr_target[episode])
            o = o_init
            ep_reward = 0
            # stop_step = 0

            for j in range(MAX_STEP):
                # if episode > 20:
                #     a = td3.get_action(o, td3.act_noise) * 2
                # else:
                #     a = env.action_space.sample()
                # a = td3.get_action(o, td3.act_noise) * 2

                a = td3.get_action(o[16:], 0.001)
                # ======================
                # next state
                # o2, r, d, obj_err = step(o, a, label)
                # o2, reward, d, obj_err, reward_acc= step(o, a, label)
                o2, r, d, obj_err, reward_acc = step(o, a, label)
                # ======================
                td3.replay_buffer.store(o, a, r, o2, d)
                if episode >= start_update and j % update_every == 0:
                    td3.update(batch_size, update_every)

                o = o2
                ep_reward += reward_acc
                # stop_step = j
                if d:
                    j_converge_list.append(j)
            notil_gamma = o[-act_dim:]
            err = np.linalg.norm(notil_gamma - label)
            converge_step = min(j_converge_list)

            print('Episode:', episode, 'notil_gamma:', notil_gamma, 'label:', label, '==========', 'Reward:', ep_reward, 'err:',
                  err, '---', 'obj_err:', obj_err, 'j:', converge_step)
            if math.isnan(ep_reward):
                print("a:", a)
            rewardList.append(ep_reward)
            err_list.append(err)
            obj_err_list.append(obj_err)
            stop_step_list.append(converge_step)
            if episode > 50 and converge_step >= 1000:
                outlier_list.append(episode)

        print("rewardList:", rewardList)
        print("err_list:", err_list)
        print("stop_step_list:", stop_step_list)
        print("outlier_list:", outlier_list)

        plt.figure(figsize=(18, 4))

        plt.subplot(1, 4, 1)
        plt.plot(np.arange(len(rewardList)), rewardList)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Reward", fontsize=10)

        plt.subplot(1, 4, 2)
        plt.plot(np.arange(len(err_list)), err_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("SINR error", fontsize=10)

        plt.subplot(1, 4, 3)
        plt.plot(np.arange(len(obj_err_list)), obj_err_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Sum rate error", fontsize=10)

        plt.subplot(1, 4, 4)
        plt.plot(np.arange(len(stop_step_list)), stop_step_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Iteration steps", fontsize=10)

        title_name = f"main_sr_rewardv3_v2_epoch_{epoch}"
        plt.title(title_name)
        plt.savefig(filename)

        with open('res/reward_rewardv3_v2.csv', 'a', newline='') as file:
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerow(filename)
            mywriter.writerow(rewardList)

        with open('res/err_rewardv3_v2.csv', 'a', newline='') as file:
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerow(filename)
            mywriter.writerow(err_list)

        with open('res/obj_err_rewardv3_v2.csv', 'a', newline='') as file:
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerow(filename)
            mywriter.writerow(obj_err_list)

        with open('res/step_rewardv3_v2.csv', 'a', newline='') as file:
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerow(filename)
            mywriter.writerow(stop_step_list)

