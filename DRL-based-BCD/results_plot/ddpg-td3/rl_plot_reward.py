import pandas as pd
import seaborn as sns
import seaborn.timeseries
import numpy as np
import matplotlib.pyplot as plt
import csv

start = 10
end = 40
my_list = list()
df=pd.read_csv('res1.csv', header=None, sep=',')
for i in range(start,len(df)-end):
    my_list.append(list(df.iloc[i]))

my_arr = np.array(my_list)


ddpg_list = list()
df=pd.read_csv('reward_ddpg.csv', header=None, sep=',')
for i in range(start,len(df)-end):
    ddpg_list.append(list(df.iloc[i]))

ddpg_arr = np.array(ddpg_list)


ppo_list = list()
df=pd.read_csv('reward_30.csv', header=None, sep=',')
for i in range(start,len(df)-end-30):
    ppo_list.append(list(df.iloc[i]))

ppo_arr = np.array(ppo_list)


from matplotlib import pyplot as plt
plt.figure(figsize=(11, 6.5))
ci = 60
fontsize = 23
axis_size = 15
ax2 = sns.tsplot(data = ddpg_arr.T,
           err_style='ci_band', #误差数据风格，可选：ci_band, ci_bars, boot_traces,
           #boot_kde, unit_traces, unit_points
           interpolate = True,  #设置连线
           # ci = [40, 70, 90],   #设置误差区间
            ci = ci,   #设置误差区间
            color = 'tomato' ,         #设置颜色
           )


# ax3 = sns.tsplot(data = ppo_arr.T,
#            err_style='ci_band', #误差数据风格，可选：ci_band, ci_bars, boot_traces,
#            #boot_kde, unit_traces, unit_points
#            interpolate = True,  #设置连线
#            # ci = [40, 70, 90],   #设置误差区间
#             ci = ci,   #设置误差区间
#             color = 'green' ,         #设置颜色
#            )

ax1 = sns.tsplot(data = my_arr.T,
           err_style='ci_band', #误差数据风格，可选：ci_band, ci_bars, boot_traces,
           #boot_kde, unit_traces, unit_points
           interpolate = True,  #设置连线
           # ci = [40, 70, 90],   #设置误差区间
            ci = ci,   #设置误差区间
            color = 'b' ,         #设置颜色
           )

# plt.legend([ax1, ax2,ax3], labels=["DDPG-BCD", "A3C-BCD","TD3-BCD"],loc='down right',bbox_to_anchor=(0.95, .25),fontsize=axis_size)
plt.legend([ax1, ax2], labels=["TD3-BCD","DDPG-BCD"],loc='down right',bbox_to_anchor=(0.95, .25),fontsize=axis_size)
plt.xlabel("Episode", fontsize=fontsize)
plt.ylabel("Reward", fontsize=fontsize)
plt.xticks(fontsize=axis_size)
plt.yticks(fontsize=axis_size)
plt.savefig("reward.pdf")

plt.show()



