import matplotlib.pyplot as plt
import numpy as np
import glob
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams["font.size"] = 30

data_list, target_list = [], []
for filename in glob.glob("1000trials/out*.txt"):
    with open(filename,"r") as f:
        lines = f.readlines()
    data, target = {}, {}
    column_name = lines[0].split()
    for name in column_name:
        data[name] = []
        target[name] = []
    for line in lines[1:]:
        values = list(map(float, line.split()))
        if len(values) != len(column_name): break
        if values[0] >= 0:
            for i,name in enumerate(column_name):
                data[name].append(values[i])
        else:
            for i,name in enumerate(column_name):
                target[name].append(values[i])
    data_list.append(data)
    target_list.append(target)


fig, ax = plt.subplots(1,1,figsize=(13,9))
# fig, axes = plt.subplots(2,1,figsize=(13,9))
# ax = axes[0]
# ax2 = ax.twinx()

for data in data_list:
    ax.plot(data['epoch'], data['loss'], color='k')
    # ax.plot(data['epoch'], data['W1MaxOvl'], color='b')
    # ax.plot(data['epoch'], data['ZMaxOvl'], color='g')
    # ax.plot(data['epoch'], data['HMaxOvl'], color='r')
    # try:
    #     ax2.plot(data['epoch'], data['HOSV0'], color='gray')
    #     ax2.plot(data['epoch'], data['HOSV1'], color='gray')
    #     ax2.plot(data['epoch'], data['HOSV2'], color='gray')
    # except:
    #     pass
border = 0.4
# ax.plot(data['epoch'], [border]*len(data['epoch']), linestyle='--', color='b')

'''
max_epoch_idx = 1000 // 1

trapped_ratio = np.array([sum([data['loss'][i] > border for data in data_list]) for i in range(max_epoch_idx)]) / len(data_list)
axes[1].plot(data['epoch'][:max_epoch_idx], trapped_ratio, color='b', lw=2)
axes[1].set_ylim(0, 1)
axes[1].set_yticks(np.linspace(0, 1, 3))
axes[1].set_ylabel('trapped proportion')
'''
outfilename = 'summary_simple'

ax.set_yscale('log')
# ax.set_ylim(-0.01, 1.01)
ax.set_ylim(1e-10, 1.01)
ax.set_xlim(1, 1000);
# axes[1].set_xlim(1, 1000)
ax.set_ylabel('training loss')

# axes[1].set_xlabel('epoch')
ax.set_xlabel('epoch')

ax.set_xscale('log');
# axes[1].set_xscale('log')
plt.savefig(outfilename + '.pdf')
ax.set_xscale('linear');
# axes[1].set_xscale('linear')
plt.savefig(outfilename + '_lin.pdf')
plt.show()


