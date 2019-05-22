import matplotlib.pyplot as plt
import numpy as np
import glob
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams["font.size"] = 40

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


fig, ax = plt.subplots(1,1,figsize=(12,8))
ax2 = ax.twinx()

xl = None
for data, target in zip(data_list, target_list):
    ax2.plot(data['epoch'], data['loss'], color='k')
    ax.plot(data['epoch'], data['W1MaxOvl'], color='b')
    ax.plot(data['epoch'], data['ZMaxOvl'], color='g')
    ax.plot(data['epoch'], data['HMaxOvl'], color='r')
    if xl is None: xl = plt.xlim()
    try:
        ax2.plot(data['epoch'], data['HOSV0'], color='gray')
        ax2.plot(data['epoch'], data['HOSV1'], color='gray')
        ax2.plot(data['epoch'], data['HOSV2'], color='gray')
        if len(target['epoch']) > 0:
            ax2.scatter(data['epoch'][-1], target['HOSV0'][0], s=40, c='gray')
            ax2.scatter(data['epoch'][-1], target['HOSV1'][0], s=40, c='gray')
            ax2.scatter(data['epoch'][-1], target['HOSV2'][0], s=40, c='gray')
    except:
        pass


outfilename = 'summary'

ax2.set_yscale('log')
ax.set_ylim(-0.01, 1.01)
ax.set_xscale('log')
ax.set_xlim(None, xl[1])
plt.savefig(outfilename + '.pdf')
ax.set_xscale('linear')
ax.set_xlim(xl)
plt.savefig(outfilename + '_lin.pdf')
plt.show()



