import matplotlib.pyplot as plt
import numpy as np
import glob
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams["font.size"] = 30


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

    flg, ax = plt.subplots(1,1,figsize=(12,8))
    ax2 = ax.twinx()

    ax2.plot(data['epoch'], data['loss'], color='k')
    ax2.set_yscale('log')

    ax.plot(data['epoch'], data['W1MaxOvl'])
    ax.plot(data['epoch'], data['ZMaxOvl'])
    ax.plot(data['epoch'], data['HMaxOvl'])
    xl = plt.xlim() 
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
    ax.set_ylim(-0.01, 1.01)
    ax.set_xscale('log')
    ax.set_xlim(None, xl[1])
    plt.savefig(filename + '.pdf')
    ax.set_xscale('linear')
    ax.set_xlim(xl)
    plt.savefig(filename + '_lin.pdf')
    # plt.show()


