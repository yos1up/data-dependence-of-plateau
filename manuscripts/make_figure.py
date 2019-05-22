import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams["font.size"] = 20
import numpy as np
import glob

def set_label(ax, label, loc='upper right', borderpad=-1, bbox_to_anchor=(0, 1), **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN, np.NaN, color='none',label=label)
    label_legend = ax.legend(handles=[line], loc=loc, handlelength=0, handleheight=0, handletextpad=0, borderaxespad=0, borderpad=borderpad, frameon=False, bbox_to_anchor=bbox_to_anchor, **kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()


# Figure 1: iris_with_deep_learning.ipynb 参照．

# Figure 2: microscopic_experiments から必要なデータをプロットする．
def make_fig2():
    gauss_path = '../microscopic_experiments/gauss_cpp/1000trials'
    # out{0--19}.txt
    iris_path = '../microscopic_experiments/iris_cpp/iris150_4-2-1(s-t)_squared_sigmoid_lr0.005/1000000trials'
    mnist_path = '../microscopic_experiments/mnist_cpp/mnist60000_784_2_1(s-t)_squared_sigmoid_lr0.005/1000trials'
    # out{0--29}.txt

    def load_cpp_output(filenames):
        """
        Args:
            filenames (iterable of str)
        Returns:
            [(data, target), ...]
                data (dict {str:list})
                target (dict {str:list})
        """
        ret = []
        for filename in filenames:
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
                else:  # エポック数の列に負の値が書いてあることがある．これはターゲット情報として読み込む．．
                    for i,name in enumerate(column_name):
                        target[name].append(values[i])
            ret.append((data, target))
        return ret

    gauss_data = load_cpp_output(
        [(gauss_path + '/out{}.txt').format(i) for i in range(20)]
    )
    iris_data = load_cpp_output(
        [(iris_path + '/out{}.txt').format(i) for i in range(30)]
    )
    mnist_data = load_cpp_output(
        [(mnist_path + '/out{}.txt').format(i) for i in range(30)]
    )

    plt.figure(figsize=(14, 4))
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.95, bottom=0.2)
    for i, data_list in enumerate([iris_data, mnist_data, gauss_data]):
        plt.subplot(1, 3, 1+i)
        for data, target in data_list: # 1ファイルごとのループ
            plt.plot(data['epoch'], data['loss'], color='b')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(1e-9, 1e-1)
        plt.xlim([1e2, 1e0, 1e0][i], [1e5, 1e3, 1e3][i])
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        set_label(plt.gca(), ['(a)', '(b)', '(c)'][i], fontsize=32)
    plt.savefig('../figures/stwvid_1x3.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

# Figure 3: WinTpic で昔に作成したものであるが，Keynote あたりで作り直した．




make_fig2()
