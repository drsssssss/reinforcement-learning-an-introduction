#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# 导入matplotlib库
import matplotlib
# 导入matplotlib.pyplot库
import matplotlib.pyplot as plt
# 导入numpy库
import numpy as np
# 导入matplotlib.table库
from matplotlib.table import Table

# 设置matplotlib使用Agg后端
matplotlib.use('Agg')

# 定义世界大小
WORLD_SIZE = 4

# 行列+-1 # left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
# 定义动作概率
ACTION_PROB = 0.25


# 定义状态是否为终止状态函数
def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)  #bool


# 定义状态转移函数
def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


# 定义绘制图像函数
def draw_image(image):
    # 创建一个子图
    fig, ax = plt.subplots()
# 关闭坐标轴
    ax.set_axis_off()
# 创建一个表格
    tb = Table(ax, bbox=[0, 0, 1, 1]) #[left, bottom, width, height] table占据整个figure

# 获取图像的行数和列数
    nrows, ncols = image.shape
# 计算每个单元格的宽度
    width, height = 1.0 / ncols, 1.0 / nrows

# 遍历图像中的每个单元格
    for (i, j), val in np.ndenumerate(image):
# 添加单元格，指定单元格的文本、位置、颜色
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

# 添加行和列的标签
        # Row and column labels...
    for i in range(len(image)):
# 添加行标签，指定标签的文本、位置、颜色
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
# 添加列标签，指定标签的文本、位置、颜色
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
# 将表格添加到子图中
    ax.add_table(tb)


# 定义计算状态值函数
def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values   #异步进行，即时更新，立刻用于下一个格子的计算
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


# 定义绘制图4.1函数
def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    _, asycn_iteration = compute_state_value(in_place=True) #异步
    values, sync_iteration = compute_state_value(in_place=False) #同步
    draw_image(np.round(values, decimals=2))
    print('In-place: {} iterations'.format(asycn_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

# 异步（async）迭代
# 定义：在异步迭代中，状态值的更新是即时发生的。即每次计算新的状态值后，这个值立即被用于后续的状态值计算。
# 代码实现：在程序中，compute_state_value函数中的in_place=True参数就是指异步迭代。在这种情况下，state_values = new_state_values使得每次更新的值立即用于后续计算。
# 同步（sync）迭代
# 定义：在同步迭代中，所有状态的更新是基于前一次完整迭代的值。这意味着在当前迭代中计算的新状态值不会立即用于该迭代的后续计算。
# 代码实现：在程序中，通过compute_state_value函数的in_place=False实现同步迭代。这里，state_values = new_state_values.copy()确保了使用的是上一次迭代的值，而不是即时更新的值。
# 区别和应用
# 效率：异步迭代可能会更快收敛，因为它使用最新的信息来更新状态值。
# 稳定性：同步迭代通常更稳定，因为它在每次迭代中使用一致的状态值集合。
# 在强化学习中，选择哪种方法取决于具体问题和计算资源。有时，异步方法因其更快的收敛速度而更受青睐，但在某些情况下，同步方法因其更高的稳定性而更合适。

    plt.savefig('../images/figure_4_1.png')
    plt.close()


# 调用函数
if __name__ == '__main__':
    figure_4_1()