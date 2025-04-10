
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd

# 设置字体和大小

plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# 加载数据
file1 = r"E:\Data_temp\MSWEP_V28\nomasked_avg_precip_2011_2020.xlsx"
file2 = r"E:\Data_temp\MSWEP_V28\nomasked_avg_precip_1980_2010.xlsx"
file3 = r"E:\Data_temp\MSWEP_V28\masked_avg_precip_2011_2020.xlsx"
file4 = r"E:\Data_temp\MSWEP_V28\masked_avg_precip_1980_2010.xlsx"

# 使用pandas打开Excel文件
precip1 = pd.read_excel(file1, header=None).values.flatten()
precip2 = pd.read_excel(file2, header=None).values.flatten()
masked_precip1 = pd.read_excel(file3, header=None).values.flatten()
masked_precip2 = pd.read_excel(file4, header=None).values.flatten()

# 移除非有限值（例如NaN值）
precip1 = precip1[np.isfinite(precip1)]
precip2 = precip2[np.isfinite(precip2)]
masked_precip1 = masked_precip1[np.isfinite(masked_precip1)]
masked_precip2 = masked_precip2[np.isfinite(masked_precip2)]

# 计算平均值
mean1 = np.mean(precip1)
mean2 = np.mean(precip2)
mean3 = np.mean(masked_precip1)
mean4 = np.mean(masked_precip2)

# 创建图形和轴
fig, axs = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [6, 4]})

# 定义直方图的bins
bins = np.arange(0, 11.2, 0.4)

# 绘制直方图和拟合曲线
for ax, precip, mean, ylim in zip(axs.flatten(), 
                                  [precip1, precip2, masked_precip1, masked_precip2], 
                                  [mean1, mean2, mean3, mean4], 
                                  [(0, 10000), (0, 10000), (0, 500), (0, 500)]):
    # 绘制直方图，density设为False，获取计数和bin信息
    counts, bins, patches = ax.hist(precip, bins=bins, color='lightblue', edgecolor='black', alpha=0.7)
    
    # 拟合正态分布曲线
    (mu, sigma) = norm.fit(precip)
    
    # 生成更平滑的x值，确保覆盖0到11.2的范围
    x = np.linspace(0, 11.2, 100)
    
    # 计算正态分布曲线，调整y值的计算方式
    y = norm.pdf(x, mu, sigma) * len(precip) * np.diff(bins)[0]
    ax.plot(x, y, 'darkblue', linewidth=2)

    
    # 添加垂直线表示平均值
    ax.axvline(x=mean, color='gray', linestyle='dashed', linewidth=2)
    
    # 去掉上方的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置轴标签和标题
    ax.set_xlabel('Average precipitation (mm/day)')
    ax.set_ylabel('Count')
    ax.yaxis.set_label_coords(-0.12, 0.5)  # 调整y轴标题位置
    ax.set_xlim(0, 11.2)
    ax.set_ylim(ylim)
    
    # 设置刻度
    if ylim[1] == 10000:
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 2000))  # 上面两个子图刻度间隔为1500
    else:
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 100))    # 下面两个子图刻度间隔为80
    ax.tick_params(axis='both', which='both', direction='out')

# 显示图形
plt.tight_layout()
plt.show()
fig.savefig('new_MSWEP直方图.png', dpi=600, bbox_inches='tight')
fig.savefig('new_MSWEP直方图.pdf', dpi=600, bbox_inches='tight')