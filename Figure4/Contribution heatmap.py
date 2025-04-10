import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import xarray as xr
import cv2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 加载模型
model_path = r"E:\Data_temp\CESM2_LENS\1850-2100\my_model.h5"
model = load_model(model_path)


def compute_grad_cam(model, input_data, layer_name='conv2d_4'):
    # 获取模型的指定卷积层的输出
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_output = last_conv_layer.output

    # 创建一个模型，映射模型输入到最后一个卷积层的输出和模型输出
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer_output, model.output])

    # 前向传播
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        loss = predictions[:, tf.argmax(predictions[0])]  # 假设分类索引为模型预测的最大值索引

    # 梯度
    grads = tape.gradient(loss, conv_outputs)

    # 梯度权重
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 应用权重到卷积层输出上
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap[0]  # 直接返回处理后的数组

# 定义时间范围
start_year = 2011
end_year = 2020

# 初始化数组以存储所有天的热图
heatmap_accumulator = None
day_count = 0

import calendar

# 遍历每一年的每一天
for year in range(start_year, end_year + 1):
    # 确定这一年的天数（考虑闰年）
    days_in_year = 366 if calendar.isleap(year) else 365
    for day in range(1, days_in_year + 1):
        filename = f"{year}{day:03d}.nc"
        file_path = f"E:\\Data_temp\\CESM2_LENS\\Anomaly_prect\\{filename}"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        ds = xr.open_dataset(file_path)
        prect_data = ds['prect'].values
        prect_data = np.expand_dims(np.expand_dims(prect_data, axis=-1), axis=0)  # 添加批次和通道维度

        # 计算单日的Grad-CAM
        heatmap = compute_grad_cam(model, prect_data)

        # 累加热图并计数
        if heatmap_accumulator is None:
            heatmap_accumulator = np.zeros_like(heatmap)
        heatmap_accumulator += heatmap
        day_count += 1

# 计算平均热图
average_heatmap = heatmap_accumulator / day_count

# 获取经纬度数据
ds = xr.open_dataset(file_path)  # 重新加载最后一个文件以获取维度信息
lon = ds['lon'].values
lat = ds['lat'].values
lon2d, lat2d = np.meshgrid(lon, lat)  # 创建二维网格

# 上采样 average_heatmap 以匹配 lon2d 和 lat2d 的尺寸
heatmap_resized = cv2.resize(average_heatmap, (lon2d.shape[1], lon2d.shape[0]), interpolation=cv2.INTER_LINEAR)

# 加载掩膜文件
mask_file_path = r"E:\Data_temp\CESM2_LENS\city_mask_matrix.npy"
city_mask = np.load(mask_file_path)

# 调整掩膜插值为二阶的双线性插值，进一步平滑掩膜边界
if city_mask.shape != heatmap_resized.shape:
    city_mask = cv2.resize(city_mask, (heatmap_resized.shape[1], heatmap_resized.shape[0]), interpolation=cv2.INTER_LINEAR)
# 确保掩膜不进行平滑处理，直接二值化（0和1）
city_mask_smoothed = np.where(city_mask > 0.5, 1, 0)  # 使用0.5阈值，确保只有严格的0和1

# 创建一个新的数组，保存掩膜应用后的热图
masked_heatmap = np.full_like(heatmap_resized, np.nan)
masked_heatmap[city_mask_smoothed == 1] = heatmap_resized[city_mask_smoothed == 1]

# 设置地图和投影
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
ax.set_global()
ax.coastlines()

from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 自定义颜色映射
colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']  
cmap = LinearSegmentedColormap.from_list(name='custom', colors=colors, N=100)
cmap_bounds = np.linspace(0.10, 0.26, 100)
norm = plt.Normalize(vmin=0.10, vmax=0.26)

# 确定等高线级别
levels = np.arange(0.10, 0.26, 0.003)  # 或者您可以根据您的数据选择不同的间隔或范围

# 2. 处理掩膜区域，确保掩膜外部区域没有空值
masked_heatmap = np.copy(heatmap_resized)
# masked_heatmap[city_mask_smoothed == 0] = np.nan  # 将非掩膜区域的值设为最小值，避免空白
# 绘制掩膜区域的热图
con = ax.contourf(lon2d, lat2d, masked_heatmap, levels=levels, cmap=cmap, norm=norm, extend='both')

# 1. 绘制非掩膜区域的透明度层
non_masked_heatmap = np.copy(heatmap_resized)
non_masked_heatmap[city_mask_smoothed == 1] = np.nan

# 使用灰色和透明度 0.3 绘制非掩膜区域
ax.contourf(lon2d, lat2d, non_masked_heatmap, levels=np.linspace(0.10, 0.26, 30), colors='#bababa', alpha=1, extend='both')


# 3. 绘制掩膜之外的区域（添加斜线填充）
ax.contourf(lon2d, lat2d, np.where(city_mask_smoothed == 0, 1, np.nan), levels=[0, 1], colors='none', alpha=0, hatches=['//'], extend='both')

ax.coastlines(linewidth=1.5) # 添加海岸线
ax.set_xticks(np.arange(-180, 181, 60), crs=projection)
ax.set_yticks(np.arange(-60, 91, 30), crs=projection)

# 设置 ticklabels 格式
lon_formatter = LongitudeFormatter(degree_symbol='°',
                                  dateline_direction_label=True)
lat_formatter = LatitudeFormatter(degree_symbol='°')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

# 设置坐标轴标签字体大小（可根据需要调整大小）
ax.tick_params(axis='both', which='major', labelsize=16, length=8)  # 例如，这里将字体大小设置为16

# 色标放置在图形底部
cb = fig.colorbar(con, orientation='horizontal', pad=0.1, aspect=60)

# 设置 colorbar 的刻度
cb_tick_values = np.arange(0.10, 0.26, 0.03)  # 从 -1.5 到 1.5 的刻度值
cb.set_ticks(cb_tick_values)

# 设置 colorbar 的刻度标签并格式化为一位小数
cb_tick_labels = [f"{tick*1:.2f}" for tick in cb_tick_values]
cb.set_ticklabels(cb_tick_labels)

# 设置 colorbar 标题
cb.set_label(r'Precipitation anomalies as a proportion of temperature projections (RIHA highlighted)', fontsize=16)

# 设置 colorbar 的刻度标签垂直居中对齐
for label in cb.ax.get_yticklabels():
    label.set_verticalalignment('center')

# 其余的 colorbar 设置保持不变
cb.ax.tick_params(axis='y', which='both', labelsize=16, length=5, direction='out')

for label in cb.ax.get_xticklabels():
    label.set_fontsize(16)


# 调整色标的位置和长度，使其与图形的宽度匹配
position = ax.get_position()
cb.ax.set_position([position.x0, position.y0 - 0.07, position.width, 0.02])

# 设置整体图形的标题栏
fig.suptitle('Contribution of precipitation anomalies to temperature projections in RIHA (2011-2020)', fontsize=16,
             y=0.92, verticalalignment='top')

# 保存文件, dpi用于设置图形分辨率, bbox_inches 尽量减小图形的白色区域
fig.savefig('Contribution of precipitation anomalies to temperature projections in RIHA (2011-2020).png', dpi=600, bbox_inches='tight')
fig.savefig('Contribution of precipitation anomalies to temperature projections in RIHA (2011-2020).pdf', dpi=600, bbox_inches='tight')

