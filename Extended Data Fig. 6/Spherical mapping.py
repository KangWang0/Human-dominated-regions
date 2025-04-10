import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import netCDF4 as nc


# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18  # 设置全局字体大小

# 读取Excel文件中的数据
# grid_data1 = pd.read_excel(r'E:\Data_temp\MSWEP_V28\avg_precip_2011_2020.xlsx', sheet_name=0, header=None)  # 假设矩阵数据没有表头
# grid_data2 = pd.read_excel(r'E:\Data_temp\MSWEP_V28\avg_precip_1980_2010.xlsx', sheet_name=0, header=None)  # 假设矩阵数据没有表头

grid_data = pd.read_excel(r'E:\Data_temp\CESM2_LENS\avg_precip_2011_2020.xlsx', sheet_name=0, header=None)  # 假设矩阵数据没有表头
# grid_data2 = pd.read_excel(r'E:\Data_temp\CESM2_LENS\avg_precip_1980_2010.xlsx', sheet_name=0, header=None)  # 假设矩阵数据没有表头

# grid_data = (grid_data1 - grid_data2)/grid_data2

# # 将DataFrame转换为numpy数组
grid_data = grid_data.values

# 使用自定义的颜色映射
colors = [(243/255, 249/255, 254/255, 0), (166/255, 206/255, 227/255, 1), (65/255, 143/255, 198/255,0.8), (31/255, 114/255, 181/255, 0.8)]
cmap = LinearSegmentedColormap.from_list("Custom_cmap", colors)
norm = matplotlib.colors.Normalize(vmin=0, vmax=12)

# # 自定义颜色映射
# colors = ['#e66101', '#fdb863', '#f5f5f5', '#b2abd2', '#5e3c99']  
# cmap = LinearSegmentedColormap.from_list(name='custom', colors=colors, N=256)
# cmap_bounds = np.linspace(-1.0, 1.1, 256)
# norm = plt.Normalize(vmin=-1.0, vmax=1.1)

# 读取掩膜文件
city_mask = np.load(r'E:\Data_temp\CESM2_LENS\city_mask_matrix.npy')

# 将掩膜矩阵中的非零值设置为边界
city_mask_boundaries = np.where(city_mask > 0, 1, np.nan)

# 创建1行2列的图形布局，每个子图用于显示一个半球
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9), dpi=600,
                         subplot_kw={'projection': ccrs.Orthographic(central_latitude=0)})

lon = np.linspace(0, 360, 360)
lat = np.linspace(-90, 90, 180)

# 对数据进行滚动，使其与调整后的经度范围对应
grid_data = np.roll(grid_data, shift=180, axis=1)
city_mask = np.roll(city_mask, shift=180, axis=1)

# 西半球
ax_north = axes[0]
ax_north.projection = ccrs.Orthographic(central_longitude=-110, central_latitude=0) 
ax_north.set_global()
ax_north.add_feature(cfeature.LAND, facecolor='white')
ax_north.coastlines(color='#676767')
ax_north.gridlines(draw_labels=True, linewidth=1, color='gray', linestyle='--', x_inline=False, y_inline=False)
ax_north.set_title('Western Hemisphere')
mesh_north = ax_north.pcolormesh(lon, lat, grid_data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

# 在西半球的图像上添加掩膜边界
ax_north.contour(lon, lat, city_mask, levels=[0.5], colors='red', linewidths=0.8, transform=ccrs.PlateCarree(), alpha=0.7)

# 东半球
ax_south = axes[1]
ax_south.projection = ccrs.Orthographic(central_longitude=70, central_latitude=0) 
ax_south.set_global()
ax_south.add_feature(cfeature.LAND, facecolor='white')
ax_south.coastlines(color='#676767')
ax_south.gridlines(draw_labels=True, linewidth=1, color='gray', linestyle='--', x_inline=False, y_inline=False)
ax_south.set_title('Eastern Hemisphere')
mesh_south = ax_south.pcolormesh(lon, lat, grid_data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

# 在东半球的图像上添加掩膜边界
ax_south.contour(lon, lat, city_mask, levels=[0.5], colors='red', linewidths=0.8, transform=ccrs.PlateCarree(), alpha=0.7)

# 添加颜色条
cbar = fig.colorbar(mesh_north, ax=axes.ravel().tolist(), shrink=0.7, location='right', label='Average Precipitation (mm/day)',ticks=np.arange(0, 12, 2))
cbar.ax.tick_params(labelsize=18)

# # 设置 colorbar 的刻度
# cb_tick_values = np.arange(-1.0, 1.1, 0.4)  # 从 -1.0 到 1.0 的刻度值
# cbar.set_ticks(cb_tick_values)

# 设置 colorbar 的刻度
cb_tick_values = np.arange(0, 13, 2)  # 从 -1.0 到 1.0 的刻度值
cbar.set_ticks(cb_tick_values)

# 设置 colorbar 的刻度标签并格式化为一位小数
cb_tick_labels = [f"{tick*1:.1f}" for tick in cb_tick_values]
cbar.set_ticklabels(cb_tick_labels)

plt.show()

# 保存文件
# fig.savefig('prect variable_2011-2020 compare 1980-2010_CESM.svg', dpi=600, bbox_inches='tight')
# fig.savefig('prect variable_2011-2020 compare 1980-2010_CESM.pdf', dpi=600, bbox_inches='tight')
# fig.savefig('prect_ave2011-2020_MSWEP_with_mask.png', dpi=600, bbox_inches='tight')
# fig.savefig('prect_ave2011-2020_MSWEP_with_mask.pdf', dpi=600, bbox_inches='tight')
fig.savefig('prect_ave2011-2020_CESM_with_mask.png', dpi=600, bbox_inches='tight')
fig.savefig('prect_ave2011-2020_CESM_with_mask.pdf', dpi=600, bbox_inches='tight')