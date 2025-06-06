import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# --------------------
# 1. 设置分辨率与经纬度
# --------------------
nlat, nlon = 1800, 3600
lat = np.linspace(90 - 0.05, -90 + 0.05, nlat)   # 从北到南
lon = np.linspace(-179.95, 179.95, nlon)            # 从西向东
lon2d, lat2d = np.meshgrid(lon, lat)  # 默认就是 indexing='xy'，更符合图像绘制顺序


# --------------------
# 2. 读取两个Excel文件为矩阵
# --------------------
file_1981_1990 = r"E:\Data_temp\MSWEP_V28\Mean_Precip_1981_1990.xlsx"
file_2011_2020 = r"E:\Data_temp\MSWEP_V28\Mean_Precip_2011_2020.xlsx"

value_1981_1990 = pd.read_excel(file_1981_1990, header=None).values
value_2011_2020 = pd.read_excel(file_2011_2020, header=None).values

# --------------------
# 3. 计算差值与相对变化
# --------------------
abs_diff = value_2011_2020 - value_1981_1990
rel_diff = np.divide(abs_diff, value_1981_1990, out=np.full_like(abs_diff, np.nan), where=value_1981_1990 != 0)
rel_diff_percent = rel_diff * 100


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# --------------------
# 4. 绘图函数
# --------------------
def plot_map(data, title, output_file, cmap, vmin, vmax, cbar_label, insets=False):
    fig = plt.figure(figsize=(16, 9))
    # ✅ 缩小主图范围，避免 inset 遮挡
    main_ax = fig.add_axes([0.05, 0.13, 0.69, 0.80], projection=ccrs.PlateCarree())
    main_ax.set_global()
    main_ax.coastlines()
    
    # 经纬度网格
    main_ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    main_ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    main_ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
    main_ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
    main_ax.tick_params(axis='both', which='major', labelsize=16, length=8)

    # 主图
    levels = np.linspace(vmin, vmax, 100)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    con = main_ax.contourf(lon2d, lat2d, data, levels=levels, cmap=cmap, norm=norm, extend='both')

    # ✅ 更美观的 colorbar：居中放在主图下方
    cb_ax = fig.add_axes([0.08, 0.12, 0.72, 0.03])
    cb = fig.colorbar(con, cax=cb_ax, orientation='horizontal')
    cb.set_label("Relative change in precipitation (%)", fontsize=16)
    cb.ax.tick_params(labelsize=16)
    
    # ✅ 设置 colorbar 刻度格式
    import matplotlib.ticker as mticker
    if "relative" in cbar_label.lower():
        ticks = np.linspace(vmin, vmax, 11)  # 11个刻度：-100 到 100
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{x:.1f}" for x in ticks])
    else:
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle(title, fontsize=18, x=0.4, y=0.88)
    

    
    # 🔧 添加 inset 小图，仅当 insets=True
    if insets:
        regions = [
            {"name": "RE1", "lon": (-78, -73), "lat": (-10, -15), "loc": [0.74, 0.69, 0.18, 0.18]},
            {"name": "RE2", "lon": (4, 9), "lat": (50, 55), "loc": [0.74, 0.44, 0.18, 0.18]},
            {"name": "RE3", "lon": (133, 138), "lat": (32, 37), "loc": [0.74, 0.19, 0.18, 0.18]},
        ]
        for region in regions:
            inset_ax = fig.add_axes(region["loc"], projection=ccrs.PlateCarree())  # ✅ 使用 cartopy 的 GeoAxes
            inset_ax.set_extent([*region["lon"], *region["lat"]], crs=ccrs.PlateCarree())
            inset_ax.contourf(lon2d, lat2d, data, levels=levels, cmap=cmap, norm=norm, extend='both')
            inset_ax.coastlines(resolution='10m', linewidth=0.5)
            inset_ax.set_title(region["name"], fontsize=12)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            import matplotlib.ticker as mticker
            from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
            
            tick_font = 12  # 主图16
            
            gl = inset_ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.MultipleLocator(1)
            gl.ylocator = mticker.MultipleLocator(1)
            gl.xlabel_style = {'size': tick_font}
            gl.ylabel_style = {'size': tick_font}

        
    # 保存图像
    fig.savefig(output_file + ".png", dpi=600, bbox_inches='tight')
    # fig.savefig(output_file + ".pdf", dpi=600, bbox_inches='tight')
    plt.close()

# --------------------
# 5. 自定义配色方案
# --------------------
# colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
colors = ['#023858', '#0570b0', '#fff7ec', '#d7301f', '#7f0000']
custom_cmap = LinearSegmentedColormap.from_list(name='custom', colors=colors, N=100)

# --------------------
# 6. 绘制图像
# --------------------
# plot_map(
#     data=abs_diff,
#     title="Absolute Change in Precipitation (2011–2020 vs 1981–1990)",
#     output_file=r"E:\Data_temp\MSWEP_V28\abs_change_precip",
#     cmap=custom_cmap,
#     vmin=-0.1,
#     vmax=0.1,
#     cbar_label="Absolute change in precipitation (mm/day)"
# )

plot_map(
    data=rel_diff_percent,
    title="Relative Change in Precipitation (2011–2020 vs 1981–1990)",
    output_file=r"E:\Data_temp\MSWEP_V28\rel_change_precip1",
    cmap=custom_cmap,
    vmin=-100.0,
    vmax=100.0,
    cbar_label="Relative change in precipitation (%)",
    insets=True  # 添加 inset 小图
)
