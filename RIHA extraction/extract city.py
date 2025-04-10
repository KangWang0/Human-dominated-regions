


# import geopandas as gpd

# # 读取 shapefile
# shapefile_path = r"E:\Data_temp\CESM2_LENS\wildareasv32009\wildareasv32009_exported.shp"
# gdf = gpd.read_file(shapefile_path)

# # 查看 shapefile 的坐标系
# print("原始坐标系：", gdf.crs)

# # 如果不是 WGS84 (EPSG:4326)，则进行转换
# if gdf.crs != 'EPSG:4326':
#     gdf = gdf.to_crs('EPSG:4326')
#     print("转换后的坐标系：", gdf.crs)

# # 保存转换后的 shapefile
# converted_shapefile_path = r"E:\Data_temp\CESM2_LENS\wildareasv32009\wildareasv32009_converted.shp"
# gdf.to_file(converted_shapefile_path)


import geopandas as gpd
import numpy as np
import netCDF4 as nc
from shapely.geometry import Point

# 读取转换后的 shapefile
converted_shapefile_path = r"E:\Data_temp\CESM2_LENS\wildareasv32009\wildareasv32009_exported.shp"
gdf = gpd.read_file(converted_shapefile_path)

# 读取示例 netCDF 文件以获取经纬度信息
example_nc_file = r"E:\Data_temp\CESM2_LENS\Anomaly_prect\1850001.nc"
with nc.Dataset(example_nc_file, 'r') as src:
    lons = src.variables['lon'][:]
    lats = src.variables['lat'][:]

# 创建掩膜矩阵
mask = np.full((len(lats), len(lons)), False)

# 检查每个网格点是否在 shapefile 范围内
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        point = Point(lon, lat)
        if gdf.contains(point).any():
            mask[i, j] = True

# 保存掩膜矩阵
np.save(r"E:\Data_temp\CESM2_LENS\city_mask_matrix.npy", mask)
print("掩膜矩阵已保存。")

# import geopandas as gpd
# import numpy as np
# import netCDF4 as nc
# from shapely.geometry import Point

# # 读取转换后的 shapefile
# shapefile_path = r"E:\Data_temp\CESM2_LENS\wildareasv32009\wildareasv32009_exported.shp"
# gdf = gpd.read_file(shapefile_path)

# # 读取示例 netCDF 文件以获取经纬度信息
# example_nc_file = r"E:\Data_temp\CESM2_LENS\Anomaly_prect\1850001.nc"
# with nc.Dataset(example_nc_file, 'r') as src:
#     lons = src.variables['lon'][:]
#     lats = src.variables['lat'][:]

# # 创建掩膜矩阵
# mask = np.full((len(lats), len(lons)), False)

# # 检查每个网格点是否在 shapefile 范围内
# for i, lat in enumerate(lats):
#     for j, lon in enumerate(lons):
#         point = Point(lon, lat)
#         if gdf.contains(point).any():
#             mask[i, j] = True

# # 保存掩膜矩阵
# np.save(r"E:\Data_temp\CESM2_LENS\mask_matrix.npy", mask)
# print("掩膜矩阵已保存。")




import os
import pandas as pd
import numpy as np
import netCDF4 as nc

# 载入预计算的掩膜矩阵
mask = np.load(r"E:\Data_temp\CESM2_LENS\city_mask_matrix.npy")

# 定义输入和输出路径
input_nc_dir = r"E:\Data_temp\CESM2_LENS\Anomaly_prect"
output_nc_dir = r"E:\Data_temp\CESM2_LENS\Anomaly_prect_cities"

# 创建输出目录（如果不存在）
os.makedirs(output_nc_dir, exist_ok=True)

# 遍历年份和日期
for year in range(1850, 2100):
    for day in range(1, 366):
        try:
            date = pd.to_datetime(f"{year}-{day}", format='%Y-%j')
            filename = f"{year}{day:03d}.nc"
            input_file_path = os.path.join(input_nc_dir, filename)
            output_file_path = os.path.join(output_nc_dir, filename)

            # 打开 netCDF 文件
            with nc.Dataset(input_file_path, 'r') as src:
                prect = src.variables['prect'][:]

                # 应用掩膜，将不在范围内的点设为 NaN
                prect_masked = np.where(mask, prect, np.nan)

                # 写入新的 netCDF 文件
                with nc.Dataset(output_file_path, 'w', format='NETCDF4') as dst:
                    # 创建维度
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
                    
                    # 创建变量
                    for name, variable in src.variables.items():
                        x = dst.createVariable(name, variable.datatype, variable.dimensions)
                        dst[name].setncatts(src[name].__dict__)
                        if name == 'prect':
                            dst[name][:] = prect_masked
                        else:
                            dst[name][:] = src[name][:]
                    
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

