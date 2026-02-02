# 依赖于conda中的drawMap环境
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.transform import from_origin
import os

# Define your data path
etopo_path = r"D:\Documents\ArcGIS\etopo1_ice_g_f4\etopo1_ice_g_f4.flt"

# Define the region of interest
lon_min, lon_max = 105.3, 118.0
lat_min, lat_max = 12.0, 23.0

small_scale_lon_min, small_scale_lon_max = 110.5, 113.5
small_scale_lat_min, small_scale_lat_max = 17.5, 20.5

def draw_large_scale_bathymetry_map():
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Plot bathymetry data
    max_depth = -4500
    max_height = 1500
    rate = np.abs(max_depth) / (np.abs(max_depth) + max_height)
    levels = np.linspace(max_depth, max_height, 3000)
    # Create ocean-focused colormap
    colors = np.vstack((plt.cm.ocean(np.linspace(0.3, 1, int(256*rate))), plt.cm.terrain(np.linspace(0.2, 1, int(256*(1-rate))))))
    bathymap = ListedColormap(colors)
    # Plot bathymetry data
    contour = ax.contourf(lon, lat, bathymetry_data, levels=levels,
                          cmap=bathymap, extend='both')
    # Add contour lines
    ax.contour(lon, lat, bathymetry_data, levels=np.arange(max_depth, max_height, 500), colors='black', linewidths=0.5, linestyles='solid', zorder=4)

    # 在图上110.5-113.5E, 17.5-20.5N位置上添加标记，用于指示研究
    ax.add_patch(plt.Rectangle((small_scale_lon_min, small_scale_lat_min), 3.0, 3.0, linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree(), zorder=5))
    # 在矩形范围内绘制等间隔的64*64个格点
    lons = np.arange(small_scale_lon_min, small_scale_lon_max + 0.0001, 3.0/64)   # 注意要包含终点
    lats = np.arange(small_scale_lat_min, small_scale_lat_max + 0.0001, 3.0/64)
    lon2d, lat2d = np.meshgrid(lons, lats)
    ax.scatter(lon2d, lat2d,
           s=0.05, color='red', marker='.',
           transform=ccrs.PlateCarree(), zorder=6)

    ax.text(
        110.5, 20.8,
        'ML training area: \nThe spatial resolution is\n(3/64)°x(3/64)°',
        color='red',
        fontsize=12,
        transform=ccrs.PlateCarree(),
        zorder=5,
        bbox=dict(
            facecolor='white',      # 背景颜色
            alpha=0.6,              # 透明度 0.6
            edgecolor='none',       # 无边框（最常用）
            # edgecolor='gray',     # 如果想要细边框可以打开这行
            pad=0.4,                # 文字到框的内边距（单位：点/字体高度的倍数）
            boxstyle='round,pad=0.3' # 可选：圆角矩形（更美观）
        )
    )
    
    
    # Add land mask
    # ax.add_feature(cfeature.LAND, facecolor='#d3bc8a', zorder=2)
    
    # Add geographical features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    # ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    # ax.add_feature(cfeature.OCEAN, zorder=1)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.05, orientation='horizontal')
    cbar.set_label('Elevation/Depth (meters)', fontsize=12)
    ticks = list(range(0, int(max_depth)-1, -1000))  # 0到max_depth（向下）
    ticks.reverse()      # 反转使深度从大到小排序
    ticks += list(range(0, int(max_height)+1, 1000))  # 0到max_height（向上）
    cbar.set_ticks(ticks)
    
    # Add title
    # plt.title('Bathymetry Map: South China Sea Region\n(12°N to 23°N, 105.3°E to 118°E)', fontsize=14, pad=20)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('../figs/south_china_sea_bathymetry.tiff', dpi=600, bbox_inches='tight')
    plt.show()

def draw_small_scale_bathymetry_map():
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([small_scale_lon_min, small_scale_lon_max, small_scale_lat_min, small_scale_lat_max], crs=ccrs.PlateCarree())
    
    # Plot bathymetry data
    max_depth = -100
    max_height = 50
    rate = np.abs(max_depth) / (np.abs(max_depth) + max_height)
    levels = np.linspace(max_depth, max_height, 3000)
    # Create ocean-focused colormap
    colors = plt.cm.ocean(np.linspace(0.3, 1, int(256*rate)))
    bathymap = ListedColormap(colors)
    # Plot bathymetry data
    contour = ax.contourf(lon, lat, bathymetry_data, levels=levels,
                          cmap=bathymap, extend='both')
    # Add contour lines
    ax.contour(lon, lat, bathymetry_data, levels=np.arange(max_depth, 0, 100), colors='black', linewidths=0.5, linestyles='solid', zorder=4)

    # 在图上110.5-113.5E, 17.5-20.5N位置上添加标记，用于指示研究
    ax.add_patch(plt.Rectangle((small_scale_lon_min, small_scale_lat_min), 3.0, 3.0, linewidth=12, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree(), zorder=5))
    # 在矩形范围内绘制等间隔的64*64个格点
    lons = np.arange(small_scale_lon_min, small_scale_lon_max + 0.0001, 3.0/64)   # 注意要包含终点
    lats = np.arange(small_scale_lat_min, small_scale_lat_max + 0.0001, 3.0/64)
    lon2d, lat2d = np.meshgrid(lons, lats)
    ax.scatter(lon2d, lat2d,
           s=10, color='red', marker='.',
           transform=ccrs.PlateCarree(), zorder=6)
    
    # Add land mask
    ax.add_feature(cfeature.LAND, facecolor='#d3bc8a', zorder=6)
    
    # Add geographical features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=6)
    # ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
    # ax.add_feature(cfeature.OCEAN, zorder=1)
    
    # Add gridlines
    # gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {'size': 12}
    # gl.ylabel_style = {'size': 12}
    
    # Add colorbar
    # cbar = plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.05, orientation='horizontal')
    # cbar.set_label('Elevation/Depth (meters)', fontsize=12)
    # ticks = list(range(0, int(max_depth)-1, -50))  # 0到max_depth（向下）
    # ticks.reverse()      # 反转使深度从大到小排序
    # ticks += list(range(0, int(max_height)+1, 50))  # 0到max_height（向上）
    # cbar.set_ticks(ticks)
    
    # Add title
    # plt.title('Bathymetry Map: South China Sea Region\n(12°N to 23°N, 105.3°E to 118°E)', fontsize=14, pad=20)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('../figs/Hainan_sea_bathymetry.tiff', dpi=600, bbox_inches='tight')
    plt.show()


try:
    # Open the ETOPO1 data file
    with rasterio.open(etopo_path) as src:
        # Calculate pixel coordinates for the region
        row_start, col_start = src.index(lon_min, lat_max)
        row_stop, col_stop = src.index(lon_max, lat_min)
        
        # Read the data within the bounding box
        window = rasterio.windows.Window.from_slices(
            (row_start, row_stop),
            (col_start, col_stop)
        )
        bathymetry_data = src.read(1, window=window)
        
        # Calculate new transform for the subset
        transform = rasterio.windows.transform(window, src.transform)
        
        # Get coordinates
        x = np.linspace(transform.c, transform.c + transform.a * bathymetry_data.shape[1], bathymetry_data.shape[1])
        y = np.linspace(transform.f, transform.f + transform.e * bathymetry_data.shape[0], bathymetry_data.shape[0])
        lon, lat = np.meshgrid(x, y)

    # Draw the large-scale bathymetry map
    draw_large_scale_bathymetry_map()
    # Draw the small-scale bathymetry map
    # draw_small_scale_bathymetry_map()



except Exception as e:
    print(f"Error processing data: {e}")
    print("Possible causes:")
    print("1. File not found at specified path")
    print("2. Corrupted or incomplete data file")
    print("3. Missing .hdr header file in the same directory")
    print("4. Insufficient memory for data processing")