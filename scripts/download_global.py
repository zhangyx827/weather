import os
import time
import warnings
import cdsapi
import xarray as xr

# 1. 拦截废弃警告，保持控制台绝对整洁
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

c = cdsapi.Client()

# ========================================================
# STCast 官方项目配置 (严格对齐 config/*.yaml)
# ========================================================
GRID_RESOLUTION = '1.4' 

# STCast 要求的 13 个高空等压面
PRESSURE_LEVELS = [
    '50', '100', '150', '200', '250', '300', '400', 
    '500', '600', '700', '850', '925', '1000'
]

# 目标存储文件夹
single_dir = './era5_global_single_2022_6h'
pressure_dir = './era5_global_pressure_2022_6h'

os.makedirs(single_dir, exist_ok=True)
os.makedirs(pressure_dir, exist_ok=True)

# 变量配置
SURFACE_VARS = [
    '2m_temperature', 
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind', 
    'mean_sea_level_pressure'
]

ATMOS_VARS = [
    'u_component_of_wind', 
    'v_component_of_wind', 
    'geopotential', 
    'temperature', 
    'specific_humidity'
]

# 6小时抽样时次
HOURS_6H = ['00:00', '06:00', '12:00', '18:00']


# ========================================================
# 核心辅助函数
# ========================================================
def preprocess_era5(ds):
    """
    ERA5/ERA5T 数据预处理：融合并消除 expver 维度，避免后续合并冲突
    """
    # 1. 如果 expver 作为一个多值维度存在
    if 'expver' in ds.dims:
        if len(ds.expver) > 1:
            try:
                # 💡 关键改动：使用 drop=True 彻底丢弃 expver 标量坐标，防止 combine_first 时发生冲突
                ds_1 = ds.sel(expver=1, drop=True)
                ds_5 = ds.sel(expver=5, drop=True)
                ds = ds_1.combine_first(ds_5)
            except KeyError:
                # 备用方案：若 sel 失败则直接取第一个并丢弃该维度坐标
                ds = ds.isel(expver=0, drop=True)
        else:
            # 如果只有单个值，直接 squeeze 压缩掉该维度并彻底丢弃
            ds = ds.squeeze('expver', drop=True)
            
    # 2. 彻底清理可能残留的 expver 和 number 坐标/变量
    drop_vars = ['expver', 'number']
    ds = ds.drop_vars([v for v in drop_vars if v in ds.coords or v in ds.variables], errors='ignore')
        
    return ds


def safe_retrieve(dataset_name, request_params, output_path, max_retries=5):
    """
    带有自动超时解锁、临时文件保护及键盘中断清理机制的 CDS 下载函数
    """
    tmp_path = output_path + ".tmp"
    
    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            print(f"   -> 正在尝试下载 (第 {attempt}/{max_retries} 次)...")
            c.retrieve(dataset_name, request_params, tmp_path)
            
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output_path)
                return True
            else:
                raise FileNotFoundError("CDS 提示成功但未找到临时下载文件。")
                
        except KeyboardInterrupt:
            print("\n👋 接收到终止信号，正在清理临时文件并退出...")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise 
            
        except Exception as e:
            print(f"   ⚠️ [第 {attempt}/{max_retries} 次尝试失败]: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            if attempt < max_retries:
                sleep_time = attempt * 30  # 递增等待时长，给服务器缓冲时间
                time.sleep(sleep_time)
            else:
                return False


# ========================================================
# 步骤一：下载全球地面数据（数据量小，直接下载单月，不进行拆分）
# ========================================================
print(f"====== 开始下载地面数据集 (分辨率: {GRID_RESOLUTION}) ======")
for month in range(1, 13):
    month_str = f"{month:02d}"
    output_filename = os.path.join(single_dir, f"era5_global_single_2022_{month_str}.nc")
    
    if os.path.exists(output_filename):
        print(f"【跳过】地面数据已存在: {output_filename}")
        continue
        
    print(f"\n>>>> 正在请求 2022年{month_str}月 全球地面变量...")
    
    params = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf', 
        'variable': SURFACE_VARS,
        'year': '2022',
        'month': month_str,
        'day': [f"{d:02d}" for d in range(1, 32)],
        'time': HOURS_6H,
        'grid': [GRID_RESOLUTION, GRID_RESOLUTION],
    }
    
    safe_retrieve('reanalysis-era5-single-levels', params, output_filename)


# ========================================================
# 步骤二：下载全球高空数据（采用“隐式拆分 + 自动清洗合并”策略）
# ========================================================
print(f"\n====== 开始下载高空数据集 (分辨率: {GRID_RESOLUTION}) ======")
DAY_CHUNKS = [
    ("part1", [f"{d:02d}" for d in range(1, 16)]),
    ("part2", [f"{d:02d}" for d in range(16, 32)])
]

for month in range(1, 13):
    month_str = f"{month:02d}"
    for var in ATMOS_VARS:
        final_filename = os.path.join(pressure_dir, f"era5_global_pl_2022_{month_str}_{var}.nc")
        
        # 如果最终合并后的标准单月文件已存在，直接跳过整个月的该变量
        if os.path.exists(final_filename):
            print(f"【跳过】完整高空文件已存在: {month_str}月 【{var}】")
            continue
            
        print(f"\n>>>> 正在处理 2022年{month_str}月 【{var}】...")
        
        p1_path = final_filename.replace('.nc', '_part1.nc')
        p2_path = final_filename.replace('.nc', '_part2.nc')
        
        success = True
        # 分段下载前半月和后半月
        for part_label, days in DAY_CHUNKS:
            part_path = p1_path if part_label == "part1" else p2_path
            
            # 如果分段文件已经存在，无需重复下载
            if os.path.exists(part_path):
                print(f" 📥 [分段已存在] {part_label} (天数: {days[0]}~{days[-1]})")
                continue
            
            print(f" 📥 正在下载 {part_label} (天数: {days[0]}~{days[-1]})...")
            params = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': [var], 
                'pressure_level': PRESSURE_LEVELS,
                'year': '2022',
                'month': month_str,
                'day': days,
                'time': HOURS_6H,
                'grid': [GRID_RESOLUTION, GRID_RESOLUTION],
            }
            if not safe_retrieve('reanalysis-era5-pressure-levels', params, part_path):
                success = False
                break
        
        # 两部分均下载完成，开始现场清洗、合并及清理垃圾
        if success and os.path.exists(p1_path) and os.path.exists(p2_path):
            print(f" 🧬 两部分下载完成，正在现场清洗并融合成完整月文件...")
            try:
                # 1. 加载临时文件
                ds1 = xr.open_dataset(p1_path)
                ds2 = xr.open_dataset(p2_path)
                
                # 2. 调用预处理函数，强行统一并消除坐标污染
                ds1_clean = preprocess_era5(ds1)
                ds2_clean = preprocess_era5(ds2)
                
                # 3. 在时间维度上融合
                # 💡 关键改动：加入 compat='override' 和 combine_attrs='override' 强行绕过任何潜在的非对齐冲突
                ds_combined = xr.concat(
                    [ds1_clean, ds2_clean], 
                    dim='time', 
                    data_vars='minimal', 
                    coords='minimal',
                    compat='override',
                    combine_attrs='override'
                )
                
                # 4. 写入最终目标 NC 文件
                ds_combined.to_netcdf(final_filename)
                
                # 5. 关闭句柄，释放内存
                ds1.close()
                ds2.close()
                ds_combined.close()
                
                # 6. 过河拆桥：删除两部分临时的切分文件，保持目录绝对干净
                os.remove(p1_path)
                os.remove(p2_path)
                print(f" 🎉 【成功】完整文件已就位，临时垃圾已清理！")
                
            except Exception as e:
                print(f" ❌ 现场合并失败: {e}")

print("\n====== ✨ STCast 全球数据集一体化流水线执行完毕！ ======")