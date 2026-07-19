import os
import time
import warnings
import cdsapi
import xarray as xr

# 1. 拦截废弃警告，保持控制台绝对整洁
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 调整底层连接池和重试参数，缩短断流后的等待时间
c = cdsapi.Client(
    timeout=60,       # 60秒无响应则超时
    retry_max=10,     # 内部最大重试次数（不用默认的500次那么夸张）
    sleep_max=30      # 失败后最多等待30秒就再次尝试，而不是120秒
)

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
single_dir = './era5_global_single_2019_6h'
pressure_dir = './era5_global_pressure_2019_6h'

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
    if 'expver' in ds.dims:
        if len(ds.expver) > 1:
            try:
                ds_1 = ds.sel(expver=1, drop=True)
                ds_5 = ds.sel(expver=5, drop=True)
                ds = ds_1.combine_first(ds_5)
            except KeyError:
                ds = ds.isel(expver=0, drop=True)
        else:
            ds = ds.squeeze('expver', drop=True)
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
                sleep_time = attempt * 30  # 递增等待时长
                time.sleep(sleep_time)
            else:
                return False


# ========================================================
# 步骤一：下载全球地面数据（整月，不分段）
# ========================================================
print(f"====== 开始下载地面数据集 (分辨率: {GRID_RESOLUTION}) ======")
for month in range(1, 13):
    month_str = f"{month:02d}"
    output_filename = os.path.join(single_dir, f"era5_global_single_2019_{month_str}.nc")
    
    if os.path.exists(output_filename):
        print(f"【跳过】地面数据已存在: {output_filename}")
        continue
        
    print(f"\n>>>> 正在请求 2019年{month_str}月 全球地面变量...")
    
    params = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf', 
        'variable': SURFACE_VARS,
        'year': '2019',
        'month': month_str,
        'day': [f"{d:02d}" for d in range(1, 32)],
        'time': HOURS_6H,
        'grid': [GRID_RESOLUTION, GRID_RESOLUTION],
    }
    
    safe_retrieve('reanalysis-era5-single-levels', params, output_filename)


# ========================================================
# 步骤二：下载全球高空数据（整月，不分段，直接请求全部日期）
# ========================================================
print(f"\n====== 开始下载高空数据集 (分辨率: {GRID_RESOLUTION}) ======")
for month in range(1, 13):
    month_str = f"{month:02d}"
    for var in ATMOS_VARS:
        final_filename = os.path.join(pressure_dir, f"era5_global_pl_2019_{month_str}_{var}.nc")
        
        # 如果最终文件已存在，跳过
        if os.path.exists(final_filename):
            print(f"【跳过】完整高空文件已存在: {month_str}月 【{var}】")
            continue
            
        print(f"\n>>>> 正在下载 2019年{month_str}月 【{var}】完整月份...")
        params = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': [var],
            'pressure_level': PRESSURE_LEVELS,
            'year': '2019',
            'month': month_str,
            'day': [f"{d:02d}" for d in range(1, 32)],   # 一次性请求全部日期
            'time': HOURS_6H,
            'grid': [GRID_RESOLUTION, GRID_RESOLUTION],
        }
        success = safe_retrieve('reanalysis-era5-pressure-levels', params, final_filename)

        # 若下载成功，可对文件进行可选的 expver 清洗
        if success:
            try:
                # 使用 with 语句并在内部 .load()，确保退出 block 时文件句柄已被完全释放
                with xr.open_dataset(final_filename) as ds:
                    ds_clean = preprocess_era5(ds).load() 
                
                # 此时文件已解锁，可以安全覆盖写入
                ds_clean.to_netcdf(final_filename)
                ds_clean.close()
                print(f"   ✅ 【{var}】下载完成并已清洗。")
            except Exception as e:
                print(f"   ⚠️ 清洗时发生错误: {e}")
        else:
            print(f"   ❌ 【{var}】下载失败，请检查网络或稍后重试。")

print("\n====== ✨ STCast 全球数据集一体化流水线执行完毕！ ======")