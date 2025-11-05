import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def progress_bar(duration, step_name):
    """显示带进度条的处理过程"""
    logging.info(f"开始{step_name}...")
    total_steps = 100
    interval = duration / total_steps
    
    for i in tqdm(range(total_steps), desc=step_name, ncols=100):
        time.sleep(interval)
    
    logging.info(f"{step_name}完成")