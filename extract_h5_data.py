"""
从 HDF5 文件中提取和可视化数据
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def extract_and_visualize(filename):
    """提取并可视化HDF5文件中的数据"""
    print(f"正在读取文件: {filename}")
    
    with h5py.File(filename, 'r') as f:
        # 读取增强信号
        if 'enhanced_signal' in f:
            enhanced_signal = f['enhanced_signal'][:]
            print(f"✅ 增强信号: {enhanced_signal.shape}")
            
            # 绘制信号
            plt.figure(figsize=(12, 4))
            plt.plot(enhanced_signal[:1000], label='增强后信号')
            plt.title('增强后的信号波形')
            plt.xlabel('采样点')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('extracted_signal.png', dpi=150)
            print("📊 信号图已保存到: extracted_signal.png")
            plt.close()
        
        # 读取特征矩阵
        if 'feature_matrix' in f:
            feature_matrix = f['feature_matrix'][:]
            print(f"✅ 特征矩阵: {feature_matrix.shape}")
            print(f"   前5行数据:\n{feature_matrix[:5]}")
        
        # 读取质量矩阵
        if 'quality_matrix' in f:
            quality_matrix = f['quality_matrix'][:]
            print(f"✅ 质量矩阵: {quality_matrix.shape}")
            
            # 可视化质量评分
            quality_scores = quality_matrix[:, :, 1].mean(axis=1)
            plt.figure(figsize=(10, 5))
            plt.plot(quality_scores, marker='o')
            plt.title('信号质量评分趋势')
            plt.xlabel('时间窗口')
            plt.ylabel('质量评分')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('quality_trend.png', dpi=150)
            print("📊 质量趋势图已保存到: quality_trend.png")
            plt.close()
        
        # 读取参数
        if 'parameters' in f:
            params = f['parameters']
            print("\n✅ 优化参数:")
            for key in params.keys():
                value = params[key][:]
                print(f"   {key}: {value}")
        
        # 显示文件元数据
        print("\n📝 文件元数据:")
        for key, value in f.attrs.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    file_path = "bluetooth_optimization/adaptive-ble-receiver/data/optimization_results.h5"
    extract_and_visualize(file_path)
    print("\n✅ 数据提取完成！")
