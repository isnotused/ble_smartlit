"""
查看 HDF5 文件内容的脚本
"""
import h5py
import numpy as np

def view_h5_file(filename):
    """查看HDF5文件的结构和内容"""
    print(f"正在读取文件: {filename}")
    print("=" * 60)
    
    try:
        with h5py.File(filename, 'r') as f:
            print("\n文件结构:")
            print("-" * 60)
            
            def print_structure(name, obj):
                """递归打印HDF5文件结构"""
                if isinstance(obj, h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  形状: {obj.shape}")
                    print(f"  数据类型: {obj.dtype}")
                    print(f"  大小: {obj.size} 个元素")
                    
                    # 显示部分数据预览
                    if obj.size > 0:
                        if obj.size <= 10:
                            print(f"  数据: {obj[...]}")
                        else:
                            print(f"  数据预览: {obj[...][:10] if len(obj.shape) == 1 else obj[...]}")
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"分组: {name}/")
                    print()
            
            f.visititems(print_structure)
            
            # 显示根级别的属性
            if f.attrs:
                print("\n文件属性:")
                print("-" * 60)
                for key, value in f.attrs.items():
                    print(f"{key}: {value}")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 查看项目中的HDF5文件
    files = [
        "bluetooth_optimization/adaptive-ble-receiver/data/optimization_results.h5",
        "data/optimization_results.h5"
    ]
    
    for file in files:
        try:
            view_h5_file(file)
            print("\n" + "=" * 60 + "\n")
        except FileNotFoundError:
            print(f"文件不存在: {file}\n")
