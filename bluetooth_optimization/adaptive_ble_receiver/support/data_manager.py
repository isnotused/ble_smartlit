import numpy as np
import pickle
import json
import h5py
from typing import Dict, List, Optional
from datetime import datetime
import os

class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._ensure_data_directory()
        
    def save_optimization_result(self, result: Dict, filename: str = None) -> str:
        """
        保存优化结果
        Args:
            result: 优化结果数据
            filename: 文件名
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_result_{timestamp}.h5"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with h5py.File(filepath, 'w') as f:
                # 保存基本元数据
                f.attrs['timestamp'] = datetime.now().isoformat()
                f.attrs['version'] = '1.0'
                
                # 保存信号数据
                if 'enhanced_signal' in result:
                    signal_data = result['enhanced_signal']
                    f.create_dataset('enhanced_signal', data=signal_data)
                
                # 保存特征矩阵
                if 'feature_matrix' in result:
                    feature_data = result['feature_matrix']
                    f.create_dataset('feature_matrix', data=feature_data)
                
                # 保存质量评估结果
                if 'quality_assessment' in result:
                    quality_data = result['quality_assessment']['quality_matrix']
                    f.create_dataset('quality_matrix', data=quality_data)
                
                # 保存参数配置
                if 'new_parameters' in result:
                    parameters = result['new_parameters']
                    param_group = f.create_group('parameters')
                    for key, value in parameters.items():
                        if isinstance(value, (int, float)):
                            param_group.attrs[key] = value
                        elif isinstance(value, list):
                            param_group.create_dataset(key, data=value)
                
            return filepath
            
        except Exception as e:
            print(f"保存优化结果失败: {e}")
            return self._save_fallback(result, filename)
    
    def load_optimization_result(self, filename: str) -> Optional[Dict]:
        """
        加载优化结果
        Args:
            filename: 文件名
        Returns:
            优化结果数据
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            result = {}
            
            with h5py.File(filepath, 'r') as f:
                # 加载元数据
                result['timestamp'] = f.attrs.get('timestamp', '')
                result['version'] = f.attrs.get('version', '')
                
                # 加载信号数据
                if 'enhanced_signal' in f:
                    result['enhanced_signal'] = f['enhanced_signal'][:]
                
                # 加载特征矩阵
                if 'feature_matrix' in f:
                    result['feature_matrix'] = f['feature_matrix'][:]
                
                # 加载质量矩阵
                if 'quality_matrix' in f:
                    result['quality_assessment'] = {
                        'quality_matrix': f['quality_matrix'][:]
                    }
                
                # 加载参数配置
                if 'parameters' in f:
                    param_group = f['parameters']
                    parameters = {}
                    
                    # 加载属性参数
                    for key in param_group.attrs:
                        parameters[key] = param_group.attrs[key]
                    
                    # 加载数据集参数
                    for key in param_group:
                        parameters[key] = param_group[key][:].tolist()
                    
                    result['new_parameters'] = parameters
            
            return result
            
        except Exception as e:
            print(f"加载优化结果失败: {e}")
            return None
    
    def save_training_data(self, features: np.ndarray, 
                          parameters: List[Dict], 
                          qualities: List[float],
                          filename: str = None) -> str:
        """
        保存训练数据
        Args:
            features: 环境特征数据
            parameters: 参数配置列表
            qualities: 质量评分列表
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.h5"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with h5py.File(filepath, 'w') as f:
                # 保存特征数据
                f.create_dataset('environment_features', data=features)
                
                # 保存参数数据
                param_group = f.create_group('optimal_parameters')
                for i, param_dict in enumerate(parameters):
                    subgroup = param_group.create_group(f'sample_{i}')
                    for key, value in param_dict.items():
                        if isinstance(value, (int, float)):
                            subgroup.attrs[key] = value
                        elif isinstance(value, list):
                            subgroup.create_dataset(key, data=value)
                
                # 保存质量数据
                f.create_dataset('quality_scores', data=qualities)
                
                # 保存元数据
                f.attrs['sample_count'] = len(features)
                f.attrs['timestamp'] = datetime.now().isoformat()
            
            return filepath
            
        except Exception as e:
            print(f"保存训练数据失败: {e}")
            return ""
    
    def load_training_data(self, filename: str) -> Optional[Dict]:
        """
        加载训练数据
        Args:
            filename: 文件名
        Returns:
            训练数据
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                
                # 加载特征数据
                data['environment_features'] = f['environment_features'][:]
                
                # 加载参数数据
                param_group = f['optimal_parameters']
                parameters = []
                for key in param_group:
                    subgroup = param_group[key]
                    param_dict = {}
                    
                    # 加载属性参数
                    for attr_key in subgroup.attrs:
                        param_dict[attr_key] = subgroup.attrs[attr_key]
                    
                    # 加载数据集参数
                    for data_key in subgroup:
                        param_dict[data_key] = subgroup[data_key][:].tolist()
                    
                    parameters.append(param_dict)
                
                data['optimal_parameters'] = parameters
                
                # 加载质量数据
                data['quality_scores'] = f['quality_scores'][:]
                
                return data
                
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return None
    
    def export_configuration(self, config: Dict, filename: str) -> str:
        """
        导出系统配置
        Args:
            config: 配置字典
            filename: 文件名
        Returns:
            导出的文件路径
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return filepath
        except Exception as e:
            print(f"导出配置失败: {e}")
            return ""
    
    def import_configuration(self, filename: str) -> Optional[Dict]:
        """
        导入系统配置
        Args:
            filename: 文件名
        Returns:
            配置字典
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"导入配置失败: {e}")
            return None
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """清理过期数据"""
        current_time = datetime.now()
        
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (current_time - file_time).days
                
                if age_days > max_age_days:
                    try:
                        os.remove(filepath)
                        print(f"已删除过期文件: {filename}")
                    except Exception as e:
                        print(f"删除文件失败 {filename}: {e}")
    
    def get_data_statistics(self) -> Dict:
        """获取数据统计信息"""
        total_size = 0
        file_count = 0
        file_types = {}
        
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.isfile(filepath):
                file_count += 1
                total_size += os.path.getsize(filepath)
                
                # 统计文件类型
                file_ext = os.path.splitext(filename)[1].lower()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': file_types,
            'data_directory': self.data_dir
        }
    
    def _ensure_data_directory(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _save_fallback(self, result: Dict, filename: str) -> str:
        """降级保存方法"""
        try:
            # 使用pickle作为备选方案
            filepath = os.path.join(self.data_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            return filepath
        except Exception as e:
            print(f"降级保存也失败: {e}")
            return ""