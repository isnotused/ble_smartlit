import numpy as np
from scipy import linalg
from typing import Union, Tuple

class MathUtils:
    """数学计算工具函数"""
    
    @staticmethod
    def matrix_exponential(A: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        计算矩阵指数
        Args:
            A: 输入矩阵
            t: 时间参数
        Returns:
            矩阵指数
        """
        return linalg.expm(A * t)
    
    @staticmethod
    def compute_eigen_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算矩阵特征分解
        Args:
            matrix: 输入矩阵
        Returns:
            eigenvalues: 特征值
            eigenvectors: 特征向量
        """
        eigenvalues, eigenvectors = linalg.eig(matrix)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def solve_lyapunov_equation(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        求解李雅普诺夫方程 AX + XA^T + Q = 0
        Args:
            A: 系统矩阵
            Q: 对称矩阵
        Returns:
            解矩阵X
        """
        X = linalg.solve_continuous_lyapunov(A, -Q)
        return X
    
    @staticmethod
    def compute_matrix_logarithm(matrix: np.ndarray) -> np.ndarray:
        """
        计算矩阵对数
        Args:
            matrix: 输入矩阵
        Returns:
            矩阵对数
        """
        return linalg.logm(matrix)
    
    @staticmethod
    def weighted_norm(vector: np.ndarray, 
                     weights: np.ndarray = None, 
                     p: float = 2.0) -> float:
        """
        计算加权范数
        Args:
            vector: 输入向量
            weights: 权重向量
            p: 范数阶数
        Returns:
            加权范数值
        """
        if weights is None:
            weights = np.ones_like(vector)
            
        weighted_vector = vector * weights
        norm_value = np.linalg.norm(weighted_vector, ord=p)
        return norm_value
    
    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> float:
        """
        计算概率分布的熵
        Args:
            probabilities: 概率分布
        Returns:
            熵值
        """
        # 移除零概率避免log(0)
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
        return entropy
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        计算KL散度
        Args:
            p: 概率分布P
            q: 概率分布Q
        Returns:
            KL散度
        """
        # 确保概率分布有效
        p_safe = np.clip(p, 1e-10, 1.0)
        q_safe = np.clip(q, 1e-10, 1.0)
        
        kl = np.sum(p_safe * np.log(p_safe / q_safe))
        return kl
    
    @staticmethod
    def compute_covariance_matrix(data: np.ndarray, 
                                rowvar: bool = False) -> np.ndarray:
        """
        计算协方差矩阵
        Args:
            data: 输入数据矩阵
            rowvar: 是否每行代表一个变量
        Returns:
            协方差矩阵
        """
        if rowvar:
            data = data.T
            
        covariance = np.cov(data, rowvar=False)
        return covariance
    
    @staticmethod
    def mahalanobis_distance(x: np.ndarray, 
                           mean: np.ndarray, 
                           cov: np.ndarray) -> float:
        """
        计算马氏距离
        Args:
            x: 观测向量
            mean: 均值向量
            cov: 协方差矩阵
        Returns:
            马氏距离
        """
        diff = x - mean
        inv_cov = linalg.inv(cov)
        distance = np.sqrt(diff.T @ inv_cov @ diff)
        return distance
    
    @staticmethod
    def compute_gradient(f: callable, 
                        x: np.ndarray, 
                        h: float = 1e-6) -> np.ndarray:
        """
        数值计算函数梯度
        Args:
            f: 目标函数
            x: 输入点
            h: 步长
        Returns:
            梯度向量
        """
        n = len(x)
        gradient = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        return gradient
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, 
                          b: np.ndarray, 
                          method: str = 'lu') -> np.ndarray:
        """
        求解线性方程组
        Args:
            A: 系数矩阵
            b: 右侧向量
            method: 求解方法
        Returns:
            解向量
        """
        if method == 'lu':
            # LU分解法
            solution = linalg.solve(A, b)
        elif method == 'qr':
            # QR分解法
            Q, R = linalg.qr(A)
            solution = linalg.solve_triangular(R, Q.T @ b)
        elif method == 'svd':
            # SVD分解法
            U, s, Vh = linalg.svd(A)
            solution = Vh.T @ (U.T @ b / s)
        else:
            raise ValueError(f"不支持的求解方法: {method}")
            
        return solution