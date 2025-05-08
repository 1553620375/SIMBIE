from matplotlib import contour
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal, getcontext
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

getcontext().prec = 6
import matplotlib.pyplot as plt
from typing import Optional
from numpy.typing import NDArray
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import os
import re


def process_matrix(Final_D, level, select,factor =0.3):
    """
    处理矩阵，根据颜色阈值映射数据

    参数:
    Final_D: 原始矩阵
    level: 颜色等级数量
    select: 选择的颜色等级（从1开始）

    返回:
    modified_D: 修改后的矩阵
    """
    # 获取矩阵的最小值和最大值
    min_val = np.min(Final_D)
    max_val = np.max(Final_D)

    # 创建颜色映射的归一化器
    norm = Normalize(vmin=min_val, vmax=max_val)

    # 计算颜色等级的边界值
    levels = np.linspace(min_val, max_val, level + 1)

    # 确保select在有效范围内
    select = max(1, min(level, select))

    # 计算select_threshold (对应select颜色层的最低值)
    select_threshold = levels[int(select) - 1]

    # 计算mid_threshold (对应jet(0.5)的值)
    # jet(0.5)对应归一化后的0.5，需要反归一化回原始数据范围
    mid_threshold = min_val + factor * (max_val - min_val)

    # 创建修改后的矩阵副本
    modified_D = Final_D.copy()

    # 对小于select_threshold的值进行映射
    mask = Final_D < select_threshold
    if np.any(mask):
        # 将小于select_threshold的值从[min_val, select_threshold]映射到[min_val, mid_threshold]
        modified_D[mask] = min_val + (Final_D[mask] - min_val) * (mid_threshold - min_val) / (
                    select_threshold - min_val)

    # # 可视化原始矩阵和修改后的矩阵（可选）
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #
    # # 原始矩阵的等高线图
    # contour1 = axes[0].contourf(Final_D, levels, cmap='jet')
    # axes[0].set_title('原始矩阵')
    # fig.colorbar(contour1, ax=axes[0])
    #
    # # 修改后矩阵的等高线图
    # contour2 = axes[1].contourf(modified_D, levels, cmap='jet')
    # axes[1].set_title('修改后矩阵')
    # fig.colorbar(contour2, ax=axes[1])
    #
    # # 在图上标记阈值
    # axes[0].text(0.05, 0.95, f'select_threshold: {select_threshold:.2f}', transform=axes[0].transAxes,
    #              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    # axes[0].text(0.05, 0.85, f'mid_threshold: {mid_threshold:.2f}', transform=axes[0].transAxes,
    #              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    #
    # plt.tight_layout()
    # plt.show()

    return modified_D, select_threshold, mid_threshold
def normalize(Final_D):
    # 使用 MinMaxScaler 进行归一化到 [0,1] 区间
    scaler = MinMaxScaler(feature_range=(0, 1))
    Final_D_N = scaler.fit_transform(Final_D.reshape(-1, 1)).flatten()

    # 重新整形为原始维度
    return Final_D_N.reshape(Final_D.shape)


def normalize_data(Final_D):
    min_value = np.min(Final_D)
    max_value = np.max(Final_D)
    if max_value == min_value:
    # 防止除以零的情况
        return np.zeros_like(Final_D)
    return (Final_D - min_value) / (max_value - min_value)


def remap_values(Final_D, select_threshold, factor = 0.5):
    # 创建 Final_D 的副本，避免修改原始数据
    result = Final_D.copy()

    # 找出小于阈值的元素
    mask_below = Final_D < select_threshold
    values_below = Final_D[mask_below]

    if len(values_below) > 0:  # 确保有需要重映射的值
        # 将小于阈值的值重新映射到 0-0.5 范围
        # 使用 MinMax 归一化公式: (x - min) / (max - min) * (new_max - new_min) + new_min
        min_val = np.min(values_below)
        max_val = np.max(values_below)

        if max_val != min_val:  # 避免除以零
            result[mask_below] = (values_below - min_val) / (max_val - min_val) * factor
        else:
            result[mask_below] = factor/2  # 如果所有值相等，映射到factor的中点

    return result
def read_time(file_path, start_row, num_intervals=None, end_row=None):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 获取文件扩展名
        file_extension = os.path.splitext(file_path)[1].lower()

        def read_excel_time(file_path, start_row, num_intervals=None, end_row=None):
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                df = pd.read_excel(file_path, header=None)
                if len(df.columns) != 1:
                    raise ValueError("Error: The time file should have only one column.")

                if num_intervals is not None:
                    if not isinstance(num_intervals, int) or num_intervals <= 0 or start_row + num_intervals > df.shape[
                        0]:
                        raise ValueError(
                            "num_intervals must be a positive integer and within the range of the DataFrame.")
                    end_index = start_row - 1 + num_intervals
                elif end_row is not None:
                    if not isinstance(end_row, int) or end_row <= start_row or end_row > df.shape[0]:
                        raise ValueError(
                            "end_row must be an integer greater than start_row and within the range of the DataFrame.")
                    end_index = end_row - 1
                else:
                    end_index = None  # 读取到文件末尾

                matrix_data = df.iloc[start_row - 1:end_index, :].values.astype(np.float64).reshape(-1, 1)
                return matrix_data
            except Exception as e:
                print(f"An error occurred while reading the Excel file: {e}")
                raise

        def read_txt_time(file_path, start_row, num_intervals=None, end_row=None):
            """TXT文件读取方法"""
            try:
                data = np.loadtxt(file_path)

                if len(data.shape) > 1 and data.shape[1] != 1:
                    raise ValueError("Error: The time file should have only one column.")

                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)

                if num_intervals is not None:
                    if not isinstance(num_intervals, int) or num_intervals <= 0 or start_row + num_intervals > \
                            data.shape[0]:
                        raise ValueError("num_intervals must be a positive integer and within the range of the data.")
                    end_index = start_row - 1 + num_intervals
                elif end_row is not None:
                    if not isinstance(end_row, int) or end_row <= start_row or end_row > data.shape[0]:
                        raise ValueError(
                            "end_row must be an integer greater than start_row and within the range of the data.")
                    end_index = end_row - 1
                else:
                    end_index = None

                matrix_data = data[start_row - 1:end_index, :]
                return matrix_data.astype(np.float64)

            except Exception as e:
                print(f"An error occurred while reading the txt file: {e}")
                raise

        # 根据扩展名调用相应的读取方法
        if file_extension in ['.xlsx', '.xls']:
            print(f"Detected Excel file: {file_path}")
            return read_excel_time(file_path, start_row, num_intervals, end_row)

        elif file_extension == '.txt':
            print(f"Detected TXT file: {file_path}")
            return read_txt_time(file_path, start_row, num_intervals, end_row)

        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only .txt, .xlsx, and .xls are supported.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

class ThresholdNormalize(mcolors.Normalize):
    """
    Normalize data values based on a threshold.
    Values below threshold map linearly to [0, 0.5] of the colormap range.
    Values above threshold map linearly to [threshold_norm, 1.0] of the
    colormap range, where threshold_norm is the original normalized
    position of the threshold value.
    """
    def __init__(self, threshold, vmin=None, vmax=None, clip=False):
        self.threshold = threshold
        # Initialize Normalize with vmin, vmax, clip
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Ensure vmin and vmax are set (might be done by the plotting function)
        if self.vmin is None or self.vmax is None:
            # Attempt to autoscale if possible, otherwise raise error
             raise ValueError("vmin and vmax must be set before calling Normalize")
            # Alternatively, implement logic to get vmin/vmax from value if it's the full dataset

        # Handle scalar vs array input if necessary (though Normalize usually handles arrays)
        # Calculate original normalized position of the threshold
        # Avoid division by zero if vmin == vmax
        if self.vmax == self.vmin:
            threshold_norm = 0.5 # Or handle as error / special case
        else:
            # Clip threshold to data range before calculating norm
            threshold_clipped = np.clip(self.threshold, self.vmin, self.vmax)
            threshold_norm = (threshold_clipped - self.vmin) / (self.vmax - self.vmin)

        # Create masked arrays for safe calculations
        x = np.ma.masked_invalid(value)
        result = np.ma.zeros_like(x, dtype=float)

        # --- Apply the piecewise linear mapping ---

        # Mask for values below threshold
        mask_lt = x < self.threshold
        # Denominator for the lower range mapping
        denom_lt = self.threshold - self.vmin
        # Avoid division by zero if threshold is at vmin
        if denom_lt > 1e-9: # Using tolerance for float comparison
            result[mask_lt] = 0.0 + 0.5 * (x[mask_lt] - self.vmin) / denom_lt
        else:
            # If threshold == vmin, map values exactly at vmin to 0.0
            # (Technically no values are < vmin if clipping applied by caller)
             result[mask_lt & (x == self.vmin)] = 0.0


        # Mask for values at or above threshold
        mask_ge = x >= self.threshold
        # Denominator for the upper range mapping
        denom_ge = self.vmax - self.threshold
        # Avoid division by zero if threshold is at vmax
        if denom_ge > 1e-9: # Using tolerance
             result[mask_ge] = threshold_norm + (1.0 - threshold_norm) * (x[mask_ge] - self.threshold) / denom_ge
        else:
             # If threshold == vmax, map values exactly at vmax to 1.0
             result[mask_ge & (x == self.vmax)] = 1.0

        # Clip the result to [0, 1] range if required
        if self.clip or clip:
             # Use np.ma.clip for masked arrays
             result = np.ma.clip(result, 0, 1)

        # Fill masked values (e.g., NaNs) with a default (optional, often handled by plotting func)
        # result = result.filled(np.nan) # Or fill with 0?

        return result
class FrequencyCalculator:  # SIBIE网格
    def __init__(self, time, width, depth, d, cp, p, q, x0, y0, file_path,
                 start_row):  # 初始化参数包括：宽度、深度、单元格尺寸、波速、敲击点位置（p,q）、接收点位置(x0,y0)单位m
        if not isinstance(time, np.ndarray) or time.ndim != 2 or time.shape[1] != 1:
            raise ValueError("time must be a 2D numpy array with a single column.")
        self.signal = None
        self.time = time
        self.width = width
        self.depth = depth
        self.d = d
        self.cp = cp
        self.p = p
        self.q = q
        self.x0 = x0
        self.y0 = y0
        print(f'敲击点位置（{self.p},{self.q}）、接收点位置({self.x0},{self.y0})')
        self.file_path = file_path
        self.start_row = start_row
        self.step = self.d  # 列向量计算增量=单元格大小

        self.X = (self.width - self.d) / 2
        self.Y1 = self.d / 2
        self.Y2 = float(Decimal(str(self.depth)) - Decimal(
            str(self.d)))  # python中二进制计算十进制的数会产生浮点数精度问题，比如0.65-0.0025=0.647500000000001，导致后面L2维度不正确，需要decimal控制精度,再转换为浮点数类型
        self.L1 = np.arange(-self.X, self.X + self.step, self.step)  # arange（）不包含结束值，因此要加step
        self.file_name = ''
        global L1
        L1 = self.L1
        self.L2 = np.arange(self.Y1, self.Y2 + self.step, self.step)
        global L2
        L2 = self.L2
        self.Z = np.zeros((len(self.L2), len(self.L1)))
        self.Final_D = np.zeros((len(self.L2), len(self.L1)))

    def read_column(self, num_intervals=None, end_row=None):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            # 获取文件扩展名
            file_extension = os.path.splitext(self.file_path)[1].lower()

            def read_txt_column(file_path, start_row, num_intervals=None, end_row=None):
                """TXT文件读取方法"""
                try:
                    data = np.loadtxt(file_path)

                    if num_intervals is not None:
                        if not isinstance(num_intervals, int) or num_intervals <= 0 or start_row + num_intervals > \
                                data.shape[0]:
                            raise ValueError(
                                "num_intervals must be a positive integer and within the range of the data.")
                        end_index = start_row - 1 + num_intervals
                    elif end_row is not None:
                        if not isinstance(end_row, int) or end_row <= start_row or end_row > data.shape[0]:
                            raise ValueError(
                                "end_row must be an integer greater than start_row and within the range of the data.")
                        end_index = end_row - 1
                    else:
                        end_index = None

                    return data[start_row - 1:end_index, :].astype(np.float64)
                except Exception as e:
                    print("Error reading Excel file:", str(e))

            def read_excel_column(file_path, start_row, num_intervals=None, end_row=None):
                # 读取Excel文件
                try:
                    df = pd.read_excel(file_path, header=None)

                    # 确定要读取的行范围
                    if num_intervals is not None:
                        if not isinstance(num_intervals, int) or num_intervals <= 0 or start_row + num_intervals > \
                                df.shape[0]:
                            raise ValueError("num_intervals must be a positive integer.")
                        end_index = start_row - 1 + num_intervals
                    elif end_row is not None:
                        if not isinstance(end_row, int) or end_row <= start_row or end_row > df.shape[0]:
                            raise ValueError("end_row must be an integer greater than start_row.")
                        end_index = end_row - 1
                    else:
                        end_index = None  # 读取到文件末尾

                    return df.iloc[start_row - 1:end_index, :].values.astype(np.float64).reshape(-1, 1)

                except Exception as e:
                    print("Error reading Excel file:", str(e))


            if file_extension in ['.xlsx', '.xls']:
                print(f"Detected Excel file: {self.file_path}")
                self.signal = read_excel_column(self.file_path, start_row, num_intervals, end_row)

            elif file_extension == '.txt':
                print(f"Detected TXT file: {self.file_path}")
                self.signal = read_txt_column(self.file_path, start_row, num_intervals, end_row)

            else:
                raise ValueError(
                    f"Unsupported file format: {file_extension}. Only .txt, .xlsx, and .xls are supported.")

            self.file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            global title_list, title_Stacking_method
            title_list = re.split(r'_', self.file_name)
            title_Stacking_method = title_Stacking_method + title_list[2]


        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            raise

    def grid_frequency(self, FFT=False):
        """
        SIBIE网格和计算频率，返回输入信号对应的矩阵self.Final_D
        """
        ###计算网格理论频率###
        m = 0
        n = 0
        for b in self.L2:
            for a in self.L1:
                c1 = (a - self.p) ** 2 + (b - self.q) ** 2
                c2 = (a - self.x0) ** 2 + (b - self.y0) ** 2
                R = np.sqrt(c1) + np.sqrt(c2)  # 计算直线距离
                F = self.cp / (R/1000)  # 单位Hz，如果使用KHz需要注意下面插值f_interp = interpolate.interp1d(f, P1, kind='linear', )中f也要变为KHz（默认Hz）
                self.Z[n, m] = F
                m += 1
            m = 0
            n += 1
        # 已经得到理论频率矩阵self.Z

        n = self.signal.shape[1]
        # 循环处理每一列信号
        try:
            for i in range(n):  # 循环n次，Python的索引从0开始，因此若
                y = self.signal[:, i]
                print(y.shape)
                L = len(self.time)
                T = (self.time[-1, 0] - self.time[0, 0]) / (L - 1)/1000# 默认单位s，有限元不用除1000，冲击回拨仪单位ms需要除1000
                print(f"T = {T} s")
                Fs = 1 / T
                f = Fs / L * np.arange(L // 2 + 1)  # Fs/N得到采样频率，np.arange(N/2+1)得到从0到N/2+1的数组，即从初始0至最高频率——奈奎斯特频率
                S = y[:L]
                Y = np.fft.fft(S)
                P2 = np.abs(Y / L)
                P1 = P2[:L // 2 + 1]
                P1[1:-1] = 2 * P1[1:-1]
                print(self.file_name + '第' + str(i + 1) + '列信号')
                print(f"Frequency range: {f[0]:.2f} - {f[-1]:.2f} Hz")
                max_freq_index = np.argmax(P1)
                print(f"Frequency with max amplitude: {f[max_freq_index]:.2f} Hz")
                P1 = P1 / np.max(np.abs(P1))

                # 傅里叶变换成像
                if FFT == True:
                    plt.plot(f, P1)

                    peaks, _ = find_peaks(P1)
                    # 获取前3个最大值的索引
                    top_3_indices = peaks[np.argsort(P1[peaks])[-3:]]
                    top_3_frequencies = f[top_3_indices]
                    top_3_amplitudes = P1[top_3_indices]
                    # 在图上标记这些点
                    plt.plot(top_3_frequencies, top_3_amplitudes, 'ro', markersize=4)

                    # 添加标记文本，使用小字体和淡蓝色
                    for freq, amp in zip(top_3_frequencies, top_3_amplitudes):
                        plt.annotate(f'({freq:.2f}, {amp:.2f})',
                                     xy=(freq, amp),
                                     xytext=(5, 5),  # 减小文本偏移量
                                     textcoords='offset points',
                                     fontsize=8,  # 设置小字体
                                     color='lightblue',  # 设置淡蓝色
                                     ha='left',
                                     va='bottom')
                    plt.show()


                ###
                f_interp = interpolate.interp1d(f, P1, kind='linear', bounds_error=False,
                                                fill_value=0)  # 创建的插值函数但超出范围时，用0填充
                '''注意：若插值矩阵为零矩阵，请检查T采样率单位'''
                Density = f_interp(self.Z)  # 此时Density是二维矩阵不含NaN
                C = Density.copy()

                # 使用 MinMaxScaler 进行归一化到 [0,1] 区间
                scaler = MinMaxScaler(feature_range=(0, 1))
                C2 = scaler.fit_transform(Density.reshape(-1, 1)).flatten()

                # 重新整形为原始维度
                Density = C2.reshape(C.shape)
                # plot_SIBIE(Density, self.L1, self.L2, condition, flow, title_Stacking_method)
                self.Final_D = self.Final_D + Density
            return self.Final_D
        except Exception:
            print('请检查变量n')

def defect_plot(defect):
    if defect == 'r-1':

        # 定义三个点的坐标
        p1 = (-25, 190)  # 点1
        p2 = (25, 190)  # 点2
        p3 = (25, 170)  # 点3
        p4 = (-25, 170)  # 点4

        # 创建三条虚线
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p4[0]], [p3[1], p4[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line4 = Line2D([p4[0], p1[0]], [p4[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        return [line1, line2, line3, line4]

    if defect == 'r-2':
        p1 = (-50, 190)  # 点1
        p2 = (50, 190)  # 点2
        p3 = (50, 150)  # 点3
        p4 = (-50, 150)  # 点4
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p4[0]], [p3[1], p4[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line4 = Line2D([p4[0], p1[0]], [p4[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        return [line1, line2, line3, line4]

    if defect == 'r-3':
        # 定义三个点的坐标
        p1 = (-75, 190)  # 点1
        p2 = (-25, 190)  # 点2
        p3 = (-25, 170)  # 点3
        p4 = (-75, 170)  # 点4
        # 创建三条虚线
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p4[0]], [p3[1], p4[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line4 = Line2D([p4[0], p1[0]], [p4[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        p1 = (25, 190)  # 点1
        p2 = (75, 190)  # 点2
        p3 = (75, 170)  # 点3
        p4 = (25, 170)  # 点4
        # 创建三条虚线
        line5 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line6 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line7 = Line2D([p3[0], p4[0]], [p3[1], p4[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line8 = Line2D([p4[0], p1[0]], [p4[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        return [line1, line2, line3, line4, line5, line6, line7, line8]

    if defect == 'x-1':
        # 定义三个点的坐标
        p1 = (-25, 190)  # 点1
        p2 = (25, 190)  # 点2
        p3 = (25, 170)  # 点3

        # 创建三条虚线
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p1[0]], [p3[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)

        return [line1, line2, line3]

    if defect == 'x-3':

        # 定义三个点的坐标
        p1 = (-75, 190)  # 点1
        p2 = (-25, 190)  # 点2
        p3 = (-25, 170)  # 点3
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p1[0]], [p3[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        p1 = (25, 190)  # 点1
        p2 = (75, 190)  # 点2
        p3 = (75, 170)  # 点3

        # 创建三条虚线
        line4 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line5 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line6 = Line2D([p3[0], p1[0]], [p3[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)

        return [line1, line2, line3, line4, line5, line6]

    if defect == 'x-2':
        p1 = (-50, 190)  # 点1
        p2 = (50, 190)  # 点2
        p3 = (50, 150)  # 点3
        line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line2 = Line2D([p2[0], p3[0]], [p2[1], p3[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        line3 = Line2D([p3[0], p1[0]], [p3[1], p1[1]],
                       linestyle='--', color='black', alpha=0.4, linewidth=2)
        return [line1, line2, line3]

def _evaluate_imaging_accuracy(contour, Lc, S, D_true, W_true, ax):
    """
    内部函数：评估SIBIE成像精度并选择最优阈值，同时在图上可视化评估过程
    
    参数:
    contour: matplotlib轮廓对象
    Lc: 轮廓层级数
    S: 缩放因子集合
    D_true: 缺陷的真实深度（单位：mm）
    W_true: 缺陷的真实宽度（单位：mm）
    ax: matplotlib轴对象，用于绘图
    
    返回:
    Tc: 最优阈值
    E_D_det: 检测深度误差
    E_W_det: 检测宽度误差
    D_est: 估计深度
    W_est: 估计宽度
    """
    # 生成候选阈值列表
    T = [int(np.floor(s * Lc)) for s in S]
    T = [t for t in T if t > 0]  # 确保阈值大于0
    
    # 存储评估结果
    results = []
    
    # 获取轮廓线集合
    contour_paths = contour.collections
    
    # 创建图例句柄列表
    legend_handles = []
    
    # 评估每个阈值
    for Tk in T:
        if Tk >= len(contour_paths):
            continue  # 跳过超出范围的阈值
        
        # 获取对应阈值的轮廓线路径
        path_collection = contour_paths[Tk]
        
        # 提取所有路径中的坐标点
        all_points_x = []
        all_points_y = []
        
        for path in path_collection.get_paths():
            vertices = path.vertices
            # 只考虑y坐标小于等于240的点
            valid_indices = vertices[:, 1] <= 220
            all_points_x.extend(vertices[valid_indices, 0])
            all_points_y.extend(vertices[valid_indices, 1])
        
        # 如果没有点，跳过此阈值
        if not all_points_x or not all_points_y:
            continue
        
        # 计算坐标的最大值和最小值
        x_max_k = max(all_points_x)
        x_min_k = min(all_points_x)
        y_max_k = max(all_points_y)
        y_min_k = min(all_points_y)
        
        # 计算估计深度和宽度
        Dk = (y_max_k + y_min_k) / 2  # 深度取中点
        Wk = x_max_k - x_min_k  # 宽度
        
        # 计算相对误差
        E_Dk = abs((Dk - D_true) / D_true)
        E_Wk = abs((Wk - W_true) / W_true)
        
        # 存储结果
        results.append({
            'Tk': Tk,
            'Dk': Dk,
            'Wk': Wk,
            'E_Dk': E_Dk,
            'E_Wk': E_Wk,
            'combined_error': E_Dk + E_Wk,  # 综合误差
            'x_max_k': x_max_k,
            'x_min_k': x_min_k,
            'y_max_k': y_max_k,
            'y_min_k': y_min_k
        })
    
    # 如果没有有效结果，返回默认值
    if not results:
        return 1, 1.0, 1.0, 0.0, 0.0
    
    # 按综合误差排序
    results.sort(key=lambda x: x['combined_error'])
    
    # 选择综合误差最小的阈值作为最优阈值
    best_result = results[0]
    Tc = best_result['Tk']
    E_D_det = best_result['E_Dk']
    E_W_det = best_result['E_Wk']
    D_est = best_result['Dk']
    W_est = best_result['Wk']
    
    # 在图上可视化评估结果
    for i, result in enumerate(results):
        Tk = result['Tk']
        
        # 设置透明度，最优结果用不透明白色，其他用半透明
        alpha = 1.0 if Tk == Tc else 0.5
        color = 'white' if Tk == Tc else plt.cm.tab10(i % 10)
        
        # 绘制轮廓线上的最大值和最小值点
        ax.plot(result['x_max_k'], result['y_max_k'], 'o', color=color, alpha=alpha, markersize=5)
        ax.plot(result['x_min_k'], result['y_min_k'], 'o', color=color, alpha=alpha, markersize=5)
        
        # 标注最大值和最小值坐标
        ax.annotate(f'({result["x_max_k"]:.2f}, {result["y_max_k"]:.2f})', 
                   (result['x_max_k'], result['y_max_k']), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, color=color, alpha=alpha)
        ax.annotate(f'({result["x_min_k"]:.2f}, {result["y_min_k"]:.2f})', 
                   (result['x_min_k'], result['y_min_k']), 
                   xytext=(5, -10), textcoords='offset points', 
                   fontsize=8, color=color, alpha=alpha)
        
        # 绘制估计深度和宽度
        # 深度线
        ax.plot([result['x_min_k'], result['x_max_k']], 
                [result['Dk'], result['Dk']], 
                '-', color=color, alpha=alpha, linewidth=2)
        # 宽度线
        ax.plot([result['x_min_k'], result['x_min_k']], 
                [result['y_min_k'], result['y_max_k']], 
                '-', color=color, alpha=alpha, linewidth=2)
        ax.plot([result['x_max_k'], result['x_max_k']], 
                [result['y_min_k'], result['y_max_k']], 
                '-', color=color, alpha=alpha, linewidth=2)
        
        # 标注估计深度和宽度
        ax.annotate(f'Dk={result["Dk"]:.4f}m', 
                   ((result['x_min_k'] + result['x_max_k'])/2, result['Dk']), 
                   xytext=(0, 10), textcoords='offset points', 
                   fontsize=8, color=color, alpha=alpha, ha='center')
        ax.annotate(f'Wk={result["Wk"]:.4f}m', 
                   ((result['x_min_k'] + result['x_max_k'])/2, (result['y_min_k'] + result['y_max_k'])/2), 
                   xytext=(0, 0), textcoords='offset points', 
                   fontsize=8, color=color, alpha=alpha, ha='center')
        
        # 添加图例
        legend_handles.append(Line2D([0], [0], color=color, marker='o', linestyle='-', 
                                    label=f'Tk={Tk}, E_Dk={result["E_Dk"]:.2%}, E_Wk={result["E_Wk"]:.2%}',
                                    alpha=alpha))
    
    # 添加图例
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)
    
    return Tc, E_D_det, E_W_det, D_est, W_est

def plot_SIBIE(Final_D, L1, L2, condition=None, flow=None, title_Stacking_method=None, level=25, 
                save=False, select=None, factor=0.3, defect=None, bar=False, evaluate=False, 
                Lc=None, S=None, D_true=None, W_true=None):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形和网格
    fig = plt.figure(figsize=(6, 10))

    # 调整子图之间的间距
    plt.subplots_adjust(right=0.85)

    # 绘制等高线图
    if select == None:
        #     #正常绘图不用过滤
        ax = plt.subplot(111)
        contour=ax.contourf(L1, L2, Final_D,
                          levels=level,
                          antialiased=True,
                          cmap='jet')


        print(contour.levels)
        print("Min:", np.min(Final_D), "Max:", np.max(Final_D))
        # plt.colorbar(contour)
    elif not 0 <= select <= level:
        raise ValueError(f"'select' must be between 0 and {level}, inclusive. Got {select}")
    else:
        Final_D, select_threshold, mid_threshold = process_matrix(Final_D, level, select=select,factor = factor)
        ax = plt.subplot(111)
        contour=ax.contourf(L1, L2, Final_D,
                          levels=level,
                          antialiased=True,
                          cmap='jet')



    # 设置坐标轴
    #统一设置标签等样式
    ax.xaxis.tick_top()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Width/mm', fontsize=18)
    ax.set_ylabel('Depth/mm', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)

    # 评估成像精度
    if evaluate:
        if D_true is None or W_true is None or S is None:
            raise ValueError("评估模式需要提供D_true、W_true和S参数")
        
        # 如果未指定Lc，则使用level
        if Lc is None:
            Lc = level
        
        # 调用内部评估函数
        Tc, E_D_det, E_W_det, D_est, W_est = _evaluate_imaging_accuracy(contour, Lc, S, D_true, W_true, ax)
        
        # 在图上显示评估结果
        info_text = f"Tc: {Tc}\nE_Dk: {E_D_det:.2%}\nE_Wk: {E_W_det:.2%}\nDk: {D_est:.4f}mm\nWk: {W_est:.4f}mm"
        plt.figtext(0.02, 0.02, info_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # 打印评估结果
        print(f"最优阈值Tc: {Tc}")
        print(f"深度误差: {E_D_det:.2%}")
        print(f"宽度误差: {E_W_det:.2%}")
        print(f"估计深度: {D_est:.4f}mm")
        print(f"估计宽度: {W_est:.4f}mm")

    # 获取主图的位置信息
    pos = ax.get_position()

    if bar == True:
        # 创建颜色条，并设置其位置与主图等高
        cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.03, pos.height])
        plt.colorbar(contour, cax=cbar_ax)

    # 设置标题
    title = []
    if condition is not None:
        title.append(str(condition))
    if flow is not None:
        title.append(str(flow))
    if title_Stacking_method is not None:
        title.append(str(title_Stacking_method))
    title = "   ".join(title)

    ax.text(0.5, -0.1, title,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=30)

    # 保存点和标注的列表
    points = []
    annotations = []



    def onclick(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata

            for point in points:
                point.remove()
            for annotation in annotations:
                annotation.remove()
            points.clear()
            annotations.clear()

            point = ax.plot(x, y, 'ro', markersize=5)[0]
            points.append(point)

            annotation = ax.annotate(
                f'({x:.3f}, {y:.3f})',
                xy=(x, y),
                xytext=(20, 20),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->')
            )
            annotations.append(annotation)

            plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    if defect != None:
        if defect == 'c-3':
            # 画圆
            circle1 = plt.Circle((-50, 165), 25, fill=False, linestyle='--', color='black', alpha=0.4, linewidth=2)
            ax.add_patch(circle1)
            circle2 = plt.Circle((50, 165), 25, fill=False, linestyle='--', color='black', alpha=0.4, linewidth=2)
            ax.add_patch(circle2)
        elif defect == 'c-1':
            circle1 = plt.Circle((0, 165), 25, fill=False, linestyle='--', color='black', alpha=0.4, linewidth=2)
            ax.add_patch(circle1)
        elif defect == 'c-2':
            circle1 = plt.Circle((0, 140), 50, fill=False, linestyle='--', color='black', alpha=0.4, linewidth=2)
            ax.add_patch(circle1)
        else:
            for i in defect_plot(defect):
                ax.add_line(i)
    if save:
        plt.savefig(r"D:\研究生\小论文\SIBIE成像\标准试件矩形1阶.tif",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    format='tiff')

    plt.show()



def recycle_stake(Final_D, location, spacing, FFT=False):
    time = read_time(file_path_time, start_row, num_intervals, end_row)
    i = 0
    for file_name in file_names:
        full_path = os.path.join(base_path, file_name)

        signal_process = FrequencyCalculator(time, width, depth, d, cp,
                                             float(Decimal(str(location)) - Decimal(str(i * spacing))), 0,
                                             float(Decimal(str(location)) - Decimal(str((i + 1) * spacing))), 0,
                                             full_path, start_row)
        signal_process.read_column(num_intervals, end_row)
        i += 1
        final_D = signal_process.grid_frequency(FFT)
        if i == 1:
            Final_D = final_D
        else:
            Final_D = Final_D + final_D
    return Final_D


def free_stack(Final_D, impact_points, receiver_points, FFT=False):
    try:
        if len(impact_points) == len(receiver_points) == len(file_names):
            time = read_time(file_path_time, start_row, num_intervals, end_row)
            i = 0
            for file_name, impact_point, receiver_point in zip(file_names, impact_points, receiver_points):
                full_path = os.path.join(base_path, file_name)
                signal_process = FrequencyCalculator(time, width, depth, d, cp, impact_point, 0, receiver_point, 0,
                                                     full_path, start_row)
                signal_process.read_column(num_intervals, end_row)
                i += 1
                final_D = signal_process.grid_frequency(FFT)
                if i == 1:
                    Final_D = final_D
                else:
                    Final_D = Final_D + final_D
            return Final_D

    except:
        raise ValueError("impact_point、receiver_point和file_names长度不一致")


def same_file_stack_recycle(Final_D, file_names, timeANDsignals, location, spacing, FFT=False):
    def split_range(s):
        # 辅助函数：解析类似 "2-4" 的字符串为列表 [2,3,4]
        if '-' not in str(s):
            return [int(s)]
        start, end = map(int, s.split('-'))
        return list(range(start, end + 1))

    all_columns = []
    for item in timeANDsignals:
        all_columns.extend(split_range(item))

    df = pd.read_excel(file_path_time, header=None, dtype='float64')

    # 检查是否超出 DataFrame 的列范围
    max_col = max(all_columns)
    if max_col > df.shape[1]:
        raise ValueError(f"需要的列数 {max_col} 超出了DataFrame的实际列数 {df.shape[1]}")
    ###判断有效性###
    i = 0

    if end_row is not None:
        time = df.iloc[(start_row - 1):end_row,0].to_numpy().reshape(-1, 1)
    elif num_intervals is not None and (start_row - 1 + num_intervals) < df.shape[0]:
        time =  df.iloc[(start_row - 1):(start_row - 1 + num_intervals),0].to_numpy().reshape(-1, 1)
    else:
        time = df.iloc[(start_row - 1):, 0].to_numpy().reshape(-1, 1)

    for timeANDsignal in timeANDsignals[1:]:

        if timeANDsignal != '1':
            cols = split_range(timeANDsignal)

            signal_process = FrequencyCalculator(time, width, depth,
                                                 d, cp, float(Decimal(str(location)) - Decimal(str(i * spacing))), 0,
                                                 float(Decimal(str(location)) - Decimal(str((i + 1) * spacing))), 0,
                                                 file_names, start_row)
            if end_row is not None:
                signal_process.signal = df.iloc[(start_row - 1):end_row, [x - 1 for x in cols]].to_numpy()
            elif num_intervals is not None and (start_row - 1 + num_intervals) < df.shape[0]:
                signal_process.signal = df.iloc[(start_row - 1):(start_row - 1 + num_intervals),[x - 1 for x in cols]].to_numpy()
            else:
                signal_process.signal = df.iloc[(start_row - 1):, [x - 1 for x in cols]].to_numpy()
            final_D = signal_process.grid_frequency(FFT)
            # plot_SIBIE(final_D,L1, L2, condition, flow, start_row, level=25, save=False)
            i += 1
            if i == 1:
                Final_D = final_D
            else:
                Final_D = Final_D + final_D

    return Final_D


def same_file_stack_free(Final_D, timeANDsignals, impact_points, receiver_points, file_names, FFT=False):
    def split_range(s):
        # 辅助函数：解析类似 "2-4" 的字符串为列表 [2,3,4]
        if '-' not in str(s):
            return [int(s)]
        start, end = map(int, s.split('-'))
        return list(range(start, end + 1))

    all_columns = []
    for item in timeANDsignals:
        all_columns.extend(split_range(item))

    df = pd.read_excel(file_path_time, header=None, dtype='float64')

    # 检查是否超出 DataFrame 的列范围
    max_col = max(all_columns)
    if max_col > df.shape[1]:
        raise ValueError(f"需要的列数 {max_col} 超出了DataFrame的实际列数 {df.shape[1]}")
    ###判断有效性###
    i = 0

    for timeANDsignal, impact_point, receiver_point in zip(timeANDsignals, impact_points, receiver_points):
        if timeANDsignal != '1':
            try:
                if len(impact_points) == len(receiver_points) == len(timeANDsignals):
                    cols = split_range(timeANDsignal)
                    signal_process = FrequencyCalculator(df.iloc[(start_row - 1):, 0].to_numpy().reshape(-1, 1), width,
                                                         depth, d, cp, impact_point, 0, receiver_point, 0,
                                                         file_names, start_row)
                    a = [x - 1 for x in cols]
                    signal_process.signal = df.iloc[:, [x - 1 for x in cols]].to_numpy()
                    final_D = signal_process.grid_frequency(FFT)
                    i += 1
                    if i == 1:
                        Final_D = final_D
                    else:
                        Final_D = Final_D + final_D
                return Final_D
            except:
                raise ValueError("impact_point、receiver_point和timeANDsignals长度不一致")


# 创建全局变量
condition = '(h)EA23'
flow = '5'
title_Stacking_method = '163-16个数据'
width = 200#mm
depth = 300#mm225
d = 5#mm
cp =4100#m/s3727
# start_row = 120
num_intervals = None
end_row = None

Final_D: Optional[NDArray] = None
L1 = None
L2 = None

title_list = []

# same_file_stack_free和free_stack会读取
impact_points = [0.1, 0.075, 0.05, 0.025, 0, -0.025, -0.05, -0.075]
receiver_points = [0.075, 0.05, 0.025, 0, -0.025, -0.05, -0.075, -0.1]

# same_file_stack_free会读取
"""所有数据写入统一文件中，每列数据当作一个位置的采集信号"""
timeANDsignals = ['1','5-7','8-10','8-10', '11-13','6-9','5-7'] #'5-7','20-22','11-13','5-7','20-22','11-13','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','2-4','2-4','2-4','5-8','9-13','9-13','9-13'
# timeANDsignals = ['1'] + [f'{i}-{i+2}' for i in range(2, 20, 4)]

# 所有情况都会读取
file_path_time = r'D:\研究生\褚金超\2024毕业材料备份\数据处理\A23-15cm矩形\A23.xlsx'  # D:\研究生\小论文\有限元模拟\32.5\矩形2_32.5cm_time.xlsx

# same_file_stack_free和same_file_stack_recycle不会读取
base_path = r'D:\研究生\小论文\有限元模拟\标准试块'  # D:\研究生\小论文\有限元模拟\32.5
"""多文件叠加方法，命名规则:工况_缺陷实际深度_信号方法（A_B_C形式）"""
file_names = ['矩形_30_敲1采3.xlsx', '矩形_30_敲1采3.xlsx', '矩形_30_敲3采5.xlsx', '矩形_30_敲3采5.xlsx',
              '矩形_30_敲3采5.xlsx', '矩形_30_敲3采5.xlsx', '矩形_30_敲1采3.xlsx',
              '矩形_30_敲1采3.xlsx']  # ,'A12_14_2.xlsx','A12_13_3.xlsx','A12_14_2.xlsx','A12_15_1.xlsx'

for start_row in range(70, 600, 1):

    if __name__ == '__main__':
        """计算成像矩阵"""
        Final_D = same_file_stack_recycle(Final_D, file_names, timeANDsignals, location=-60, spacing=-20, FFT=False)
        # Final_D = normalize_data(Final_D) 单独归一化Final_D，测试发现归一化后会造成成像精度下降
        """初步成像，图像名称包含过滤信号数目，不包含精度评估"""
        # plot_SIBIE(Final_D, L1, L2, condition=condition, flow=flow, title_Stacking_method=start_row, level=25, save=True, bar = True,defect='r-1')#,select=15,defect='c-1',默认factor=0.3,bar = True
        # plot_SIBIE(Final_D, L1, L2, condition=condition, level=25, save=True, bar = True,defect='r-2')
        """系统成像，图像名称包含过滤信号数目，包含精度评估"""
        # plot_SIBIE(Final_D, L1, L2, condition=condition, flow=flow, title_Stacking_method=start_row,defect='x-1',
        # level=25, save=True, bar = True, evaluate=True, S=[0.6,0.65,0.7,0.75,0.8, 0.85, 0.9, 0.95, 1],D_true=170,W_true=50)
        """最终成像，图像名称自定义，包含精度评估"""
        plot_SIBIE(Final_D, L1, L2, condition=condition,defect='r-2',level=25, save=True, bar = True, 
        evaluate=True, S=[0.8, 0.85, 0.9, 0.95, 1],D_true=150,W_true=100)

        # Final_D = same_file_stack_free(Final_D,timeANDsignals,impact_points,receiver_points,file_names,FFT = False)
        # plot_SIBIE(Final_D, L1, L2, condition=condition,  flow=flow, title_Stacking_method=start_row,level=20, save=True)

        # Final_D = recycle_stake(Final_D,location = 0.1,spacing = 0.025, FFT = False)#location最远敲击点m,spacing敲击点和采集点距离m
        # plot_SIBIE(Final_D, L1, L2, condition, flow, title_Stacking_method)

        # Final_D = free_stack(Final_D,impact_points,receiver_points,FFT = False)
        # plot_SIBIE(Final_D, L1, L2, condition, flow, title_Stacking_method)








