import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import sem

# 读取数据
file_path = 'Breast-Rad.xlsx'  # 请替换为你的文件路径
data = pd.read_excel(file_path)

# 提取设备类型和数据类型（假设设备类型在Name列中并以F、S、X区分）
data['Device'] = data['Name'].str.extract(r'([FSX])')
data['DataType'] = data['Name'].str.extract(r'([LT])')

features = data.columns[4:]

# 初始化结果字典
icc_results = {}

# 计算ICC函数
def calculate_icc(df, feature):
    # 数据准备
    df = df[['Device', feature]].dropna()
    
    # 确保特征列是数值类型
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # 检查特征值是否全为NaN或为空
    if df[feature].isna().all():
        raise ValueError(f"All values for feature {feature} are NaN")
    
    if df[feature].empty:
        raise ValueError(f"No valid values for feature {feature}")
    
    # 数据格式转换
    melted = df.melt(id_vars=['Device'], var_name='Feature', value_name='Value')
    melted['Device'] = melted['Device'].astype('category')
    
    # 检查是否有足够的设备种类
    if melted['Device'].nunique() < 2:
        raise ValueError("Not enough device categories to calculate ICC")
    
    # 混合效应模型计算ICC
    model = smf.mixedlm("Value ~ 1", data=melted, groups=melted["Device"])
    result = model.fit()
    
    # 提取方差分量
    vc = result.cov_re
    if vc.size == 0:
        raise ValueError(f"Variance components for feature {feature} is empty")
    
    var_device = vc.values[0][0]
    var_residual = result.scale
    
    # 计算ICC
    icc = var_device / (var_device + var_residual)
    
    # 计算95% CI
    ci_low = icc - 1.96 * sem([var_device, var_residual])
    ci_high = icc + 1.96 * sem([var_device, var_residual])
    
    return icc, (ci_low, ci_high)

# 过滤出三个设备都有的数据
common_data_L = data[data['Name'].str.contains('_L')]
common_data_L = common_data_L[common_data_L.groupby('Name')['Device'].transform('nunique') == 3]

common_data_T = data[data['Name'].str.contains('_T')]
common_data_T = common_data_T[common_data_T.groupby('Name')['Device'].transform('nunique') == 3]

# 进一步过滤有效的特征列
valid_features_L = [feature for feature in features if not common_data_L[feature].isna().all()]
valid_features_T = [feature for feature in features if not common_data_T[feature].isna().all()]

# 打印有效特征列表以调试
print("Valid features for L data:", valid_features_L)
print("Valid features for T data:", valid_features_T)

# 逐个特征计算ICC
for feature in valid_features_L:
    try:
        icc, ci = calculate_icc(common_data_L, feature)
        icc_results[f'{feature}_L'] = {'ICC': icc, '95% CI': ci}
    except Exception as e:
        icc_results[f'{feature}_L'] = {'ICC': None, '95% CI': None, 'Error': str(e)}

for feature in valid_features_T:
    try:
        icc, ci = calculate_icc(common_data_T, feature)
        icc_results[f'{feature}_T'] = {'ICC': icc, '95% CI': ci}
    except Exception as e:
        icc_results[f'{feature}_T'] = {'ICC': None, '95% CI': None, 'Error': str(e)}

# 显示结果
icc_df = pd.DataFrame(icc_results).T
print(icc_df.head())  # 打印输出头部数据以查看

# 保存结果到Excel
icc_df.to_excel('ICC_Results_filtered.xlsx')
