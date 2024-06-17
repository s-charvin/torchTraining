import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
# Define the data
data = {
    'Experiment Name': [
        '音频基线', '音频基线', 
        '视频基线', '视频基线', 
        'BER_ISF', 'BER_ISF', 
        'BER_SISF', 'BER_SISF'
    ],
    'Task Type': [
        '音频', '音频', 
        '视频', '视频', 
        '音视频联合', '音视频联合', 
        '音视频联合', '音视频联合'
    ],
    'WA (%)': [
        63.02, 47.64, 
        74.67, 61.29, 
        77.86, 63.49,
        81.05, 65.69
    ],
    'UA (%)': [
        64.76, 45.36, 
        74.45, 60.95, 
        78.27, 62.05, 
        81.11, 65.46
    ],
    'ACC (%)': [
        63.89, 46.50, 
        74.56, 61.12, 
        78.06, 62.77, 
        81.08, 65.58
    ],
    'Macro F1 (%)': [
        63.68, 44.54, 
        74.43, 60.72, 
        78.10, 62.27, 
        81.23, 65.22
    ],
    'Weighted F1 (%)': [
        62.65, 46.55, 
        74.68, 61.24, 
        77.83, 63.24, 
        81.05, 65.85
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Separate data for four and six categories
four_cat_df = df.iloc[[0, 2, 4, 6]]
six_cat_df = df.iloc[[1, 3, 5, 7]]

# Plot settings
fig, axes = plt.subplots(2, 1, figsize=(9, 12), dpi=600)

# Color settings for plots
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']

# Plot for four categories
four_cat_df.plot(kind='bar', ax=axes[0], x='Experiment Name', y=['WA (%)', 'UA (%)', 'ACC (%)', 'Macro F1 (%)', 'Weighted F1 (%)'], color=colors)
axes[0].set_title('四分类消融实验')
axes[0].set_ylabel('准确率 (%)')
axes[0].tick_params(axis='x', labelrotation=0)
axes[0].set_ylim(0, 100)  # Adjust Y-axis scale
axes[0].grid(True)
axes[0].legend(loc='upper left')
axes[0].set_xlabel('')

# Plot for six categories
six_cat_df.plot(kind='bar', ax=axes[1], x='Experiment Name', y=['WA (%)', 'UA (%)', 'ACC (%)', 'Macro F1 (%)', 'Weighted F1 (%)'], color=colors)
axes[1].set_title('六分类消融实验')
axes[1].set_ylabel('准确率 (%)')
axes[1].tick_params(axis='x', labelrotation=0)
axes[1].set_ylim(0, 100)  # Adjust Y-axis scale
axes[1].grid(True)
axes[1].legend(loc='upper left')
axes[1].set_xlabel('')
# Adjust layout
plt.tight_layout()

plt.savefig('./test.svg')
plt.savefig('./test.png')