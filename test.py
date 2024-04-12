import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
# Define the data
data = {
    'Experiment Name': [
        '四分类音频基线', '六分类音频基线', 
        '四分类视频基线', '六分类视频基线', 
        '四分类 BER_ISF', '六分类 BER_ISF', 
        '四分类 BER_SISF', '六分类 BER_SISF'
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
        81.31, 58.23, 
        81.05, 65.69
    ],
    'UA (%)': [
        64.76, 45.36, 
        74.45, 60.95, 
        81.94, 56.44, 
        81.11, 65.46
    ],
    'ACC (%)': [
        63.89, 46.50, 
        74.56, 61.12, 
        81.63, 57.34, 
        81.08, 65.58
    ],
    'Macro F1 (%)': [
        63.68, 44.54, 
        74.43, 60.72, 
        81.57, 56.88, 
        81.23, 65.22
    ],
    'Weighted F1 (%)': [
        62.65, 46.55, 
        74.68, 61.24, 
        81.30, 58.38, 
        81.05, 65.85
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Separate data for four and six categories
four_cat_df = df[df['Experiment Name'].str.contains('四分类')]
six_cat_df = df[df['Experiment Name'].str.contains('六分类')]

# Plot settings
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=600)

# Color settings for plots
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']

# Plot for four categories
four_cat_df.plot(kind='bar', ax=axes[0], x='Experiment Name', y=['WA (%)', 'UA (%)', 'ACC (%)', 'Macro F1 (%)', 'Weighted F1 (%)'], color=colors)
axes[0].set_title('Performance for Four Categories')
axes[0].set_ylabel('Performance (%)')
axes[0].set_xlabel('Experiment Configuration')
axes[0].set_ylim(0, 100)  # Adjust Y-axis scale
axes[0].grid(True)
axes[0].legend(loc='upper left')

# Plot for six categories
six_cat_df.plot(kind='bar', ax=axes[1], x='Experiment Name', y=['WA (%)', 'UA (%)', 'ACC (%)', 'Macro F1 (%)', 'Weighted F1 (%)'], color=colors)
axes[1].set_title('Performance for Six Categories')
axes[1].set_ylabel('Performance (%)')
axes[1].set_xlabel('Experiment Configuration')
axes[1].set_ylim(0, 100)  # Adjust Y-axis scale
axes[1].grid(True)
axes[1].legend(loc='upper left')

# Adjust layout
plt.tight_layout()

plt.savefig('./test.svg')
plt.savefig('./test.png')


# # Plot line graphs for the performance trends across different experiment configurations

# # Prepare data for line plots
# metrics = ['WA (%)', 'UA (%)', 'ACC (%)', 'Macro F1 (%)', 'Weighted F1 (%)']
# exp_configs = range(len(df))

# # Separate metrics for easier plotting
# four_cat_metrics = four_cat_df[metrics].values.T
# six_cat_metrics = six_cat_df[metrics].values.T

# # Create line plots
# fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=600)

# # Four categories line plot
# for i, metric in enumerate(metrics):
#     axes[0].plot(exp_configs[:4], four_cat_metrics[i], marker='o', label=metric, color=colors[i])
# axes[0].set_title('Performance Trend for Four Categories')
# axes[0].set_xlabel('Experiment Configuration')
# axes[0].set_ylabel('Performance (%)')
# axes[0].set_xticks(exp_configs[:4])
# axes[0].set_xticklabels(four_cat_df['Experiment Name'], rotation=45, ha='right')
# axes[0].set_ylim(0, 100)  # Adjust Y-axis scale
# axes[0].grid(True)
# axes[0].legend(loc='best')

# # Six categories line plot
# for i, metric in enumerate(metrics):
#     axes[1].plot(exp_configs[4:], six_cat_metrics[i], marker='o', label=metric, color=colors[i])
# axes[1].set_title('Performance Trend for Six Categories')
# axes[1].set_xlabel('Experiment Configuration')
# axes[1].set_ylabel('Performance (%)')
# axes[1].set_xticks(exp_configs[4:])
# axes[1].set_xticklabels(six_cat_df['Experiment Name'], rotation=45, ha='right')
# axes[1].set_ylim(0, 100)  # Adjust Y-axis scale
# axes[1].grid(True)
# axes[1].legend(loc='best')for i, metric in enumerate(metrics):
#     axes[1].plot(exp_configs[4:], six_cat_metrics[i], marker='o', label=metric, color=colors[i])
# axes[1].set_title('Performance Trend for Six Categories')
# axes[1].set_xlabel('Experiment Configuration')
# axes[1].set_ylabel('Performance (%)')
# axes[1].set_xticks(exp_configs[4:])
# axes[1].set_xticklabels(six_cat_df['Experiment Name'], rotation=45, ha='right')
# axes[1].set_ylim(0, 100)  # Adjust Y-axis scale
# axes[1].grid(True)
# axes[1].legend(loc='best')

# # Adjust layout
# plt.tight_layout()

# plt.show()
