import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置全局参数
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams.update({
    'font.size': 20,  # 字体大小
    'font.family': 'arial',  # 字体类型
})

# 读取 CSV 文件
df = pd.read_csv("test_final_clean_315-0708.csv")
actuals = df['Actual']
predictions = df['Predicted']

# 计算评估指标
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

# 输出测试指标
print(f"Test R²: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE: {mae:.3f}")

# 创建包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制第一个子图（散点图）
scatter_plot = ax1.scatter(actuals, predictions, color="#b4e2d3", zorder=2)

x = np.linspace(-5, 4, 100)
line1 = ax1.plot(x, x, color="#86a1b4", linestyle="-", linewidth=2.5, label='y = x', zorder=3)[0]
line2 = ax1.plot(x, x + 2 * rmse, color="#86a1b4", linestyle="--", linewidth=2, label=f'y = x ± 2$RMSE$', zorder=3)[0]
line3 = ax1.plot(x, x - 2 * rmse, color="#86a1b4", linestyle="--", linewidth=2, zorder=3)[0]

ax1.set_xlabel('Experimental $log_{10}{LC_{50}}$', fontdict={'family': 'arial', 'size': 20})
ax1.set_ylabel('Predicted $log_{10}{LC_{50}}$', fontdict={'family': 'arial', 'size': 20})

ax1.tick_params(axis='both', which='major', labelsize=20, labelcolor='black')
ax1.tick_params(axis='both', which='major', width=2)  # 设置刻度线的粗细

# 设置坐标轴数字的字体属性
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontname('arial')
    label.set_fontsize(20)

ax1.set_xlim(-5, 4)  # 设置 x 轴的范围
ax1.set_ylim(-5, 4)  # 设置 y 轴的范围

# 创建第一个图例，包含直线的信息
lines_handles = [line1, line2]
lines_labels = [line1.get_label(), line2.get_label()]
first_legend = ax1.legend(lines_handles, lines_labels, fontsize=20, loc='upper left')

# 去除第一个图例的边框
frame = first_legend.get_frame()
frame.set_linewidth(0)

# 将第一个图例添加到当前轴
ax1.add_artist(first_legend)

# 创建第二个图例，包含 R2、RMSE 和 MAE 的值
metrics_labels = [f'$RMSE$ = 0.55', f'$MAE$ = 0.39', f'$R^2$ = 0.84']
empty_handle = [plt.Line2D([], [], color='none') for _ in range(len(metrics_labels))]
second_legend = ax1.legend(empty_handle, metrics_labels, fontsize=20, loc='lower right', handlelength=0)

# 去除第二个图例的边框
frame = second_legend.get_frame()
frame.set_linewidth(0)

# 绘制第二个子图（直方图）
# 计算真实值和预测值之间的差异
difference = actuals - predictions

# 计算百分比
total_data = len(difference)

# 计算大于 1，2，3 的百分比
percent_right_1 = (difference > 1).sum() / total_data * 100
percent_right_2 = (difference > 2).sum() / total_data * 100
percent_right_3 = (difference > 3).sum() / total_data * 100

# 计算小于 -3，-2，-1 的百分比
percent_left_minus_3 = (difference < -3).sum() / total_data * 100
percent_left_minus_2 = (difference < -2).sum() / total_data * 100
percent_left_minus_1 = (difference < -1).sum() / total_data * 100

# 设置 bin 的范围为整数
bins = np.arange(int(difference.min()), int(difference.max()) + 2, 0.5)

# 绘制差异的直方图
ax2.hist(difference, bins=bins, color="#b4e2d3", edgecolor="#86a1b4", linewidth=2)

# 设置图形标题和标签
ax2.set_xlabel('Absolute error', fontdict={'family': 'arial', 'size': 24})
ax2.set_ylabel('Count', fontdict={'family': 'arial', 'size': 24})

# 设置坐标轴的字体大小和颜色
ax2.tick_params(axis='both', which='major', labelsize=20, labelcolor='black')

# 设置坐标轴数字的字体属性
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontname('arial')
    label.set_fontsize(16)

# 设置 x 轴刻度为 0.5 的间隔
xticks = np.arange(np.floor(difference.min()), np.ceil(difference.max()) + 1, 1)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticks, fontsize=20)
yticks = [0,100,200,300,400,500,600]
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticks, fontsize=20)
ax2.set_xlim(-4, 4)

# 设置图四个方向框线的粗细
ax2.spines['top'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)

# 添加垂直线并显示百分比
ax2.axvline(1, color='#b0b7c6', linestyle='--', linewidth=1.5)
ax2.axvline(2, color='#b0b7c6', linestyle='--', linewidth=1.5)
ax2.axvline(3, color='#b0b7c6', linestyle='--', linewidth=1.5)

# 在直线的右侧显示百分比
ax2.text(1 + 0.1, ax2.get_ylim()[1] * 0.9, f'{percent_right_1:.1f}%', fontsize=20, color='black')
ax2.text(2 + 0.1, ax2.get_ylim()[1] * 0.9, f'{percent_right_2:.1f}%', fontsize=20, color='black')
ax2.text(3 + 0.1, ax2.get_ylim()[1] * 0.9, f'{percent_right_3:.1f}%', fontsize=20, color='black')

# 添加负值的垂直线和百分比
ax2.axvline(-1, color='#b0b7c6', linestyle='--', linewidth=1.5)
ax2.axvline(-2, color='#b0b7c6', linestyle='--', linewidth=1.5)
ax2.axvline(-3, color='#b0b7c6', linestyle='--', linewidth=1.5)

# 在直线的左侧显示百分比
ax2.text(-1 - 0.85, ax2.get_ylim()[1] * 0.9, f'{percent_left_minus_1:.1f}%', fontsize=20, color='black')
ax2.text(-2 - 0.85, ax2.get_ylim()[1] * 0.9, f'{percent_left_minus_2:.1f}%', fontsize=20, color='black')
ax2.text(-3 - 0.85, ax2.get_ylim()[1] * 0.9, f'{percent_left_minus_3:.1f}%', fontsize=20, color='black')
fig.text(0.07, 1, '(A)', fontsize=20,  va='top', ha='left')
fig.text(0.57, 1, '(B)', fontsize=20,  va='top', ha='left')
plt.tight_layout()
plt.savefig("yyd-white-retrain.svg", format="svg", dpi=300)
plt.savefig("yyd-white-retrain.png", format="png", dpi=300)
plt.show()
