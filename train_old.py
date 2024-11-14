import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import ResNet50_1D, ResNet18_1D, SimpleCNN, SimpleMLP
import numpy as np
from sklearn.decomposition import PCA

# 自定义数据集类
class RadiomicsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)  # 转换为Tensor
        self.labels = torch.LongTensor(labels)  # 转换为Tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 读取数据
data = pd.read_csv('data/ct_features_1.csv')
features = data.iloc[:, :-1].values
# 提取标签（最后一列）
labels = data.iloc[:, -1].values
num_features = len(features[0])
num_channels = 1
num_classes = 2

# 数据标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# # # 使用PCA进行降维
# num_features = 200
# pca = PCA(n_components=num_features)  # 假设将数据降到50维，你可以调整这个参数
# features = pca.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# 检查标签分布
# print(f"Training labels distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
# print(f"Test labels distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# 创建训练集和测试集的DataLoader
train_dataset = RadiomicsDataset(X_train, y_train)
test_dataset = RadiomicsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)  # 训练集的DataLoader
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)    # 测试集的DataLoader

# 示例：遍历数据加载器
# for features_batch, labels_batch in train_loader:
#     print(features_batch, labels_batch)  # 输出批量数据
#     break  # 只打印第一批数据

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet18(num_channels,num_classes).to(device)
# model = SimpleCNN(num_channels, num_classes, num_features).to(device)
model = SimpleMLP(num_features, num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 设置训练的轮数
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化累计损失
    for features_batch, labels_batch in train_loader:
        features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
        # 前向传播
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累计损失
        running_loss += loss.item()

    # 每个 epoch 结束后，计算平均损失
    avg_train_loss = running_loss / len(train_loader)

    # 计算训练集的准确率、敏感性、特异性、PPV、NPV 和 AUC
    model.eval()  # 切换为评估模式
    y_true_train = []  # 存储真实标签
    y_pred_train = []  # 存储预测标签
    y_prob_train = []  # 存储预测概率

    with torch.no_grad():
        for features_batch, labels_batch in train_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            outputs = model(features_batch)
            _, predicted = torch.max(outputs, 1)

            y_true_train.extend(labels_batch.cpu().numpy())  # 添加真实标签
            y_pred_train.extend(predicted.cpu().numpy())  # 添加预测标签
            y_prob_train.extend(outputs[:, 1].cpu().numpy())  # 第二列为正类概率

    # 计算混淆矩阵，显式指定 labels=[0, 1]
    cm = confusion_matrix(y_true_train, y_pred_train, labels=[0, 1])
    print(cm)
    if cm.size == 4:  # 确保得到完整的混淆矩阵（4个值：TN, FP, FN, TP）
        tn, fp, fn, tp = cm.ravel()

        # 计算各项指标
        sensitivity_train = tp / (tp + fn)  # 敏感性
        specificity_train = tn / (tn + fp)  # 特异性
        ppv_train = tp / (tp + fp)  # 阳性预测值
        npv_train = tn / (tn + fn) if (tn + fn) != 0 else np.nan  # 阴性预测值，防止除零错误
        auc_train = roc_auc_score(y_true_train, y_prob_train)  # AUC

        # 打印训练集评估指标
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {100 * (tp + tn) / (tp + tn + fp + fn):.2f}%, '
              f'Sensitivity: {sensitivity_train:.4f}, '
              f'Specificity: {specificity_train:.4f}, '
              f'PPV: {ppv_train:.4f}, '
              f'NPV: {npv_train:.4f}, '
              f'AUC: {auc_train:.4f}')
    else:
        print(f"Warning: Confusion matrix has a single class in the prediction, skipping metrics calculation.")

    # 计算测试集的准确率、敏感性、特异性、PPV、NPV 和 AUC
    y_true_test = []  # 存储测试集的真实标签
    y_pred_test = []  # 存储测试集的预测标签
    y_prob_test = []  # 存储测试集的预测概率

    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            outputs = model(features_batch)
            _, predicted = torch.max(outputs, 1)

            y_true_test.extend(labels_batch.cpu().numpy())  # 添加真实标签
            y_pred_test.extend(predicted.cpu().numpy())  # 添加预测标签
            y_prob_test.extend(outputs[:, 1].cpu().numpy())  # 第二列为正类概率

    # 计算混淆矩阵，显式指定 labels=[0, 1]
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])

    if cm_test.size == 4:  # 确保得到完整的混淆矩阵（4个值：TN, FP, FN, TP）
        tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

        # 计算各项指标
        sensitivity_test = tp_test / (tp_test + fn_test)  # 敏感性
        specificity_test = tn_test / (tn_test + fp_test)  # 特异性
        ppv_test = tp_test / (tp_test + fp_test)  # 阳性预测值
        npv_test = tn_test / (tn_test + fn_test) if (tn_test + fn_test) != 0 else np.nan  # 阴性预测值，防止除零错误
        auc_test = roc_auc_score(y_true_test, y_prob_test)  # AUC

        # 打印测试集评估指标
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Test Accuracy: {100 * (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test):.2f}%, '
              f'Sensitivity: {sensitivity_test:.4f}, '
              f'Specificity: {specificity_test:.4f}, '
              f'PPV: {ppv_test:.4f}, '
              f'NPV: {npv_test:.4f}, '
              f'AUC: {auc_test:.4f}')
    else:
        print(f"Warning: Test confusion matrix has a single class in the prediction, skipping metrics calculation.")

    # 绘制 ROC 曲线（训练集与测试集）
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_prob_test)

    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch+1}')
    plt.legend(loc="lower right")
    plt.show()