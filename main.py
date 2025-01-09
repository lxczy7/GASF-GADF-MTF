import os
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from scipy.interpolate import interp1d

# 设置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 编码字符串列到数值值
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 分离特征和标签
    X_df = df.iloc[:, :-1]  # 所有列除了最后一列作为特征
    y = df.iloc[:, -1]  # 最后一列作为标签

    return X_df, y


def extract_sequences(X_df, y):
    """
    提取具有相同天气类别的子序列。
    """
    sequences = []
    labels = []
    current_label = None
    current_sequence = []

    for i in range(len(y)):
        if y[i] != current_label:
            if current_label is not None:
                sequences.append(current_sequence)
                labels.append(current_label)
            current_label = y[i]
            current_sequence = [X_df.iloc[i].values]
        else:
            current_sequence.append(X_df.iloc[i].values)

    # 添加最后一个序列
    if current_label is not None:
        sequences.append(current_sequence)
        labels.append(current_label)

    return sequences, labels


def linear_interpolate(sequence, target_length):
    """
    对给定的时间序列进行线性插值，使其达到目标长度。
    """
    sequence = np.array(sequence)
    interpolated_sequence = np.zeros((target_length, sequence.shape[1]))

    for feature in range(sequence.shape[1]):
        original_indices = np.linspace(0, len(sequence) - 1, num=len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, num=target_length)
        interpolated_sequence[:, feature] = np.interp(new_indices, original_indices, sequence[:, feature])

    return interpolated_sequence


def generate_gaf_images(X, window_size, gaussian_blur=False):
    """
    生成 GASF 和 GADF 图像。
    """
    images_GASF, images_GADF = [], []

    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]
        scaled_feature = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))

        # Piecewise Aggregation Approximation (PAA)
        paa_feature = [scaled_feature[
                       t * (len(scaled_feature) // window_size):(t + 1) * (len(scaled_feature) // window_size)].mean()
                       for t in range(window_size)]

        # Convert to cosine and sine
        cos_paa = np.cos(np.pi * np.array(paa_feature))
        sin_paa = np.sin(np.pi * np.array(paa_feature))

        # Generate GASF and GADF matrices
        matrix_GASF = np.outer(cos_paa, cos_paa) - np.outer(sin_paa, sin_paa)
        matrix_GADF = np.outer(sin_paa, cos_paa) - np.outer(cos_paa, sin_paa)

        # Apply Gaussian Blur if required
        if gaussian_blur:
            matrix_GASF = cv2.GaussianBlur(matrix_GASF, (5, 5), 0)
            matrix_GADF = cv2.GaussianBlur(matrix_GADF, (5, 5), 0)

        images_GASF.append(matrix_GASF)
        images_GADF.append(matrix_GADF)

    return np.array(images_GASF), np.array(images_GADF)


def plot_images(images_GASF, images_GADF, feature_names, window_size, save_dir=None):
    """
    绘制 GASF 和 GADF 图像。
    """
    num_features = len(images_GASF)
    fig, axes = plt.subplots(num_features, 2, figsize=(16, 4 * num_features))
    fig.suptitle(f"GASF and GADF Images for Each Feature (Window Size: {window_size})")

    for i in range(num_features):
        ax1, ax2 = axes[i] if num_features > 1 else (axes[0], axes[1])
        ax1.set_title(f'Feature {feature_names[i]} - GASF')
        cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.2)
        im1 = ax1.imshow(images_GASF[i], cmap=cm.jet)
        fig.colorbar(im1, cax=cax1)

        ax2.set_title(f'Feature {feature_names[i]} - GADF')
        cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.2)
        im2 = ax2.imshow(images_GADF[i], cmap=cm.jet)
        fig.colorbar(im2, cax=cax2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"gaf_images_window_{window_size}.png"))
    plt.show()


class WeatherDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, window_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(int(window_size / 2), 1), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(16 * int((window_size / 2) / 2) * 7,
                             num_classes)  # Adjusted based on output dimensions after conv and pool layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test set: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return best_accuracy


def main():
    # 配置参数
    file_path = r'E:\LX\GIT\git\test\GASF-GADF-MTF\data\weather_classification_data.csv'  # 替换为你的CSV文件路径
    gaussian_blur = False  # 是否应用高斯模糊
    save_dir = 'output_images'  # 输出图像的目录

    # 创建输出目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载数据
    X_df, y = load_csv_data(file_path)

    # 提取具有相同天气类别的子序列
    sequences, labels = extract_sequences(X_df, y)

    # 计算最长子序列的长度
    max_seq_len = max(len(seq) for seq in sequences)
    logger.info(f"Maximum sequence length: {max_seq_len}")

    # 使用线性插值对齐所有子序列到最大长度
    aligned_sequences = [linear_interpolate(seq, max_seq_len) for seq in sequences]
    aligned_sequences = np.stack(aligned_sequences)

    # 标准化数据
    scaler = StandardScaler()
    normalized_X_all = scaler.fit_transform(aligned_sequences.reshape(-1, aligned_sequences.shape[-1])).reshape(
        aligned_sequences.shape)

    # 定义要尝试的PAA窗口大小
    window_sizes = [16, 32, 64]
    best_accuracy = 0.0
    best_window_size = None

    for window_size in window_sizes:
        logger.info(f"Trying window size: {window_size}")

        # 生成GASF和GADF图像
        images_GASF, images_GADF = generate_gaf_images(normalized_X_all, window_size, gaussian_blur)

        # 绘制图像
        feature_names = X_df.columns.tolist()
        plot_images(images_GASF, images_GADF, feature_names, window_size, save_dir)

        # 将图像转换为适合卷积神经网络的格式
        X_cnn = np.stack((images_GASF, images_GADF), axis=1)  # Shape: (num_samples, 2, height, width)
        X_cnn = torch.tensor(X_cnn, dtype=torch.float32)
        y_cnn = torch.tensor(labels, dtype=torch.long)

        # 划分训练集和测试集
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
        train_indices, test_indices = next(splitter.split(X_cnn, y_cnn))
        train_dataset = WeatherDataset(X_cnn[train_indices], y_cnn[train_indices])
        test_dataset = WeatherDataset(X_cnn[test_indices], y_cnn[test_indices])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = SimpleCNN(input_channels=2, num_classes=len(np.unique(y)), window_size=window_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练和评估模型
        accuracy = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_window_size = window_size

    logger.info(f"Best window size: {best_window_size}, Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()