import os
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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
from torch.nn import functional as F

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


def remove_constant_features(X_df):
    """
    删除所有常数特征。
    """
    constant_features = []
    for col in X_df.columns:
        if X_df[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        logger.info(f"Removing constant features: {constant_features}")
        X_df.drop(columns=constant_features, inplace=True)
    else:
        logger.info("No constant features found.")

    return X_df


def extract_sequences(X_df, y):
    """
    提取具有相同天气类别的子序列，并为每个子序列分配唯一的标签。
    """
    sequences = []
    labels = []
    current_label = None
    current_sequence = []

    for i in range(len(y)):
        if y[i] != current_label:
            if current_label is not None:
                sequences.append(current_sequence)
                labels.append(current_label)  # 使用第一个标签作为子序列的标签
            current_label = y[i]
            current_sequence = [X_df.iloc[i].values]
        else:
            current_sequence.append(X_df.iloc[i].values)

    # 添加最后一个序列
    if current_label is not None:
        sequences.append(current_sequence)
        labels.append(current_label)  # 使用第一个标签作为子序列的标签

    logger.info(f"Extracted {len(sequences)} sequences with corresponding labels.")
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


def global_min_max_scale(data):
    """
    对整个数据集进行最小最大归一化，使其范围在 [-1, 1] 之间。
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def generate_gaf_images(X, window_size, gaussian_blur=False):
    """
    生成 GASF 图像。
    """
    images_GASF = []
    valid_labels = []  # 存储有效的标签

    for sample_idx in range(X.shape[0]):
        gafs = []
        for feature_idx in range(X.shape[2]):
            feature_values = X[sample_idx, :, feature_idx]

            # Piecewise Aggregation Approximation (PAA)
            paa_feature = [feature_values[t * (len(feature_values) // window_size):(t + 1) * (
                        len(feature_values) // window_size)].mean()
                           for t in range(window_size)]

            # Convert to cosine and sine
            cos_paa = np.cos(np.pi * np.clip(paa_feature, -1, 1))  # Clip values to avoid numerical instability
            sin_paa = np.sin(np.pi * np.clip(paa_feature, -1, 1))  # Clip values to avoid numerical instability

            # Generate GASF matrix
            matrix_GASF = np.outer(cos_paa, cos_paa) - np.outer(sin_paa, sin_paa)

            # Apply Gaussian Blur if required
            if gaussian_blur:
                matrix_GASF = cv2.GaussianBlur(matrix_GASF, (5, 5), 0)

            gafs.append(matrix_GASF)

        if gafs:  # 只有当至少有一个有效的GASF矩阵时才添加
            images_GASF.append(gafs)
            valid_labels.append(sample_idx)

    if not images_GASF:
        raise ValueError("No valid GASF images generated. All features may have constant values across samples.")

    logger.info(f"Generated {len(images_GASF)} valid GASF images.")
    return np.array(images_GASF), valid_labels


def generate_gadf_images(X, window_size, gaussian_blur=False):
    """
    生成 GADF 图像。
    """
    images_GADF = []
    valid_labels = []  # 存储有效的标签

    for sample_idx in range(X.shape[0]):
        gadfs = []
        for feature_idx in range(X.shape[2]):
            feature_values = X[sample_idx, :, feature_idx]

            # Piecewise Aggregation Approximation (PAA)
            paa_feature = [feature_values[t * (len(feature_values) // window_size):(t + 1) * (
                        len(feature_values) // window_size)].mean()
                           for t in range(window_size)]

            # Convert to cosine and sine
            cos_paa = np.cos(np.pi * np.clip(paa_feature, -1, 1))  # Clip values to avoid numerical instability
            sin_paa = np.sin(np.pi * np.clip(paa_feature, -1, 1))  # Clip values to avoid numerical instability

            # Generate GADF matrix
            matrix_GADF = np.outer(cos_paa, sin_paa) - np.outer(sin_paa, cos_paa)

            # Apply Gaussian Blur if required
            if gaussian_blur:
                matrix_GADF = cv2.GaussianBlur(matrix_GADF, (5, 5), 0)

            gadfs.append(matrix_GADF)

        if gadfs:  # 只有当至少有一个有效的GADF矩阵时才添加
            images_GADF.append(gadfs)
            valid_labels.append(sample_idx)

    if not images_GADF:
        raise ValueError("No valid GADF images generated. All features may have constant values across samples.")

    logger.info(f"Generated {len(images_GADF)} valid GADF images.")
    return np.array(images_GADF), valid_labels


def generate_position_encoding(window_size):
    """
    生成非对称的位置编码图像。
    """
    position_encoding = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            position_encoding[i, j] = i / (window_size - 1) + j / (window_size - 1) ** 2
    return position_encoding


def plot_images(images, image_type, feature_names, window_size, save_dir=None):
    """
    绘制图像。
    """
    num_samples = images.shape[0]

    for sample_idx in range(min(num_samples, 5)):  # 仅绘制前5个样本以节省空间
        num_features = images[sample_idx].shape[0]
        fig, axes = plt.subplots(num_features, 1, figsize=(8, 4 * num_features))
        fig.suptitle(f"{image_type} Images for Sample {sample_idx} (Window Size: {window_size})")

        for i in range(num_features):
            ax = axes[i] if num_features > 1 else axes
            ax.set_title(f'Feature {feature_names[i]} - {image_type}')
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.2)
            im = ax.imshow(images[sample_idx, i], cmap=cm.jet)
            fig.colorbar(im, cax=cax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"{image_type.lower()}_images_sample_{sample_idx}_window_{window_size}.png"))
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


# Residual模块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


# 简化的ResNet模型
class SimpleResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.b2 = nn.Sequential(*resnet_block(16, 16, 1, first_block=True))  # 修改此处：保持输入输出通道一致
        self.b3 = nn.Sequential(*resnet_block(16, 16, 1))  # 修改此处：保持输入输出通道一致
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, num_classes)  # 修改此处：与最后的通道数一致

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):  # first_block用于判断是否是第一个block
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
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

        # Step the scheduler
        scheduler.step(running_loss / len(train_loader))

    return best_accuracy


def main():
    # 配置参数
    file_path = 'E:\\LX\\GIT\\git\\test\\GASF-GADF-MTF\\data\\weather_classification_data.csv'  # 替换为你的CSV文件路径
    gaussian_blur = False  # 是否应用高斯模糊
    save_dir = r'E:\LX\GIT\git\test\GASF-GADF-MTF\img'  # 输出图像的目录

    # 创建输出目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载数据
    X_df, y = load_csv_data(file_path)

    # 删除常数特征
    X_df = remove_constant_features(X_df)

    # 提取具有相同天气类别的子序列
    sequences, labels = extract_sequences(X_df, y)

    # 计算最长子序列的长度
    max_seq_len = max(len(seq) for seq in sequences)
    logger.info(f"Maximum sequence length: {max_seq_len}")

    # 使用线性插值对齐所有子序列到最大长度
    aligned_sequences = [linear_interpolate(seq, max_seq_len) for seq in sequences]
    aligned_sequences = np.stack(aligned_sequences)
    logger.info(f"Aligned sequences shape: {aligned_sequences.shape}")

    # 标准化数据
    flattened_sequences = aligned_sequences.reshape(-1, aligned_sequences.shape[-1])
    normalized_X_all_flattened, scaler = global_min_max_scale(flattened_sequences)
    normalized_X_all = normalized_X_all_flattened.reshape(aligned_sequences.shape)
    logger.info(f"Normalized X all shape: {normalized_X_all.shape}")

    # 定义窗口大小为最长序列长度
    window_size = max_seq_len
    logger.info(f"Using window size: {window_size}")

    try:
        # 生成GASF图像
        images_GASF, valid_labels = generate_gaf_images(normalized_X_all, window_size, gaussian_blur)
        logger.info(f"Images GASF shape: {images_GASF.shape}")

        # 生成GADF图像
        images_GADF, _ = generate_gadf_images(normalized_X_all, window_size, gaussian_blur)
        logger.info(f"Images GADF shape: {images_GADF.shape}")

        # 生成位置编码图像
        position_encoding = generate_position_encoding(window_size)
        position_encoding_expanded = np.expand_dims(position_encoding, axis=0)  # Shape: (1, window_size, window_size)
        position_encoding_expanded = np.repeat(position_encoding_expanded, images_GASF.shape[1],
                                               axis=0)  # Repeat for each feature
        position_encoding_expanded = np.expand_dims(position_encoding_expanded,
                                                    axis=0)  # Shape: (1, num_features, window_size, window_size)
        position_encoding_expanded = np.repeat(position_encoding_expanded, images_GASF.shape[0],
                                               axis=0)  # Repeat for each sample
        logger.info(f"Position encoding expanded shape: {position_encoding_expanded.shape}")

        # 计算GASF和GADF的差值并进行指数化处理
        diff_images = images_GASF - images_GADF
        exp_diff_images = np.exp(diff_images)
        logger.info(f"Exponentiated difference images shape: {exp_diff_images.shape}")

        # 添加位置编码
        final_images = exp_diff_images + position_encoding_expanded
        logger.info(f"Final images shape after adding position encoding: {final_images.shape}")

        # 绘制图像
        feature_names = X_df.columns.tolist()
        plot_images(final_images, "Final Exponentiated Difference with Position Encoding", feature_names, window_size,
                    save_dir)

        # 将图像转换为适合卷积神经网络的格式
        # 形状应为 (num_samples, num_features, height, width)，其中 height 和 width 是 window_size
        X_cnn = final_images.transpose(0, 1, 2, 3)  # Shape: (num_samples, num_features, height, width)
        X_cnn = torch.tensor(X_cnn, dtype=torch.float32)

        # 更新标签以匹配有效样本
        y_cnn = torch.tensor([labels[lbl] for lbl in valid_labels], dtype=torch.long)
        logger.info(f"X cnn shape: {X_cnn.shape}, y cnn shape: {y_cnn.shape}")

        # 检查是否有足够的样本进行训练
        if len(X_cnn) < 2:
            logger.error("Not enough valid samples to perform training and evaluation.")
            return

        # 划分训练集和测试集
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
        try:
            train_indices, test_indices = next(splitter.split(X_cnn, y_cnn))
        except ValueError as e:
            logger.error(f"Error during splitting: {e}")
            return

        train_dataset = WeatherDataset(X_cnn[train_indices], y_cnn[train_indices])
        test_dataset = WeatherDataset(X_cnn[test_indices], y_cnn[test_indices])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 初始化模型、损失函数和优化器
        num_classes = len(np.unique(y_cnn.numpy()))
        num_features = X_cnn.shape[1]
        model = SimpleResNet(input_channels=num_features, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)  # 减少学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # 训练和评估模型
        accuracy = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        logger.info(f"Best accuracy: {accuracy:.2f}%")

    except ValueError as ve:
        logger.error(f"ValueError encountered: {ve}")


if __name__ == "__main__":
    main()