import torch.nn as nn
import torch.nn.functional as F
import torch
 
class BasicBlock(nn.Module):
    # 由两个卷积层组成，每个卷积层后跟一个 Batch Normalization 和 ReLU 激活函数。

    expansion = 1  # 定义扩展因子，用于调整输出通道数
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 定义第一个卷积层，输入通道为 in_planes，输出通道为 planes，卷积核大小为3，步幅为 stride，padding 为1，不使用偏置
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential() # 初始化 shortcut（捷径）层
        # 如果步幅不为1或输入通道数不等于扩展因子乘以输出通道数，则创建一个捷径卷积层
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.bn2(self.conv2(out))
        #print(out.shape)
        out += self.shortcut(x)
        #print(out.shape)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    # 由三个卷积层组成：第一个卷积层用来减少通道数（1x1 卷积），第二个卷积层进行深度卷积（3x3 卷积），第三个卷积层用来恢复通道数（1x1 卷积）。
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
 ## https://blog.csdn.net/TXK_Kevin/article/details/125453827
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 定义每个块的步幅
        layers = []  # 存储层的列表
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建块并添加到层列表
            self.in_planes = planes * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)  # 返回一个顺序容器，解包，相当于 nn.Sequential(layer1, layer2, layer3）
    
    def forward(self, x):
        # 输入形状: (batch_size, in_channels, height, width) 估计输入输出都是32
        out = F.relu(self.bn1(self.conv1(x)))
        # 第一层卷积后的输出形状: (batch_size, 64, height, width)，因为输出通道数为64
        out = self.layer1(out)  # 特征图个数没变，输入是64 输出也是64 在shortcut中不需要调整
        # 第一层的输出形状: (batch_size, 64, height, width)
        out = self.layer2(out)
        # 第二层的输出形状: (batch_size, 128, height/2, width/2)，因为步幅为2，通道数为128
        out = self.layer3(out)
        # 第三层的输出形状: (batch_size, 256, height/4, width/4)，因为步幅为2，通道数为256
        out = self.layer4(out)
        # 第四层的输出形状: (batch_size, 512, height/8, width/8)，因为步幅为2，通道数为512
        out = F.avg_pool2d(out, 4)
        # 平均池化后的输出形状: (batch_size, 512, 1, 1)，将特征图缩小到1x1
        out = out.view(out.size(0), -1)
        # 扁平化后的输出形状: (batch_size, 512)，将特征图展平为一维向量
        out = self.linear(out)
        # 线性层输出形状: (batch_size, num_classes)，最终分类输出
        return out

 
def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
 
 
def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

class BasicBlock1D(nn.Module):
    expansion = 1  # 定义扩展因子

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        # 定义第一个卷积层，1D 卷积，输入通道为 in_planes，输出通道为 planes，卷积核大小为3，步幅为 stride，padding 为1，不使用偏置
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # 定义第二个卷积层，卷积核大小为3，步幅为1，padding 为1，不使用偏置
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        # 定义 shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积层 + BN + ReLU
        out = self.bn2(self.conv2(out))  # 第二个卷积层 + BN
        out += self.shortcut(x)  # 添加 shortcut 路径
        out = F.relu(out)  # 最后的 ReLU
        return out

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 64
        
        # 一维卷积层处理一维特征
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输出通道数设置为 64
        self.bn1 = nn.BatchNorm1d(64)  # 对应的 BatchNorm 通道数也设置为 64
        
        # 按照 ResNet50 配置构建 Bottleneck 模块层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 将输入维度转成 (batch_size, in_channels, feature_length)
        x = x.unsqueeze(1)  # 假设 x 是 (batch_size, feature_length)，加上 in_channels 维度
        
        # 通过初始卷积层
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 逐层经过 ResNet50 的 Bottleneck 模块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 平均池化和展平操作
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        
        # 全连接层输出
        out = self.linear(out)
        return out

# 定义一维特征的 ResNet50
def ResNet50_1D(in_channels, num_classes):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


def ResNet18_1D(in_channels, num_classes):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_classes, num_features):
        super(SimpleCNN, self).__init__()
        
        # 卷积层的定义
        self.conv_layer = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),  # 1->16通道
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 池化将特征长度减半
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 16->32通道
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 池化再次将特征长度减半
        )

        # 动态计算卷积层输出的特征长度
        conv_output_length = self._get_conv_output_size(num_features)
        
        # 定义全连接层
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_length, 64),  # 动态计算的卷积输出长度
            nn.ReLU(),
            nn.Linear(64, output_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

    def _get_conv_output_size(self, num_features):
        # 输入假设数据形状 (batch_size=1, channels=1, num_features)
        x = torch.zeros(1, 1, num_features)  # 用零初始化输入数据
        x = self.conv_layer(x)  # 通过卷积层
        return x.numel()  # 返回卷积后输出的总元素数量

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_classes):
        super(SimpleMLP, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_classes)
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x