import torch
import torch.nn as nn
import torch.nn.functional as F   
import torch.optim as optim
from torchvision import datasets, transforms   
from torch.optim.lr_scheduler import StepLR  # 新增：学习率调度器

# 1. 数据预处理（保持不变，适配CIFAR-10）
transform = transforms.Compose([
    transforms.ToTensor(),              
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)    
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)     
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)    
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)      



# 2. 定义ResNet核心组件：残差块（BasicBlock）,卷积--批归一化--激活--卷积---批归一化---残差连接（可能需要在sikp connection上加一个卷积层变换通道数）
class BasicBlock(nn.Module):   
    expansion = 1  # 通道扩展系数（BasicBlock为1，Bottleneck为4）  

    def __init__(self, in_channels, out_channels, stride=1):  #in_channels表示输入残差块前的通道数，out_channels表示希望经过残差块之后得到的通道数
        super(BasicBlock, self).__init__()  
        # 第一个卷积：3×3，步长由stride决定，用于下采样缩小尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # BN层：加速训练，缓解梯度消失
        # 第二个卷积：3×3，步长固定为1,第二个卷积层保持通道数和形状大小不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)#仅做特征提取，不改变尺寸和通道。
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差连接的维度匹配：如果输入和输出的通道/尺寸不一致，用1×1卷积调整
        self.shortcut = nn.Sequential()   
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
            

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 卷积→BN→ReLU
        out = self.bn2(self.conv2(out))        # 卷积→BN（无ReLU）
        out += self.shortcut(x)                # 残差连接：核心！
        out = F.relu(out)                      # 最后ReLU
        return out



# 3. 定义适配CIFAR-10的ResNet整体架构
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 输入层：适配CIFAR-10（32×32），不用7×7卷积，改用3×3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 4个残差层（CIFAR-10尺寸小，步长调整为1/2/2/2）
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 全局平均池化：替代扁平化，减少参数，避免过拟合
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层：输出10类（CIFAR-10）
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # 只有第一个块用stride下采样
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32×32→32×32，64通道
        out = self.layer1(out)                 # 32×32→32×32，64通道
        out = self.layer2(out)                 # 32×32→16×16，128通道
        out = self.layer3(out)                 # 16×16→8×8，256通道
        out = self.layer4(out)                 # 8×8→4×4，512通道
        out = self.avg_pool(out)               # 4×4→1×1，512通道
        out = out.flatten(1)                   # 展平：[batch,512]
        out = self.fc(out)                     # 输出10类
        return out

# 构建ResNet18（适配CIFAR-10的轻量版）
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 4. 初始化模型与优化器（调整ResNet的训练参数）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)  # 替换原SimpleCNN为ResNet18

criterion = nn.CrossEntropyLoss()
# 调整优化器：ResNet需要更大的学习率+权重衰减（防止过拟合）
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# 新增：学习率调度器（阶梯下降，训练后期降低学习率）
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 5. 训练循环（仅新增学习率调度）
def train(epoch):
    model.train()
    train_loss_sum = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss_sum += loss.item() * len(data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.4f}')

    avg_epoch_loss = train_loss_sum / len(train_dataset)
    print(f'Train Epoch: {epoch}\tAverage Loss:{avg_epoch_loss:.4f}')
    # 学习率调度
    scheduler.step()

# 6. 测试循环（保持不变）
def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.2f}%)\n')

# 开始运行（增加训练轮数，ResNet需要更多轮次收敛）
if __name__ == '__main__':
    for epoch in range(1, 61):  # 训练60轮（ResNet收敛慢，需更多轮次）
        train(epoch)
        test()