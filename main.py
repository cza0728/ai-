import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# ==========================================
# 1. 定义数据预处理 (这里有大改动！)
# ==========================================
# ResNet 是在 ImageNet 上训练的，它习惯看 224x224 的大图
# 而且它习惯特定的均值和方差，我们必须配合它
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 【关键】强行放大！CIFAR-10 原图只有 32x32
    transforms.ToTensor(),
    # 下面这串数字是 ImageNet 的标准均值和方差，死记硬背即可
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def main():
    # ==========================================
    # 2. 准备数据
    # ==========================================
    # batch_size=32: 图片变大了，一次少吃点，防止显存爆炸
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    # ==========================================
    # 3. 请出大神 (加载预训练模型)
    # ==========================================
    print("正在加载 ResNet18 预训练模型...")
    # weights='DEFAULT' 会自动下载最好的预训练参数
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 【关键步骤】冻结骨干网络 (Freeze Backbone)
    # 告诉 PyTorch：前面的卷积层都是大神练好的，别动它们！
    for param in model.parameters():
        param.requires_grad =True

    # ==========================================
    # 4. 篡改最后一层 (Replace the Head)
    # ==========================================
    # ResNet18 最后一层全连接的输入是 512
    num_ftrs = model.fc.in_features
    
    # 把它换成输出 10 类 (CIFAR-10)
    # 注意：新加的层默认 requires_grad=True，所以它会参与训练
    model.fc = nn.Linear(num_ftrs, 10)

    # 把它搬到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已就绪，使用设备: {device}")

    # ==========================================
    # 5. 定义训练工具
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：我们只需要优化最后一层 model.fc 的参数
    # 因为前面的参数都被冻结了 (requires_grad=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ==========================================
    # 6. 开始训练
    # ==========================================
    print("开始迁移学习训练...")
    
    # 只跑 3 轮试试看，因为是大神模型，学得极快
    for epoch in range(3): 
        running_loss = 0.0
        model.train() # 确保在训练模式 (Dropout等生效)
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 每 100 个 batch 打印一次
            if i % 100 == 99:    
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('训练完成！')

    # ==========================================
    # 7. 测试准确率
    # ==========================================
    correct = 0
    total = 0
    model.eval() # 切换到评估模式 (关掉 Dropout)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'ResNet-18 迁移学习准确率: {100 * correct / total}%')
    save_path = './resnet18_cifar10.pth'
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

if __name__ == '__main__':
    main()