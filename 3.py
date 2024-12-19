import os
import glob
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torch.nn as nn
import copy
import time
from tqdm import tqdm
import graphviz


# 设置种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 设置随机数种子
setup_seed(2021)

# 数据预处理
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice - cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}


def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    feature_list = []
    label_list = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            label_name = os.path.basename(os.path.dirname(fn))
            if label_name in label_dict:
                label_list.append(label_dict[label_name])
            else:
                print(f"Warning: Unexpected label name {label_name} found, skipping this file.")
                continue
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mels = np.mean(librosa.power_to_db(mel).T, axis=0)
            feature_list.append(mels)

    feature_np = np.array(feature_list)
    label_np = np.array(label_list)
    return feature_np, label_np


# 自己更改目录
parent_dir = './train/'
sub_dirs = ['aloe', 'burger', 'cabbage', 'candied_fruits',
            'carrots', 'chips', 'chocolate', 'drinks', 'fries',
            'grapes', 'gummies', 'ice - cream', 'jelly', 'noodles', 'pickles',
            'pizza', 'ribs', 'salmon', 'soup', 'wings']

# 获取特征feature以及类别的label
feature_data, label_data = extract_features(parent_dir, sub_dirs, max_file=100)

# 获取特征
X = feature_data

# 获取标签
Y = label_data
Y = np.asarray(Y, 'int64')
print()
print('X的特征尺寸是：', X.shape)
print('X的特征尺寸是：', Y.shape)

# 数据集制作
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.05)
print('训练集的大小', len(X_train))
print('测试集的大小', len(X_test))

X_train = X_train.reshape(-1, 1, 16, 8)
Y_train = np.asarray(Y_train, 'int64')
X_test = X_test.reshape(-1, 1, 16, 8)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
train_data = Data.TensorDataset(X_train, Y_train)
X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test, dtype=torch.int64)
test_data = Data.TensorDataset(X_test, Y_test)

## 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=128,  # 批处理样本大小
    shuffle=False,  # 每次迭代前不乱数据
)

test_loader = Data.DataLoader(
    dataset=test_data,  # 使用的数据集
    batch_size=128,  # 批处理样本大小
    shuffle=False,  # 每次迭代前不乱数据
)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
print(b_x.shape)
print(b_y.shape)
print(b_x.dtype)
print(b_y.dtype)

## 可视化一个batch的图像
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
show_num = min(len(batch_y), 64)  # 确定要展示的子图数量，最多为64个
plt.figure(figsize=(12, 5))
for ii in np.arange(show_num):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(batch_y[ii], size=9)
    plt.axis("off")


# 定义模型
class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        ## 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入的feature map
                out_channels=64,  # 输出的feature map
                kernel_size=3,  # 卷积核尺寸
                stride=1,  # 卷积核步长
                padding=5,  # 进行填充
            ),  # 卷积后： (1*28*28) ->(16*28*28)
            nn.Tanh(),  # 激活函数
            nn.AvgPool2d(
                kernel_size=2,  # 平均值池化层,使用 2*2
                stride=1,  # 池化步长为2
            ),  # 池化后：(16*28*28)->(16*14*14)
        )
        ## 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Tanh(),  # 激活函数
            nn.AvgPool2d(2, 1)
        )
        ## 定义第二个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Tanh(),  # 激活函数
            nn.AvgPool2d(2, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Tanh(),  # 激活函数
            nn.AvgPool2d(2, 1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Tanh(),  # 激活函数
            nn.AvgPool2d(2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(13376, 5012),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(5012, 1024),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 20),
        )
        self.dropout_c = nn.Dropout(p=0.5)  # dropout训练

    ## 定义网络的向前传播路径
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout_c(x)
        x = self.conv2(x)
        x = self.dropout_c(x)
        x = self.conv3(x)
        x = self.dropout_c(x)
        x = self.conv4(x)
        x = self.dropout_c(x)
        x = self.conv5(x)
        x = self.dropout_c(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        output = self.classifier(x)
        return output


model = MyConvNet()

# 指定Graphviz可执行文件路径相关操作
graphviz_bin_path = "C:\\Program Files\\Graphviz\\bin"
os.environ["PATH"] += os.pathsep + graphviz_bin_path

# 创建一个Graphviz的Digraph对象用于生成模型图
dot = graphviz.Digraph(comment='Model Graph', format='pdf', engine='dot',
                       graph_attr={
                           'rankdir': 'LR',  # 这里先尝试修改布局方向为从左到右，可按需调整
                           'nodesep': '0.5',  # 增大节点之间的水平间距，可根据实际效果调整数值
                           'ranksep': '1.0',  # 增大不同层级之间的垂直间距，可调整数值
                           'size': '10,10'  # 设置页面大小为宽10英寸、高10英寸，可根据实际情况增大数值
                       },
                       node_attr={
                           'shape': 'box',
                           'fontsize': '10'  # 适当减小字体大小，避免文字太大占空间，也可调整合适数值
                       },
                       edge_attr={})

# 创建一个随机输入张量，用于追踪模型的计算图，形状需符合模型输入要求
x = torch.randn(1, 1, 16, 8).requires_grad_(True)
# 将模型和输入张量传入以构建计算图相关内容
output = model(x)

# 获取模型所有层的名称列表
layer_names = []
for name, module in model.named_children():
    layer_names.append(name)

# 为输入张量添加节点，并添加详细信息
dot.node('input', 'Input\nShape: 1x1x16x8')

# 遍历每一层，添加节点并连接边，同时在节点标签中添加详细信息
for i in range(len(layer_names)):
    layer_name = layer_names[i]
    layer = getattr(model, layer_name)
    if isinstance(layer, nn.Conv2d):
        label_info = f"{layer_name}\n{layer}\nOutput Shape: {_get_output_shape(layer, x)}"
    elif isinstance(layer, nn.Linear):
        label_info = f"{layer_name}\n{layer}\nOutput Shape: {_get_output_shape(layer, x)}"
    else:
        label_info = layer_name
    dot.node(layer_name, label_info)
    if i == 0:
        dot.edge('input', layer_name)
    else:
        prev_layer_name = layer_names[i - 1]
        dot.edge(prev_layer_name, layer_name)

# 为输出添加节点并连接最后一层
dot.node('output', 'Output\nShape: 20')
last_layer_name = layer_names[-1]
dot.edge(last_layer_name, 'output')


# 辅助函数，用于计算卷积层或全连接层的输出形状
def _get_output_shape(layer, x):
    if isinstance(layer, nn.Conv2d):
        x = layer(x)
        return x.shape[1:]
    elif isinstance(layer, nn.Linear):
        x = x.view(x.size(0), -1)
        x = layer(x)
        return x.shape[1:]


# 保存生成的模型图文件（这里保存到当前用户的文档文件夹下，可根据实际情况调整保存路径及文件名等）
save_path = "model_graph.pdf"
dot.render(save_path, view=False)

# 训练部分
device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10,...], [10,...], [10,...] on 3 GPUs
    model = nn.DataParallel(model)
    model.to(device)
print(model)
model = model.to(device)

## 对模型进行训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()  # 损失函数
num_epochs = 150
train_rate = 0.8

## 计算训练使用的批次数量
batch_num = len(train_loader)
train_batch_num = round(batch_num * train_rate)
## 复制模型的参数
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_loss_all = []
train_acc_all = []
val_loss_all = []
val_acc_all = []
since = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # 每个epoch有两个训练阶段
    train_loss = 0.0
    train_corrects = 0
    train_num = 0
    val_loss = 0.0
    val_corrects = 0
    val_num = 0

    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()
        output = model(b_x)
        pre_lab = torch.argmax(output, 1)
        loss = criterion(output, b_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab == b_y.data)
        train_num += b_x.size(0)

    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        model.eval()  # 设置模型为训练模式评估模式
        output = model(b_x)
        pre_lab = torch.argmax(output, 1)
        loss = criterion(output, b_y)
        val_loss += loss.item() * b_x.size(0)
        val_corrects += torch.sum(pre_lab == b_y.data)
        val_num += b_x.size(0)

    ## 计算一个epoch在训练集和验证集上的的损失和精度
    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(train_corrects.double().item() / train_num)
    val_loss_all.append(val_loss / val_num)
    val_acc_all.append(val_corrects.double().item() / val_num)
    print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
        epoch, train_loss_all[-1], train_acc_all[-1])
    )
    print('{} Val Loss: {:.4f}  val Acc: {:.4f}'.format(
        epoch, val_loss_all[-1], val_acc_all[-1])
    )
    # 拷贝模型最高精度下的参数
    if val_acc_all[-1] > best_acc:
        best_acc = val_acc_all[-1]
        best_model_wts = copy.deepcopy(model.state_dict())
    time_use = time.time() - since
    print("Train and val complete in {:.0f}m {:.0f}s".format(
        time_use // 60, time_use % 60))

# 使用最好模型的参数
model.load_state_dict(best_model_wts)
# torch.save(model,'./cnn.pkl')

train_process = pd.DataFrame(
    data={"epoch": range(num_epochs),
          "train_loss_all": train_loss_all,
          "val_loss_all": val_loss_all,
          "train_acc_all": train_acc_all,
          "val_acc_all": val_acc_all})

## 可视化模型训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all,
         "r-", label="Train loss")
plt.plot(train_process.epoch, train_process.val_loss_all,
         "b-", label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all,
         "r-", label="Train acc")
plt.plot(train_process.epoch, train_process.val_acc_all,
         "b-", label="Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.grid()
plt.show()

test_dir = './test_a'


def extract_features(test_dir, file_ext="*.wav"):
    feature_list = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):  # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mels = np.mean(librosa.power_to_db(mel).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
        feature_list.append(mels)

    feature_np = np.array(feature_list)
    return feature_np


X_test = extract_features(test_dir)
X_test = torch.tensor(np.vstack(X_test))
X_test = X_test.to(device)
predictions = model(X_test.reshape(-1, 1, 16, 8)).cpu()

preds = np.argmax(predictions.detach().numpy(), axis=1)
preds = [label_dict_inv[x] for x in preds]

path = glob.glob('./test_a/*.wav')
result = pd.DataFrame({'name': path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('./submit.csv', index=None)