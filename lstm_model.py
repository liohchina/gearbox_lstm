import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 读取数据
df = pd.read_csv('data.csv')
train_df = pd.DataFrame()
test_df = pd.DataFrame()
new_class_dfs = [df[df['label'] == i] for i in range(4)]

# 分层划分数据集
test_ratio = 0.2
for class_df in new_class_dfs:
    num_samples = len(class_df)
    num_test_samples = int(num_samples * test_ratio)

    # 随机选择样本索引
    random_indices = random.sample(range(num_samples), num_samples)

    # 根据比例分割样本
    test_indices = random_indices[:num_test_samples]
    train_indices = random_indices[num_test_samples:]

    # 获取训练集和测试集的子DataFrame
    test_class_df = class_df.iloc[test_indices]
    train_class_df = class_df.iloc[train_indices]

    # 将每个类别的子DataFrame添加到相应的集合中
    test_df = pd.concat([test_df, test_class_df], ignore_index=True)
    train_df = pd.concat([train_df, train_class_df], ignore_index=True)
# print(train_df, test_df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
# 转换成lstm输入形式
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).reshape(1120,1024,1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).reshape(280,1024,1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# 不设置drop_last=True的话模型训练会报错
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

class lstmclf(nn.Module):
    def __init__(self):
        super(lstmclf, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # h0(layers*directions, batch, hidden_size)
        h0 = torch.randn(2, 32, 8).to(device)
        c0 = torch.randn(2, 32, 8).to(device)
        # out(seq_len,batch,hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        y = self.fc(out)
        return y

model = lstmclf()
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(epochs):

    train_loss = 0
    train_accuracy = 0

    model.train()

    train_bar = tqdm(train_loader)

    for i, (inputs, labels) in enumerate(train_bar):
        train_bar.set_description("epoch {}".format(epoch + 1))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == labels).float().mean().cpu().numpy()
        train_accuracy += acc / len(train_loader)
        train_loss += loss.item() / len(train_loader)

    train_losses.append(train_loss)
    train_accs.append(train_accuracy)

    test_loss = 0
    test_acc = 0
    model.eval()

    with torch.no_grad():  # 不跟踪梯度
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss = loss.item()
            acc = (outputs.argmax(dim=1) == labels).float().mean().cpu().numpy()

            test_loss += loss / len(test_loader)
            test_acc += acc / len(test_loader)

    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print("=> loss: {:.4f}   acc: {:.4f}   test_loss: {:.4f}   test_acc: {:.4f}".
          format(train_loss, train_accuracy, test_loss, test_acc))

# 绘制损失函数和准确率曲线
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.plot(np.arange(len(train_losses)), train_losses,label="train loss")
plt.plot(np.arange(len(train_accs)), train_accs, label="train acc")
plt.plot(np.arange(len(test_losses)), test_losses, label="valid loss")
plt.plot(np.arange(len(test_accs)), test_accs, label="valid acc")
plt.legend() #显示图例
plt.xlabel('epoches')
#plt.ylabel("epoch")
plt.title('Model accuracy&loss')
plt.show()

# 准备绘制混淆矩阵
model.eval()
correct = 0
total = 0
y_test=[]
y_pred=[]
with torch.no_grad():  # 不跟踪梯度
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        for i in range(len(labels)):
            y_test.append(labels[i].item())
        outputs = model(images)
        outputs = torch.exp(outputs)
#         返回数组最大值以及最大值处的索引
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            y_pred.append(predicted[i].item())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('在测试集上的准确率为：{:.3f}%'.format(correct*100/total))
print(y_test)

C = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])  # 可将'1'等替换成自己的类别，如'cat'。

plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.xticks(range(0,4), labels=['正常','磨损','裂纹','断齿']) # 将x轴或y轴坐标，刻度 替换为文字/字符
plt.yticks(range(0,4), labels=['正常','磨损','裂纹','断齿'])
plt.show()
