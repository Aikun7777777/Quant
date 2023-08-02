import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 假设你已经有了一个Pandas DataFrame，其中包含比特币的历史价格
# df = pd.read_csv('your_bitcoin_data.csv')

# 为了简化，我们创建一个假设的数据
data = {
    'date': pd.date_range(start='1/1/2023', periods=30),
    'price': [45000, 45300, 44700, 45500, 46000, 46500, 47000, 47500, 48000, 48500, 
              49000, 49500, 50000, 50500, 51000, 51500, 52000, 52500, 53000, 53500, 
              54000, 54500, 55000, 55500, 56000, 56500, 57000, 57500, 58000, 58500]
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# 数据预处理
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(df['price'].values.reshape(-1, 1))

# 创建数据集
def create_dataset(data, look_back=5):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 5
dataX, dataY = create_dataset(data_normalized, look_back)

# 划分训练集和测试集
train_size = int(len(dataY) * 0.67)
test_size = len(dataY) - train_size
trainX = torch.Tensor(dataX[:train_size])
trainY = torch.Tensor(dataY[:train_size])
testX = torch.Tensor(dataX[train_size:])
testY = torch.Tensor(dataY[train_size:])

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# 训练模型
num_epochs = 20000
learning_rate = 0.0001
input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

lstm = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm(trainX.unsqueeze(-1))
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# 预测未来7天的价格
lstm.eval()
train_predict = lstm(torch.Tensor(dataX).unsqueeze(-1))

data_predict = train_predict.data.numpy()
dataY_plot = dataY

data_predict = scaler.inverse_transform(data_predict)
dataY_plot = scaler.inverse_transform(dataY_plot.reshape(-1, 1))

# 打印预测的价格
print(data_predict)
