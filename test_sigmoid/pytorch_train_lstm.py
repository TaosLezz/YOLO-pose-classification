import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from lstm_model import LSTMModel
# Đọc dữ liệu
normal_df = pd.read_csv(r'E:\aHieu\YOLO_pose_sleep\NORMAL.txt')
sleep_df = pd.read_csv(r'E:\aHieu\YOLO_pose_sleep\SLEEP.txt')
X = []
y = []
no_of_timesteps = 10


dataset = sleep_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)
dataset = normal_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
# Mã hóa one-hot cho nhãn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Chuyen du lieu thanh tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#tao dataloader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Đặt thông số
input_dim = X.shape[2]
hidden_dim = 50
num_layers = 4
output_dim = 1
num_epochs = 16
learning_rate = 0.001

# Khởi tạo mô hình, hàm mất mát và bộ tối ưu
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Đánh giá mô hình
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predicted = (outputs.squeeze() > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

# Lưu mô hình
torch.save(model.state_dict(), 'model1.pth')