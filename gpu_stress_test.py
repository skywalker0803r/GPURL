# filename: gpu_stress_test.py

import torch
import torch.nn as nn
import torch.optim as optim

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定義一個較大的模型讓GPU有更多運算量
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

# 初始化模型和優化器
model = LargeModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 模擬的輸入與目標資料
x = torch.randn(64, 4096).to(device)
y = torch.randn(64, 4096).to(device)

# 無限循環進行訓練，直到手動中斷（Ctrl+C）
try:
    while True:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")
except KeyboardInterrupt:
    print("Stopped by user.")
