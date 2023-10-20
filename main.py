import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Модель
class CalculatorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CalculatorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
input_dim = 3
hidden_dim = 64
output_dim = 1
# Загрузка модели
model = CalculatorModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('calculator_model.pth'))
model.eval()  # Переводим модель в режим оценки

# Использование модели
data = np.array([[5, 10, 0]])  # Пример данных
inputs = torch.from_numpy(data).float()
outputs = model(inputs)
print(outputs)