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

# Инициализация модели и оптимизатора
input_dim = 3
hidden_dim = 64
output_dim = 1
model = CalculatorModel(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Функция потерь
criterion = nn.L1Loss()

# Генерация данных для обучения
num_samples = 100000
input_data = np.random.randint(1, 100, (num_samples, 2)) 
operation_data = np.random.randint(0, 3, (num_samples, 1))
input_data = np.concatenate((input_data, operation_data), axis=1)

# Вычисление целевых данных
target_data = np.zeros((num_samples, 1))
for i in range(num_samples):
    if operation_data[i] == 0:  # сложение
        result = input_data[i, 0] + input_data[i, 1]
    elif operation_data[i] == 1:  # вычитание
        result = input_data[i, 0] - input_data[i, 1]

    if abs(result) < 1e6:
        target_data[i] = np.round(result, 10)

# Обучение модели
for epoch in range(100000):  # количество эпох
    inputs = torch.from_numpy(input_data).float()
    targets = torch.from_numpy(target_data).float().view(-1, 1)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        example_index = np.random.randint(num_samples)
        example_input = input_data[example_index]
        example_target = target_data[example_index]
        example_output = model(torch.from_numpy(example_input).float().unsqueeze(0))
        print(f'Example: {example_input}, Target: {example_target}, Output: {example_output.item()}')

    torch.save(model.state_dict(), 'D:\\AI-test\\AI-test\\model\\model_1.pth')

# Тестирование модели
test_data = np.array([[5, 10, 0], [20, 30, 1], [50, 50, 2]])
test_inputs = torch.from_numpy(test_data).float()
test_outputs = model(test_inputs)
print(test_outputs.item())