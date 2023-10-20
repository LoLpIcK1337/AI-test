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
criterion = nn.MSELoss()

# Генерация данных для обучения
num_samples = 10000
input_data = np.random.randint(1, 100, (num_samples, 2))  # избегаем нулей
operation_data = np.random.randint(0, 6, (num_samples, 1))
input_data = np.concatenate((input_data, operation_data), axis=1)

# Вычисление целевых данных
target_data = np.zeros((num_samples, 1))
for i in range(num_samples):
    if operation_data[i] == 0:  # сложение
        result = input_data[i, 0] + input_data[i, 1]
    elif operation_data[i] == 1:  # вычитание
        result = input_data[i, 0] - input_data[i, 1]
    elif operation_data[i] == 2:  # умножение
        result = input_data[i, 0] * input_data[i, 1]
    elif operation_data[i] == 3:  # деление

        if input_data[i, 1] != 0:  # избегаем деления на ноль
            result = input_data[i, 0] / input_data[i, 1]
    elif operation_data[i] == 4:  # возведение в степень
        result = input_data[i, 0] ** input_data[i, 1]

    elif operation_data[i] == 5:  # логарифмирование

        if input_data[i, 0] > 0:  # избегаем логарифма от нуля
            result = np.log(input_data[i, 0])
    elif operation_data[i] == 6:  # квадратный корень

        if input_data[i, 0] >= 0:  # избегаем корня от отрицательного числа
            result = np.sqrt(input_data[i, 0])

    elif operation_data[i] == 7:  # экспонента
        result = np.exp(input_data[i, 0])

    # Ограничиваем размер чисел до 6 цифр и округляем до 10 знаков после точки
    if abs(result) < 1e6:
        target_data[i] = np.round(result, 10)

# Обучение модели
for epoch in range(1000):  # количество эпох
    inputs = torch.from_numpy(input_data).float()
    targets = torch.from_numpy(target_data).float().view(-1, 1)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Выводим loss каждые 10 эпох
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Выводим пример и ответ
        example_index = np.random.randint(num_samples)  # выбираем случайный пример
        example_input = input_data[example_index]
        example_target = target_data[example_index]
        example_output = model(torch.from_numpy(example_input).float().unsqueeze(0))
        print(f'Example: {example_input}, Target: {example_target}, Output: {example_output.item()}')

    torch.save(model.state_dict(), 'D:\\AI-test\\AI-test\\model\\model_1.pth')

# Тестирование модели
test_data = np.array([[5, 10, 0], [20, 30, 1], [50, 50, 2]])
test_inputs = torch.from_numpy(test_data).float()
test_outputs = model(test_inputs)
print(test_outputs)