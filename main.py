from transformers import LlamaForCausalLM, LlamaConfig
import torch
# Создаем конфигурацию для модели
config = LlamaConfig(
    vocab_size=4024,  # размер словаря
    d_model=512,  # размерность векторов слов
    num_attention_heads=8,  # количество голов внимания
    num_layers=12,  # количество слоев в модели
    eos_token_id=4023,  # ID токена конца предложения
    pad_token_id=4023,  # ID токена паддинга
)

# Создаем модель с этой конфигурацией
model = LlamaForCausalLM(config)

# Сохраняем модель
torch.save(model.state_dict(), 'D:\\AI-test\\model\\model.pth')