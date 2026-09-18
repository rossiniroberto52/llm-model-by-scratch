import torch
from transformers import AutoModelForCausalLM
import os

# O modelo mais leve e moderno da Meta (tem ~1.2 bilhões de parâmetros)
model_id = "meta-llama/Llama-3.2-1B"
output_file = "llama_weights.bin"

print(f"Baixando/Carregando {model_id} do Hugging Face...")
# Forçamos float32 para facilitar a nossa vida no C++ agora (evita lidar com conversão de ponteiros de 16-bits)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

print(f"\nExtraindo matrizes e exportando para {output_file}...")
with open(output_file, "wb") as f:
    state_dict = model.state_dict()
    
    for name, tensor in state_dict.items():
        # Imprime o nome da camada e as dimensões só pra acompanharmos o progresso
        print(f"Gravando: {name: <40} | Shape: {list(tensor.shape)}")
        
        # Converte o tensor do PyTorch num array Numpy linear e escreve os bytes crus no disco
        f.write(tensor.numpy().tobytes())

# Pega o tamanho do arquivo gerado
tamanho_gb = os.path.getsize(output_file) / (1024 ** 3)
print(f"\nFeito! O monstro foi engarrafado: {tamanho_gb:.2f} GB de tensores crus.")