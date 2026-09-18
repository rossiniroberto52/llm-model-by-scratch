import torch
from transformers import AutoModelForCausalLM
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_bin = "tinyllama_weights.bin"
output_meta = "tinyllama_metadata.txt"

print(f"Carregando {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

offset = 0 # Em quantidade de floats, não bytes! Facilita muito no C++
with open(output_bin, "wb") as f_bin, open(output_meta, "w") as f_meta:
    state_dict = model.state_dict()
    
    for name, tensor in state_dict.items():
        num_elements = tensor.numel()
        shape_str = ",".join(map(str, tensor.shape))
        
        # Salva o mapa: Nome | Offset | Tamanho | Dimensões
        f_meta.write(f"{name} {offset} {num_elements} {shape_str}\n")
        
        # Salva os bytes crus
        f_bin.write(tensor.numpy().tobytes())
        
        offset += num_elements

print("Pesos e metadados exportados com sucesso!")