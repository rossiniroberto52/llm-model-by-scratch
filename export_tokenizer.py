from transformers import AutoTokenizer
import struct

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading Tokenizer fromm: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

vocab_size = 32000
vocab = [""] * vocab_size

for token, idx in tokenizer.get_vocab().items():
    if idx < vocab_size:
        # Substituímos o caractere especial " " (usado pelo SentencePiece para espaços)
        # por um espaço real para facilitar a nossa vida no C++
        vocab[idx] = token.replace(' ', ' ')

with open("tokenizer.bin", "wb") as f:
    # 1. Escrevemos o tamanho total do vocabulário (int de 32 bits)
    f.write(struct.pack("i", vocab_size))
    
    # 2. Escrevemos cada token
    for i in range(vocab_size):
        token_bytes = vocab[i].encode('utf-8')
        f.write(struct.pack("i", len(token_bytes))) # Escreve o tamanho
        f.write(token_bytes)                        # Escreve os bytes

print("tokenizer.bin gerado com sucesso!")