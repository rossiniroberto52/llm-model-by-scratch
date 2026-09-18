from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(len(tok))                     # provavelmente > 32000
print(tok.convert_tokens_to_ids("<|assistant|>"))   # id >= 32000?

print(tok.tokenize("<|assistant|>"))
print(tok.tokenize("Olá, quem é você?"))
print(repr(tok.convert_ids_to_tokens([300, 301, 302])))  # olha alguns tokens crus do vocab
print(tok.convert_tokens_to_ids("</s>"))