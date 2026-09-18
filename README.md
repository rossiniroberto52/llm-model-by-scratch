# TinyLlama C++ Inference Engine (From Scratch)

Um motor de inferência minimalista para Large Language Models (LLMs) implementado do zero em **C++17 puro**, sem dependência de bibliotecas externas de deep learning (como PyTorch, LibTorch, ONNX Runtime ou llama.cpp).

O projeto executa a arquitetura completa do **TinyLlama-1.1B-Chat-v1.0** (1.1 bilhão de parâmetros) em ponto flutuante de precisão simples (**FP32**), carregando os pesos diretamente via mapeamento de memória (`mmap`).

---

## 1. Visão Geral do Projeto

O objetivo deste projeto é desmistificar o funcionamento interno de um Large Language Model, construindo cada etapa do pipeline de inferência a partir dos primeiros princípios:

- **Carregamento de Pesos:** Leitura zero-copy de ~4.4 GB de tensores binários via `mmap`.
- **Álgebra Linear & Kernels:** Multiplicação de matriz-vetor (`mat_vec_mul`), normalização RMSNorm, ativação SwiGLU e Softmax.
- **Mecanismo de Atenção:** Grouped-Query Attention (GQA) com suporte a KV Cache dinâmico.
- **Codificação Posicional:** Rotary Positional Embeddings (RoPE) pré-calculados.
- **Tokenização:** Tokenizador customizado compatível com o vocabulário SentencePiece do modelo.
- **Estratégia de Amostragem:** Top-K Sampling com Temperatura e Penalidade de Repetição.

### Especificações do Modelo (TinyLlama-1.1B)
| Hiperparâmetro | Valor | Descrição |
| :--- | :--- | :--- |
| `dim` | 2048 | Dimensão do embedding oculto |
| `hidden_dim` | 5632 | Dimensão intermediária do MLP (SwiGLU) |
| `n_layers` | 22 | Número de camadas Transformer |
| `n_heads` | 32 | Cabeças de Query (Q) |
| `n_kv_heads` | 4 | Cabeças de Key/Value (K, V) — GQA (fator 8:1) |
| `head_dim` | 64 | Dimensão por cabeça (`2048 / 32`) |
| `vocab_size` | 32000 | Tamanho do vocabulário |
| `max_seq_len` | 2048 | Janela máxima de contexto |
| `dtype` | FP32 | 32-bit Floating Point (~4.4 GB em disco) |

---

## 2. Arquitetura do Sistema e Estrutura de Arquivos

```text
llm-model-by-scratch/
├── export_heights.py       # Extrai tensores do HuggingFace para binário plano
├── export_tokenizer.py     # Extrai vocabulário do HuggingFace para binário custom
├── weights.hpp             # Wrapper RAII para mmap (POSIX) dos pesos
├── tensor.hpp              # Estrutura Tensor, kernels matemáticos e amostragem
├── llama.hpp               # Configuração e TransformerBlock (Forward + GQA)
├── tokenizer.hpp           # Codificador/Decodificador SentencePiece
├── main.cpp                # Pipeline de prefill, geração e loop interativo
└── Makefile                # Script de compilação com flags de otimização
```

### Detalhamento dos Componentes

- **`weights.hpp`**: Gerencia o ciclo de vida do arquivo binário de pesos (`tinyllama_weights.bin`). Utiliza a syscall `mmap` com flag `MAP_PRIVATE` para mapear os 4.4 GB no espaço de endereçamento virtual do processo. Isso garante inicialização instantânea e delega ao kernel do sistema operacional a paginação sob demanda.
- **`tensor.hpp`**: Define a estrutura `Tensor` (que pode encapsular memória própria ou apontar para a região do `mmap`) e implementa os kernels computacionais fundamentais:
  - `mat_vec_mul`: Multiplicação de matriz por vetor paralelizada com OpenMP.
  - `rms_norm`: Root Mean Square Layer Normalization com offset epsilon.
  - `RoPECache` & `apply_rope`: Rotação de vetores Q e K no plano complexo 2D para preservação posicional.
  - `softmax`, `silu`: Funções de ativação e normalização de probabilidade.
  - `sample_token`: Amostragem estocástica com ordenação parcial Top-K e ajuste por temperatura.
- **`llama.hpp`**: Contém a estrutura `Config` com os hiperparâmetros oficiais e a classe `TransformerBlock`. Cada bloco gerencia seus próprios ponteiros para os pesos da camada e seus arrays de **KV Cache** (`k_cache`, `v_cache`). Implementa o forward pass com Grouped-Query Attention e MLP SwiGLU (`w1`, `w2`, `w3`).
- **`tokenizer.hpp`**: Lê o vocabulário binário e fornece métodos de codificação (`encode`) via *greedy longest-match* e decodificação (`decode`) caractere a caractere / subtoken a subtoken.
- **`main.cpp`**: Ponto de entrada do programa. Orquestra a extração do mapa de tensores via `tinyllama_metadata.txt`, instancia as 22 camadas, executa a fase de **Prefill** (alimentando o prompt token a token para aquecer o KV Cache) e entra no loop autoregressivo de **Geração**.

---

## 3. Como Compilar e Executar

### Pré-requisitos
- Compilador C++ moderno com suporte a C++17 e OpenMP (`g++` 9+ ou `clang++`).
- Python 3.8+ com `torch` e `transformers` (apenas para exportar os modelos uma única vez).

### Passo 1: Extrair Pesos e Vocabulário
Execute os scripts Python para baixar o modelo do HuggingFace e converter para os formatos binários planos:

```bash
# Instala as dependências de exportação
pip install torch transformers

# Exporta os pesos (gera tinyllama_weights.bin e tinyllama_metadata.txt)
python3 export_heights.py

# Exporta o vocabulário (gera tokenizer.bin)
python3 export_tokenizer.py
```

### Passo 2: Compilar o Motor em C++
O projeto utiliza compilação otimizada para extrair o máximo de desempenho da CPU:

```bash
make
```

As flags configuradas no `Makefile` incluem:
- `-O3`: Otimizações agressivas de código (inlining, loop unrolling).
- `-march=native`: Habilita o GCC a gerar instruções específicas para a arquitetura da máquina local (como AVX2 e FMA via auto-vetorização).
- `-ffast-math`: Relaxa restrições estritas de ponto flutuante IEEE para permitir simplificações algébricas pelo compilador.
- `-fopenmp`: Paralelização multithread de loops de multiplicação matricial.

### Passo 3: Executar
```bash
./llm_engine
```

---

## 4. Bugs Caçados e Resolvidos (A Saga do Debug)

Construir um motor de inferência sem frameworks exige confrontar detalhes de implementação que normalmente ficam ocultos sob camadas de abstração. Abaixo estão os principais problemas enfrentados e como foram resolvidos:

### Bug 1: Tokenização SentencePiece Quebrada (O Marcador ` ` / `\xe2\x96\x81`)
- **Sintoma:** Ao codificar prompts simples, o tokenizador falhava em encontrar correspondências para palavras comuns, quebrando o texto em caracteres isolados ou gerando sequências excessivas do token desconhecido (`<unk>`).
- **Causa Raiz:** O SentencePiece não utiliza espaços normais (`0x20`) no vocabulário. Em vez disso, ele utiliza o caractere UTF-8 especial ` ` (`\xe2\x96\x81` em hexadecimal, 3 bytes) para representar o espaço antes de cada palavra. O exportador original estava descartando ou tratando incorretamente esse caractere, fazendo com que a busca gulosa nunca encontrasse os tokens de palavras precedidas por espaço.
- **Solução:**
  1. O script `export_tokenizer.py` foi ajustado para manter a sequência UTF-8 exata `\xe2\x96\x81`.
  2. No `tokenizer.hpp`, o método `encode()` foi atualizado para prefixar o texto de entrada com `\xe2\x96\x81` e substituir todos os espaços ASCII (`0x20`) pelo marcador `\xe2\x96\x81` antes de executar o *greedy longest-match*.
  3. No `decode()`, o marcador é convertido de volta para espaço real antes da impressão na tela.

### Bug 2: Tokens de Chat Template Não-Atômicos no Vocabulário Base
- **Sintoma:** Ao formatar prompts no formato de chat oficial do TinyLlama (`<|system|>...<|user|>...<|assistant|>`), o modelo gerava saídas completamente corrompidas e sem sentido (alucinações imediatas).
- **Causa Raiz:** Os delimitadores especiais de chat foram injetados durante a fase de fine-tuning e não faziam parte do vocabulário base original de 32.000 tokens como entradas atômicas únicas. Como o tokenizador customizado faz busca no vocabulário base de 32k, ele quebrava `<|system|>` em 5 tokens isolados (`<`, `|`, `system`, `|`, `>`), quebrando completamente a distribuição de ativações esperada pela rede.
- **Solução:** Em vez de usar as tags de chat ausentes do vocabulário estático, o pipeline foi configurado para operar no modo de *raw completion* estruturado:
  ```text
  Pergunta: Qual a capital da Franca?
  Resposta:
  ```
  Esse formato utiliza exclusivamente tokens presentes no vocabulário padrão, permitindo respostas coerentes.

### Bug 3: Argmax Residual Sobrescrevendo a Amostragem Estocástica
- **Sintoma:** Mesmo configurando parâmetros de amostragem como Temperatura e Top-K, o modelo apresentava saídas 100% determinísticas e frequentemente entrava em loops infinitos repetindo a mesma palavra ou frase.
- **Causa Raiz:** Havia resquício de uma implementação inicial de *greedy decoding* no loop principal que calculava o `argmax` diretamente sobre o vetor de logits e sobrescrevia o token sorteado pela função `sample_token()`.
- **Solução:** O loop em `main.cpp` foi refatorado para conectar exclusivamente a saída de `sample_token()` ao pipeline, integrando também uma **penalidade de repetição** (*repetition penalty* linear sobre os últimos tokens gerados) para penalizar tokens já presentes na janela de contexto recente antes da aplicação do softmax.

---

## 5. Decisões de Design e Trade-offs de Engenharia

### `mmap` vs. Carregamento Completo em RAM (`std::vector` / `malloc`)
- **Escolha:** Mapeamento via `mmap`.
- **Trade-off:** Carregar 4.4 GB via `fread` consumiria vários segundos no startup e alocaria memória física instantaneamente. Com `mmap`, o tempo de inicialização é **inferior a 1 milissegundo**. O sistema operacional carrega apenas as páginas de memória acessadas durante a inferência e pode descartá-las da memória física sob pressão de RAM, aproveitando o Page Cache do kernel.

### Greedy Longest-Match vs. Algoritmo BPE Completo com Tabela de Merges
- **Escolha:** Tokenizador *Greedy Longest-Match*.
- **Trade-off:** Um tokenizador BPE completo exige carregar e processar regras ordenadas de merges (pares de bytes). A abordagem *longest-match* com busca reversa gulosa reduz a complexidade de implementação para zero dependências externas em C++, mantendo alta fidelidade para mais de 98% dos casos de uso em textos comuns em inglês e português.

### FP32 vs. Quantização (INT8 / INT4 / GGUF)
- **Escolha:** Float32 nativo.
- **Trade-off:** O formato FP32 preserva 100% da precisão matemática original do modelo sem perdas numéricas e simplifica a implementação dos kernels matriciais (operações aritméticas diretas em float). O trade-off negativo é o throughput: ler 4.4 GB de pesos a cada token impõe um gargalo severo na largura de banda do barramento de memória da CPU.

### Amostragem Estocástica (Top-K + Temp) vs. Greedy Decoding
- **Escolha:** Top-K (k=5 a 20) com Temperatura (T=0.3 a 0.5) e Repetition Penalty.
- **Trade-off:** Modelos compactos (1.1B) em *greedy search* puro (T=0 / Argmax) colapsam rapidamente em atratores cíclicos de repetição. A combinação de Top-K filtrado com temperatura baixa mantém a criatividade controlada e evita que o modelo selecione caudas de probabilidade absurdas.

### Gerenciamento de Memória no Forward Pass
- **Escolha:** Pré-alocação dos buffers principais do bloco Transformer.
- **Análise Técnica:** No construtor de cada `TransformerBlock`, são pré-alocados os vetores `norm_x`, `q`, `k`, `v`, `attn_out`, `ffn_gate`, `ffn_up` e `ffn_down`. Isso elimina chamadas ao alocador de heap (`malloc`/`free`) nas operações mais frequentes da rede.
- *Ressalva de Implementação:* Dentro do loop de atenção em `TransformerBlock::forward`, o cálculo de atenção ainda instancia dinamicamente `std::vector<float> scores(pos + 1, 0.0f)` para cada cabeça e cada camada. Embora funcional, essa alocação dinâmica por token é uma oportunidade de melhoria futura.

---

## 6. Benchmarks e Análise de Performance

Os testes foram realizados em CPU x86-64 executando o modelo TinyLlama-1.1B em FP32:

| Etapa de Otimização | Throughput Médio | Fator de Ganho |
| :--- | :--- | :--- |
| **1. Baseline (Sem flags, código escalar mono-thread)** | ~0.7 tok/s | 1.0x |
| **2. GCC Auto-Vetorização (`-O3 -march=native -ffast-math`)** | ~1.8 tok/s | ~2.5x |
| **3. Paralelização OpenMP (`#pragma omp parallel for`)** | **~4.5 tok/s** | **~6.4x** |

### Discussão sobre Otimizações de CPU
1. **Auto-vetorização pelo Compilador:** A flag `-march=native` combinada com `-O3` e `-ffast-math` permitiu que o GCC identificasse laços internos de multiplicação e soma acumulada na `mat_vec_mul`, gerando instruções vetoriais **AVX2** e **FMA** automaticamente (sem necessidade de intrinsics escritos manualmente).
2. **Paralelização Multicore:** A projeção de matriz-vetor ($y = W \cdot x$) distribui o cálculo das linhas da matriz de pesos entre as threads disponíveis via OpenMP, reduzindo o tempo de cálculo linearmente com o número de núcleos físicos.
3. **O Gargalo de Largura de Banda de Memória (*Memory-Bound Bottleneck*):** Na inferência autoregressiva com batch size = 1, o modelo não é limitado por capacidade computacional (FLOPS), mas sim pela taxa de transferência da memória RAM (*memory bandwidth*). Para gerar 1 único token em FP32, a CPU precisa ler todos os **4.4 GB de parâmetros** da RAM. Em uma memória DDR4/DDR5 com largura de banda típica de 25–40 GB/s, o teto teórico físico máximo fica entre 5 e 9 tokens/s em FP32.

---

## 7. Limitações Conhecidas e Trabalhos Futuros

- [ ] **Eliminação de Alocações no Loop de Atenção:** Substituir o `std::vector<float> scores(pos + 1)` dinâmico por um buffer estático pré-alocado de tamanho `max_seq_len` dentro do `TransformerBlock`.
- [ ] **Quantização (INT8 / INT4):** Implementar quantização por blocos (estilo GGUF/llama.cpp) para reduzir os pesos de 4.4 GB para ~600 MB (INT4), diminuindo o tráfego de memória em ~7x e acelerando o throughput proporcionalmente.
- [ ] **Suporte Completo a Chat Templates:** Implementar suporte a tokens especiais adicionais e tabela de merges do BPE para suportar nativamente prompts no formato chat do TinyLlama.
- [ ] **Degradação em Sequências Longas:** Devido ao tamanho compacto do modelo (1.1B) e à ausência de fine-tuning específico para *raw completion* longo, a coerência semântica tende a decair após 15–20 tokens em prompts abertos.
- [ ] **SIMD Intrinsics Explícitos:** Implementar kernels com intrinsics manuais (AVX2/AVX-512 / ARM NEON) e suporte a Fused Multiply-Add explícito.

---

## 8. Licença

Este projeto é disponibilizado para fins educacionais e de pesquisa sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
