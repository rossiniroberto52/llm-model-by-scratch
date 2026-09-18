#include <iostream>
#include "tensor.hpp"

int main() {
    // Criando matriz A (2x3) e Matriz B (3x2)
    Tensor A({2, 3});
    Tensor B({3, 2});
    Tensor C({2, 2}); // O resultado será 2x2

    // Preenchendo com uns números aleatórios só pra testar
    A.data = {1.0, 2.0, 3.0, 
              4.0, 5.0, 6.0};
              
    B.data = {7.0, 8.0, 
              9.0, 10.0, 
              11.0, 12.0};

    matmul_2d(A, B, C);

    // C deve dar: 
    // [58, 64]
    // [139, 154]
    std::cout << "Resultado C[0,0]: " << C.data[0] << "\n";
    std::cout << "Resultado C[1,1]: " << C.data[3] << "\n";

    // Inicializando o cache do RoPE pro Llama 3
    RoPECache rope(8192, 128);

    // ==========================================
    // TESTE 2: Self-Attention (O Cérebro)
    // ==========================================
    int seq_len = 2;   // Vamos simular que a LLM já leu 2 tokens
    int head_dim = 4;  // Uma "cabeça" minúscula de 4 dimensões (na Llama 3 são 128)

    // O vetor Query (Q) do token atual que estamos tentando prever
    std::vector<float> q = {1.0f, 0.0f, 1.0f, 0.0f};

    // O KV Cache simulando o histórico da conversa (2 tokens)
    // Token 0 na primeira linha, Token 1 na segunda
    std::vector<float> k_cache = {
        1.0f, 0.1f, 0.1f, 0.1f, // Token 0
        0.1f, 1.0f, 0.1f, 1.0f  // Token 1
    };
    
    std::vector<float> v_cache = {
        10.0f, 10.0f, 10.0f, 10.0f, // Valores do Token 0
        20.0f, 20.0f, 20.0f, 20.0f  // Valores do Token 1
    };

    // Buffer para guardar o resultado
    std::vector<float> out(head_dim, 0.0f);

    std::cout << "\nTestando Self-Attention...\n";

    // 1. Aplicar RoPE no token atual (simulando que ele está na posição 1)
    // Passamos o ponteiro puro (.data()) para a nossa função
    apply_rope(q.data(), k_cache.data() + head_dim, 1, head_dim, rope);

    // 2. Rodar a Atenção
    self_attention(q.data(), k_cache.data(), v_cache.data(), out.data(), seq_len, head_dim);

    // 3. Imprimir resultado
    std::cout << "Resultado da Atencao (Out): [";
    for (int i = 0; i < head_dim; i++) {
        std::cout << out[i] << (i == head_dim - 1 ? "" : ", ");
    }
    std::cout << "]\n";

    return 0;
}