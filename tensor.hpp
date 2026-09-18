#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data; // Buffer 1D

    // Construtor: Calc the total size
    Tensor(std::vector<int> s) : shape(s) {
        int total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size, 0.0f); // fill with zeros, Zero-Alloc
    }

    // return the total size of the tensor
    int size() const {
        return data.size();
    }
};

// dumb matmul (2d)
// A[M, K] * B[K, N] -> C[M, N]

void matmul_2d(const Tensor& A, const Tensor& B, Tensor& C){
    assert(A.shape.size() == 2 && B.shape.size() == 2 && C.shape.size() == 2);

    int M = A.shape[0];
    int K = A.shape[1];

    int K_B = B.shape[0];
    int N = B.shape[1];

    assert(K == K_B);
    assert(C.shape[0] == M && C.shape[1] == N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // Acesso ao array 1D fingindo que é 2D: indice = linha * num_colunas + coluna
                float a_val = A.data[i * K + k];
                float b_val = B.data[k * N + j];
                sum += a_val * b_val;
            }
            C.data[i * N + j] = sum;
        }
    }
}

struct RoPECache {
    std::vector<float> cos_cache;
    std::vector<float> sin_cache;

    // max_seq_len: Tamanho máximo do contexto (ex: 8192 para Llama 3)
    // head_dim: Tamanho de cada "cabeça" de atenção (ex: 128)
    RoPECache(int max_seq_len, int head_dim, float base = 500000.0f) { // Llama 3 usa base 500k
        cos_cache.resize(max_seq_len * (head_dim / 2));
        sin_cache.resize(max_seq_len * (head_dim / 2));

        for (int pos = 0; pos < max_seq_len; pos++) {
            for (int i = 0; i < head_dim / 2; i++) {
                // A fórmula insana do artigo original do RoPE
                float freq = 1.0f / std::pow(base, (float)(2 * i) / head_dim);
                float val = (float)pos * freq;
                
                int index = pos * (head_dim / 2) + i;
                cos_cache[index] = std::cos(val);
                sin_cache[index] = std::sin(val);
            }
        }
        std::cout << "RoPE Cache pre-computado com sucesso!\n";
    }
};

void apply_rope(float* q, float* k, int pos, int head_dim, const RoPECache& cache) {
    for (int i = 0; i < head_dim; i += 2) {
        // Encontrar o índice correto no cache pré-computado
        int cache_idx = pos * (head_dim / 2) + (i / 2);
        float cos_val = cache.cos_cache[cache_idx];
        float sin_val = cache.sin_cache[cache_idx];

        // Rotação da Query
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        // Rotação da Key
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

void softmax(std::vector<float>& x, int size) {
    // 1. find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // 2. Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // 3. Normalization (slash by sum)
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// q: Vetor Query atual [head_dim]
// k_cache / v_cache: Histórico das Keys e Values [seq_len, head_dim]
// out: Onde vamos guardar o resultado [head_dim]
void self_attention(float* q, float* k_cache, float* v_cache, float* out, int seq_len, int head_dim) {
    // Vetor temporário para guardar os scores de atenção (este sim, alocamos por segurança do tamanho do contexto)
    std::vector<float> scores(seq_len, 0.0f);
    
    // Fator de escala clássico da matemática do Transformer
    float scale = 1.0f / std::sqrt((float)head_dim);

    // Passo 1: Q * K^T (Quão parecido o token atual é com os anteriores?)
    for (int t = 0; t < seq_len; t++) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q[i] * k_cache[t * head_dim + i];
        }
        scores[t] = score * scale;
    }

    // Passo 2: Transformar scores em probabilidades
    softmax(scores, seq_len);

    // Passo 3: Multiplicar as probabilidades pela matriz de Values (V)
    for (int i = 0; i < head_dim; i++) {
        out[i] = 0.0f; // Limpa o lixo de memória
        for (int t = 0; t < seq_len; t++) {
            out[i] += scores[t] * v_cache[t * head_dim + i];
        }
    }
}