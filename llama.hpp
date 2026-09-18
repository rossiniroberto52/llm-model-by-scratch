#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include "tensor.hpp"

// Hiperparâmetros oficiais do TinyLlama-1.1B
struct Config {
    int dim = 2048;          // Tamanho do embedding
    int hidden_dim = 5632;   // Tamanho da camada densa (MLP)
    int n_layers = 22;       // Quantidade de blocos Transformer
    int n_heads = 32;        // Cabeças de Query (Q)
    int n_kv_heads = 4;      // Cabeças de Key/Value (K, V) -> O famoso GQA!
    int vocab_size = 32000;
    int max_seq_len = 2048;  // Tamanho máximo do contexto (memória)
    
    // Função auxiliar que calcula o tamanho de cada "cabeça"
    int head_dim() const { return dim / n_heads; } 
};

class TransformerBlock {
public:
    // Ponteiros dos pesos
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq; float* wk; float* wv; float* wo;
    float* w1; float* w2; float* w3;

    // Memória do Chat (KV Cache)
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    // --- BUFFERS TEMPORÁRIOS (Alocados 1 vez só, Zero Alocação no Forward!) ---
    std::vector<float> norm_x, q, k, v, attn_out, ffn_gate, ffn_up, ffn_down;

    TransformerBlock(int layer_idx, const std::unordered_map<std::string, Tensor>& weights, const Config& cfg) {
        std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

        rms_att_weight = weights.at(prefix + "input_layernorm.weight").data;
        rms_ffn_weight = weights.at(prefix + "post_attention_layernorm.weight").data;
        
        wq = weights.at(prefix + "self_attn.q_proj.weight").data;
        wk = weights.at(prefix + "self_attn.k_proj.weight").data;
        wv = weights.at(prefix + "self_attn.v_proj.weight").data;
        wo = weights.at(prefix + "self_attn.o_proj.weight").data;
        
        w1 = weights.at(prefix + "mlp.gate_proj.weight").data;
        w2 = weights.at(prefix + "mlp.down_proj.weight").data;
        w3 = weights.at(prefix + "mlp.up_proj.weight").data;

        // KV Cache
        int kv_dim = cfg.n_kv_heads * cfg.head_dim();
        int kv_size = cfg.max_seq_len * kv_dim;
        k_cache.resize(kv_size, 0.0f);
        v_cache.resize(kv_size, 0.0f);

        // Alocando os Buffers pro Forward
        norm_x.resize(cfg.dim, 0.0f);
        q.resize(cfg.dim, 0.0f);
        k.resize(kv_dim, 0.0f);
        v.resize(kv_dim, 0.0f);
        attn_out.resize(cfg.dim, 0.0f);
        ffn_gate.resize(cfg.hidden_dim, 0.0f);
        ffn_up.resize(cfg.hidden_dim, 0.0f);
        ffn_down.resize(cfg.dim, 0.0f);
    }

    // A MÁGICA ACONTECE AQUI
    // x: vetor do token atual (tamanho dim). pos: posição na frase.
    void forward(float* x, int pos, const RoPECache& rope, const Config& cfg) {
        int dim = cfg.dim;
        int hidden_dim = cfg.hidden_dim;
        int head_dim = cfg.head_dim();
        int kv_dim = cfg.n_kv_heads * head_dim;

        // ==========================================
        // 1. BLOCO DE ATENÇÃO
        // ==========================================
        // Normaliza a entrada
        rms_norm(norm_x.data(), x, rms_att_weight, dim);

        // Projeta Q, K, V
        mat_vec_mul(q.data(), norm_x.data(), wq, dim, dim);
        mat_vec_mul(k.data(), norm_x.data(), wk, kv_dim, dim);
        mat_vec_mul(v.data(), norm_x.data(), wv, kv_dim, dim);

        // Aplica o RoPE para a IA saber a ordem das palavras
        apply_rope(q.data(), k.data(), pos, head_dim, rope);

        // Salva K e V no Cache da frase inteira
        int cache_offset = pos * kv_dim;
        copy_vector(k_cache.data() + cache_offset, k.data(), kv_dim);
        copy_vector(v_cache.data() + cache_offset, v.data(), kv_dim);

        // Self-Attention com GQA (Grouped Query Attention)
        int q_heads_per_kv = cfg.n_heads / cfg.n_kv_heads; // No TinyLlama = 8
        
        for (int h = 0; h < cfg.n_heads; h++) {
            // Descobre qual cache KV essa cabeça Q vai usar
            int kv_h = h / q_heads_per_kv;
            
            // Ponteiros pro miolo exato de cada cabeça
            float* q_head = q.data() + h * head_dim;
            float* out_head = attn_out.data() + h * head_dim;
            
            // Aqui a gente precisaria de uma versão customizada do self_attention que 
            // saiba pular de linha no cache (stride). Como é um pouco complexo, 
            // pra rodar AGORA, vamo fazer o self_attention mais brutal direto aqui:
            std::vector<float> scores(pos + 1, 0.0f);
            float scale = 1.0f / std::sqrt((float)head_dim);

            for (int t = 0; t <= pos; t++) {
                float* k_head = k_cache.data() + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_head[i];
                }
                scores[t] = score * scale;
            }

            softmax(scores, pos + 1);

            for (int i = 0; i < head_dim; i++) {
                out_head[i] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    float* v_head = v_cache.data() + t * kv_dim + kv_h * head_dim;
                    out_head[i] += scores[t] * v_head[i];
                }
            }
        }

        // Projeção final da atenção e Conexão Residual (soma com a entrada x)
        mat_vec_mul(norm_x.data(), attn_out.data(), wo, dim, dim); // Reusando norm_x como buffer temp
        add_vectors(x, x, norm_x.data(), dim);

        // ==========================================
        // 2. BLOCO DENSO (FEED FORWARD / SwiGLU)
        // ==========================================
        rms_norm(norm_x.data(), x, rms_ffn_weight, dim);

        // SwiGLU: gate = silu(w1 * x) * (w3 * x)
        mat_vec_mul(ffn_gate.data(), norm_x.data(), w1, hidden_dim, dim);
        silu(ffn_gate.data(), hidden_dim);
        
        mat_vec_mul(ffn_up.data(), norm_x.data(), w3, hidden_dim, dim);
        mul_vectors(ffn_gate.data(), ffn_gate.data(), ffn_up.data(), hidden_dim); // Reusando ffn_gate pro resultado

        // Projeção final pra voltar pro tamanho original
        mat_vec_mul(ffn_down.data(), ffn_gate.data(), w2, dim, hidden_dim);
        
        // Última Conexão Residual
        add_vectors(x, x, ffn_down.data(), dim);
    }
};