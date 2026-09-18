#include <iostream>
#include <chrono>
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"
#include "llama.hpp"

std::unordered_map<std::string, Tensor> load_weights_map(const std::string& metadata_path, ModelWeights& mmap_weights) {
    std::unordered_map<std::string, Tensor> tensors;
    std::ifstream file(metadata_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Falha ao abrir o ficheiro de metadados!");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string name;
        size_t offset;
        int num_elements;
        std::string shape_str;

        ss >> name >> offset >> num_elements >> shape_str;

        std::vector<int> shape;
        std::stringstream shape_ss(shape_str);
        std::string dim;
        while (std::getline(shape_ss, dim, ',')) {
            shape.push_back(std::stoi(dim));
        }

        float* tensor_ptr = mmap_weights.data + offset;
        tensors.emplace(name, Tensor(shape, tensor_ptr));
    }

    std::cout << "Fatiamento do modelo concluido! " << tensors.size() << " tensores prontos a usar.\n";
    return tensors;
}

int main() {
    std::cout << "Iniciando motor LLM...\n";
    
    try {
        ModelWeights weights("./tinyllama_weights.bin"); 
        auto model_tensors = load_weights_map("tinyllama_metadata.txt", weights);
        Tokenizer tokenizer("tokenizer.bin");
        
        Config config; 
        
        // Inicializa a tabela de senos e cossenos do RoPE
        RoPECache rope(config.max_seq_len, config.head_dim());

        std::cout << "\nMontando a arquitetura Transformer...\n";
        std::vector<TransformerBlock> layers;
        for (int i = 0; i < config.n_layers; i++) {
            layers.emplace_back(i, model_tensors, config);
        }
        std::cout << "Montagem concluida! " << layers.size() << " camadas prontas.\n";

        

        // ==========================================
        // LOOP AUTOREGRESSIVO COM BENCHMARK
        // ==========================================
        std::cout << "\n=======================================\n";
        std::cout << "             GERANDO TEXTO               \n";
        std::cout << "=========================================\n";

            //    std::string prompt_text =
            //"<|system|>\nVocê é um assistente útil.\n</s>\n"
            //"<|user|>\nOlá, quem é você?\n</s>\n"
            //"<|assistant|>\n";
            std::string prompt_text = "Pergunta: Olá, quem é você?\nResposta:";

        std::vector<int> prompt_ids = tokenizer.encode(prompt_text);
        prompt_ids.insert(prompt_ids.begin(), 1); // BOS

        std::vector<float> x(config.dim, 0.0f);
        std::vector<float> x_final(config.dim, 0.0f);
        std::vector<float> logits(config.vocab_size, 0.0f);

        Tensor& embed_tokens = model_tensors.at("model.embed_tokens.weight");
        Tensor& final_norm = model_tensors.at("model.norm.weight");
        Tensor& lm_head = model_tensors.at("lm_head.weight");

        int pos = 0;
        int token = 0;

        // --- PREFILL: passa o prompt inteiro pelo forward, sem imprimir nada ---
        for (size_t idx = 0; idx < prompt_ids.size(); idx++) {
            token = prompt_ids[idx];
            copy_vector(x.data(), embed_tokens.data + (token * config.dim), config.dim);
            for (int i = 0; i < config.n_layers; i++) {
                layers[i].forward(x.data(), pos, rope, config);
            }
            pos++;
        }

        std::cout << "\n debug: ";
        std::cout << prompt_text << "\n";

        std::cout << "IA: ";

        int total_tokens_gerados = 40;
        std::vector<int> generated_so_far;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < total_tokens_gerados; step++) {
            rms_norm(x_final.data(), x.data(), final_norm.data, config.dim);
            mat_vec_mul(logits.data(), x_final.data(), lm_head.data, config.vocab_size, config.dim);

            // --- PENALIDADE DE REPETIÇÃO ---
            for (int prev_tok : generated_so_far) {
                logits[prev_tok] -= 0.5f;
            }

            int next_token = sample_token(logits, config.vocab_size, 0.3f, 5);

            if (next_token == 2) break; // </s>

            generated_so_far.push_back(next_token); // <-- registra

            std::cout << tokenizer.decode(next_token) << std::flush;

            token = next_token;
            copy_vector(x.data(), embed_tokens.data + (token * config.dim), config.dim);
            for (int i = 0; i < config.n_layers; i++) {
                layers[i].forward(x.data(), pos, rope, config);
            }
            pos++;
        }

// PARA O CRONÓMETRO E CALCULA A VELOCIDADE
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double segundos = elapsed.count();
        double tokens_por_segundo = total_tokens_gerados / segundos;

        std::cout << "\n\n---------------------------------------\n";
        std::cout << "Estatísticas de Performance:\n";
        std::cout << "Tempo total: " << segundos << " segundos\n";
        std::cout << "Velocidade:  " << tokens_por_segundo << " tokens/segundo\n";
        std::cout << "---------------------------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL: " << e.what() << '\n';
        return 1;
    }

    return 0;
}