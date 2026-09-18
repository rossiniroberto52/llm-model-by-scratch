#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

struct Tokenizer {
    std::vector<std::string> vocab;
    int vocab_size;

    Tokenizer(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Falha ao abrir o tokenizer.bin!");

        // Lê o tamanho do vocabulário (os primeiros 4 bytes)
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        vocab.resize(vocab_size);

        // Lê token a token
        for (int i = 0; i < vocab_size; i++) {
            int len;
            file.read(reinterpret_cast<char*>(&len), sizeof(int));
            
            std::string token(len, ' ');
            file.read(&token[0], len);
            vocab[i] = token;
        }
        std::cout << "Tokenizador carregado! Vocabulario: " << vocab_size << " palavras.\n";
    }

    // A função mágica que transforma o cuspe da rede neural em texto legível
    std::string decode(int id) {
        if (id < 0 || id >= vocab_size) return "";
        
        std::string token = vocab[id];
        
        // O Llama usa uns tokens de sistema para quebra de linha tipo "<0x0A>"
        // Vamos limpá-los para imprimir bonitinho no terminal
        if (token == "<0x0A>") return "\n";
        
        return token;
    }
};