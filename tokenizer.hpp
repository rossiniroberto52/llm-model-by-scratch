#include <unordered_map>
#include <algorithm>

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab_to_id;
    int vocab_size;
    int unk_id = 0;

    const std::string SPACE_MARKER = "\xe2\x96\x81"; // ▁ em UTF-8 (3 bytes)

    Tokenizer(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Falha ao abrir o tokenizer.bin!");

        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        vocab.resize(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            int len;
            file.read(reinterpret_cast<char*>(&len), sizeof(int));
            std::string token(len, ' ');
            file.read(&token[0], len);
            vocab[i] = token;
            vocab_to_id[token] = i;
        }

        auto it = vocab_to_id.find("<unk>");
        if (it != vocab_to_id.end()) unk_id = it->second;

        std::cout << "Tokenizador carregado! Vocabulario: " << vocab_size << " palavras.\n";
    }

    // decode: troca o marcador ▁ por espaço real na hora de imprimir
    std::string decode(int id) {
        if (id < 0 || id >= vocab_size) return "";
        std::string token = vocab[id];
        if (token == "<0x0A>") return "\n";

        size_t pos = token.find(SPACE_MARKER);
        if (pos != std::string::npos) {
            token.replace(pos, SPACE_MARKER.size(), " ");
        }
        return token;
    }

    // encode: prefixa a entrada com ▁ e troca espaços normais por ▁,
    // depois faz greedy longest-match igual antes
    std::vector<int> encode(const std::string& raw_text) {
        std::string text = SPACE_MARKER + raw_text; // SentencePiece sempre prefixa a 1ª palavra

        // troca cada espaço normal (0x20) por ▁ (3 bytes)
        std::string text_marked;
        for (char c : text) {
            if (c == ' ') text_marked += SPACE_MARKER;
            else text_marked += c;
        }

        std::vector<int> ids;
        size_t i = 0;
        while (i < text_marked.size()) {
            size_t max_len = std::min(text_marked.size() - i, (size_t)24);
            int best_id = -1;
            size_t best_len = 1;

            for (size_t len = max_len; len >= 1; len--) {
                auto it = vocab_to_id.find(text_marked.substr(i, len));
                if (it != vocab_to_id.end()) {
                    best_id = it->second;
                    best_len = len;
                    break;
                }
            }

            ids.push_back(best_id == -1 ? unk_id : best_id);
            i += best_len;
        }
        return ids;
    }
};