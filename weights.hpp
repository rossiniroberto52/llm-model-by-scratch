#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

struct ModelWeights {
    float* data;   // O ponteiro mágico para os 4GB de pesos
    size_t size;   // Tamanho do ficheiro em bytes
    int fd;        // File descriptor

    ModelWeights(const char* filename) {
        // 1. Abrir o ficheiro em modo de leitura
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Falha ao abrir o ficheiro de pesos! Ele existe?");
        }

        // 2. Descobrir o tamanho exato do ficheiro
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            throw std::runtime_error("Falha ao ler status do ficheiro!");
        }
        size = sb.st_size;

        // 3. A Bruxaaria: mmap!
        // PROT_READ: Vamos apenas ler. MAP_PRIVATE: Mapeamento privado.
        void* mapped = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            throw std::runtime_error("Falha colossal no mmap! (Falta de memoria virtual?)");
        }

        // Fazemos o cast dos bytes crus para ponteiros de float (já que extraimos como float32)
        data = static_cast<float*>(mapped);
        
        std::cout << "Pesos carregados na veia via mmap! Tamanho: " 
                  << size / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    }

    // O Destrutor para sermos bons cidadãos e limparmos a memória quando o programa fechar
    ~ModelWeights() {
        if (data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};