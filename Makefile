CXX = g++
# -O3 é crucial aqui para ativar as otimizações agressivas de velocidade do compilador
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# Nome do executável final
TARGET = llm_engine

# Ficheiros fonte (por agora só temos o main.cpp, já que o tensor.hpp é incluído diretamente)
SRCS = main.cpp

# Regra principal
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Regra para limpar o repositório
clean:
	rm -f $(TARGET)