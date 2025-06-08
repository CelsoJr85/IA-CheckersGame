# Checkers Game 🔴⚪

Um jogo de damas completo desenvolvido em Python usando Pygame, com inteligência artificial simples integrada.

## 🎮 Características

- **Interface Gráfica Moderna**: Tabuleiro visual com peças coloridas e animações
- **IA Inteligente**: Sistema de inteligência artificial usando algoritmo Minimax com poda Alfa-Beta
- **Dois Modos de Jogo**: 
  - Jogador vs IA
  - Jogador vs Jogador (usando mesmo mouse)
- **Regras Completas de Damas**:
  - Capturas obrigatórias
  - Capturas múltiplas
  - Coroação de peças (reis)
  - Movimentação avançada para reis
- **Indicadores Visuais**: Movimentos válidos destacados em cores diferentes
- **Sistema de Pontuação**: Avaliação inteligente de posições

## 📋 Pré-requisitos

- Python 3.7 ou superior
- Pygame 2.0 ou superior

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/checkers-game.git
cd checkers-game
```

2. Instale as dependências:
```bash
pip install pygame
```

3. Execute o jogo:
```bash
python checkers.py
```

## 🎯 Como Jogar

### Controles
- **Mouse**: Clique para selecionar e mover peças
- **R**: Reiniciar o jogo
- **M**: Alternar entre modo IA e dois jogadores

### Regras
1. **Peças Vermelhas** (Jogador) começam primeiro
2. **Peças Brancas** são controladas pela IA (no modo IA)
3. Movimentos diagonais apenas em casas escuras
4. **Capturas são obrigatórias** quando disponíveis
5. Peças que alcançam a linha oposta tornam-se **Damas** (coroa dourada)
6. Damas podem mover-se em todas as direções diagonais

### Objetivo
Capture todas as peças do oponente ou bloqueie todos os seus movimentos para vencer!

## 🤖 Inteligência Artificial

A IA utiliza:
- **Algoritmo Minimax** com profundidade configurável
- **Poda Alfa-Beta** para otimização de performance
- **Função de Avaliação** que considera:
  - Número de peças
  - Posição no tabuleiro
  - Peças coroadas
  - Proximidade à promoção

## 🏗️ Estrutura do Código

```
checkers.py
├── Piece         # Classe para peças individuais
├── Board         # Lógica do tabuleiro e regras
├── AI            # Sistema de inteligência artificial
└── Game          # Loop principal e interface
```

## 🎨 Screenshots

### Tela Principal
- Tabuleiro 8x8 com peças posicionadas
- Indicadores visuais para movimentos válidos
- Interface limpa e intuitiva

### Movimentos
- **Círculos Azuis**: Movimentos normais
- **Círculos Vermelhos**: Movimentos de captura (com número de peças)

## 🔧 Customização

Você pode ajustar facilmente:
- **Dificuldade da IA**: Modifique `depth` na classe `AI`
- **Cores do Tabuleiro**: Altere as constantes de cor
- **Tamanho da Janela**: Modifique `BOARD_SIZE`

```python
# Exemplo: IA mais difícil
self.ai = AI(depth=6)  # Padrão é 4
```

## 🤝 Contribuindo

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- Pygame community pela excelente biblioteca
- Algoritmos clássicos de IA em jogos
- Regras oficiais de damas internacionais

---

⭐ Se você gostou do projeto, deixe uma estrela no repositório!
