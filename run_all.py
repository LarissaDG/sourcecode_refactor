#!/bin/bash

#SBATCH --job-name=my_little_job  # Job name
#SBATCH --time=20:00:00           # Time limit hrs:min:sec
#SBATCH -w gorgona6
#SBATCH -N 1                        # Number of nodes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x # all comands are also outputted

#Faz o download da base de dados

#Configura o ambiente
echo "Nucleo 1 - Análise exploratória dos dados do APDDv2"
#Notebook
echo "Experimento 1 - Análise do impacto da amostragem"
#Faz a amostragem
echo "Núcleo 2 - Posicionamento do ArtCLIP na literatura de avaliação
estética"
#Discursivo - Talvez passar para parte de trabalhos relacionados
echo " Núcleo 3- Consistência e Capacidade Discriminativa da Heurística
Estética"
echo "Experimento 3.1 - Comparação dos Escores Estético"
#Aquela tabela de +/- do paper
echo "Experimento 3.2 - Consistência"
#Base vs. Janus - gráfico de barras paper
echo "Experimento 3.2 - Validação em acervo artístico real (Base
Portinari)"
#comparação do APDDv2 com o portinari
echo "Núcleo 4- Vieses, Limitações e Fronteiras Ontológicas da Heurística 73
O que este experimento "
echo "Experimento 4.1 - Viés Chinês"
#A base de dados tem um viés chines? Codigo que calcula as estetisticas e os boxplots
echo "Experimento 4.2 - Limitação ontológica do modelo"
#MNIST - para mostrar o APDDv2 vs. o MNIST e como ele funciona em dados não artisticos
echo "Núcleo 5 - Aplicação em Processos Criativos"
echo "Experimento 5.1 - Grammarly para Artistas"
#GIFs
echo "Experimento 5.2 - Mockup Knights Tour"
#Design generativo


#Vamos remodelar com essa estrutura:
* Coleta de dados -> Discursivo
* Análise exploratória -> Notebook
* Pré-processamento -> Amostragem/ Análise do impacto da amostragem

Parte 1 (APDDv2 viabilidade "Teorica")
* Experimento 1 -> 3.1 -> RQ1: Como a métrica se comporta em pinturas humanas vs IA (pequeno vs. grande) do mesmo estilo? 
* Experimento 2 -> 3.2 -> RQ2: Qual o impacto do tamanho do modelo de geração de imagens na qualidade estética das pinturas sintéticas?
* Experimento 3 -> 4.1 -> RQ3: Como a métrica se comporta em pinturas humanas de estilos diferentes? 
RQ4: As descrições de pinturas geradas por IA são expressivas o suficiente para produzir versões sintéticas? (Acho que ele queria contrastar o uso do deepseek com a base do portinari que são descrições humanas). Era para entrar aqui a análise de prompts
* Experimento 4 -> 4.2

Parte 2 (Experimental "Prática")
* Experimento 5 -> 5.1 -> RQ5: O quanto a métrica é sensível a ruído (modificações locais)?
* Experimento 6 -> 5.2 


#Coleta aquelas métricas

#
#get descriptions

#gera as imagens 

#gera os scores 

#gera as métricas 

#Manda e-mail

