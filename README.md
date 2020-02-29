# Airbnb - Previsão do preço da estadia

Todos os algoritmos foram implementados em Python, utilizando pacotes numpy, pandas, scikit-learn e xgboost.

Dataset:
- Dataset utilizado: listings.csv.gz, 22 de November, 2019
- Dataset disponível em [Link](http://insideairbnb.com/get-the-data.html)

## Como foi a definição da sua estratégia de modelagem?

Após fazer um tratamento do dataset, por exemplo, remoção de missing data, imputação de dados, a base de dados foi particionada em treino, validação e teste. No conjunto de treino, iniciou-se a análise exploratória de dados que resultou na criação de features com base em hipóteses:

- Acredita-se que o atributo nominal 'neighbourhood' influencia o preço, dado que reigões mais nobres tendem a ter um preço elevado. Para esse fim, os dados foram agrupados em função do atributo e a média de cada grupo foi obtida. Em seguida, as médias foram ordenadas e, então, um valor númerico atríbuido a uma entrada  foi definido como a posição no vetor ordenado. Deste modo, transformou-se o atributo nominal textual em um atributo numérico ordinal. Uma alternativa seria executar um algoritmo de clusterização (k-means ou hierarchical clustering) para definir os rótulos.
- Acredita-se que os atributos nominais 'bed_type' e 'room_type' influenciam o preço. Por exemplo, um sofá fornece gerelmente um conforto inferior comparado a uma cama, impactando, então, o preço; um quaro privativo, por sua vez, geralmente é mais caro comparado a um quarto coletivo. Portanto, foram criados  atributos numéricos com uma faixa de valores para modelar esso conceito ordinal; estratégia similar àquela usada no atributo 'neighbourhood', descrito anteriormente. Além dessa estratégia, tentamos a estratégia one-hot encoding. Contudo, o aumento no número de atributos tornou o treinamento lento. A tabela abaixo ilustra o preço médio por 'room_type':

|  Tipo | Preço  |
| ----- | ------ |
| Entire home/apt   | 838.34 |
| Hotel room  | 704.67 |
| Private room   |  269.81 |
| Shared room |  251.66 | 

- O atributo 'amenities', que é uma lista de palavras, representa comodidades que a hospdagem possui. Portanto, criou-se uma feature que consiste no número de amenities que a hospedagem fornece. Este novo atributo baseia-se na hipótese de que o preço é diretamente proporcional ao número de comodidades.

Vale ressaltar que neste projeto, descartamos os campos de texto corrido tais como 'Summary', Standard', entre outros. Nesse caso, é possível realizar análise de sentimento sobre texto ou extrair features bag-of-words após eliminar stop words. Essas features podem aumentar o poder preditivo dos modelos.

Em seguida, foi avaliada a correlação linear de Pearson entre as features e o preço. Dentre as variáveis tratadas ou criadas, as variáveis 'room_type', 'bed_type'  e 'neighbourhood' apresentaram coeficientes de correlação iguais a 0.25,  0.03 e 0.23, respectivamente. Por fim, outliers foram removidos do treino com base no critério do valor absoluto do z-score ser maior ou igual a três unidades. Features com coeficiente de correlação inferior a 0.01 foram removidas do modelo. Ao total, há 26 features.



## Como foi definida a função de custo utilizada?

Os seguintes algoritmos para regressão foram avaliados:
- Regressão Linear: A função de custo consiste na soma dos erros quadráticos, em outras palavras, minimiza-se o Mean Squared Error (MSE)
- Regresão Linear Polinomial: Não há diferença da função de custo comparada à Regressão Linear vanilla.
- K-Nearest Neighbors (KNN): Apesar de ser um algoritmo lazy, isto é, onde não há treino, a função objetivo escolhida consistiu em minimizar a distância euclidiana.
- Gradient Boosting (GB): A função objetivo consiste em minimizar o erro quadrático. Porém, o erro é um função do erro (gradiente) do estimador da iteração anterior a partir da segunda iteração. A saber, utilizamos o XGBoost como implementação.

Além desses, utilizamos um baseline padrão que consiste em prever constamente a média do preço dentre as instâncias no treino. Esse é, então, um weak learner e serve como baseline de comparação. Por fim, utilizamos a técnica de Stacking Ensemble para combinar os três algoritmos utilizados. Vale ressaltar que o GB não é um Weak Learner, porém a sua utilização apresentou bons resultados.

## Qual foi o critério utilizado na seleção do modelo final?

Para cada algoritmo avaliado, utilizamos a métrica Root Mean Squared Error (RMSE) no conjunto de validação para escolher o modelo final. Para o algoritmo Regresão Linear Polinomial, investigamos polinômios de graus dois e três. Para o algoritmo KNN, avaliamos por meio de Grid Search com base nas configurações do parâmetro K (número de vizinhos utilizados para regressão) assumindo valores no conjunto {1, 5, 10}, e pesos uniformes ou pesos com base no inverso da distância entre os pontos. Para o algoritmo GB, avaliamos por meio de Grid Search os seguintes parâmetros: taxa de aprendizado, número de estimadores, profundidade máxima, taxa de amostragem dos dados, taxa de amostragem das colunas; o 'early stopping rounds' foi fixado em 10.


## Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Para validar o modelo, utilzamos a técnica Random Train/Test Split onde o conjunto de dados'é aleatoriamente dividido entre treino e teste. Posteriormente, o conjunto de treino é subvidido de modo a formar o conjunto de treino final e o conjunto de validação. O conjunto de treino é utilizado para aprender os parâmetros. O conjunto de validação é utilizado para o  calibração de hiperparâmetros. O conjunto de teste, por fim, é utilizado para reportar os resultados finais. Notem que desta forma não há vazamento de informações do conjunto de teste. No meu entedimento, o melhor método é o k-fold cross validation, o qual é útil para reduzir o viés e variância da estimativa do erro. Porém, por questões de tempo de processamento, não tive condições de utilizar esse método. A saber, o algoritmo GB toma um tempo considerável ao utilizar o Grid Search. Na verdade, o método utilizado, isto é, o Random Train/Test Split é uma versão simplificada do K-fold cross validation quando K=1. 

## Quais evidências você possui de que seu modelo é suficientemente bom?

A tabela abaixo apresenta o RMSE para cada um dos modelos apresentados. O baseline mais fraco, que consiste em prever constamente a média, apresenta um RMSE 18.41% superior ao nosso melhor algoritmo, a saber, obtido por meio da técnica Stacking Ensemble. Pode-se concluir que a introdução de features polinomiais não melhora o desempenho do algortimo Regressão Linear. O algoritmo KNN apresentou resultados superiores àqueles fornecidos pela Regressão Linear. Por fim, o algoritmo GB apresenta RMSE 3.23% superior ao Stacking.

|  Método | RMSE  |
| ------------ | ------------ |
| Mean | 1752.62 (18.41%) |
| Regressão Linear | 1671.12 (12.90%) |
| Regressão Linear Polinomial   |  1691.55 (14.29%)  |
| KNN | 1590.00 (7.42%) | 
| GB | 1528.02 (3.23%) |
| Stacking | 1480.10 |

Em problemas de regressão, apesar de inalcançável, zero é o RMSE ideal. Claramente o melhor RMSE obtido, igual a 1480.10, está longe de zero. Ao analisar estatísticas da variável preço, verifica-se que, mesmo após a remoção de outliers, há uma grande amplitude,  a saber, igual 5702 em que os valores mínimo e máximo são iguais a 29 e 5731, respectivamente. Além disso, há o desvio padrão é igual a 690.52. Por fim, no caso de Regressão Linear, é possível analisar os resíduos para avaliar os ajustes, como também estudar transformações log nas features ou variável de resposta (modelos log-linear, linear-log).

Acredito que o desempenho foi prejudicado pelo descarte, por motivo de missing data, de colunas possivelmente relevantes para o problema. Mantivemos apenas features com porcentagem máxima de missing data  de 50% e imputamos com o valor zero. Outras estratégias de imputação (média, mediana, KNN, entre outras) podem ser investigadas. Por questões de tempo, uma vez que há dificuldades em utilizar GPU/TPU no Google Colab, não foi possível avaliar modelos de Deep Learning, sobretudo aqueles que explorem as features textuais descartadas. Por fim, vale ressaltar que foi possível reduzir o valor do RMSE ao remover instâncias com preço acima de 3000. Porém, acreditamos que esta não é uma estratégia justa, uma vez já havíamos removido os outliers com base no critério do z-score.