# Airbnb - Previsão do preço da estadia

Todos os algoritmos foram implementados em Python, utilizando pacotes numpy, pandas, scikit-learn e XGBoost.

Dataset:
- Dataset utilizado: listings.csv.gz, 22 de Novembro, 2019
- Dataset disponível em [Link](http://insideairbnb.com/get-the-data.html)

## Como foi a definição da sua estratégia de modelagem?

Após fazer um tratamento do dataset, por exemplo, remoção de missing data, imputação de dados, a base de dados foi particionada em treino, validação e teste. No conjunto de treino, iniciou-se a análise exploratória de dados que resultou na criação de features com base em hipóteses:

- Acredita-se que o atributo nominal 'neighbourhood' influencia o preço, dado que reigões mais nobres tendem a ter um preço elevado. Para esse fim, os dados foram agrupados em função do atributo e a média de cada grupo foi obtida. Em seguida, as médias foram ordenadas e, então, um valor numérico atribuído a uma entrada foi definido como a posição no vetor ordenado. Deste modo, transformou-se o atributo nominal textual em um atributo numérico ordinal. Uma alternativa seria executar um algoritmo de clusterização (k-means ou hierarchical clustering) para definir os rótulos.
- Acredita-se que os atributos nominais 'bed_type' e 'room_type' influenciam o preço. Por exemplo, um sofá fornece geralmente um conforto inferior comparado a uma cama, impactando, então, o preço; um quarto privativo, por sua vez, geralmente é mais caro comparado a um quarto coletivo. Portanto, foram criados atributos numéricos com uma faixa de valores para modelar esse conceito ordinal; estratégia similar àquela usada no atributo 'neighbourhood', descrito anteriormente. Além dessa estratégia, tentamos a estratégia one-hot encoding. Contudo, o aumento no número de atributos tornou o treinamento lento. 

- O atributo 'amenities', que é uma lista de palavras, representa comodidades que a hospedagem possui. Portanto, criou-se uma feature que consiste no número de amenities que a hospedagem fornece. Este novo atributo se baseia na hipótese de que o preço é diretamente proporcional ao número de comodidades.

Vale ressaltar que neste projeto, descartamos os campos de texto corrido tais como 'Summary', 'Standard', entre outros. Nesse caso, é possível realizar análise de sentimento sobre texto ou extrair features bag-of-words após eliminar stop words. Essas features podem aumentar o poder preditivo dos modelos.

Em seguida, foi avaliada a correlação linear de Pearson entre as features e o preço. Outliers foram removidos do treino caso o valor absoluto do z-score seja supoerior a 2 unidades de desvio padrão. Features com coeficiente de correlação inferior a 0.01 foram removidas do modelo.


## Como foi definida a função de custo utilizada?

Os seguintes algoritmos para regressão foram avaliados:
- Regressão Linear: A função de custo consiste na soma dos erros quadráticos, em outras palavras, minimiza-se o Mean Squared Error (MSE)
- Regressão Linear Polinomial: Não há diferença da função de custo comparada à Regressão Linear vanilla.
- K-Nearest Neighbors (KNN): Apesar de ser um algoritmo lazy, isto é, onde não há treino, a função objetivo escolhida consistiu em minimizar a distância euclidiana.
- Gradient Boosting (GB): A função objetivo consiste em minimizar o erro quadrático. Porém, o erro é uma função do erro (gradiente) do estimador da iteração anterior a partir da segunda iteração. A saber, utilizamos o XGBoost como implementação.

Além desses, utilizamos um baseline padrão que consiste em prever constantemente a média do preço dentre as instâncias no treino. Esse é, então, um weak learner e serve como baseline de comparação. Por fim, utilizamos a técnica de Stacking Ensemble para combinar os três algoritmos utilizados. Vale ressaltar que o GB não é um Weak Learner, porém a sua utilização apresentou bons resultados.

## Qual foi o critério utilizado na seleção do modelo final?

Para cada algoritmo avaliado, utilizamos as métricas Root Mean Squared Error (RMSE) e Mean Absolute Percentage Error (MAPE) no conjunto de validação para escolher o modelo final a ser usado no conjunto de teste. Para o algoritmo Regressão Linear Polinomial, investigamos polinômios de graus dois apenas por limitação de memória (hardware). Para o algoritmo KNN, avaliamos por meio de Grid Search com base nas configurações do parâmetro K (número de vizinhos utilizados para regressão) assumindo valores no conjunto {1, 5, 10}, e pesos uniformes ou pesos com base no inverso da distância entre os pontos. Para o algoritmo GB, avaliamos por meio de Grid Search os seguintes parâmetros: taxa de aprendizado, número de estimadores, profundidade máxima, taxa de amostragem dos dados, taxa de amostragem das colunas; o 'early stopping rounds' foi fixado em 10.

## Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Para validar o modelo, utilizamos a técnica Random Train/Test Split onde o conjunto de dados é aleatoriamente dividido entre treino e teste. Posteriormente, o conjunto de treino é subdividido de modo a formar o conjunto de treino final e o conjunto de validação. O conjunto de treino é utilizado para aprender os parâmetros. O conjunto de validação é utilizado para o calibração de hiperparâmetros. O conjunto de teste, por fim, é utilizado para reportar os resultados finais. Notem que desta forma não há vazamento de informações do conjunto de teste. No meu entedimento, o melhor método é o k-fold cross validation, o qual é útil para reduzir o viés e variância da estimativa do erro. Porém, por questões de tempo de processamento, não tive condições de utilizar esse método. A saber, o algoritmo GB toma um tempo considerável ao utilizar o Grid Search. Na verdade, o método utilizado, isto é, o Random Train/Test Split é uma versão simplificada do K-fold cross validation quando K=2. 

## Quais evidências você possui de que seu modelo é suficientemente bom?

A tabela abaixo apresenta os resultaodos obtidos para cada um dos modelos apresentados. O baseline mais fraco, que consiste em prever constantemente a média, apresenta um RMSE 17.5% superior ao nosso melhor algoritmo, a saber, obtido por meio da técnica Stacking Ensemble. Pode-se concluir que a introdução de features polinomiais não melhora o desempenho do algortimo Regressão Linear em termos de RMSE. Embora, em termos de MAPE, a introdução features polinomiais de fato melhora o poder preditivo da Regressão Linear vanilla. KNN apresentou resultados superiores àqueles fornecidos pela Regressão Linear em ambas as métricas. Se por um lado GB apresenta RMSE 4.1% superior ao Stacking, por outro lado,  Stacking apresenta um MAPE 80.23% superior àquele apresentado por GB. Com base nos resultados, concluímos que o modelo a ser utilizado em produção deve ser GB. Vale ressaltar que não levamos em conta a complexidade do modelo para decidir o modelo final. A saber, há métricas, como Akaike Information Criterion (AIC), para escolher o modelo levando em conta sua complexidade, dada em função do número de parâmetros; vale ressaltar que AIC não é aplicável nos algoritmos utilizados, pois os modelos são discrimanativos em vez de generativos, em outras palavras, não é possível avaliar a verossimilhança.

| Método | RMSE | MAPE |
| --- | --- | --- |
| Mean | 1755.09 (17.5%) | 167.55 (154.2%) |
| LR | 1658.31 (11.0%) | 97.38  (47.7%)| 
| KNN | 1618.34 (8.3%)| 79.27 (20.2%) |
| Poly | 1695.15 (13.5%) | 81.86 (24.2%) |
| GB | 1555.02 (4.1%) | **65.92** |
| Stacking | **1493.61** | 118.81 (80.23%) |

Em problemas de regressão, apesar de inalcançável, zero é o RMSE ideal. Claramente o melhor RMSE obtido está longe de zero. Ao analisar estatísticas da variável preço, verifica-se que, mesmo após a remoção de outliers, há uma grande amplitude nos preços e elevado desvio padrão. Por fim, ao analisar os resíduos da regressão linear para avaliar os ajustes, é possível concluir a violação de algumas suposições. Para remediar a situação, podemos investigar transformações log nos preditores ou variável de resposta (modelos log-linear, linear-log) ou realizar análise breakdown, muito comum na área de sistema de recomendação, para entender em quais cenários os modelos vem sistematicamente errando. Por exemplo, pela análise breakdown pode-se entender se o modelo erra sistematicamente em determinadas faixas de preços ou se tende a superestimar/subestimar os preços.

Acredito que o desempenho foi prejudicado pelo descarte, por motivo de missing data, de colunas possivelmente relevantes para o problema. Mantivemos apenas features com porcentagem máxima de missing data de 50% e imputamos com o valor zero. Outras estratégias de imputação (média, mediana, KNN, entre outras) podem ser investigadas. Por questões de tempo, uma vez que há dificuldades em utilizar GPU/TPU no Google Colab, uma vez que o processo pode ser morto a qualquer momento, não foi possível avaliar modelos de Deep Learning, sobretudo aqueles que explorem aquelas features textuais descartadas. Por fim, vale ressaltar que foi possível reduzir o valor do RMSE ao remover instâncias com preço acima de 3000. Porém, acreditamos que esta não é uma estratégia justa, uma vez já havíamos removido os outliers com base no critério do z-score.