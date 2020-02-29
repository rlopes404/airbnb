# Airbnb - Previsão do preço da estadia

##Como foi a definição da sua estratégia de modelagem?

Após fazer um tratamento do dataset, por exemplo, remoção de missing data, imputação de dados, a base de dados foi dividida em treino, validação e teste. No conjunto de treino, iniciou a análise exploratória de dados que resultou nas seguintes hipóteses e criação de features com base em hipóteses:

- O atributo nominal 'neighbourhood' influencia o preço, dado que reigões mais nobres tendem a ter um preço elevado. Portanto, faixas de valores foram criadas de modo a criar atributos binários. Para esse fim, os dados foram agrupados em função do atributo e a média de cada grupo foi obtida. Em seguida, as médias foram ordenadas e, então, o valor númerico associado a um valor do atributo foi definido como a posição no vetor ordenado. Deste modo, transformou-se o atributo nominal textual em um atributo numérico ordinal.
- Os atributos nominais 'bed_type' e 'room_type' influenciam o preço. Por exemplo, um sofá fornece um conforto inferior àquele fornecido por cama, enquanto um quaro privativo geralmente é mais caro comparado a um quarto coletivo. Portanto, foram criados  atributos numéricos com uma faixa de valores para modelar esso conceito ordinal; estratégia similar àquela usada no atributo 'neighbourhood', descrito anteriormente.
- O atributo 'amenities', que é uma lista de palavras, representa comodidades que a hospdagem possui. Portanto, criou-se uma feature que consiste no número de amenities que a hospedagem fornece. Este novo atributo baseia-se na hipótese de que o preço é diretamente proporcional ao número de comodidades.

Vale ressaltar que neste projeto, descartamos os campos de texto corrido tais como 'Summary', Standard', entre outros. Neste caso, é possível realizar análise de sentimento sobre texto ou extrair features bag-of-words após eliminar stop words. Essas features podem ser aumentar o poder preditivo dos modelos.

Em seguida, foi avaliada a correlação linear de Pearson entre as features e o preço. Dentre as variáveis tratadas ou criadas, as variáveis 'room_type', 'bed_type'  e 'neighbourhood' apresentaram coeficientes de correlação iguais a 0.25,  0.03 e 0.23, respectivamente. Por fim, outliers foram removidos do treino com base no critério do valor absoluto do z-score ser maior ou igual a três unidades. Features com coeficiente de correlação inferior a 0.01 foram removidas do modelo. Ao total, há 26 features.

## Como foi definida a função de custo utilizada?

Os seguintes algoritmos para regressão foram avaliados:
- Regressão Linear: A função de custo consiste na soma dos erros quadráticos, em outras palavras, minimiza-se o Mean Squared Error
- Regresão Linear Polinomial: Não há diferença da função de custo comparada à Regressão Linear vanilla.
- K-Nearest Neighbors (KNN): Apesar de ser um algoritmo lazy, isto é, onde não há treino, a função objetivo escolhida consistiu em minimizar a distância euclidiana.
- Gradient Boosting (GB): A função objetivo consiste em minimizar o erro quadrático. Porém, o erro consiste em função do erro (gradiente) do estimador da iteração anterior a partir da segunda iteração.
Além desses, utilizamos um baseline padrão que consiste em prever a média do preço dentre as instâncias no treino. Este é um weak learner e servirá como basline de comparação. Por fim, utilizamos a técnica de Ensemble, a saber, Stacking para combinar os três algoritmos utilizados. Vale ressaltar que o GB não é um Weak Learner, porém a sua utilização apresentou bons resultados.

## Qual foi o critério utilizado na seleção do modelo final?

Para cada algoritmo avaliado, utilizamos a métrica Root Mean Squared Error (RMSE) no conjunto de validação para escolher o modelo final. Para o algoritmo Regresão Linear Polinomial, investigamos polinômios de graus dois e três. Para o algoritmo KNN, avaliamos por meio de Grid Search com base nas configurações do parâmetro K (número de vizinhos utilizados para regressão) assumindo valores no conjunto {1, 5, 10} e pesos uniformes ou ou pesos com base no inverso da distância entre os pontos. Para o algoritmo GB, dentre os diversos parâmetros, avaliamos por meio de Grid Search os seguintes parâmetros: taxa de aprendizado, número de estimadores, profundidade máxima, taxa de amostragem dos dados, taxa de amostragem das colunas. Por fim, o 'early stopping rounds' foi definido em 10.


## Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Para validar o modelo, utilzamos a técnica Random Train/Test Split onde o conjunto de dados'é aleatoriamente dividido entre treino e teste. Posteriormente, o conjunto de treino é subvidido de modo a formar o conjunto de treino final e um outro conjunto denominado validação. O conjunto de treino é utilizado para aprender os parâmetros. O conjunto de validação é utilizado para o  calibração de hiperparâmetros. O conjunto de teste, por fim, é utilizado para reportar os resultados finais. No meu entedimento, o melhor método é o k-fold cross validation. Porém, por questões de tempo de processamento, não tive condições de utilizar este método. O método Train/Test Split é uma versão simplificada do K-fold cross validation com o valor de k igual a um. O

## Quais evidências você possui de que seu modelo é suficientemente bom?



|  Método | RMSE  |
| ------------ | ------------ |
| Mean | 1848.13 |
| Linear Regression | 1745.47  |
| Polynomial Linear Regression   |  1701.13  |
| K-Nearest Neighbor Regression| 1702 |
| XGBoost | 1659.99 |
| Stacking | 1625.51 |

Vale ressaltar que os pesos 88%, 9%, 1%.