# BCN_temp
Estudio de les precipitaciones y temperatures des de 1780 en la ciutat de Barcelona
Barcelona accumulated precipitations
Analisis historico sobre las temperaturas y precipitaciones en Barcelona ciudad. Projecto diseñado para poder predecir los pròximos dos años.


Contenido
Dataframe
Dataset downloaded from OpenData Barcelona site.



1. Introducción a Serie de tiempo 1.1 ¿Qué es una serie de tiempo?
2. Evaluación de Modelos
3. Modelo Arima 3.1 Definición Matemática de Modelo Arima 3.2 Leemos datos 3.3 Análisis Exploratorio de Datos (EDA) 3.4 Prueba de Dickey Fuller Aumentada (ADF) 3.5 División de datos para entrenamiento y prueba 3.6 Modelo con Auto-Arima 3.7 Implementación del modelo
4. Modelo LSTM 4.1 Estandarización de los datos 4.2 Modelación con LSTM 4.3 Evaluación del modelo
5. Modelo Random Forest 5.1 Idea intuitiva detrás de los Bosques Aleatorios 5.2 Feature Change 5.3 Implementación del Modelo Random Forest 5.4 Evaluación del modelo
6. Modelo Prophet 6.1 Modelación del modelo Prophet 6.2 Evaluación del modelo
7. Conclusión


1. Introducción a Serie de tiempo 1.1 ¿Qué es una serie de tiempo?
Una serie temporal o cronológica es una sucesión de datos medidos en determinados momentos y ordenados cronológicamente.
Componentes de una serie temporal

Los componentes que forman una serie temporal son los siguientes:

Tendencia: Se puede definir como un cambio a largo plazo que se produce en relación al nivel medio, o el cambio a largo plazo de la media. La tendencia se identifica con un movimiento suave de la serie a largo plazo.
Estacionalidad: Se puede definir como cierta periodicidad de corto plazo, es decir, cuando se observa en la serie un patrón sistemático que se repite periódicamente (cada año, cada mes, etc., dependiendo de las unidades de tiempo en que vengan recogidos los datos). Por ejemplo, el paro laboral aumenta en general en invierno y disminuye en verano.
Ciclo: Similar a la estacionalidad, ya que se puede definir como una fluctuación alrededor de la tendencia, pero de una duración irregular (no estrictamente periódica).
Irregular: Son factores que aparecen de forma aleatoria y que no responden a un comportamiento sistemático o regular y por tanto no pueden ser predecidos. No se corresponden a la tendencia ni a la estacionalidad ni a los ciclos.
Tipos de series temporales


Como se puede observar en la Ecuación 2.1, cada una de las componentes de las series temporales,
tanto en casos de descomposición clásica como en los más elaborados, tiene unas características y una
morfología particulares y reconocibles.
Componente de estacionalidad: Es una curva cíclica, en la descomposición clásica tiene periodo constante, aunque en algunos de los métodos más elaborados puede adaptarse con respecto a cambios de estacionalidad con el tiempo. Se puede calcular, entre otros métodos, con
transformaciones matemáticas como la de Fourier [7].
Componente de tendencia: Suele ser una curva suavizada, la cual no responde a cambios
bruscos en la serie temporal sino a la tendencia general. Suele estar presente en todo tipo de
series, y es fácilmente calculable con métodos basados en media móvil (Moving Average) o
similares.
Componente de ruido: Corresponde a la resta de las otras dos componentes a la serie, y por
tanto corresponde a un error de ajuste. Como tal, se suele aceptar que los valores de esta componente, cuando las demás están calculadas adecuadamente, deben seguir una distribución normal
o gaussiana 


Además, las series temporales se pueden dividir en:

Estacionarias: es aquella en la que las propiedades estadísticas de la serie son estables, no varían con el tiempo, más en concreto su media, varianza y covarianza se mantienen constantes a lo lardo del tiempo. Si una serie temporal tiene una media constante a lo largo del tiempo, decimos que es estacionaria con respecto a la media. Si tiene varianza constante con respecto al tiempo, decimos que es estacionaria en varianza.
No estacionarias: son aquellas en las que las propiedades estadísticas de la serie sí varían con el tiempo. Estas series pueden mostrar cambio de varianza, tendencia o efectos estacionales a lo largo del tiempo. Una serie es no estacionaria en media cuando muestra una tendencia, y una serie es no estacionaria en varianza cuando la variabilidad de los datos es diferente a lo largo de la serie.
La importancia de esta división reside en que la estacionaridad (e
¿Por qué son importantes las series de tiempo estacionarias?
La razón por la que estas series son importantes es que la mayoría de los modelos de series de tiempo funcionan bajo el supuesto de que la serie es estacionaria. Intuitivamente, podemos suponer que si una serie tiene un comportamiento particular en el tiempo, hay una probabilidad muy alta de que se comportamiento continúe en el futuro. Además, las teorías relacionadas con las series estacionarias son más maduras y más fáciles de implementar en comparación con series no estacionarias. A pesar de que el supuesto de que la serie es estacionaria se utiliza en muchos modelos, casi ninguna de las series de tiempo que encontramos en la práctica son estacionarias. Por tal motivo la estadística tuvo que desarrollar varias técnicas para hacer estacionaria, o lo más cercano posible a estacionaria, a una serie.


Enfoque de los Alisados o Suavizados
Los métodos de suavizado o alisado se basan en modelos paramétricos deterministas que se ajustan a la evolución de la serie. Son técnicas de tipo predictivo más que descriptivo (resultan más adecuados para pronosticar). Estos modelos se pueden emplear en:

Series temporales sin tendencia ni estacionalidad

Este tipo de series tienen un comportamiento más o menos estable que sigue un patrón subyacente salvo fluctuaciones aleatorias (comportamiento estacionario), a este tipo de series se le pueden aplicar:

Modelos "naive" o ingenuos: según la importancia que se le de a las observaciones se tiene:
Se otorga la misma importancia a todas las observaciones a la hora de predecir, de esta forma la previsión vendrá dada por la media de las observaciones.
Se da importancia únicamente a la última de las observaciones ignorando el resto, de forma que el ajuste de la serie es su “sombra”, es la misma serie pero retardada en una unidad de periodo.
Modelos de médias móviles: se basan en considerar únicamente las últimas k observaciones. De esta forma se da el mismo peso a los últimos k datos y nada al resto. Este procedimiento no es tan extremo como los anteriores, y al sustituir cada dato por una media de los k últimos la serie se suaviza y se elimina ruido, obteniendo el patrón subyacente de la misma. Cuantas más observaciones relevantes (k) tomemos al aplicar este tipo de ajuste más se suavizará la serie.
Modelos de suavizado exponencial simple: consisten en dar importancia a todos los datos anteriores, pero concediéndoles diferentes pesos, ya que los datos más relevantes a la hora de efectuar una previsión son los últimos de los que se dispone, disminuyendo la importancia conforme nos alejamos de ellos. De esta manera se sustituye cada dato de la serie por una media ponderada de las observaciones anteriores, considerando que los pesos de las mismas decaen de forma exponencial conforme éstas se alejan en el tiempo (la fórmula del ajuste es recursiva). La cantidad de alisado (de nivel) depende de un parámetro, alpha, que modula la importancia que tienen las observaciones pasadas sobre el presente. Su valor oscila entre 0 y 1. Si alpha toma un valor próximo a 0 las predicciones a lo largo de la serie son muy similares entre sí, y se modifican poco con la nueva información. El caso extremo se produce cuando alpha es cero, lo que implicaría que la predicción es una constante a lo largo del tiempo. Si alpha, por el contrario, toma un valor próximo a 1 la predicción se va adaptando al último valor observado, por lo que se puede decir que los valores alejados en el tiempo no tienen mucha influencia sobre la predicción. El suavizado exponencial simple es el más similar a un modelo ARIMA con cero órdenes de autorregresión, un orden de diferenciación y un orden de media móvil.


Existen numerosos algoritmos de predicción de series temporales, pero se pueden agrupar en dos familias: modelos estadístico y modelos de redes neuronales. Gran parte de los modelos estadísticos suponen que las relaciones subyacentes entre los datos son lineales, mientras que modelos estadísticos basados en funciones no lineales y los modelos neuronales permiten descubrir relaciones no lineales. De entre los métodos estadísticos destacan la regresión lineal, la familia de modelos ARIMA (Autoregressive Integrated Moving Average) con AR, MA, ARMA, ARIMA, ARIMAX y SARIMAX, y finalmente Prophet





https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html
https://github.com/FrancisArgnR/SeriesTemporalesEnCastellano
https://relopezbriega.github.io/blog/2016/09/26/series-de-tiempo-con-python/
https://www.analyticsvidhya.com/blog/2021/07/introduction-to-time-series-modeling-with-arima/

