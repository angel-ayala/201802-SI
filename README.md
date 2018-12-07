# Sistemas Inteligentes

En este repositorio encontrarán los codigos fuentes utilizados en el laboratorio del curso de Sistemas Inteligentes para el período 2018-02 de la Universidad Central de Chile.

Estos algoritmos presentan agentes cognitivos que resuelven problemas de Reinforcement Learning, se han trabajado dos agentes principalmente:

### AgenteGymDiscreto
Este agente se ha probado con el entorno de CartPole (v0 y v1), el cual simula el problema de *pole balancing*, siendo resuelto en los primeros 200 episodios.

1.**Política** de selección de acción ~~aleatoria~~, epsilon-greedy.
2.Vector de **Estados** en representación Discreta.
3.Disminucion en valores iniciales de alpha y epsilon.
4.Selección de acción interactiva (Interactive Reinforcement Learning)

Para la implementación del entorno se utiliza [Gym](https://github.com/openai/gym/) de OpenAI con sus funciones de envoltura (Wrapper) para modificar funcionalidades.

### AgenteTD
Este agente fué desarrollado para el entorno de CliffWalking para comparar los diferentes métodos de Temporal Difference Learning.

1.Implementa el método de SARSA.
2.Implementado el método de Q-learning.
3.Selección de acción interactiva (Interactive Reinforcement Learning)
