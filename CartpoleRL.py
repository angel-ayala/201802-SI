#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import gym
import math
import matplotlib.pyplot as plt
from gym.wrappers.monitor import load_results
from agentes.AgenteGymDiscreto import ObservationDiscretize, AgenteGymDiscreto

# entorno
cartpole = gym.make("CartPole-v0")

# discretizar
limites = list(zip(cartpole.observation_space.low, cartpole.observation_space.high))
limites[1] = [-0.5, 0.5]
limites[3] = [-math.radians(50), math.radians(50)]
rangos = (1, 1, 6, 3)
# aplicar el Wrapper
entorno = ObservationDiscretize(cartpole, limites, rangos)

# inicio monitor de datos
carpeta = 'resultados'
entorno = gym.wrappers.Monitor(entorno, # entorno cartpole discetizado
                    carpeta, # carpeta de guardado
                    video_callable=False, # grabacion de video
                    force=True) # eliminacion de datos anteriores
# entrenamiento del agente
episodios = 300
# instanciacion y entrenamiento del agente
alpha = 0.5
epsilon = 0.9
gamma = 1
agente = AgenteGymDiscreto(entorno, alpha=alpha, epsilon=epsilon, gamma=gamma)

print('Comenzando entrenamiento RL')
agente.entrenar(episodios)
# fin del monitoreo
entorno.close()
print('Fin')
# carga de resultados
results = load_results(carpeta)
recompensa = results['episode_rewards']
# graficar
plt.plot(recompensa)
#plt.legend()
plt.show()
