#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:03:01 2018

@author: angel4ayala@gmail.com
"""
import numpy as np
import gym
import math
import matplotlib.pyplot as plt
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

# entrenamiento de agentes
agentes = 2
episodios = 500
autonomo = np.zeros(episodios)
interactive = np.zeros(episodios)

# parametros del agente autónomo
alpha = 0.5
epsilon = 0.9
gamma = 0.99

print('Comenzando entrenamiento RL...')
for r in range(agentes):
    print('Entrenando agente', r)
    agente = AgenteGymDiscreto(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    [recompensas, epsilones, alphas] = agente.entrenar(episodios)
    autonomo += recompensas

#interactive    
#TODO: Selección de agente entrenador
feedbackProbabilidad = 0.3

print('Comenzando entrenamiento IRL...')
for r in range(agentes):    
    print('Entrenando agente', r)
    agente = AgenteGymDiscreto(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    [recompensas, epsilones, alphas] = agente.entrenar(episodios, agente, feedbackProbabilidad)
    interactive += recompensas
    
#promedio de recompensa
autonomo /= agentes
interactive /= agentes

#plot
plt.plot(autonomo, label='Autónomo')
plt.plot(interactive, label='Interactivo')
plt.xlabel('Episodios')
plt.ylabel('Recompensas durante episodios')
plt.legend()
plt.show()
