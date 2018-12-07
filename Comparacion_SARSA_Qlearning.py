#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from entornos.CliffWalking import CliffWalking
from agentes.AgenteTD import AgenteSarsa
from agentes.AgenteTD import AgenteQlearning

entorno = CliffWalking(12,4)

episodios = 500
agentes = 50
epsilon = 0.5
alpha = 0.5
gamma = 0.99

qlearning = np.zeros(episodios)
sarsa = np.zeros(episodios)

for r in range(agentes):
    print('Entrenando agente', r)
    agente = AgenteQlearning(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    qlearning += agente.entrenar(episodios)
    
    agente = AgenteSarsa(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    sarsa += agente.entrenar(episodios)
    
#promedio de recompensa
qlearning /= agentes
sarsa /= agentes

#plot
plt.plot(qlearning, label='Q-Learning')
plt.plot(sarsa, label='Sarsa')
plt.xlabel('Episodios')
plt.ylabel('Recompensas durante episodios')
plt.ylim([-100, -15])
plt.legend()