#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers.monitor import load_results

# Wrapper para discretizar el estado
class ObservationDiscretize(gym.ObservationWrapper):

    def __init__(self, env, states_boundaries, states_fold):
        super(ObservationDiscretize, self).__init__(env)
        self.sb = states_boundaries
        self.sf = states_fold
        self.unwrapped.rangos = states_fold

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        obs = self.observation(observation)
        self.env.state = obs
        return obs

    def observation(self, obs):
        bucket_indice = []
        for i in range(len(obs)):
            if obs[i] <= self.sb[i][0]:
                bucket_index = 0
            elif obs[i] >= self.sb[i][1]:
                bucket_index = self.sf[i] - 1
            else:
                bound_width = self.sb[i][1] - self.sb[i][0]
                offset = (self.sf[i]-1) *  self.sb[i][0] / bound_width
                scaling = (self.sf[i]-1) / bound_width
                bucket_index = int(round(scaling * obs[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)


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
episodios = 1000
t_max = 250 # segun doc gym

for e in range(episodios):
    estadoActual = entorno.reset()

    for t in range(t_max):
        # mostrar el escenario
        # self.entorno.render()
        # seleccionar accion
        accion = entorno.action_space.sample()
        # ejecutar accion
        estadoActual, reward, fin, info = entorno.step(accion)
        # fin del escenario
        if fin:
            # print("fin episodio t=", t+1, 'r=', reward, 's=', estadoActual)
            break
# fin del monitoreo
entorno.close()
# carga de resultados
results = load_results(carpeta)
recompensa = results['episode_rewards']
# graficar
plt.plot(recompensa)
plt.legend()
plt.show()
