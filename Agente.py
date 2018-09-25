#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import gym
import math
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

# Agente
class CartpoleDiscreto:
    def __init__(self, entorno, alpha = 0.5, epsilon = 0.9, gamma = 1):#0.99):
        self.entorno = entorno
        self.nEstados = entorno.observation_space.shape[0]
        self.nAcciones = entorno.action_space.n

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.random.uniform(-1, 1, entorno.unwrapped.rangos + (entorno.action_space.n, ))
        
        # print(self.Q)
        self.MIN_ALPHA = 0.1
        self.MIN_EPSILON = 0.01
    # end __init__
    
    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado])
    # end seleccionarAccion
    
    def actualizarPolitica(self, estadoAnterior, estadoActual, accion, reward):
        # q learning
        best_q = np.amax(self.Q[estadoActual])
        self.Q[estadoAnterior + (accion, )] += self.alpha * (reward + self.gamma *
                           best_q - self.Q[estadoAnterior + (accion, )])
    # end actualizarPolitica
    
    def update_explore_rate(self, t):
        self.epsilon = max(self.MIN_EPSILON, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
    # end update_explore_rate

    def update_learning_rate(self, t):
        self.alpha = max(self.MIN_ALPHA, min(self.alpha, 1.0 - math.log10((t+1)/25)))
    # end update_learning_rate
    
    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []

        for e in range(episodios):
            estadoAnterior = estadoActual = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                # self.entorno.render()
                accion = self.seleccionarAccion(estadoActual)
                estadoActual, reward, fin, info = self.entorno.step(accion)
                recompensa += reward

                #actualizar valor Q
                self.actualizarPolitica(estadoAnterior, estadoActual, accion, reward)
                estadoAnterior = estadoActual

            print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)

            self.update_explore_rate(e)
            self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar

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
# instanciacion y entrenamiento del agente
alpha = 0.5
epsilon = 0.9
agente = CartpoleDiscreto(entorno, alpha=alpha, epsilon=epsilon)
agente.entrenar(episodios)
# fin del monitoreo
entorno.close()
# carga de resultados
results = load_results(carpeta)
recompensa = results['episode_rewards']
# graficar
plt.plot(recompensa)
#plt.legend()
plt.show()
