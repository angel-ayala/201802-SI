#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import gym
import math
import numpy as np

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
class AgenteGymDiscreto:
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
    
    def seleccionarAccionFeedback(self, estado, entrenador, feedbackProbabilidad):
        # consejo
        if np.random.rand() <= feedbackProbabilidad: #consejo
            return np.argmax(entrenador.Q[estado])
        else: #accion agente
            return self.seleccionarAccion(estado)
    # end seleccionarAccionFeedback
    
    def actualizarPolitica(self, estado, estado_sig, accion, reward):
        # q learning
        td_target = reward + self.gamma * np.amax(self.Q[estado_sig])        
        td_error = td_target - self.Q[estado + (accion, )]        
        self.Q[estado + (accion, )] += self.alpha * td_error
    # end actualizarPolitica
    
    def update_explore_rate(self, t):
        self.epsilon = max(self.MIN_EPSILON, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
    # end update_explore_rate

    def update_learning_rate(self, t):
        self.alpha = max(self.MIN_ALPHA, min(self.alpha, 1.0 - math.log10((t+1)/25)))
    # end update_learning_rate
    
    def entrenar(self, episodios, entrenador=None, feedbackProbabilidad=0):
        recompensas = []
        epsilones = []
        alphas = []

        for e in range(episodios):
            estado = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                # self.entorno.render()
                accion = self.seleccionarAccionFeedback(estado, entrenador, feedbackProbabilidad)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                recompensa += reward

                #actualizar valor Q
                self.actualizarPolitica(estado, estado_sig, accion, reward)
                estado = estado_sig

#            print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)

            self.update_explore_rate(e)
            self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar
