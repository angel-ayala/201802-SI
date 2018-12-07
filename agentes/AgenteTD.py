#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""

import numpy as np
from abc import abstractmethod

class AgenteDiscreto(object):
    def __init__(self, entorno, alpha = 0.5, epsilon = 0.1, gamma = 1):#0.99):
        self.entorno = entorno
        self.nEstados = [entorno.ancho, entorno.alto]
        self.nAcciones = 4

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([self.nEstados[0], self.nEstados[1], self.nAcciones ])        
    # end __init__
    
    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return np.random.randint(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado[0], estado[1], :])
    # end seleccionarAccion
    
    def seleccionarAccionFeedback(self, estado, entrenador, feedbackProbabilidad):
        # consejo
        if np.random.rand() <= feedbackProbabilidad: #consejo
            return np.argmax(entrenador.Q[estado[0], estado[1], :])
        else: #accion agente
            return self.seleccionarAccion(estado)
    # end seleccionarAccionFeedback
    
    @abstractmethod
    def entrenar(self, episodios, entrenador=None, feedbackProbabilidad=0):
        pass
        

class AgenteSarsa(AgenteDiscreto):
    
    def __init__(self, *args, **kwds):
        AgenteDiscreto.__init__(self, *args, **kwds)
    # end __init__
    
    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return np.random.randint(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado[0], estado[1], :])
    # end seleccionarAccion
    
    def sarsa(self, estado, estado_sig, accion, accion_sig, reward):
        td_target = reward + self.gamma * np.max(self.Q[estado_sig[0], estado_sig[1], accion_sig])        
        td_error = td_target - self.Q[estado[0], estado[1], accion]        
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error
    # end sarsa
    
    def entrenar(self, episodios, entrenador=None, feedbackProbabilidad=0):
        recompensas = []

        for e in range(episodios):
            estado = self.entorno.reset()
            accion = self.seleccionarAccionFeedback(estado, entrenador, feedbackProbabilidad) # for sarsa
            recompensa = 0
            fin = False

            while not fin:
                estado_sig, reward = self.entorno.actuar(accion)
                accion_sig = self.seleccionarAccionFeedback(estado_sig, entrenador, feedbackProbabilidad) # for sarsa
                recompensa += reward
                
                fin = estado == self.entorno.goalPos

                #actualizar valor Q
                self.sarsa(estado, estado_sig, accion, accion_sig, reward)
                estado = estado_sig
                accion = accion_sig # for sarsa

#            print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)

        return recompensas
    # end entrenar


class AgenteQlearning(AgenteDiscreto):
    
    def __init__(self, *args, **kwds):
        AgenteDiscreto.__init__(self, *args, **kwds)
    # end __init__    
    
    def QLearning(self, estado, estado_sig, accion, reward):
        td_target = reward + self.gamma * np.max(self.Q[estado_sig[0], estado_sig[1], :])        
        td_error = td_target - self.Q[estado[0], estado[1], accion]        
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error
    #end Qlearning
    
    def entrenar(self, episodios, entrenador=None, feedbackProbabilidad=0):
        recompensas = []

        for e in range(episodios):
            estado = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                accion = self.seleccionarAccionFeedback(estado, entrenador, feedbackProbabilidad)
                estado_sig, reward = self.entorno.actuar(accion)
                recompensa += reward
                
                fin = estado == self.entorno.goalPos

                #actualizar valor Q
                self.QLearning(estado, estado_sig, accion, reward)
                estado = estado_sig

#            print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)

        return recompensas
    # end entrenar