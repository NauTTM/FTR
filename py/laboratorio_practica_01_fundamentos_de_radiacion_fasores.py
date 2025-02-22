#!/usr/bin/env python
# coding: utf-8

# # Práctica 01: Fundamentos de radiación

# In[ ]:


# Esto se emplea para que la salida se adapte al formato del notebook
get_ipython().run_line_magic('matplotlib', 'notebook')
# A continuación importamos librerías
import numpy as np  # cálculo numérico
import matplotlib.pyplot as plt  # gráficos
import matplotlib.transforms as mtransforms


# # La ecuación de ondas escalar homogénea en una dimensión

# Hemos visto que a partir de las ecuaciones de Maxwell se puede obtener la ecuación de ondas.
# Vamos a estudiar la ecuación de ondas homogénea escalar como una ecuación diferencial con coeficientes constantes
# $$\displaystyle 
# \phi^{\prime\prime}+k^2 \phi = 0
# $$
# 
# Se clasifica como ecuación lineal de orden 2:
# $$\displaystyle 
# \phi^{\prime\prime}+ p \phi^\prime+ q\phi = 0
# $$
# 
# La solución general de este tipo de ecuaciones se obtiene resolviendo la ecuación característica correspondiente:
# $$\displaystyle 
# \lambda^2 + p\lambda + q = (\lambda - \lambda_1)(\lambda - \lambda_2)
# $$
# Si $\lambda_1$ y $\lambda_2$ son números complejos, la solución general se expresa así:
# $$\displaystyle 
# \phi(x) = c_1 e^{\lambda_1 x} + c_2 e^{\lambda_2 x},
# $$
# donde $c_1$ y $c_2$ son constantes complejas arbitrarias.
# En nuestro caso la ecuación característica es:
# $$\displaystyle 
# \lambda^2 + k^2 = 0 \Rightarrow
# \lambda=\pm jk
# $$ 
# Finalmente la soluciones son:
# $$\displaystyle 
# \phi(x) = c_1 e^{-jkx} + c_2 e^{jkx},
# $$
# donde $k$ es el número de ondas y $x$ es la dirección en la que se propaga la onda. Para obtener la solución en el dominio del tiempo:
# $$\displaystyle 
# \phi(x,t)=\Re\{\phi(x)e^{j\omega t}\} = \Re\{c_1 e^{-jkx}e^{j\omega t} + c_2 e^{jkx}e^{j\omega t}\}
# $$
# 
# El primer sumando es una onda plana progresiva (va hacia $x=\infty$), si $c_1$ y $c_2$ son números reales:
# $$\displaystyle 
# \Re\{c_1 e^{-jkx}e^{j\omega t} \}=c_1\cos(\omega t -kx)
# $$
# </div>
# El segundo sumando es una onda plana regresiva (va hacia $x=-\infty$):
# $$\displaystyle 
# \Re\{c_1 e^{jkx}e^{j\omega t} \}=c_2\cos(\omega t +kx)
# $$
# -

# De ahora en adelante particularizamos:
# - $c_1=c_2=0.1$ 
# - $k=\omega=2\pi$

# Se debe ir rellenando el código de las funciones ''respuesta_0?()''
# con código fuente en python de tal forma que se satisfaga el
# comentario que define lo que hace la función. A menudo la siguiente
# celda sirve para visualizar la respuesta y poder razonar sobre el
# dibujo de ayuda. Así que hay que observar siempre si a continuación de
# la respuesta existe código para visualizar la respuesta.

# ## Onda progresiva inicial

# Esta función debe devolver una onda plana progresiva en t=0 
# con valor máximo $C_1=0.1$ y frecuencia espacial y temporal $2\pi$
# El formato de los datos de salida es un array numpy con la 
# onda evaluada en las posiciones del array x

# In[ ]:


def respuesta_01():
    π=np.pi       # número pi
    x = np.arange(-1, 1,0.01)
    # Edita a partir de aquí  
    onda_progresiva = x
    return(onda_progresiva)        

respuesta_01()


# In[ ]:


# Para visualizar la respuesta se puede ejecutar esta celda
from ftrlearn01 import muestra_onda
muestra_onda(respuesta_01(),'onda progresiva inicial')    


# ## Onda progresiva un cuarto de periodo temporal más tarde

# Esta función debe devolver la onda plana progresiva anterior
# en el instante t=0.25 
# con amplitud instantánea máxima 0.1 y frecuencia espacial y temporal 2 pi

# In[ ]:


def respuesta_02():
    x = np.arange(-1, 1,0.01)
    # Edita a partir de aquí
    onda_progresiva = x
    return(onda_progresiva)

respuesta_02()
muestra_onda(respuesta_02(),'Onda progresiva 1/4 de periodo más tarde')


# ## Onda regresiva inicial

# Esta función debe devolver la onda plana regresiva en t=0 
# con amplitud 0.1 y frecuencia espacial y temporal 2$\pi$

# In[ ]:


def respuesta_03():
    x = np.arange(-1, 1,0.01)
    # Edita a partir de aquí
    onda_regresiva = x
    return(onda_regresiva)

muestra_onda(respuesta_03(),'onda regresiva inicial')   


# ## Onda regresiva un cuarto de periodo temporal más tarde

# Esta función debe devolver la onda plana regresiva en $t=0.25$
# con amplitud $0.1$ y frecuencia espacial y temporal $2\pi$

# In[ ]:


def respuesta_04():
    # Esta función debe devolver la onda plana regresiva en t=0.25 
    # con amplitud 0.1 y frecuencia espacial y temporal 2 pi
    x = np.arange(-1, 1,0.01)
    # Edita a partir de aquí   
    onda_regresiva = x
    return(onda_regresiva)

muestra_onda(respuesta_04(),'onda regresiva 1/4 de periodo más tarde')    


# In[ ]:





# ## Fasores

# Esta función debe devolver los fasores A B y C mostrados abajo, 
# de la onda plana progresiva de la respuesta_01 (en t=0 con 
# módulo 0.1 y frecuencia espacial y temporal 2 pi) en las posiciones
# espaciales que se muestran en el gráfico generado a continuación, es
# decir, en las posiciones: $x=-\lambda/4$, $x=0$, $x=\lambda/4$

# In[ ]:


def respuesta_05():
    # Edita a partir de aquí   
    fasorA=1j   
    fasorB=1j
    fasorC=1j
    return([fasorA,fasorB,fasorC])

from ftrlearn01 import muestra_onda_con_fasores
muestra_onda_con_fasores(respuesta_01(),'onda plana progresiva inicial', respuesta_05())


# ## Fasores onda progresiva más tarde

# Esta función debe devolver los fasores A B y C mostrados abajo, 
# de la onda plana progresiva de la respuesta_02 (en t=0.25 con 
# módulo 0.1 y frecuencia espacial y temporal 2 pi)

# In[ ]:


def respuesta_06():
    # Edita a partir de aquí  
    fasorA=1j
    fasorB=1j
    fasorC=1j
    return( [fasorA,fasorB,fasorC])

muestra_onda_con_fasores(respuesta_02(),'onda plana progresiva 1/4 de periodo más tarde',respuesta_06())


# ## Fasores onda regresiva inicial

# Esta función debe devolver los fasores A B y C mostrados abajo, 
# de la onda plana regresiva de la respuesta_03 (en t=0 con 
# módulo 0.1 y frecuencia espacial y temporal 2 pi)

# In[ ]:


def respuesta_07():
    # Edita a partir de aquí  
    fasorA=1j
    fasorB=1j
    fasorC=1j
    return( [fasorA,fasorB,fasorC])

muestra_onda_con_fasores(respuesta_03(),'onda plana regresiva inicial',respuesta_07())


# In[ ]:





# ## Onda regresiva más tarde

# Esta función debe devolver los fasores A B y C mostrados abajo, 
# de la onda plana regresiva de la respuesta_04 (en t=0.25 no?
# módulo 0.1 y frecuencia espacial y temporal 2 pi)

# In[ ]:


def respuesta_08():
    # Edita a partir de aquí   
    fasorA=1j
    fasorB=1j
    fasorC=1j
    
    return( [fasorA,fasorB,fasorC])

muestra_onda_con_fasores(respuesta_04(),'onda plana regresiva tras 1/4 de periodo temporal',respuesta_08())


# In[ ]:





# ## Fasor de onda progresiva variando en espacio y tiempo

# Se debe devolver una función que exprese el fasor de la onda
# progresiva de la primera parte como función del espacio y del
# tiempo. Es necesario que se pueda evaluar usando arrays para la
# variable espacio y la variable tiempo.

# In[ ]:


def respuesta_09():
    # Edita a partir de aquí
    fasor_onda_progresiva=lambda x,t: 0*x
    return(fasor_onda_progresiva)

from ftrlearn01 import muestra_fasor_xt
muestra_fasor_xt(fasor=respuesta_09(),titulo='onda progresiva')


# ## Fasor de onda regresiva variando en espacio y tiempo

# Se debe devolver una función que exprese el fasor de la onda regresiva
# de la primera parte como función del espacio y del tiempo. Es
# necesario que se pueda evaluar usando arrays para la variable espacio
# y la variable tiempo.
# -

# In[ ]:


def respuesta_10():
    # Edita a partir de aquí
    fasor_onda_regresiva=lambda x,t: x*0
    return(fasor_onda_regresiva)

from ftrlearn01 import muestra_fasor_xt
muestra_fasor_xt(fasor=respuesta_10(),titulo='onda regresiva')

