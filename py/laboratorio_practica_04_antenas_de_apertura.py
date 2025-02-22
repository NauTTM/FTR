#!/usr/bin/env python
# coding: utf-8

# # Práctica 04: Antenas de apertura

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np  # cálculo numérico
import matplotlib.pyplot as plt  # gráficos
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import scipy.special as sc
import sympy as sp
from sympy.vector import CoordSys3D # para trabajar con vectores
N = CoordSys3D('N') # Sistema de coordenadas de referencia 
# (los vectores unitarios para x,y,z son N.i,N.j,N.k)
from sympy import fourier_transform


# ## Corrientes equivalentes en la apertura para incidencia frontal

# Esta apertura cuadrada se usará como base para los siguientes ejercicios.
# El campo incidente con fasores de campo eléctrico $E_i$ y magnético $H_i$
# corresponde a una onda plana que se propaga desde x<0 hacia x>0 a
# través de la apertura realizada en un plano conductor perfecto. La
# onda plana incidente tiene vector de propagación
# $\vec{\rm{\beta}}$. Cuando $\vec{\rm{\beta}}$ es paralelo a
# $\hat{\mathbf{x}}$ diremos que la incidencia es frontal, con el frente
# de ondas coplanar con la apertura. Con incidencia oblicua la dirección
# de llegada de la onda plana puede tener una inclinación arbitraria
# $(\theta,\phi)$.
# <img src="figuras/apertura_yz.svg" alt="apertura" class="bg-primary mb-1" width="500px">

# Lo primero que tenemos que hacer es calcular las corrientes
# equivalentes en la superficie de la apertura en la condiciones
# siguientes:
# 
# - Incidencia frontal, asumimos iluminación uniforme en la apertura,
#   como si los campos no fueran perturbados por la apertura
# - El campo eléctrico en la apertura será por tanto constante e igual a
#   $E_0 \hat{\mathbf{z}}$
# - Los vectores unitarios
#   $\hat{\mathbf{x}},\hat{\mathbf{y}},\hat{\mathbf{z}}$ son N.i,N.j,N.k
#   y se muestran en el dibujo de la apertura
# - Recordar que en sympy el producto vectorial se puede representar
#   mediante el símbolo ^
# - Versión del teorema de equivalencia: rellenando con PEC y aplicando
#   imágenes

# In[ ]:


def respuesta_01():
    # M_ss Densidad de corriente magnética superficial
    # Símbolos a utilizar, no hacen falta todos
    E_0, H_0, a, k_x, k_y, k_z, x, y, z=sp.symbols('E_0 H_0 a k_x k_y k_z x y z') 
    # Edita a partir de aquí


    M_ss=0
    
    return M_ss

respuesta_01()


# In[ ]:





# ## Diagrama normalizado de apertura cuadrada con iluminación frontal

# Se parte de la apertura del dibujo anterior. Asumimos apertura
# cuadrada de lado 3 longitudes de onda. Tenemos iluminación frontal, es
# decir el frente de onda de la onda plana incidente es coplanar con la
# apertura. Se desea representar el diagrama de radiación normalizado,
# empleando [dB] y mostrando valores de 0 a -30 dB de atenuación

# In[ ]:


def respuesta_02():
    # El objetivo de esta función es devolver 4 matrices con 
    # la superficie mallada para alimentar plot_surface:
    # X la matriz de coordenadas X de los parches a representar
    # Y la matriz de coordenadas Y de los parches a representar
    # Z la matriz de coordenadas Z de los parches a representar
    # R_log la distancia de cada punto de la malla al origen 
    # (para colorear cada parche en consecuencia)
    # Representamos únicamente en x>0, 70 segmentos en theta y
    # 140 segmentos en phi

    from numpy import sin,cos,sinc,pi
    π=pi
    # Editar a partir de aquí
    t = np.linspace(0, 1, 71)  
    p = np.linspace(0, 1, 141)
    θ,φ = np.meshgrid(t,p)


    R_log=0
    X = 0
    Y = 0
    Z = 0
    
    return X,Y,Z,R_log

from ftrlearn04 import muestra_vistas, muestra_vista
muestra_vistas(muestra_vista,X=respuesta_02()[0],Y=respuesta_02()[1],Z=respuesta_02()[2],R_log=respuesta_02()[3])


# ## Alimentación uniforme de la apertura, fase progresiva debida a incidencia oblicua

# Suponemos ahora que el vector de propagación de la onda plana
# incidente está contenido en el plano XZ. En el origen de coordenadas
# está dirigido hacia la dirección $(\theta=\pi/3,\phi=0)$. La onda
# incidente viene desde las coordenadas x negativas hacia la
# apertura. La longitud de onda es 0,3 [m]. Aproximamos el campo
# incidente en la apertura por el campo que existiría en esa superficie
# en espacio libre, sin colocar la apertura.

# El objetivo ahora consiste en expresar la fase en radianes en cada
# punto de la superficie de la apertura. Tomamos como referencia una
# fase nula en el centro de la apertura en (x=0,y=0,z=0).

# In[ ]:


def respuesta_03():   
    # fase: esta función devuelve la fase en radianes evaluada en una muestreo de puntos
    # de la apertura
    
    # Establecemos el muestreo en (x=0,y,z) de los puntos de la apertura
    x = np.linspace(-0.3*1.5,0.3*1.5, 51)
    y = np.linspace(-0.3*1.5,0.3*1.5, 51)
    z = np.linspace(-0.3*1.5,0.3*1.5, 51)
    from numpy import cos,sin,sinc,exp
    Y,Z=np.meshgrid(y,z)
    # Edita a partir de aquí
    
    fase=0
    
    return fase

# Para mostrar vuestra respuesta visualmente ejecutad la celda siguiente


# In[ ]:


from ftrlearn04 import muestra_respuesta_03
muestra_respuesta_03(respuesta_03)


# ## Apertura cuadrada con incidencia oblicua estática

# Como en la pregunta anterior el vector de propagación de la onda
# incidente está contenido en el plano XZ.

# En el origen apunta en la dirección $(\theta=\pi/3,\phi=0)$. Se desea
# representar el diagrama de radiación normalizado, empleando [dB] y
# mostrando valores de 0 a -30 dB de atenuación

# In[ ]:


def respuesta_04():
    # El objetivo de esta función es devolver cuatro matrices
    # que describen la superficie mallada para plot_surface:
    # X la matriz de coordenadas X de los parches a representar
    # Y la matriz de coordenadas Y de los parches a representar
    # Z la matriz de coordenadas Z de los parches a representar
    # R_log la distancia de cada punto de la malla al origen 
    # (para colorear cada parche en consecuencia)
    from numpy import sin,cos,sinc,pi
    π=pi
    t = np.linspace(0,π, 41)
    p = np.linspace(-π/2,π/2, 74)
    # Hacemos mallado de puntos theta y phi
    #Hacemos mallado de puntos theta y phi
    θ,φ = np.meshgrid(t,p)   

    # Editar a partir de aquí

    
    #Pasamos theta y phi a cartesianas(obtenemos todos los puntos para las mallas)
    X = 0
    Y = 0
    Z = 0
    R_log=0
    
    return X,Y,Z,R_log

from ftrlearn04 import muestra_vistas, muestra_vista
muestra_vistas(muestra_vista,X=respuesta_04()[0],Y=respuesta_04()[1],Z=respuesta_04()[2],R_log=respuesta_04()[3])


# ## Antena parabólica

# Un paraboloide de revolución se alimenta en el foco con una
# antena de diagrama de radiación
# \begin{eqnarray}
# D(\theta,\phi)= \left\{ {\begin{array}{*{20}lcl}
#    6\cos^2(\theta) & & 0\leq \theta\leq \frac{\pi}{2} \\
#    0 & & \theta>\frac{\pi}{2}  \\
#  \end{array} } \right.
# \end{eqnarray}
# 
# - Calcular la eficiencia de desbordamiento en función de $\beta$ (para el intervalo $30°\leq\beta\leq
# 90°)$

# In[ ]:


def respuesta_05():
    β = np.arange(30, 90,0.1) 
    from numpy import cos,sin,tan
    # Edita a partir de aquí
          
    neta_s=0
    
    return neta_s   

respuesta_05()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




