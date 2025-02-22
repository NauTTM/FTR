import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt  # gráficos
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from ipywidgets import interact, FloatSlider, RadioButtons
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import cm
import matplotlib.colors as mcolors
import scipy.special as sc
import sympy as sp
from sympy.vector import CoordSys3D # para trabajar con vectores
N = CoordSys3D('N') # Sistema de coordenadas de referencia 
# (los vectores unitarios para x,y,z son N.i,N.j,N.k)
from sympy import fourier_transform

def muestra_vistas(dibuja_func,X,Y,Z,R_log):
    fig = plt.figure(figsize=plt.figaspect(1), dpi=200.0) # Establecemos el tamaño de la figura
    plt.title('Vistas del diagrama de radiación',pad=15)
    plt.axis('off')
    # Subdividimos la figura en cuantro subplots para mostrar 4 vistas
    ax1 = fig.add_subplot(221, projection='3d') # 1º superior izquierdo
    ax2 = fig.add_subplot(222, projection='3d') # 2º superior derecho
    ax3 = fig.add_subplot(223, projection='3d') # 3º inferior izquierdo
    ax4 = fig.add_subplot(224, projection='3d') # 4º inferior derecho
    
    
    # Representamos la función en los cuatro ejes correspondientes
    dibuja_func(ax1,X,Y,Z,R_log)
    dibuja_func(ax2,X,Y,Z,R_log)
    dibuja_func(ax3,X,Y,Z,R_log)
    dibuja_func(ax4,X,Y,Z,R_log)
    
    ax1.set_title('Alzado') # 1ª vista
    ax1.view_init(0,0)
    ax1.set_yticklabels([]) 
    ax1.set_ylabel('')
    ax2.set_title('Perfil') # 2ª vista
    ax2.view_init(0,-90)
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax3.set_title('Planta') # 3ª vista
    ax3.view_init(90,90)
    ax3.set_zticklabels([])
    ax3.set_zlabel('')
    ax4.view_init(30,30)    # 4ª vista
    fig.canvas.draw()
    
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R_log-30)   # Eslaca en dB de la leyenda va de 0 a -30 dB
    cbar=plt.colorbar(m)
    cbar.set_label('[dB]', rotation=90)
    
    
def muestra_vista(ax,X,Y,Z,R_log):
    #Definición de parámetros    
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=R_log.min(), vmax=R_log.max()) # Escala de color
    ax.plot_surface(X, Y, Z,rstride=1, cstride=1,facecolors=cmap(norm(R_log)))

    # Escribimos las etiquetas de los ejes
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')

    #ax.auto_scale_xyz([-25, 25], [-25,25], [0,50])
    ax.auto_scale_xyz([0, 50], [-25,25], [-25,25]) # Juan



def muestra_respuesta_03(fase):
    y = np.linspace(-0.3*1.5,0.3*1.5, 51)
    #x = np.linspace(-0.3*1.5,0.3*1.5, 51)
    z = np.linspace(-0.3*1.5,0.3*1.5, 51)
    #Y,X=np.meshgrid(y,x)
    Y,Z=np.meshgrid(y,z)
    fig = plt.figure(figsize=[10,3])
    ax2 = fig.add_subplot(122, projection='3d')
    ax1 = fig.add_subplot(121)
    #ax1.contourf(Y,X, fase, 10, cmap=cm.jet)
    ax1.contourf(Y,Z, fase, 10, cmap=cm.jet)
    #ax2.plot_surface(Y,X,np.cos(fase),cmap=cm.jet)
    ax2.plot_surface(Y,Z,np.cos(fase),cmap=cm.jet)
    ax2.view_init(0,0)
    ax2.set_title('Parte real del fasor del campo normalizado\n incidente en la apertura')
    ax2.set_xlabel(r'$y$')
    #ax2.set_ylabel(r'$x$')
    ax2.set_ylabel(r'$z$')
    ax1.set_title('Comprobad la frecuencia espacial con dibujo en papel')
    ax1.set_xlabel(r'$y$')
    #ax1.set_ylabel(r'$x$')
    ax1.set_ylabel(r'$z$')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=None)
    
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(fase)
    cbar=plt.colorbar(m,ax=ax1)
    cbar.set_label('Fase', rotation=90)














