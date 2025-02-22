import matplotlib.pyplot as plt  # gráficos
import numpy as np

def muestra_onda(onda,titulo):
    fig = plt.figure(figsize=(10,3.5))
    axOnda = plt.subplot(111)
    #axOnda.set_aspect(1.)
    x = np.arange(-1, 1,0.01)
    axOnda.plot(x, onda)
    axOnda.set_ylabel('onda')
    axOnda.set_xlabel('posición x medida en $\lambda$')
    axOnda.set_title(titulo)
    axOnda.grid(True)
    circle = plt.Circle((0, 0), radius=0.005, color='red', fill=True)
    axOnda.add_artist(circle)



def muestra_onda_con_fasores(onda,titulo,fasores):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    fig = plt.figure(figsize=(10,3.5))#figsize=plt.figaspect(1))
    axFasores = plt.subplot(111)
    axFasores.set_aspect(1.)
    
    [fasorA,fasorB,fasorC]=fasores

    # create new axes on the right and on the top of the current axes.
    divider = make_axes_locatable(axFasores)
    axOnda = divider.append_axes("top", size=1, pad=0.5, sharex=axFasores)
 
    x = np.arange(-1, 1,0.01)

    # Plot a cosine wave using time and amplitude obtained for the cosine wave

    axOnda.set_ylabel('onda')
    axOnda.set_xlabel('posición x medida en $\lambda$')
    axOnda.set_title(titulo)
    axOnda.plot(x,onda)

    axOnda.grid(True)

    axFasores.arrow(-0.25, 0, 0.8*fasorA.real, 0.8*fasorA.imag, head_width=0.2*0.05, head_length=0.2*abs(fasorA), fc='k', ec='k')
    axFasores.arrow(0, 0, 0.8*fasorB.real, 0.8*fasorB.imag, head_width=0.2*0.05, head_length=0.2*abs(fasorB), fc='k', ec='k')
    axFasores.arrow(0.25, 0, 0.8*fasorC.real, 0.8*fasorC.imag, head_width=0.2*0.05, head_length=0.2*abs(fasorC), fc='k', ec='k')
    
    circle = plt.Circle((0, 0), radius=0.1, color='blue', fill=False)
    axFasores.add_artist(circle)
    #axFasores.Axes.set_xlim([-1,1])
    circle = plt.Circle((0.25, 0), radius=0.1, color='blue', fill=False)
    axFasores.add_artist(circle)
    axFasores.text(-0.25+0.11, 0, "$\Re$")
    axFasores.text(-0.25, 0.11, "$\Im$")
    axFasores.text(-0.25, -0.15, "$Fasor A$",horizontalalignment='center')
    
    circle = plt.Circle((-0.25, 0), radius=0.1, color='blue', fill=False)
    axFasores.add_artist(circle)
    axFasores.text(0.11, 0, "$\Re$")
    axFasores.text(0, 0.11, "$\Im$")
    axFasores.text(0, -0.15, "$Fasor B$",horizontalalignment='center')


    axFasores.set_xlabel('posición x medida en $\lambda$ en la que se evalúan los fasores')
    axFasores.set_ylim([-0.15,0.15])
    axFasores.grid(True)
    axFasores.text(0.25+0.11, 0, "$\Re$")
    axFasores.text(0.25, 0.11, "$\Im$")    #axOnda.xlabel('Time')
    axFasores.text(0.25, -0.15, "$Fasor C$",horizontalalignment='center')

from ipywidgets import interact, FloatSlider, RadioButtons
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class muestra_fasor_xt:
    
    def plot_fasor(self,espacio,tiempo):

        a=espacio
        b=tiempo
        
        self.onda=np.real(self.fasor(self.x,b*np.ones(len(self.x)))) 
        self.line.set_ydata(self.onda)
        self.axFasores.patches[0].remove()
        self.axFasores.patches[0].remove()
        #self.axFasores.patches.pop(0)
        #self.axFasores.patches.pop(0)
          
        patchB = plt.Arrow(a, 0, self.fasor(a,b).real, self.fasor(a,b).imag, width=0.1 )
        self.axFasores.add_patch(patchB)
        
        circle = plt.Circle((a, 0), radius=0.1, color='blue', fill=False)
        self.axFasores.add_patch(circle)
              
        
    def __init__(self,fasor,titulo):

        self.fig = plt.figure(figsize=(10,3.5))#figsize=plt.figaspect(1))
        self.axFasores = plt.subplot(111)
        self.axFasores.set_aspect(1.)

        self.x = np.arange(-1, 1,0.01)

        # create new axes on the right and on the top of the current axes.
        self.divider = make_axes_locatable(self.axFasores)
        self.axOnda = self.divider.append_axes("top", size=1, pad=0.5, sharex=self.axFasores)

        # Plot a cosine wave using time and amplitude obtained for the cosine wave
        self.axOnda.grid(True)
      
        self.titulo=titulo
        self.axOnda.set_title(titulo)
        self.fasor=fasor
        self.onda=np.real(self.fasor(self.x,0))
        self.axOnda.set_ylabel('onda')
        self.axOnda.set_xlabel('posición x medida en $\lambda$')
        self.line,=self.axOnda.plot(self.x,self.onda)
        
        patchB = plt.Arrow(0, 0, self.fasor(0,0).real, self.fasor(0,0).imag,width=0.05)
        self.axFasores.add_patch(patchB)
      
        circle = plt.Circle((0, 0), radius=0.1, color='blue', fill=False)
        self.axFasores.add_patch(circle)
 
        self.axFasores.set_xlabel('posición x medida en $\lambda$ en la que se evalúan los fasores')
        self.axFasores.set_ylim([-0.15,0.15])
        self.axFasores.grid(True)          
        interact(self.plot_fasor,espacio=FloatSlider(min=-1, max=1,step=0.001),layout={2,2},tiempo=FloatSlider(min=-1, max=1,step=0.001)) 
        #plt.show()


