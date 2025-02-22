# Esta función nos permitirá mostrar el factor de agrupación tanto en
# cartesianas como en polares. Ejecutamos la celda para cargar la
# definición de la función en memoria y así poder usarla para visualizar
# resultados posteriores.

import matplotlib.pyplot as plt 
import numpy as np
from ipywidgets import interact, FloatSlider, RadioButtons
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import cm

class muestra_FA():  
    
    def plot_fasor(self,a):
        # Borramos las flechas que indican el valor de psi y theta antiguo
        #self.axFAcartesianas.patches.pop(0)
        self.axFAcartesianas.patches[0].remove()
        #self.axFApolares.patches.pop(1)
        self.axFApolares.patches[1].remove()
         
        # Dibujamos la flecha que indica el valor psi actual    
        patchB = plt.Arrow(self.kd*np.cos(a*np.pi/180)+self.alpha, 0,0,-30 , width=0.2, color='red')
        self.axFAcartesianas.add_patch(patchB)
        kd=self.kd
        # Dibujamos la flecha que indica el valor de theta, correspondiente al valor de psi actual
        patchA = plt.Arrow(self.alpha, 0,
           kd*np.cos(a*np.pi/180),kd*np.sin(a*np.pi/180), 
           width=0.2, color='red' )
        self.axFApolares.add_patch(patchA)                    
        
    def __init__(self,factor_agrupacion,margen_visible,titulo):
        if not isinstance(factor_agrupacion(np.array([1,2])), np.ndarray):
            print("ERROR: Se debe vectorizar el FA(Ψ). Es decir, Ψ puede \
ser un vector y entonces se debe devolver un vector de resultados. Investiga np.vectorize. Cancelo")
            return
        if np.iscomplex(factor_agrupacion(np.array([1,2]))).any()==True:
            print("ERROR: Se pide el módulo del FA, no el FA complejo. Cancelo.")
            return
        self.fig=plt.figure(figsize=(10,3.8))#figsize=plt.figaspect(1))
        self.axFApolares = plt.subplot(111) 
        # Relación de aspecto 1 a 1
        self.axFApolares.set_aspect(1.)
        # Crear nuevos ejes encima de los actuales (mostramos el eje en 
        # cartesianas encima de la gráfica en polares, alineando los ejes horizontales).
        self.divider = make_axes_locatable(self.axFApolares)
        self.axFAcartesianas = self.divider.append_axes("top", size=1, pad=0.5, sharex=self.axFApolares)
        self.axFAcartesianas.set_title(titulo)
        # Creamos un array unidimensional de puntos del margen visible, para 
        # representar en esos puntos el valor del factor de agrupación
        self.x = np.arange(margen_visible[0], margen_visible[1],0.01)
        self.axFAcartesianas.set_ylabel('|$FA(\Psi)$| [dB]')
        self.axFApolares.set_ylabel('|$FA(\Theta)$| [dB]')
        self.axFAcartesianas.set_xlabel(r'$\Psi_z=kd\cos(\Theta)$+$\alpha$')
        self.axFApolares.set_xlabel('eje z. Eje de alineación del array')
        # Representamos en unidades logarítmicas para ver mejor los lóbulos pequeños
        FAlog=20*np.log10(np.abs(factor_agrupacion(self.x))/max(factor_agrupacion(self.x)))
        # Todos los valores inferiores a -30 dB son sustituidos por -30 dB
        FAlog[FAlog<-30]=-30
        
        self.line,=self.axFAcartesianas.plot(self.x,FAlog) #factor_agrupacion(self.x))
        
        # Deducimos a partir del margen visible los valores de kd y de la fase progresiva
        self.kd=abs(margen_visible[1]-margen_visible[0])/2
        kd=self.kd
        self.alpha=(margen_visible[1]+margen_visible[0])/2
        alpha=self.alpha
        
        # Función en polares del factor de agrupación, haciendo el cambio 
        # de psi en función de theta
        FApolar=lambda theta,alpha: factor_agrupacion(kd*np.cos(theta)+alpha)      
        th=np.linspace(0,np.pi,200)
        FApolarLog=20*np.log10(np.abs(FApolar(th,alpha)/max(FApolar(th,alpha))))
        FApolarLog[FApolarLog<-30]=-30
        # Normalizamos para que el máximo valga kd y encaje la proyección vertical desde la 
        # gráfica en cartesianas
        FApolarLog=(FApolarLog+30)/30*kd
        self.axFApolares.plot(FApolarLog*np.cos(th)+self.alpha,FApolarLog*np.sin(th))
        self.axFAcartesianas.grid(True)
  
        circle = plt.Circle((alpha, 0), radius=kd, color='blue', fill=False)
        self.axFApolares.add_patch(circle)
        
        # Flechas que indican la proyección del valor de arriba a la gráfica de abajo
        patchB = plt.Arrow(0, 0,0, -30,width=0.05)
        self.axFAcartesianas.add_patch(patchB)
        patchA = plt.Arrow(0, 0,margen_visible[1],0, width=0.1 )
        self.axFApolares.add_patch(patchA) 
               
        self.axFApolares.set_ylim([0,kd])
        interact(self.plot_fasor,a=FloatSlider(min=0, max=180,step=0.01, description='$\Psi_z=>\Theta$'), layout={'width': '800px'})
        
        plt.show()

def muestra_FA_cartesianas(XX,YY,ZZ,titulo):
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    # Introducimos las matrices que hemos calculado para hacer la representación
    ax.plot_surface(XX,YY,ZZ, alpha=1,rstride=1, cstride=1,cmap=cm.jet)
    # Etiquetamos los ejes
    ax.set_xlabel('Psi_x')
    ax.set_ylabel('Psi_y')
    # Damos título a la gráfica
    ax.set_title(titulo)
    # Seleccionamos un punto de vista inicial
    ax.view_init(15,45)
    # Mostramos la gráfica
    plt.show()

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

class muestra_FA_2D():  
                      
    def __init__(self,factor_agrupacion,margen_visible_psi_x,margen_visible_psi_y,titulo):
        self.fig = plt.figure(figsize=(10,4)) #figsize=plt.figaspect(1))
        
        self.axFAcartesianas = self.fig.add_subplot(121, projection='3d')
        self.axFAesfericas = self.fig.add_subplot(122, projection='3d')
                       
        self.titulo=titulo
        self.axFAcartesianas.set_title(titulo)
        
        # Establecemos la variación de los ejes en cartesianas del FA
        self.Psi_x = np.linspace(margen_visible_psi_x[0], margen_visible_psi_x[1],60)
        self.Psi_y = np.linspace(margen_visible_psi_y[0], margen_visible_psi_y[1],60)
       
        # Establecemos las etiquetas de los ejes en cartesianas del FA
        self.axFAcartesianas.set_title('|$FA(\Psi_x,\Psi_y)$| [dB]')
        self.axFAcartesianas.set_xlabel(r'$\Psi_x=kd\sin(\Theta)\cos(\phi)$+$\alpha_x$')
        self.axFAcartesianas.set_ylabel(r'$\Psi_y=kd\sin(\Theta)\sin(\phi)$+$\alpha_y$')
        
        # Creamos la malla de coordenadas para representar los parches de superficie
        PSI_X,PSI_Y=np.meshgrid(self.Psi_x,self.Psi_y)
        
        # Expresamos en dB el módulo del factor de agrupación normalizado
        FAlog=20*np.log10(np.abs(factor_agrupacion(PSI_X,PSI_Y))/np.max(np.max(factor_agrupacion(PSI_X,PSI_Y))))
        # Todos los valores más pequeños que -30 dB son sustituidos por -30 dB para la representación
        FAlog[FAlog<-30]=-30
        # Dibujamos la superfice que representa el FA bidimensional en cartesianas
        self.axFAcartesianas.plot_surface(PSI_X, PSI_Y, FAlog, rstride=1, cstride=1, alpha=1,cmap=cm.jet)
        
        # Configuramos la leyenda de la barra de color (mapa de colores empleado y etiqueta)
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(FAlog)
        #cbar=plt.colorbar(m)
        cbar=plt.colorbar(m,ax=self.axFAcartesianas)
        cbar.set_label('dB', rotation=90)
        
        # Deducimos el valor de kd_x, así como el de alpha_x
        self.kd_x=abs(margen_visible_psi_x[1]-margen_visible_psi_x[0])/2
        kd_x=self.kd_x
        self.alpha_x=(margen_visible_psi_x[1]+margen_visible_psi_x[0])/2
        alpha_x=self.alpha_x
        # Deducimos el valor de kd_y, así como el de alpha_y
        self.kd_y=abs(margen_visible_psi_y[1]-margen_visible_psi_y[0])/2
        kd_y=self.kd_y
        self.alpha_y=(margen_visible_psi_y[1]+margen_visible_psi_y[0])/2
        alpha_y=self.alpha_y
        
        
        # Cilindro indicando el margen visible de FA(Psi_x,Psi_y)
        x=np.linspace(-kd_x, kd_x, 100)
        z=np.linspace(0, -40, 100)
        Xc, Zc=np.meshgrid(x, z)
        Yc = np.sqrt((kd_x**2-Xc**2)*kd_y**2/kd_x**2)
        self.axFAcartesianas.plot_surface(Xc+alpha_x, Yc+alpha_y, Zc, alpha=0.2, rstride=1, cstride=1,color='blue')
        self.axFAcartesianas.plot_surface(Xc+alpha_x, -Yc+alpha_y, Zc, alpha=0.2, rstride=1, cstride=1,color='blue')
        # Vista inicial en planta de la representación cartesiana del FA
        self.axFAcartesianas.view_init(90,90)
             
        th=np.linspace(0.001,np.pi/2,30)
        ph=np.linspace(00.001,2*np.pi,100)
        TH,PH=np.meshgrid(th,ph)
        
        FA=lambda theta,phi,alpha_x,alpha_y: factor_agrupacion(kd_x*np.sin(TH)*np.cos(PH)+alpha_x,kd_y*np.sin(TH)*np.sin(PH)+alpha_y) 
        
        FApolarLog=20*np.log10(np.abs(FA(TH,PH,alpha_x,alpha_y)/np.max(np.max(FA(TH,PH,alpha_x,alpha_y)))))
        FApolarLog[FApolarLog<-30]=-30
        FApolarLog=(FApolarLog+30)/30
        
        #Pasamos a cartesianas:
        X=FApolarLog*(np.sin(TH))*(np.cos(PH));
        Y=FApolarLog*(np.sin(TH))*(np.sin(PH));
        Z=FApolarLog*(np.cos(TH));
        cmap = plt.get_cmap('jet')
        norm = mcolors.Normalize(vmin=FApolarLog.min(), vmax=FApolarLog.max())
        self.axFAesfericas.plot_surface(X,Y,Z,rstride=1, cstride=1, alpha=0.5,facecolors=cmap(norm(FApolarLog)))
        self.axFAesfericas.set_ylabel('y')
        self.axFAesfericas.set_xlabel('x')
        self.axFAesfericas.set_title('|$FA(\Theta,\phi)$| [dB]')
        #self.axFasores.quiver(0, 0, 0, FApolarLog*np.sin(th)*np.cos(ph), FApolarLog*np.sin(th)*np.sin(ph), FApolarLog*np.cos(th), length=0.1, normalize=True)
        self.axFAcartesianas.grid(True)

        # Dibujamos un domo encima del dibujo en esféricas
        X=FApolarLog.max()*(np.sin(TH))*(np.cos(PH));
        Y=FApolarLog.max()*(np.sin(TH))*(np.sin(PH));
        Z=FApolarLog.max()*(np.cos(TH));
        domo = self.axFAesfericas.plot_surface(X,Y,Z,rstride=1, cstride=1,alpha=0.1)
        self.axFAesfericas.auto_scale_xyz([-FApolarLog.max(),FApolarLog.max()],[-FApolarLog.max(),FApolarLog.max()],[-FApolarLog.max(),FApolarLog.max()] )
        
        plt.show()
