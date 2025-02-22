# + [code]
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt  # gráficos
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from ipywidgets import interact, FloatSlider, RadioButtons
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# -

# + [code]
def muestra_respuesta_01(X,Y,Z,R):
    if isinstance(R,(int,complex,float)):
        print('Error: no puedo dibujar un número')
        return()

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface.
    ax.plot_surface(X, Y, Z,facecolors=plt.cm.jet(R),
                    cstride=1, rstride=1, alpha=1)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(-1, 1)
    plt.title('Diagrama de radiación normalizado $t(\\theta,\phi)$')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    plt.tight_layout()
    
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R)

    #cbar=plt.colorbar(m)
    cbar=plt.colorbar(m,ax=ax)
    cbar.set_label('u.n.', rotation=90)
    plt.show
# -

def muestra_respuesta_02(X,Y,Z,R):
    if isinstance(R,(int,complex,float)):
       print('Error: no puedo dibujar un número')
       return

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibuja la superficie
    ax.plot_surface(X, Y, Z,facecolors=plt.cm.jet(R),
                    cstride=1, rstride=1, alpha=0.3)

    # Configura los límites de visualización y las etiquetas
    ax.set_zlim(-1, 1)
    plt.title('Diag. de radiación normalizado $t(\\theta,\phi)$')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    # Escalar la figura
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    # Muestra el eje de revolución
    u=1/np.sqrt(5)*np.array([1,0,-2])
    ax.plot([-u[0],u[0]], [-u[1],u[1]],[-u[2],u[2]],color='black', marker='o', linestyle='dashed')
    
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R)

    #cbar=plt.colorbar(m)
    cbar=plt.colorbar(m,ax=ax)
    cbar.set_label('u.n.', rotation=90)
    
    plt.show

# + [code]
def muestra_respuesta_03(X,Y,Z,R_log):
    if isinstance(R_log,(int,complex,float)):
        print('Error: no puedo dibujar un número')
        return

    fig = plt.figure(figsize=[10,10]) #plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface.
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=R_log.min(), vmax=R_log.max())
    ax.plot_surface(X, Y, Z,rstride=1,cstride=1,
                    facecolors=cmap(norm(R_log)),alpha=1)
    
    # Tweak the limits and add latex math labels.
    #ax.set_zlim(-1, 1)
    plt.title('Diag. de radiación normalizado $t(\\theta,\phi) [dB]$')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    ax.auto_scale_xyz([-20, 20], [-20, 20], [-20, 20])

    #ax.plot_wireframe(np.array([[-20,20],[-20,20]]), np.array([[-20,20],[-20,20]]),np.array([[0,1],[0,1]]),alpha=1)
    
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R_log-30)
    cbar=plt.colorbar(m,ax=ax)
    cbar.set_label('[dB]', rotation=90)
    
    ax.view_init(elev=90,azim=45.) 
   
    
    plt.show
    
# -

# + [code]
class muestra_rotacion():  
    
    def plot_FA(self,a):
        # Borramos las flechas que indican el valor de psi y theta antiguo
        #self.ax.collection3d.pop(0)
        self.ax.collections.remove(self.wframe)
           # R[R==0]=1.0e-20 # Evitar división por cero
        alpha=a*np.pi/180
        Rnow=np.abs(self.R(alpha))
        Rnow[Rnow==0]=1.0e-20 # Evitar división por cero
        R_log=20*np.log10(np.abs(Rnow))
        R_log[R_log<-30]=-30
        R_log=R_log+30
        self.cmap = plt.get_cmap('jet')
        self.norm = mcolors.Normalize(vmin=Rnow.min(), vmax=Rnow.max())
        self.wframe=self.ax.plot_surface(self.X(alpha),self.Y(alpha), self.Z(alpha),rstride=1, cstride=1,facecolors=self.cmap(self.norm(Rnow)),alpha=1)
        self.ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
               
    def __init__(self,X,Y,Z,R):
        self.X=X
        self.Y=Y
        self.Z=Z
        self.R=R

        if isinstance(R(1),(int,complex,float)):
            print('Error: no puedo dibujar un número')
            return
         
        fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = fig.add_subplot(111, projection='3d')
    
        RR=np.abs(self.R(0))
        RR[RR==0]=1.0e-20 # Evitar división por cero
       
        self.cmap = plt.get_cmap('jet')
        self.norm = mcolors.Normalize(vmin=RR.min(), vmax=RR.max())
        self.wframe=self.ax.plot_surface(self.X(0),self.Y(0), self.Z(0),rstride=1, cstride=1,facecolors=self.cmap(self.norm(RR)),alpha=1)
  
        # Tweak the limits and add latex math labels.
        #ax.set_zlim(-1, 1)
        plt.title('Diag. de radiación de amplitud normalizado $')
        self.ax.set_xlabel(r'$X$')
        self.ax.set_ylabel(r'$Y$')
        self.ax.set_zlabel(r'$Z$')
        self.ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(RR)
        cbar=plt.colorbar(m,ax=self.ax,orientation="horizontal")
        cbar.set_label('[u.n.]', rotation=90)
     
        self.ax.view_init(elev=90,azim=0) 
        
        interact(self.plot_FA,a=FloatSlider(min=0, max=90,step=5, description='$alpha$'), layout={'width': '800px'})
        
        plt.show()
# -

from ipywidgets import interact, FloatSlider
class muestra_lineas_de_campo_t:
            
    def plot_lineas(self,t,r):
        azimuths = np.radians(np.linspace(0, 360, 100))
        zeniths = np.linspace(0.01, r, 100)

        self.R, self.theta = np.meshgrid(zeniths, azimuths)
        self.values = self.lf(self.R,t,self.theta)
        self.values[self.values<-2]=-2/r**3
        self.values[self.values>2]=2/r**3
        self.ax.clear()
        
        self.ax.contour(self.theta, self.R, self.values,levels=self.c0.levels)
        self.ax.grid(False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_theta_zero_location("N")
        #ax.redraw_in_frame()               
        
    def __init__(self,lf):

        #plt.ion() 

        azimuths = np.radians(np.linspace(0, 360, 100))
        zeniths = np.linspace(0.01, 4*np.pi, 100)

        self.R, self.theta = np.meshgrid(zeniths, azimuths)
        self.T=self.R*0

        self.lf=lf
        self.values = self.lf(self.R,self.T,self.theta)
        self.values[self.values<-2]=-2
        self.values[self.values>2]=2    

        fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(5,4))
        tiempo=FloatSlider(min=0.0, max=2*np.pi,step=0.001,orientation='horizontal')
        alcance=FloatSlider(min=np.pi/4, max=4*np.pi,step=0.001,value=4*np.pi,orientation='horizontal')
        
        self.ax.grid(False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_theta_zero_location("N")
       
        self.c0=self.ax.contour(self.theta, self.R, self.values,levels=20)        
        self.ax.set_title("Líneas de fuerza")
        interact(self.plot_lineas,t=tiempo,
                                  r=alcance,
                                  layout={1,})
