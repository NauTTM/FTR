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







def muestra_integrales_de_fresnel():
    S=lambda nu: sc.fresnel(nu)[0]
    C=lambda nu: sc.fresnel(nu)[1]
    x=np.arange(-3,3,0.01)
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    ax.plot(x, C(x),label='C(ν)')
    ax.plot(x, S(x),'k--',label='S(ν)')
    ax.legend(fontsize=16)
    ax.set_xlabel('ν')
    ax.set_title("Integrales de Fresnel")
    ax.grid(True)

def muestra_espiral_de_cornu():
    S=lambda nu: sc.fresnel(nu)[0]
    C=lambda nu: sc.fresnel(nu)[1]
    ν=np.arange(-5,5,0.01)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    ax.plot(C(ν),S(ν))
    ax.set_xlabel('C(ν)')
    ax.set_ylabel('S(ν)')
    ax.set_title("Espiral de Cornu")
    ax.aspect=[1,1]
    ax.grid(True)

def muestra_atenuación_en_arista():
    S=lambda nu: sc.fresnel(nu)[0]
    C=lambda nu: sc.fresnel(nu)[1]
    F=lambda ν: 0.5*(1+1j)*((0.5-C(ν))-1j*(0.5-S(ν)))
    # Atenuación por difracción en arista afilada
    L_arista=lambda ν: -20*np.log10(np.abs(F(ν)))
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    ν = np.arange(-3,3,0.01)
    ax.plot(ν,L_arista(ν))
    ax.set_ylabel('L_arista [dB]')
    ax.set_xlabel('Obstrucción Normalizada: ν')
    ax.set_title("Atenuación por difracción en arista")
    ax.grid(True)

def muestra_ejemplo_fft():
    from numpy.fft import fft,ifft,fftshift
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    fc=50 # frequency of the carrier
    fs=4*fc # frecuencia de muestreo con factor de sobremuestreo=4
    t=np.arange(start = 0,stop = 2,step = 1/fs) # duración: 2 segundos
    x=np.cos(2*np.pi*fc*t) # señal en el dominio del tiempo 
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()
    ax1.plot(t,x) # dibujamos la señal
    ax1.set_title('$x[n]= cos(2 \pi 50 t)$')
    ax1.set_xlabel('$t=nT_s$')
    ax1.set_ylabel('$x[n]$')
    N=256 # Tamaño de la FFT 
    X = fft(x,N) # DFT compleja de N puntos, la salida contiene
    # CC en el índice 0
    # Frecuencia de Nyquist en N/2 
    # Frecuencia positivas desde le índice 2 hasta N/2-1 
    # Frecuencias negativas desde N/2 hasta N-1
    X2 = fftshift(X) # ordenamos las frecuencias usando fftshift
    df=fs/N # resolución en frecuencia
    sampleIndex = np.arange(start = -N//2,stop = N//2) # // para división entera
    f=sampleIndex*df # eje x convertido a frecuencias
    ax2.stem(sampleIndex,abs(X2),'r',use_line_collection=True) #result with fftshift
    ax2.set_xlabel('k')
    ax2.set_ylabel('|X(k)|')
    ax3.stem(f,abs(X2),'r' , use_line_collection=True)
    ax3.set_xlabel('Frecuencias (f)')
    ax3.set_ylabel('|X(f)|')

def muestra_desplazamiento_doppler(eje_frecuencia,espectro_r):
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    ax.set_ylim(-60,10)
    ax.plot(eje_frecuencia,10*np.log10(espectro_r)-np.max(10*np.log10(espectro_r)))
    # v=10m/s lambda=0.15 => Desplazamiento Doppler máximo v/lambda= 66.6666 Hz
    plt.axvline(x=66.666666)
    ax.set_xlabel('Desplazamiento Doppler [Hz]')
    ax.set_ylabel('Respuesta en frecuencia normalizada [dB]')
    ax.set_title('Espectro Doppler de la señal paso bajo equivalente')
    ax.grid(True)

def muestra_envolvente(eje_temporal,r):
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    #ax.set_ylim(-60,10)
    ax.plot(eje_temporal,np.abs(r))
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Magnitud de la envolvente compleja')
    ax.set_title('Modulación en amplitud de la envolvente')
    ax.grid(True)

def muestra_respuesta_07(respuesta_07,sigma=1):
    modalvalue,standarddeviation,meanvalue,medianvalue=respuesta_07()

    xaxis=np.linspace(0,4*sigma,39)
    RayleighCDF=lambda sigma,xaxis: 1-np.exp(-(xaxis**2)/(2*sigma**2))
    RayleighPDF=lambda sigma,xaxis: (xaxis/sigma**2)*np.exp(-(xaxis**2)/(2*sigma**2))
    PDF=RayleighPDF(sigma,xaxis)
    CDF=RayleighCDF(sigma,xaxis)

    fig, ax = plt.subplots()
    ax.plot(xaxis, CDF, linewidth=2.0)
    ax.plot(xaxis, PDF, linewidth=2.0)
    ax.set(xlim=(0, 4*sigma),ylim=(0, 1))

    ax.plot([modalvalue(sigma),modalvalue(sigma)],[0,0.9],'k:')
    ax.plot([standarddeviation(sigma),standarddeviation(sigma)],[0,0.95],'k:')
    ax.plot([medianvalue(sigma),medianvalue(sigma)],[0,0.85],'k:')
    ax.plot([meanvalue(sigma),meanvalue(sigma)],[0,0.8],'k:')

    ax.text(modalvalue(sigma),0.9,"moda", size=12)
    ax.text(standarddeviation(sigma),0.95,"desviación estándar", size=12)
    ax.text(medianvalue(sigma),0.85,"mediana", size=12)
    ax.text(meanvalue(sigma),0.8,"media", size=12)

    ax.set_xlabel('Variable aleatoria, r')
    ax.set_ylabel('pdf y CDF')
    ax.set_title('Distribución Rayleigh (sigma={})'.format(sigma))

    plt.show()

def muestra_serie_temporal():
    serie1=np.load('serie1.npy')
    eje_temporal=serie1[:,0]
    P=serie1[:,1]
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    ax.plot(eje_temporal,P)
    ax.set_xlabel('Tiempo transcurrido [s]')
    ax.set_ylabel('Potencia recibida [dBm]')

def muestra_comparacion_con_rayleigh(bin_centers,y):
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.subplot(111)
    ax.plot(bin_centers,y,label='CDF experimental')
    sigma=1
    CDFy=1-np.exp(-(bin_centers**2)/(2*sigma**2));
    ax.plot(bin_centers,CDFy,'k--',label='Rayleigh')
    ax.legend()



def muestra_respuesta_10(respuesta_10):
    ds=1.0
    Nsamples=200
    M=-80
    R,RFiltered=respuesta_10()
    d_axis=np.arange(0,Nsamples)*ds
    fig, ax = plt.subplots()
    gsc=ax.plot(d_axis,R+M,'k:',
               linewidth=2.0,
               label='Log-normal sin correlar')
    gc=ax.plot(d_axis,RFiltered,
               'k',linewidth=2.0,
               label='Log-normal correlada')
    ax.legend()
    ax.set_xlabel('Distancia recorrida [m]')
    ax.set_ylabel('Variaciones lentas de señal [dBm]')
    plt.show()

def muestra_autocorrelacion_respuesta_10(respuesta_10):
    ds=1.0
    Nsamples=200
    M=-80
    R,RFiltered=respuesta_10()

    if np.std(RFiltered)!=0:
        Rfwithout=(RFiltered-np.mean(RFiltered))/np.std(RFiltered)
    else:
        Rfwithout=RFiltered
    Rfcorr=np.correlate(Rfwithout,Rfwithout,"full");
    Rcorr=np.correlate(R,R,"full")
    fig, ax = plt.subplots()
    ax.plot(np.arange(-Nsamples+1,Nsamples)*ds,Rcorr,'k:',label='Log-normal sin correlar')
    ax.plot(np.arange(-Nsamples+1,Nsamples)*ds,Rfcorr,'k',label='Log-normal correlada')
    plt.show
    ax.set_ylabel('Coeficiente de autocorrelación')
    ax.set_xlabel('Espaciado entre muestras [m]')
    ax.legend()
