import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import quad
from random import seed
from scipy.special import kn, iv
import scipy as sp
seed(42)

def BWM_pdf(t_f,g_i,r_T,p_T,m_T,eta,y,phi,phi_p,dt_fdr_T,T,eta_T):
    return ((t_f*g_i*r_T*p_T)/(mt.pow(2*(mt.pi),3)))*(m_T*np.cosh(eta-y)-p_T*dt_fdr_T*np.cos(phi-phi_p))*(np.exp(-(m_T/T)*np.cosh(eta-y)*np.cosh(eta_T)+(p_T/T)*np.sinh(eta_T)*np.cos(phi-phi_p)))

def semi(x,m_T,T,alpha,R,p_T,dt_fdr_T):
    return x*((2*mt.pi*t_f*g_i*p_T)/(2*mt.pow(mt.pi,2)))*m_T*(kn(1,(m_T/T)*np.cosh((alpha*x)/R)))*(iv(0,(p_T/T)*np.sinh((alpha*x)/R)))

number=1000000
m_particle= 0.14
alpha = 1
r_Tmin = p_Tmin= 0
r_Tmax = R= 7
p_Tmax = 3
t_f = 10
g_i=1
T = 0.160
eta_min = y_min =-3
eta_max = y_max =3
phi_min =phi_pmin =0
phi_max =phi_pmax =2*mt.pi
dt_fdr_T=0
max=0.024965257841973162  # estimation of the maximum BWM
max2 = 0.8037484140919793 # estimation of the maximum Semianalytic

dndpt = []
dndpt2 = []
pt_array =[]

for i in range(number):
    r_T=random.uniform(r_Tmin,r_Tmax)
    p_T=random.uniform(p_Tmin,p_Tmax)
    phi=random.uniform(phi_min,phi_max)
    phi_p=random.uniform(phi_pmin,phi_pmax)
    eta=random.uniform(eta_min,eta_max)
    y=random.uniform(y_min,y_max)

    m_T=mt.sqrt(p_T*p_T+m_particle*m_particle)
    eta_T=(alpha*r_T)/R
    #val, abserr = sp.integrate.quad(semi, 0, 7, args=(m_Tpion,T,alpha,R,p_T,dt_fdr_T))
    probability=BWM_pdf(t_f,g_i,r_T,p_T,m_T,eta,y,phi,phi_p,dt_fdr_T,T,eta_T)

    value=random.uniform(0,max)
    value2=random.uniform(0,max2)

    if value<probability:
        dndpt.append(p_T)

    #if value2<val:
    #    dndpt2.append(p_T)
pom=0
for i in range(30):
    diff=(p_Tmax-p_Tmin)/30
    p_Tbin=p_Tmin+i*diff
    m_Tbin=mt.sqrt(p_Tbin*p_Tbin+m_particle*m_particle)
    val, abserr = sp.integrate.quad(semi, 0, 7, args=(m_Tbin,T,alpha,R,p_Tbin,dt_fdr_T))
    dndpt2.append(val)
    pt_array.append(p_Tbin)
    pom=pom+1


#n2, bins = np.histogram(dndpt2, bins=pt_array,density="True")
n, bins = np.histogram(dndpt, bins=pt_array,density="True")

bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]

fig, axes = plt.subplots(sharex=True, figsize=(9,8))
axes.set_yscale('log')
axes.scatter(bins_mean, n, color="red",label="BWM")
axes.scatter(pt_array, dndpt2, label="Semi-analytic BWM")
axes.grid(True)
axes.set_xlabel('$p_T$[GeV]')
axes.set_ylabel('$d^2N/(p_Tdp_Tdy)$[$GeV^{-2}$]')
axes.set_ylim([0.001, 5])
axes.annotate('m={} MeV'.format(m_particle*1000), (2.0, 0.8))
axes.annotate('T = {} MeV'.format(T), (2.0, 1.2))
axes.annotate('$t_f ={}\ fm$'.format(t_f), (2.5, 1.2))
axes.annotate('N={}'.format(number), (2.5, 0.8))

axes.legend(loc=0)
plt.show()
fig.savefig("blastwavemodel{}.pdf".format(m_particle*1000))
