# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:20:46 2021

@author: User1
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import special

#Neuron Simulation
sim_time = [] 
for i in range(1,1001):
    sim_time.append(i/1000)
    
print(sim_time)

#Neuron Simulation
sim_time = [] 
for i in range(1,1001):
    sim_time.append(i/1000)

def f(Vrest,I,Vth,tau):
    V_values = []
    time = []
    Vrest = 0
    V0 = 0
    t = 0
  
    for k in range(1,10):
        V = Vrest
        V_values.append(V)
        t += 1 
        time.append(t)
        for i in sim_time:
            t += 1 
            time.append(t)
            V = I - I*math.e**(-i/tau)
            if V < Vth:
                V_values.append(V)
            else:
                V += 80
                V_values.append(V)  
                break
    return V_values, time

plott = f(0,40,20,0.02)

print(f(0,30,20,0.02)[1])


plt.plot(plott[1],plott[0])
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.title("Lif Model (with constant input)")
plt.show()



def f_I(Vrest,Vreset,Vth,tau,t_ref,I):
    
    #Frequency
    f = 1/(t_ref + tau*math.log((Vreset-Vrest-I)/(Vth-Vrest-I)))
    return f

#f-I curve
sim = []  
yiota = []
ff_values = []
Vth = 0.02
for i in range(1,51):
    yiota.append(i)
    if (Vth-(i/1000))==0:
        sim.append(0)
    elif ((-(i/1000))/(Vth-(i/1000))) < 0:
        sim.append(0)
    elif ((-(i/1000))/(Vth-(i/1000))) > 0:
        sim.append(f_I(0,0,0.02,0.02,0.002,i/1000))

#Plotting
plt.plot(yiota,sim)
plt.xlabel("IR (mV)")
plt.ylabel("f (Hz)")
plt.title("Lif Model , f-I Curve")
plt.show()






#Approaching df with Simpson's method
def simpsons(n):
    ksi_values = []
    y1_values = []
    y2_values = []
    V_values = []
    time = []
   
    
    step = []
    ksi = 0
    y1 = 0
    y2 = 0
    ksii = 0
    Vrest = 0
    t = 0 
    I = 30
    tau = 0.02 
    Vth = 20
    
    #Neuron Simulation
    sim_time = [] 
    for i in range(1,1001):
        sim_time.append(i/1000)
    
    for m in range(1,11):
        mu = 0 
        sigma = 3
        V = Vrest
        V_values.append(V)
        t += 1 
        time.append(t)
        for k in sim_time:
            y2 = 0
            t += 1 
            time.append(t)
            y1 = I - I*math.e**(-k/tau)
            
            ### Ksi should be given by a Gaussian distribution
            for i in range(-1,n):
                if i == 0 or i == (n-1):                  
                    y2 += ((20*(0.001)/3) * np.random.normal(mu,sigma)) / (math.sqrt(tau))
                elif (i % 2) == 0:
                    y2 += 2*((20*(0.001)/3) * np.random.normal(mu,sigma)) / (math.sqrt(tau))
                elif (i % 2) == 1:
                    y2 += 4*((20*(0.001)/3) * np.random.normal(mu,sigma)) / (math.sqrt(tau))
                    
               
            y2_values.append(y2)
            V = y1 + y2
            
            if V < Vth:
                V_values.append(V)
            else:
                V += 80
                V_values.append(V)
                break
                
        
    return V_values, time, ksi_values , y2_values
    
print(simpsons(21)[3])

###Plotting
plot = simpsons(21)

plt.plot(plot[1],plot[0])
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.title("Lif Model (response to noisy input)")

plt.show()    
        
def f_I_noise(Vth, Vrest, Vreset, I, tau, sigma, n ):

    x = (-I/sigma) + (Vth/(sigma* n))
    xi = Vth/(sigma* n)
    v = 0 
    f = 0 
   
    for i in range(-1,n):
        if i == 0 or i == (n-1):                  
            v +=  tau * math.sqrt(math.pi) * ((x)/3)* ((Vth-Vreset)/sigma) * (math.e**((i* xi)**2) * (1 + math.erf(n)))
        elif (i % 2) == 0:
            v += 2* (tau * math.sqrt(math.pi) *((x)/3)* ((Vth-Vreset)/sigma) * (math.e**((i* xi)**2) * (1 + math.erf(n))))
        elif (i % 2) == 1:
            v += 4*(tau * math.sqrt(math.pi) *((x)/3)* ((Vth-Vreset)/sigma) * (math.e**((i* xi)**2) * (1 + math.erf(n))))
    
    f = 1/v
        
    return f


f_values = []
yiota_values = []
for k in range(1,51):
    yiota_values.append(k)
    f_values.append(f_I_noise(20,0,0,k,0.02,3,21))
    
#Plotting
plt.plot(yiota_values,f_values)
plt.xlabel("IR (mV)")
plt.ylabel("f (Hz)")
plt.show()








###################
        
def f_I_noisee(Vth, Vrest, Vreset, I, tau, sigma):

    yt = (Vth - I)/ sigma
    yr = -I/sigma 
    

    v = tau * 1/2 * (math.sqrt(math.pi) * (special.erf(yt) - special.erf(yr) + (yt - yr)) + (yt - yr))
    f = 1/v
    
        
    return  f

Vth = 20 
f_values = []
yiota_values = []
for k in range(1,51):
    yiota_values.append(k)
    if k < Vth:
        f_values.append(f_I_noisee(20,0,0,k,0.02,3))
    elif k == Vth:
        f_values.append(f_I_noisee(20,0,0,k,0.02,3))
    elif k > Vth:
        f_values.append(f_I_noisee(20,0,0,k,0.02,3) + f_I((0,0,20,0.02,0.002,k))

print(f_values)
    
#Plotting
plt.plot(yiota_values,f_values)
plt.xlabel("IR (mV)")
plt.ylabel("f (Hz)")
plt.show()
  
  

    
    




y1 = 0
y2 = 0
ksii = 0
Vrest = 0
t = 0 
I = 40
tau = 0.02 
Vth = 20
mu = 0 
sigma = 3

x= 0 
x_values = []
for i in range(1, 21):
    x += (20/3) * np.random.normal(mu,sigma) / (math.sqrt(tau))
x_values.append(x)

print(x_values)

print(simpsons(21)[2])
    
