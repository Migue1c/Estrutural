import numpy as np
import sympy as sp


##Elasticity matrix for the isotropic axisymmetrically loaded 

E = 200
upsilon = 0.2
t = 0.1
phi_user = 30

def elastM(E,t,upsilon): #pag495 pdf (eq 13.48a)
    D = (E*t)/(1-upsilon**2)*np.array([[1,upsilon, 0, 0],[upsilon, 1, 0, 0],[0, 0, (t**2/12), upsilon*(t**2/12)],[0, 0, upsilon*(t**2/12), (t**2/12)]])
    return D

D = elastM(E,t,upsilon)
print("Elasticity Matrix:\n",D)

##Transformation matrix

phi = phi_user*np.pi/180

def transM(phi): #pah 496 pdf (eq 13.51)
    T = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    return T

T = transM(phi)
print("Transformation Matrix:\n",T)

#Vars Definition
s1 = sp.Symbol("s1")     
s =  sp.Symbol("s")
h = sp.Symbol("h")
r = sp.Symbol("r")
cosphi = sp.Symbol("cosphi")
sinphi = sp.Symbol("sinphi")

#Displacements Matrix (from nodes) pag496 pdf eq 13.53
Pbar0 = np.array([[1-s1,0,0,s1,0,0],[0,2*s1**3-3*s1**2+1,s1*(1-2*s1+s1**2)*h,0,(3-2*s1)*s1**2,h*(-1+s1)*s1**2]]) 
Pbar = sp.Matrix(Pbar0) #Converts numpy matrix to sympy matrix
print(Pbar)


