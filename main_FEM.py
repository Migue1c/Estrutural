import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def nod_and_element(points:np.ndarray, nev: np.ndarray, ne:int, acum_el:np.ndarray, thicnesses: np.ndarray, matseg: np.ndarray, interpolation: np.ndarray):
    global vpn
    global vpe

    nl = len(points) - 1 # number of segments equals number of points minus 1
    nn = ne + 1          # total number of nodes equals number of elements + 1
    vpn = np.empty((nn,2), dtype=float) # 'vector' with the position of each node initialized as empty, 0 - is z and 1 - is r coordinate
    vpe = np.empty((ne, 5), dtype=float)     # 'vector' of elements and for the second index, both this and the one before are in fact matrices
                                        # 0 - initial radius aka r coordinate of the first node
                                        # 1 - phi, the inclination of the element
                                        # 2 - h, length of the element
                                        # 3 - thi, thickness of the element, isn't t to not confuse with time
                                        # 4 - m, material represented by a number 0, 1... to access material properties in another vector


    for i in np.arange(0, nl):             # for each segment
        if points[i,0] == points[i+1,0]:   # if the points have the same z coordinate
            # node part to fill vpn
            r = np.linspace(points[i, 1], points[i+1, 1], nev[i], endpoint=False) # makes a uniform distribution of r values spaced in nev[i]
            # intervals therefore, has length nev[i]
            z = np.full((nev[i]), points[i,0], dtype=float) # a vect with points[i,0] z-coordinate repeated nev[i] times
            vpn[acum_el[i]-nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r))) # the node index is equal to all the elements up to and
            # including those of a segment minus the number of elements in the segment itself up to all the elements until the end of the segment
            # element part to fill vpe
            if points[i,1] < points[i+1,1]:
                thi = np.interp(r, [points[i, 1], points[i+1, 1]], [thicnesses[i], thicnesses[i+1]])
                thi = np.concatenate((thi,np.array([thicnesses[i+1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j]+thi[j+1])/2
                thi = thi[:-1]
                vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
                vpe[acum_el[i]-nev[i]:acum_el[i],1] = np.pi/2
                vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.abs((points[i,1] - points[i+1,1])/nev[i])
                vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
                vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
            else:
                thi = np.interp(r, [points[i+1, 1], points[i, 1]], [thicnesses[i], thicnesses[i+1]])
                thi = np.concatenate((thi,np.array([thicnesses[i+1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j]+thi[j+1])/2
                thi = thi[:-1]
                vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
                vpe[acum_el[i]-nev[i]:acum_el[i],1] = -np.pi/2
                vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.abs((points[i,1] - points[i+1,1])/nev[i])
                vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
                vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
        else:
            if interpolation[i] == 0:
                # node part to fill vpn
                z = np.linspace(points[i, 0], points[i+1, 0], nev[i], endpoint=False) # z has dimension nev[i] equivalent to r in the previous case
                r = np.interp(z, [points[i, 0], points[i+1, 0]], [points[i, 1], points[i+1, 1]]) # now r is an interpolation in between both points
                # the last one is naturally excluded by the z vector size, doesn't include the endpoint
                vpn[acum_el[i]-nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r)))
                # element part to fill vpe
                thi = np.interp(z, [points[i, 0], points[i+1, 0]], [thicnesses[i], thicnesses[i+1]])
                thi = np.concatenate((thi,np.array([thicnesses[i+1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j]+thi[j+1])/2
                thi = thi[:-1]
                vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
                vpe[acum_el[i]-nev[i]:acum_el[i],1] = np.arctan((points[i+1,1] - points[i,1])/(points[i+1,0] - points[i,0]))
                vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.sqrt((points[i,0] - points[i+1,0])**2 + (points[i,1] - points[i+1,1])**2)/nev[i]
                vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
                vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
            else:
                # node part to fill vpn
                z = np.linspace(points[i, 0], points[i+1, 0], nev[i], endpoint=False) # z has dimension nev[i] equivalent to r in the previous case
                dr1 = (points[i, 1]-points[i-1, 1])/(points[i, 0]-points[i-1, 0])
                dr2 = (points[i+2, 1]-points[i+1, 1])/(points[i+2, 0]-points[i+1, 0])
                f = sp.interpolate.CubicHermiteSpline([points[i, 0], points[i+1, 0]],[points[i, 1], points[i+1, 1]],[dr1, dr2])
                r = f(z) # now r is an interpolation in between both points
                # the last one is naturally excluded by the z vector size, doesn't include the endpoint
                vpn[acum_el[i]-nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r)))
                # element part to fill vpe
                thi = np.interp(z, [points[i, 0], points[i+1, 0]], [thicnesses[i], thicnesses[i+1]])
                z = np.concatenate((z,np.array([points[i+1,0]])))
                r = np.concatenate((r,np.array([points[i+1,1]])))
                thi = np.concatenate((thi,np.array([thicnesses[i+1]])))
                for j in range(0, nev[i]):
                    vpe[acum_el[i]-nev[i]+j,0] = r[j]
                    vpe[acum_el[i]-nev[i]+j,1] = np.arctan((r[j+1] - r[j])/(z[j+1] - z[j]))
                    vpe[acum_el[i]-nev[i]+j,2] = np.sqrt((r[j+1] - r[j])**2 + (z[j+1] - z[j])**2)
                    vpe[acum_el[i]-nev[i]+j,3] = (thi[j]+thi[j+1])/2
                    vpe[acum_el[i]-nev[i]+j,4] = matseg[i]
    vpn[nn-1,:] = points[-1,:]
    #print(vpn, len(vpn))
    
    #plt.scatter(vpn[:-1,0], vpe[:,3], marker='o', c='b', s=2)
    # makes a scatter plot with first column (z) in the x axis and second column (r) in the y axis
    #plt.grid()
    #plt.axis('equal')
    #plt.show()

def Bi(s1:float, index:int, r:float) -> np.ndarray:
    global vpe
    phi = vpe[index, 1]
    h = vpe[index, 2]
    sen_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([[-1/h, 0, 0],
                     [(1-s1)*sen_phi/r, (1-3*s1**2+2*s1**3)*cos_phi/r, h*s1*(1-2*s1+s1**2)*cos_phi/r],
                     [0, 6*(1-2*s1)/(h**2), 2*(2-3*s1)/h],
                     [0, 6*s1*(1-s1)*sen_phi/(r*h), (-1+4*s1-3*s1**2)*sen_phi/r]])

def Bj(s1:float, index:int, r:float) -> np.ndarray:
    global vpe
    phi = vpe[index, 1]
    h = vpe[index, 2]
    sen_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([[1/h, 0, 0],
                     [s1*sen_phi/r, (s1**2)*(3-2*s1)*cos_phi/r, h*(s1**2)*(-1+s1)*cos_phi/r],
                     [0, -6*(1-2*s1)/(h**2), 2*(1-3*s1)/h],
                     [0, -6*s1*(1-s1)*sen_phi/(r*h), s1*(2-3*s1)*sen_phi/r]])

def elastM(index:int) -> np.ndarray:
    global vpe
    global mat
    E = mat[int(vpe[index, 4]), 1] # mat must have more than one material so that the array is 2D by default
    t = vpe[index, 3]
    upsilon = mat[int(vpe[index, 4]), 2]
    D = (E*t)/(1-upsilon**2)*np.array([[1,upsilon, 0, 0],[upsilon, 1, 0, 0],[0, 0, (t**2/12), upsilon*(t**2/12)],[0, 0, upsilon*(t**2/12), (t**2/12)]])
    return D

def transM(phi) -> np.ndarray:
    T = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    return T

def Bmatrix(s1:float, index:int, r:float, phi:float) -> np.ndarray:
    T = transM(phi)
    return np.hstack((Bi(s1,index,r)@T,Bj(s1,index,r)@T))

def Pbar(s1:float, index:int) -> np.ndarray:
    global vpe
    h = vpe[index, 2]
    return np.array([[1-s1, 0, 0, s1, 0, 0],
                     [0, 1-3*s1**2+2*s1**3, s1*(1-2*s1+s1**2)*h, 0, (s1**2)*(3-2*s1), (s1**2)*(s1-1)*h]])

def Pmatrix(s1:float, index:int, phi:float) -> np.ndarray:
    T = transM(phi)
    P = Pbar(s1, index)
    #print(P, '\n')
    Pi, Pj = np.hsplit(P, 2)
    #print(Pi)
    #print(Pj)
    return np.hstack((Pi@T, Pj@T))

def loading(ne:int, pressure:np.ndarray) -> None: # To be verified
    global load_vct
    global vpe
    load_vct = np.zeros(3*(ne+1))
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        h = vpe[i, 2]
        ef_press = np.array([0, pressure[i]])
        s1 = lambda s: (s+1)/2
        r = lambda s: ri + s1(s)*h*np.sin(phi)
        integrand = lambda s: ef_press.dot(Pmatrix(s1(s), i, phi))*(r(s))
        integral = 0.347854845*integrand(-0.861136312)+0.652145155*integrand(-0.339981044)+0.652145155*integrand(0.339981044)+0.347854845*integrand(0.861136312)
        load_vct[3*i:3*i+6] = load_vct[3*i:3*i+6] + 2*np.pi*h*integral
    #print(load for load in load_vct)
    #print(load_vct)
    #print(load_vct.shape)

def Kestacked(ne:int, ni:int, simpson=False) -> np.ndarray: # Incoeherent integration results
    kes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        h = vpe[i, 2]
        D = elastM(i)
        if simpson:
            I = np.empty((6, 6, ni+1), dtype=float)
            for j, s1 in enumerate(np.linspace(0,1,ni+1)):
                r = ri + s1*h*np.sin(phi)
                B = Bmatrix(s1,i,r,phi)
                I[:,:,j] = B.T@D@B*(r)

            ke = 2*np.pi*h*sp.integrate.simpson(I, x=None, dx=h/ni, axis=-1)
        else:
            s1_ = lambda s: (s+1)/2
            r = lambda s: ri + s1_(s)*h*np.sin(phi)
            integrand = lambda s: Bmatrix(s1_(s),i,r(s),phi).T@D@Bmatrix(s1_(s),i,r(s),phi)*(r(s))
            ke = 2*np.pi*h*((5/9)*integrand(-np.sqrt(3/5))+(8/9)*integrand(0)+(5/9)*integrand(np.sqrt(3/5)))
        kes[:,:,i] = ke
    return kes

def k_global(ne:int, ni=1200, sparse=False) -> None:
    global k_globalM
    kes = Kestacked(ne, ni)
    if sparse:
        row = []
        column = []
        data = []
        for i in range(0, ne):
            for j in range(0,6):
                for k in range(0,6):
                    row.append(3*i+j)
                    column.append(3*i+k)
                    data.append(kes[j, k, i])
        #print(row)
        #print(column)
        #print(data)
        k_globalM = sp.sparse.bsr_array((data, (row, column)), shape=(3*(ne+1), 3*(ne+1)))#.toarray()     
    else:
        k_globalM = np.zeros((3*(ne+1), 3*(ne+1)), dtype='float64')
        for i in range(0,ne):
            k_globalM[3*i:3*i+6,3*i:3*i+6] = k_globalM[3*i:3*i+6,3*i:3*i+6] + kes[:,:,i]
    #print(f'Element {i+1}')
    #for j in range(0, 3*(ne+1)):
    #    for k in range(0, 3*(ne+1)):
    #        print(f'{k_globalM[j, k]:.2}', end='   ')
    #    print()
    #print('\n\n')
    #print(k_globalM)
    #print(sparse)

def Mestacked(ne:int, ni:int, simpson=True) -> np.ndarray: # Incoeherent integration results
    mes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        rho = 2700 #Value from Material Vector
        t = vpe[i,3]
        h = vpe[i, 2]
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        if simpson:
            I = np.empty((6, 6, ni+1), dtype=float)
            for j, s1 in enumerate(np.linspace(0,1,ni+1) ):
                r = ri + s1*h*np.sin(phi)
                P = Pmatrix(s1,i,phi)
                I[:,:,j] = (r)*P.T@P

            me = rho*t*2*sp.pi*h*sp.integrate.simpson(I, x=None, dx=h/ni, axis=-1)
        mes[:,:,i] = me
    return mes

def m_global(ne:int, ni=1200, sparse=False) -> None:
    global m_globalM
    mes = Mestacked(ne, ni)
    if sparse:
        row = []
        column = []
        data = []
        for i in range(0, ne):
            for j in range(0,6):
                for k in range(0,6):
                    row.append(3*i+j)
                    column.append(3*i+k)
                    data.append(mes[j, k, i])
        #print(row)
        #print(column)
        #print(data)
        m_globalM = sp.sparse.bsr_array((data, (row, column)), shape=(3*(ne+1), 3*(ne+1)))#.toarray()     
    else:
        m_globalM = np.zeros((3*(ne+1), 3*(ne+1)), dtype='float64')
        for i in range(0,ne):
            m_globalM[3*i:3*i+6,3*i:3*i+6] = m_globalM[3*i:3*i+6,3*i:3*i+6] + mes[:,:,i]
    return m_globalM


def main():
    global mat
    mat = np.array([[2800, 70*10**9, 0.33, 200*10**6, 70*10**6],
                    [7600, 200*10**9, 0.33, 800*10**6, 200*10**6]])
    # points of interest to define each segment for interpolation
    points = np.array([[0.0,0.001],
                       [0.0,0.49],
                       [0.970,0.49],
                       [0.990,0.25],
                       [1.000, 0.20],
                       [1.100, 0.35]])
    
    nev = np.array([3, 2, 2, 2, 2]) # number of elements per segment
    interpolation = np.array([0, 0, 0, 1, 0])
    ne = np.sum(nev)                 # total number of elements
    thicnesses = np.array([0.03, 0.008, 0.008, 0.05, 0.04, 0.03])
    matseg = np.array([0, 0, 0, 0, 0])
    acum_el = np.cumsum(nev, dtype=int) # each entry in this vect is the number of elements of the present segment + those that came before

    nod_and_element(points, nev, ne, acum_el, thicnesses, matseg, interpolation)
    #print(vpn)
    #print(vpe)

    #kelements = Kestacked(ne)
    k_global(ne)
    m_global(ne)
    #print(Pmatrix(0.4,1,np.pi/2))
    #pressure = np.ones(ne)
    #loading(ne, pressure)

    natfreq = np.sqrt(sp.linalg.eigh(k_globalM,m_globalM , eigvals_only=True))/(2*np.pi)
    sortfreq = np.sort(natfreq, kind='heapsort', axis=None)
    print(sortfreq)

# Atention to units system, is it mm or m? needs to be coherent
if __name__ == '__main__':
    main()
