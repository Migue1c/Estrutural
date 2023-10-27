import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import icecream as ic

def nod_and_element(points:np.ndarray, nev: np.ndarray, ne:int, acum_el:np.ndarray, thicnesses: np.ndarray, matseg: np.ndarray):
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
                vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
                vpe[acum_el[i]-nev[i]:acum_el[i],1] = np.pi/2
                vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.abs((points[i,1] - points[i+1,1])/nev[i])
                vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
                vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
            else:
                thi = np.interp(r, [points[i+1, 1], points[i, 1]], [thicnesses[i], thicnesses[i+1]])
                vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
                vpe[acum_el[i]-nev[i]:acum_el[i],1] = -np.pi/2
                vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.abs((points[i,1] - points[i+1,1])/nev[i])
                vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
                vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
        else:
            # node part to fill vpn
            z = np.linspace(points[i, 0], points[i+1, 0], nev[i], endpoint=False) # z has dimension nev[i] equivalent to r in the previous case
            r = np.interp(z, [points[i, 0], points[i+1, 0]], [points[i, 1], points[i+1, 1]]) # now r is an interpolation in between both points
            # the last one is naturally excluded by the z vector size, doesn't include the endpoint
            vpn[acum_el[i]-nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r)))
            # element part to fill vpe
            thi = np.interp(z, [points[i, 0], points[i+1, 0]], [thicnesses[i], thicnesses[i+1]])
            vpe[acum_el[i]-nev[i]:acum_el[i],0] = np.transpose(r)
            vpe[acum_el[i]-nev[i]:acum_el[i],1] = np.arctan((points[i,1] - points[i+1,1])/(points[i,0] - points[i+1,0]))
            vpe[acum_el[i]-nev[i]:acum_el[i],2] = np.sqrt((points[i,0] - points[i+1,0])**2 + (points[i,1] - points[i+1,1])**2)/nev[i]
            vpe[acum_el[i]-nev[i]:acum_el[i],3] = np.transpose(thi)
            vpe[acum_el[i]-nev[i]:acum_el[i],4] = matseg[i]
            # thi is only interpolated to the first node of the element not the average across it
    vpn[nn-1,:] = points[-1,:]
    #print(vpn, len(vpn))
    
    #plt.scatter([node[0] for node in vpn], [node[1] for node in vpn], marker='o', c='b', s=2)
    # makes a scatter plot with first column (z) in the x axis and second column (r) in the y axis
    #plt.grid()
    #plt.axis('equal')
    #plt.show()



def main():
    global mat
    mat = np.array([2800, 70*10**9, 0.33, 200*10**6, 70*10**6])
    # points of interest to define each segment for interpolation
    points = np.array([[0.0,0.0],
                       [0.0,0.49],
                       [0.970,0.49],
                       [0.990,0.25],
                       [1.000, 0.20],
                       [1.100, 0.35]])
    
    nev = np.array([5, 10, 2, 1, 5]) # number of elements per segment
    ne = np.sum(nev)                 # total number of elements
    thicnesses = np.array([0.03, 0.008, 0.008, 0.05, 0.04, 0.03])
    matseg = np.array([0, 0, 0, 0, 0])
    acum_el = np.cumsum(nev, dtype=int) # each entry in this vect is the number of elements of the present segment + those that came before

    nod_and_element(points, nev, ne, acum_el, thicnesses, matseg)
    print(vpn)
    print(vpe)





if __name__ == '__main__':
    main()