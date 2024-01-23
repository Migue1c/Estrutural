import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def nod_and_element(points: np.ndarray, nev: np.ndarray, ne: int, acum_el: np.ndarray, thicnesses: np.ndarray,
                    matseg: np.ndarray, interpolation: np.ndarray):
    global vpn
    global vpe

    nl = len(points) - 1  # number of segments equals number of points minus 1
    nn = ne + 1  # total number of nodes equals number of elements + 1
    vpn = np.empty((nn, 2),
                   dtype=float)  # 'vector' with the position of each node initialized as empty, 0 - is z and 1 - is r coordinate
    vpe = np.empty((ne, 5),
                   dtype=float)  # 'vector' of elements and for the second index, both this and the one before are in fact matrices
    # 0 - initial radius aka r coordinate of the first node
    # 1 - phi, the inclination of the element
    # 2 - h, length of the element
    # 3 - thi, thickness of the element, isn't t to not confuse with time
    # 4 - m, material represented by a number 0, 1... to access material properties in another vector

    for i in np.arange(0, nl):  # for each segment
        if points[i, 0] == points[i + 1, 0]:  # if the points have the same z coordinate
            # node part to fill vpn
            r = np.linspace(points[i, 1], points[i + 1, 1], nev[i],
                            endpoint=False)  # makes a uniform distribution of r values spaced in nev[i]
            # intervals therefore, has length nev[i]
            z = np.full((nev[i]), points[i, 0],
                        dtype=float)  # a vect with points[i,0] z-coordinate repeated nev[i] times
            vpn[acum_el[i] - nev[i]:acum_el[i], :] = np.transpose(
                np.vstack((z, r)))  # the node index is equal to all the elements up to and
            # including those of a segment minus the number of elements in the segment itself up to all the elements until the end of the segment
            # element part to fill vpe
            if points[i, 1] < points[i + 1, 1]:
                thi = np.interp(r, [points[i, 1], points[i + 1, 1]], [thicnesses[i], thicnesses[i + 1]])
                thi = np.concatenate((thi, np.array([thicnesses[i + 1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j] + thi[j + 1]) / 2
                thi = thi[:-1]
                vpe[acum_el[i] - nev[i]:acum_el[i], 0] = np.transpose(r)
                vpe[acum_el[i] - nev[i]:acum_el[i], 1] = np.pi / 2
                vpe[acum_el[i] - nev[i]:acum_el[i], 2] = np.abs((points[i, 1] - points[i + 1, 1]) / nev[i])
                vpe[acum_el[i] - nev[i]:acum_el[i], 3] = np.transpose(thi)
                vpe[acum_el[i] - nev[i]:acum_el[i], 4] = matseg[i]
            else:
                thi = np.interp(r, [points[i + 1, 1], points[i, 1]], [thicnesses[i], thicnesses[i + 1]])
                thi = np.concatenate((thi, np.array([thicnesses[i + 1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j] + thi[j + 1]) / 2
                thi = thi[:-1]
                vpe[acum_el[i] - nev[i]:acum_el[i], 0] = np.transpose(r)
                vpe[acum_el[i] - nev[i]:acum_el[i], 1] = -np.pi / 2
                vpe[acum_el[i] - nev[i]:acum_el[i], 2] = np.abs((points[i, 1] - points[i + 1, 1]) / nev[i])
                vpe[acum_el[i] - nev[i]:acum_el[i], 3] = np.transpose(thi)
                vpe[acum_el[i] - nev[i]:acum_el[i], 4] = matseg[i]
        else:
            if interpolation[i] == 0:
                # node part to fill vpn
                z = np.linspace(points[i, 0], points[i + 1, 0], nev[i],
                                endpoint=False)  # z has dimension nev[i] equivalent to r in the previous case
                r = np.interp(z, [points[i, 0], points[i + 1, 0]],
                              [points[i, 1], points[i + 1, 1]])  # now r is an interpolation in between both points
                # the last one is naturally excluded by the z vector size, doesn't include the endpoint
                vpn[acum_el[i] - nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r)))
                # element part to fill vpe
                thi = np.interp(z, [points[i, 0], points[i + 1, 0]], [thicnesses[i], thicnesses[i + 1]])
                thi = np.concatenate((thi, np.array([thicnesses[i + 1]])))
                for j in range(0, nev[i]):
                    thi[j] = (thi[j] + thi[j + 1]) / 2
                thi = thi[:-1]
                vpe[acum_el[i] - nev[i]:acum_el[i], 0] = np.transpose(r)
                vpe[acum_el[i] - nev[i]:acum_el[i], 1] = np.arctan(
                    (points[i + 1, 1] - points[i, 1]) / (points[i + 1, 0] - points[i, 0]))
                vpe[acum_el[i] - nev[i]:acum_el[i], 2] = np.sqrt(
                    (points[i, 0] - points[i + 1, 0]) ** 2 + (points[i, 1] - points[i + 1, 1]) ** 2) / nev[i]
                vpe[acum_el[i] - nev[i]:acum_el[i], 3] = np.transpose(thi)
                vpe[acum_el[i] - nev[i]:acum_el[i], 4] = matseg[i]
            else:
                # node part to fill vpn
                z = np.linspace(points[i, 0], points[i + 1, 0], nev[i],
                                endpoint=False)  # z has dimension nev[i] equivalent to r in the previous case
                dr1 = (points[i, 1] - points[i - 1, 1]) / (points[i, 0] - points[i - 1, 0])
                dr2 = (points[i + 2, 1] - points[i + 1, 1]) / (points[i + 2, 0] - points[i + 1, 0])
                f = sp.interpolate.CubicHermiteSpline([points[i, 0], points[i + 1, 0]],
                                                      [points[i, 1], points[i + 1, 1]], [dr1, dr2])
                r = f(z)  # now r is an interpolation in between both points
                # the last one is naturally excluded by the z vector size, doesn't include the endpoint
                vpn[acum_el[i] - nev[i]:acum_el[i], :] = np.transpose(np.vstack((z, r)))
                # element part to fill vpe
                thi = np.interp(z, [points[i, 0], points[i + 1, 0]], [thicnesses[i], thicnesses[i + 1]])
                z = np.concatenate((z, np.array([points[i + 1, 0]])))
                r = np.concatenate((r, np.array([points[i + 1, 1]])))
                thi = np.concatenate((thi, np.array([thicnesses[i + 1]])))
                for j in range(0, nev[i]):
                    vpe[acum_el[i] - nev[i] + j, 0] = r[j]
                    vpe[acum_el[i] - nev[i] + j, 1] = np.arctan((r[j + 1] - r[j]) / (z[j + 1] - z[j]))
                    vpe[acum_el[i] - nev[i] + j, 2] = np.sqrt((r[j + 1] - r[j]) ** 2 + (z[j + 1] - z[j]) ** 2)
                    vpe[acum_el[i] - nev[i] + j, 3] = (thi[j] + thi[j + 1]) / 2
                    vpe[acum_el[i] - nev[i] + j, 4] = matseg[i]
    vpn[nn - 1, :] = points[-1, :]
    # print(vpn, len(vpn))

    # plt.scatter(vpn[:-1,0], vpe[:,3], marker='o', c='b', s=2)
    # makes a scatter plot with first column (z) in the x axis and second column (r) in the y axis
    # plt.grid()
    # plt.axis('equal')
    # plt.show()



#carregamento estatico
def pressao(w,nev,nl,press_est,nn):
    pressure = np.zeros(nn)
    for i in range(0, nl):
        #y = np.array([df.loc[i, 'pressure'], df.loc[i+1, 'pressure']]) #same
        y = press_est
        y_axis = np.linspace(y[i],y[i+1],nev[i], endpoint=False)#y_axis = np.linspace(max(y), min(y), nev[i] + 1)#same
        pressure[w[i] - nev[i] : w[i]] += y_axis#press_node#create a two array matrix with the pressure in each node in the whole geometry
    pressure[-1] = press_est[-1]
    #print(pressure)
    return pressure


def medium_pressure(pressao, ne):
    press_medium = np.zeros(ne)
    for i in range(0, ne):
        press_medium[i] = (pressao[i+1] + pressao[i])/2
    #print(press_medium)
    return press_medium


def loading(ne: int, pressure) -> None:  # To be verified
    global load_vct
    global vpe
    load_vct = np.zeros(3 * (ne + 1))
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        hi = vpe[i, 2]
        p = pressure[i]
        #print(phi, ri, hi, p)
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2*np.pi*hi*p*np.array([A11, A12, A13, A14, A15, A16])

        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr
    print(load_vct)
    return load_vct

# Carregamento Dinâmico
def func_carr_t (funcoes, A, B, w, b, t_final, pi, util, t_col, p_col):
    if util[0] == 1:
        n = np.size(funcoes)
        dt = 0.01
        T = np.arange(0, seg_max, dt)
        seg_max = np.max(t_final)
        #print(t)
        #print(np.size(t))
        P = np.zeros(np.size(T))
        P[0]=pi[0]
        for i in range(0,n):
            if funcoes[i] == 1:
                A1 = A[0]      #ler do excel
                B1 = B[0]
                w1 = w[0]
                b1 = b[0]
                first_zero_index1 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t1 = t_final[0] - dt*first_zero_index1
                t_sin = np.arange(0, t1, dt)
                t_iter1 = np.size(t_sin)
                last_index1 = first_zero_index1 + t_iter1

                #print(t[first_zero_index1:last_index1])
                P_1 = np.zeros(t_iter1)
                for i in range (0,t_iter1):
                    t_1 = t_sin[i]
                    P_1[i] = A1*np.sin(w1*t_1+b1) + P[first_zero_index1-1] - A1*np.sin(w1*t_sin[0]+b1)
                #print(t_sin)
                P[first_zero_index1:last_index1] = P_1


            elif funcoes[i] == 2:
                A2 = A[1]
                B2 = B[1]
                first_zero_index2 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t2 = t_final[1] - first_zero_index2*dt
                t_exp = np.arange(0, t2, dt)
                t_iter2 = np.size(t_exp)
                last_index2 = first_zero_index2 + t_iter2
                #print(t[first_zero_index2:last_index2])
                P_2 = np.zeros(t_iter2)
                for i in range (0, t_iter2):
                    t_2 = t_exp[i]
                    P_2[i] = A2*np.exp(B2*t_2) + P[first_zero_index2-1] - 1
                #print(t_2)
                P[first_zero_index2:last_index2] = P_2


            elif funcoes[i] == 3:
                A3 = A[2]
                first_zero_index3 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t3 = t_final[2] - first_zero_index3 * dt
                t_lin = np.arange(0, t3, dt)
                t_iter3 = np.size(t_lin)
                last_index3 = first_zero_index3 + t_iter3
                #print(t[first_zero_index3:last_index3])
                P_3 = np.zeros(t_iter3)
                for i in range(0, t_iter3):
                    t_3 = t_lin[i]
                    P_3[i] = A3*t_3 + P[first_zero_index3-1]

                P[first_zero_index3:last_index3] = P_3

            elif funcoes[i] == 4:
                first_zero_index4 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t4 = t_final[3] - first_zero_index4 * dt
                #print(t4)
                t_cst = np.arange(0, t4, dt)
                t_iter4 = np.size(t_cst)
                last_index4 = first_zero_index4 + t_iter4
                P_4 = np.zeros(t_iter4)
                for i in range(0, t_iter4):
                    P_4[i] = P[first_zero_index4-1]
                #print(P_4)
                P[first_zero_index4:last_index4] = P_4
    elif util[0] == 0:
        T = t_col
        P = p_col
    return P, T

def Carr_t(loading, t, T, P, press_max_est):

    p_col_adim = np.zeros(np.size(P))
    p_col_adim = P/press_max_est
    print(np.max(p_col_adim))
    P_t_adim = np.interp(t,T,p_col_adim)

    loading = loading * P_t_adim
    print(loading)
    return loading

def load_p(vpe, ne, P, pressure_nodes):

    max = np.amax(pressure_nodes)
    pressure_nodes1 = np.zeros(np.size(pressure_nodes))
    for i in range(0, ne+1):
        pressure_nodes1[i] = pressure_nodes[i] / max * P
    #pressão media
    press_medium = np.zeros(ne)
    for i in range(0, ne):
        press_medium[i] = (pressure_nodes1[i + 1] + pressure_nodes1[i]) / 2
    #vetor carregamento
    load_vct = np.zeros(3 * (ne + 1))
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        hi = vpe[i, 2]
        p = press_medium[i]
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2 * np.pi * hi * p * np.array([A11, A12, A13, A14, A15, A16])

        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr

    return load_vct


def main():
    global mat
    mat = np.array([[2800, 70 * 10 ** 9, 0.33, 200 * 10 ** 6, 70 * 10 ** 6],
                    [7600, 200 * 10 ** 9, 0.33, 800 * 10 ** 6, 200 * 10 ** 6]])
    # points of interest to define each segment for interpolation
    points = np.array([[0.0, 0.001],
                       [0.0, 0.49],
                       [0.970, 0.49],
                       [0.990, 0.25],
                       [1.000, 0.20],
                       [1.100, 0.35]])


    interpolation = np.array([0, 0, 0, 1, 0])
    thicnesses = np.array([0.03, 0.008, 0.008, 0.05, 0.04, 0.03])
    matseg = np.array([0, 0, 0, 0, 0])


    nev = np.array([5, 7, 6, 3, 2])  # number of elements per segment
    ne = np.sum(nev)  # total number of elements
    nn = ne + 1 # total numer of nodes
    acum_el = np.cumsum(nev, dtype=int)  # each entry in this vect is the number of elements of the present segment + those that came before
    nl = len(nev) # total number of segments
    t=13.91 # time
    #t_col = np.arange(0,11) # vector corresponding to the time interval
    #P_col = np.arange(0,11) # vector with the corresponding pressures
    #press_max = np.max(P_col) # max pressure value in the pressure vector
    press_est = np.array([729.35346, 27.64, 27.64, 26.995, 26.6726, 13.589206]) # pressure distribution along the geometry
    press_max_est = np.max(press_est)

    #df = pd.read_excel('C:\\Users\\Nuno\\Desktop\\Mecanica Estrutural\\Trabalhos de Mecâncica Estrutural\\Livro1.xlsx',engine='openpyxl', sheet_name='Loading')
    #print(df)
    #A = df['b']
    #print(A)
    #print(df)
    nod_and_element(points, nev, ne, acum_el, thicnesses, matseg, interpolation)

    #print(vpe[0,2])
    # print(vpn)
    # print(vpe)
    #loading(ne, medium_pressure(pressao(acum_el, nev, nl, pres, nn), ne))

    loading(ne,medium_pressure(pressao(acum_el,nev,nl,press_est,nn),ne))
    # kelements = Kestacked(ne)
    #k_global(ne)
    # print(Pmatrix(0.4,1,np.pi/2))
    # pressure = np.ones(ne)
    # loading(ne, pressure)

    df = pd.read_excel('C:\\Users\\Nuno\\Desktop\\Mecanica Estrutural\\Trabalhos de Mecâncica Estrutural\\Livro2.xlsx',
                       engine='openpyxl')
    A = df['A']
    B = df['B']
    w = df['w']
    b = df['b']
    t_final = df['tf']
    pi = df['Pi']
    funcoes = df['funções']
    util = df['utilizador']
    t_col = df['t_col']
    p_col = df['P_col']
    #print(util)

    P = func_carr_t(funcoes, A, B, w, b, t_final, pi, util, t_col, p_col)[0]
    T = func_carr_t(funcoes, A, B, w, b, t_final, pi, util, t_col, p_col)[1]


    #carregamento Dinamico

    Carr_t(loading(ne,medium_pressure(pressao(acum_el,nev,nl,press_est,nn),ne)),t, T, P, press_max_est)

    #print(A, B, w, b, t_inicial, t_final)
    fig, ax = plt.subplots()

    #Plot multiple lines on the same axis
    ax.plot(T, P)
    plt.grid(True)

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()


# Atention to units system, is it mm or m? needs to be coherent
if __name__ == '__main__':
    main()
