import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def func_carr_t (funcoes, A, B, w, b, t_final, pi, util, t_col, p_col):
    if util[0] == 1:
        n = np.size(funcoes)
        dt = 0.01
        seg_max = np.max(t_final)
        T = np.arange(0, seg_max, dt)
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

def main():
    df = pd.read_excel('C:\\Users\\Nuno\\Desktop\\Mecanica Estrutural\\Trabalhos de Mecâncica Estrutural\\Livro2.xlsx',engine='openpyxl')
    A = df['A']
    B = df['B']
    w = df['w']
    b = df['b']
    t_final = df['tf']
    pi = df['Pi']
    funcoes = df['funções']
    t_col = df['t_col']
    p_col = df['P_col']
    util = df['utilizador']
    seg_max = np.max(t_final)
    #print(seg_max)
    P = func_carr_t(funcoes, A, B, w, b, t_final, pi, util, t_col, p_col)[0]
    T = func_carr_t(funcoes, A, B, w, b, t_final, pi, util, t_col, p_col)[1]
    index_P_max = np.argmax(P)
    #print(index_P_max)

    #print(A, B, w, b, t_inicial, t_final)
    fig, ax = plt.subplots()

    #Plot multiple lines on the same axis
    ax.plot(T, P)
    plt.grid(True)
    #ax.plot(time, phi, label='phi')
    #ax.plot(time, psi, label='phi')

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()


# Atention to units system, is it mm or m? needs to be coherent
if __name__ == '__main__':
    main()
