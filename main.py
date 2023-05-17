import math
import cmath
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.integrate import simps

is_Q = 0
Img1, Img2, Img3, Img4, Img5, Img6, Img7= [], [], [], [], [], [], []
UmgAD, UmgAB, UmgCD, UmgBE, UmgEC, UmgC1, UmgL1 = [], [], [], [], [], [], []
gI1, gI2, gI3, gI4, gI5, gI6, gI7  = [], [], [], [], [], [], []
gUAD, gUAB, gUCD, gUBE, gUEC, gUC1, gUL1 = [], [], [], [], [], [], []
Idei1, Idei2, Idei3, Idei4, Idei5, Idei6, Idei7 = 0, 0, 0, 0, 0, 0, 0
UdeiAD, UdeiAB, UdeiCD, UdeiBE, UdeiEC, UdeiC1, UdeiL1 = 0, 0, 0, 0, 0, 0, 0
Q_1, Q_2, Q_3, Q_4, Q_5, Q_6, Q_7, Q_8, Q_9, Q_10, Q_11, Q_12, Q_13, Q_14 = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], []
sch = 0
n = 40
R1 = R2 = R3 = R4 = R5 = C1 = C2 = C3 = C4 = L1 = L2 = L3 = F = Q1 = Q2 = U = 0
while sch < n:
    print('Введите параметр R1')
    R1 = int(input())
    print('Введите параметр R2')
    R2 = int(input())
    print('Введите параметр R3')
    R3 = int(input())
    print('Введите параметр R4')
    R4 = int(input())
    print('Введите параметр R5')
    R5 = int(input())
    print('Введите параметр C1')
    C1 = int(input())
    print('Введите параметр C2')
    C2 = int(input())
    print('Введите параметр C3')
    C3 = int(input())
    print('Введите параметр C4')
    C4 = int(input())
    print('Введите параметр L2')
    L2 = int(input())
    print('Введите параметр L3')
    L3 = int(input())
    print('Введите параметр f')
    f = int(input())
    print('Введите параметр Q1')
    Q1 = int(input())
    print('Введите параметр Q2')
    Q2 = int(input())
    print('Введите параметр U')
    U = int(input())

    pi = math.acos(-1.)
    w = 2. * pi * f
    if is_Q == 0:
        L1 = 5.e-3

    Z1 = complex(R2, 0.)
    Z2 = complex(R1, 0.)
    Z3 = complex(0., w * L1 - 1. / (w * C3))
    Z4 = complex(R2, -1. / (w * C1))
    Z5 = complex(R4, 0.)
    Z6 = complex(R2, w * L1 - 1. / (w * C3))
    Z7 = complex(R5, w * L3 - 1. / (w * C2))
    Z12 = complex(Z1 * Z2 / (Z1 + Z2))
    Z56 = complex(Z5 * Z6 / (Z5 + Z6))
    Z356 = Z56 + Z3
    Z3456 = Z356 * Z4 / (Z356 + Z4)
    Z = Z12 + Z3456 + Z7
    ZC1 = complex(0., -1. / (w * C1))
    ZL1 = complex(0., w * L1)

    UAD = U
    Io = U / Z
    UCD = Io * Z7
    UBC = Io * Z3456
    UAB = Io * Z12
    I356 = UBC / Z356
    UBE = I356 * Z3
    UEC = I356 * Z56
    I1 = complex(UAB / Z1)
    I2 = complex(UAB / Z2)
    I3 = complex(UBE / Z3)
    I4 = complex(UBC / Z4)
    I5 = complex(UEC / Z5)
    I6 = complex(UEC / Z6)
    I7 = Io
    UL1 = I3 * ZL1
    UC1 = I4 * ZC1

    I1MAX = abs(I1)
    I2MAX = abs(I2)
    I3MAX = abs(I3)
    I4MAX = abs(I4)
    I5MAX = abs(I5)
    I6MAX = abs(I6)
    I7MAX = abs(I7)

    UADMAX = abs(UAD)
    UABMAX = abs(UAB)
    UCDMAX = abs(UCD)
    UBEMAX = abs(UBE)
    UECMAX = abs(UEC)
    UC1MAX = abs(UC1)
    UL1MAX = abs(UL1)

    FI1 = cmath.phase(I1)*180/pi
    FI2 = cmath.phase(I2)*180/pi
    FI3 = cmath.phase(I3)*180/pi
    FI4 = cmath.phase(I4)*180/pi
    FI5 = cmath.phase(I5)*180/pi
    FI6 = cmath.phase(I6)*180/pi
    FI7 = cmath.phase(I7)*180/pi

    FUAD = cmath.phase(UAD)*180/pi
    FUAB = cmath.phase(UAB)*180/pi
    FUCD = cmath.phase(UCD)*180/pi
    FUBC = cmath.phase(UBC)*180/pi
    FUBE = cmath.phase(UBE)*180/pi
    FUEC = cmath.phase(UEC)*180/pi
    FUC1 = cmath.phase(UC1)*180/pi
    FUL1 = cmath.phase(UL1)*180/pi

    t = 0.
    h = 1. / (f * n)
    if is_Q == 1:
        Img1.clear()
        Img2.clear()
        Img3.clear()
        Img4.clear()
        Img5.clear()
        Img6.clear()
        Img7.clear()
        UmgAD.clear()
        UmgAB.clear()
        UmgCD.clear()
        UmgBE.clear()
        UmgEC.clear()
        UmgC1.clear()
        UmgL1.clear()

    for i in range(0, n + 1):
        Img1.append(I1MAX * math.sin(w * t + FI1))
        Img2.append(I2MAX * math.sin(w * t + FI2))
        Img3.append(I3MAX * math.sin(w * t + FI3))
        Img4.append(I4MAX * math.sin(w * t + FI4))
        Img5.append(I5MAX * math.sin(w * t + FI5))
        Img6.append(I6MAX * math.sin(w * t + FI6))
        Img7.append(I7MAX * math.sin(w * t + FI7))

        UmgAD.append(UADMAX * math.sin(w * t + FUAD))
        UmgAB.append(UABMAX * math.sin(w * t + FUAB))
        UmgCD.append(UCDMAX * math.sin(w * t + FUCD))
        UmgBE.append(UBEMAX * math.sin(w * t + FUBE))
        UmgEC.append(UECMAX * math.sin(w * t + FUEC))
        UmgC1.append(UC1MAX * math.sin(w * t + FUC1))
        UmgL1.append(UL1MAX * math.sin(w * t + FUL1))
        t = t + h
        if is_Q == 0:
            gI1.append(Img1[i])
            gI2.append(Img2[i])
            gI3.append(Img3[i])
            gI4.append(Img4[i])
            gI5.append(Img5[i])
            gI6.append(Img6[i])
            gI7.append(Img7[i])

            gUAD.append(UmgAD[i])
            gUAB.append(UmgAB[i])
            gUCD.append(UmgCD[i])
            gUBE.append(UmgBE[i])
            gUEC.append(UmgEC[i])
            gUC1.append(UmgC1[i])
            gUL1.append(UmgL1[i])


    def mmm(a, n, fmin, fmax):
        fmax = a[0]
        fmin = a[0]
        for i in range(0, n):
            if a[i] < fmin: fmin = a[i]
            if a[i] > fmax: fmax = a[i]
        return fmin, fmax


    def aaa(w, z, x, n):
        s1 = 0.
        xmax, xmin = 0, 0
        xmax, xmin = mmm(w, n, xmax, xmin)
        for i in range(0, n):
            s1 = s1 + w[i] ** 2
        s2 = 0.
        for i in range(0, n - 1):
            s2 = s2 + w[i] ** 2
        q = math.sqrt(z * x * (xmin + xmax + 4. * s1 + 2. * s2) / 6.)
        return q


    Idei1 = aaa(Img1, f, h, 40)
    Idei2 = aaa(Img2, f, h, 40)
    Idei3 = aaa(Img3, f, h, 40)
    Idei4 = aaa(Img4, f, h, 40)
    Idei5 = aaa(Img5, f, h, 40)
    Idei6 = aaa(Img6, f, h, 40)
    Idei7 = aaa(Img7, f, h, 40)

    UdeiAD = aaa(UmgAD, f, h, 40)
    UdeiAB = aaa(UmgAB, f, h, 40)
    UdeiCD = aaa(UmgCD, f, h, 40)
    UdeiBE = aaa(UmgBE, f, h, 40)
    UdeiEC = aaa(UmgEC, f, h, 40)
    UdeiC1 = aaa(UmgC1, f, h, 40)
    UdeiL1 = aaa(UmgL1, f, h, 40)
    if is_Q == 1:
        kq = sch + 1
        Q_1.append(Idei1)
        Q_2.append(Idei2)
        Q_3.append(Idei3)
        Q_4.append(Idei4)
        Q_5.append(Idei5)
        Q_6.append(Idei6)
        Q_7.append(Idei7)

        Q_8.append(UdeiAD)
        Q_9.append(UdeiAB)
        Q_10.append(UdeiCD)
        Q_11.append(UdeiBE)
        Q_12.append(UdeiEC)
        Q_13.append(UdeiC1)
        Q_14.append(UdeiL1)
    if is_Q == 0:
        f = open('resultpython.txt', 'w')
        th = ['Parament', 'Max znach', 'dei znach', 'ugol']
        td = ['I1, A', I1MAX, Idei1, FI1,
              'I2, A', I2MAX, Idei2, FI2,
              'I3, A', I3MAX, Idei3, FI3,
              'I4, A', I4MAX, Idei4, FI4,
              'I5, A', I5MAX, Idei5, FI5,
              'I6, A', I6MAX, Idei6, FI6,
              'I7, A', I7MAX, Idei7, FI7,
              'Uad, B', UADMAX, UdeiAD, FUAD,
              'Uab, B', UABMAX, UdeiAB, FUAB,
              'Ucd, B', UCDMAX, UdeiCD, FUCD,
              'Ube, B', UBEMAX, UdeiBE, FUBE,
              'Uec, B', UECMAX, UdeiEC, FUEC,
              'UL1, B', UC1MAX, UdeiL1, FUL1,
              'UC1, B', UL1MAX, UdeiC1, FUC1]
        columns = len(th)
        table = PrettyTable(th)
        td_data = td[:]
        while td_data:
            table.add_row(td_data[:columns])
            td_data = td_data[columns:]
        f.write(str(table))
        f.write('\n')
        f.write("Мгновенные значеня I\n")
        th = ['#', 'Img1', 'Img2', 'Img3', 'Img4', 'Img5', 'Img6', 'Img7']
        table = PrettyTable(th)
        for i in range(0, n):
            table.add_row([i, gI1[i], gI2[i], gI3[i],
                           gI4[i], gI5[i], gI6[i], gI7[i]])
        f.write(str(table))
        f.write('\n')
        x = [x for x in range(0, n + 1)]
        fig, ax = plt.subplots()
        i1 = [i for i in Img1]
        i2 = [i for i in Img2]
        i3 = [i for i in Img3]
        i4 = [i for i in Img4]
        i5 = [i for i in Img5]
        i6 = [i for i in Img6]
        i7 = [i for i in Img7]
        plt.title("График изменения I")
        plt.plot(x, i1, '--', label='Img1')
        plt.plot(x, i2, '--', label='Img2')
        plt.plot(x, i3, '--', label='Img3')
        plt.plot(x, i4, '--', label='Img4')
        plt.plot(x, i5, '--', label='Img5')
        plt.plot(x, i6, '--', label='Img6')
        plt.plot(x, i7, '--', label='Img7')
        plt.legend()
        fig.savefig('График изменения I')
        plt.show()

        f.write("Мгновенные значения U\n")
        th = ['#', 'UmgAD', 'UmgAB', 'UmgCD', 'UmgBE', 'UmgEC', 'UmgC1', 'UmgL1']
        table = PrettyTable(th)
        for i in range(0, n):
            table.add_row([i, UmgAD[i], UmgAB[i], UmgCD[i], UmgBE[i], UmgEC[i],
                           UmgC1[i], UmgL1[i]])
        f.write(str(table))
        f.write('\n')
        x = [x for x in range(0, n + 1)]
        fig, ax = plt.subplots()
        u1 = [u for u in UmgAD]
        u2 = [u for u in UmgAB]
        u3 = [u for u in UmgCD]
        u4 = [u for u in UmgBE]
        u5 = [u for u in UmgEC]
        plt.title("График изменения U")
        plt.plot(x, u1, '--', label='UmgAD')
        plt.plot(x, u2, '--', label='UmgAB')
        plt.plot(x, u3, '--', label='UmgCD')
        plt.plot(x, u4, '--', label='UmgBE')
        plt.plot(x, u5, '--', label='UmgEC')
        plt.legend()
        fig.savefig('График изменения U')
        plt.show()

        f.write("\nМгновенные значения U по п.4.1в\n")
        th = ['#', 'UmgAD', 'UmgC1', 'UmgL1']
        table = PrettyTable(th)
        for i in range(0, n):
            table.add_row([i, UmgAD[i], UmgC1[i], UmgL1[i]])
        f.write(str(table))

        x = [x for x in range(0, n + 1)]
        fig, ax = plt.subplots()
        u1 = [u for u in UmgAD]
        u2 = [u for u in UmgC1]
        u3 = [u for u in UmgL1]
        plt.title("График изменения U по п.4.1в")
        plt.plot(x, i1, '--', label='UmgAD')
        plt.plot(x, i2, '--', label='UmgC1')
        plt.plot(x, i3, '--', label='UmgL1')
        plt.legend()
        plt.show()
        fig.savefig('График изменения U по п_4_1в')
        f.close()


    if is_Q == 1 and sch + 1 == n:
        f = open("resultpython.txt", "a")
        f.write('\n')
        f.write("Изменение I от Q")
        th = ['L1', 'Idei1', 'Idei2', 'Idei3', 'Idei4', 'Idei5', 'Idei6', 'Idei7']
        table = PrettyTable(th)
        for i in range(0, n):
            L1 = Q1 - hQ * i
            table.add_row([L1, Q_1[i], Q_2[i], Q_3[i],
                           Q_4[i], Q_5[i], Q_6[i], Q_7[i]])
        f.write('\n')
        f.write(str(table))
        f.write("\nИзменение U от Q\n")
        th = ['L1', 'UdeiAD', 'UdeiAB', 'UdeiCD', 'UdeiBE', 'UdeiEC', 'UdeiC1', 'UdeiL1']
        table = PrettyTable(th)
        for i in range(0, n):
            L1 = Q1 - hQ * i
            table.add_row([L1, Q_8[i], Q_9[i], Q_10[i], Q_11[i], Q_12[i], Q_13[i], Q_14[i]])
        f.write(str(table))
        x = [x for x in range(0, n)]
        fig, ax = plt.subplots()
        i1 = [i for i in Q_1]
        i2 = [i for i in Q_2]
        i3 = [i for i in Q_3]
        i4 = [i for i in Q_4]
        i5 = [i for i in Q_5]
        i6 = [i for i in Q_6]
        i7 = [i for i in Q_7]
        plt.title("График изменения I(Q)")
        plt.plot(x, i1, '--', label='Idei1')
        plt.plot(x, i2, '--', label='Idei2')
        plt.plot(x, i3, '--', label='Idei3')
        plt.plot(x, i4, '--', label='Idei4')
        plt.plot(x, i5, '--', label='Idei5')
        plt.plot(x, i6, '--', label='Idei6')
        plt.plot(x, i7, '--', label='Idei7')
        plt.legend()
        fig.savefig('График изменения I(Q)')
        plt.show()
        x = [x for x in range(0, n)]
        fig, ax = plt.subplots()
        u1 = [u for u in Q_8]
        u2 = [u for u in Q_13]
        u3 = [u for u in Q_14]
        plt.title("График изменения U(Q)")
        plt.plot(x, u1, '--', label='UdeiAD')
        plt.plot(x, u2, '--', label='UdeiC1')
        plt.plot(x, u3, '--', label='UdeiL1')
        plt.legend()
        plt.show()
        fig.savefig('График изменения U(Q)')
        f.close()

    if is_Q == 0:
        hQ = (Q1 - Q2) / n
        sch = 0
    else:
        sch = sch + 1
    L1 = (Q1 - hQ * sch)
    is_Q = 1
    if sch > n:
        f.close()
        break
