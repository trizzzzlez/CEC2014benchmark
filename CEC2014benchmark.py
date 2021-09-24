import numpy as np
import numba

@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def High_Conditioned_Elliptic(x):
    dim = x.shape[0]
    n = numba.prange(dim)
    res = 0
    for i in n:
        res += pow(1e+6, (i - 1) / (dim - 1)) * x[i]*x[i]
    return res


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Bent_Cigar(x):
    dim = x.shape[0]
    return x[0]*x[0] + 1e+6 * np.sum(np.square(x[1:dim]))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Discus(x):
    dim = x.shape[0]
    return 1e+6 * x[0]*x[0] + np.sum(np.square(x[1:dim]))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Rosenbrock(x):
    dim = x.shape[0]
    n = numba.prange(dim - 1)
    res = 0
    for i in n:
        res += (100 * np.square(x[i]*x[i] - x[i+1]) + (x[i]-1)*(x[i]-1))
    return res

@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Weierstrass(x):
    dim = x.shape[0]
    D = numba.prange(dim)
    kmax = numba.prange(20)
    res = 0
    for i in D:
        t_1 = 0
        t_2 = 0
        for k in kmax:
            a_k = pow(0.5, k)
            b_k = pow(3, k)
            t_1 += a_k * np.cos(2 * np.pi * b_k * (x[i]+0.5))
            t_2 += a_k * np.cos(np.pi * b_k)
        res += (t_1 - dim * t_2)
    return res


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Griewank(x):
    x = np.ascontiguousarray(x)
    y = np.arange(1, x.shape[0]+1, 1)
    return 0.00025 * np.dot(x, x) - np.prod(np.cos(x / np.sqrt(y))) + 1


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Rastrigin(x):
    return np.sum(np.square(x) - 10*np.cos(2*np.pi*x) + 10)


@numba.vectorize("f8(f8, f8)")
def Modified_Schwefel_g(z, d):

    if z < -500:
        new_z = abs(z) % 500 - 500
        return new_z * np.sin(np.sqrt(abs(new_z))) - (z+500)*(z+500) / (10000*d)
    elif z > 500:
        new_z = 500 - (z % 500)
        return new_z * np.sin(np.sqrt(abs(new_z))) - (z-500)*(z-500) / (10000*d)
    else:
        return z * np.sin(np.sqrt(abs(z)))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Modified_Schwefel(x):
    dim = x.shape[0]
    return 418.9829 * dim - np.sum(Modified_Schwefel_g(x+4.209687462275036e+2, dim))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Katsuura(x):
    dim = x.shape[0]
    d_1_2 = 10 / pow(dim, 1.2)
    d_2 = 10 / (dim*dim)
    D = numba.prange(dim)
    J = numba.prange(32)
    prod = 1
    for i in D:
        t_sum = 0
        for j in J:
            t1 = pow(2, j)
            t2 = t1 * x[i]
            t_sum += (np.abs(t2 - round(t2)) / t1)
        prod *= pow(i * t_sum + 1, d_1_2)
    return d_2 * (prod - 1)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def HappyCat(x):
    x = np.ascontiguousarray(x)
    dim = x.shape[0]
    t = np.dot(x, x)
    return pow(np.abs(t - dim), 0.25) + (0.5*t + np.sum(x))/dim + 0.5


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def HGBat(x):
    x = np.ascontiguousarray(x)
    dim = x.shape[0]
    t1 = np.sum(x)
    t2 = np.dot(x, x)
    return np.sqrt(np.abs(t2*t2 - t1*t1)) + (0.5*t2 + t1)/dim + 0.5


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Expended_Griewank_plus_Rosenbrock(x):
    dim = x.shape[0]
    D_1 = numba.prange(dim - 1)
    res = 0

    for i in D_1:
        res += Modified_Schwefel(np.array([Rosenbrock(np.array([x[i], x[i + 1]]))]))

    return res + Modified_Schwefel(np.array([Rosenbrock(np.array([x[dim - 1], x[0]]))]))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Expanded_Scaffer_F6(x):

    def Scaffer_F6(x1, x2):
        t = x1*x1 + x2*x2
        return 0.5 + (pow(np.sin(np.sqrt(t)), 2)-0.5) / (1 + 0.001 * t*t)

    dim = x.shape[0]
    D_1 = numba.prange(dim-1)
    res = Scaffer_F6(x[dim-1], x[0])
    for i in D_1:
        res += Scaffer_F6(x[i], x[i+1])
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Rotated_High_Conditioned_Elliptic(x, M, o):
    M = np.ascontiguousarray(M)
    return High_Conditioned_Elliptic(np.dot(M, x - o)) + 100


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Rotated_Bent_Cigar(x, M, o):
    M = np.ascontiguousarray(M)
    return Bent_Cigar(np.dot(M, x - o)) + 200


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Rotated_Discus(x, M, o):
    M = np.ascontiguousarray(M)
    return Discus(np.dot(M, x - o)) + 300


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Rosenbrock(np.dot(M, 0.02048*(x - o))) + 400


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Ackley(x, M, o):
    M = np.ascontiguousarray(M)
    return Ackley(np.dot(M, x-o)) + 500


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Weierstrass(x, M, o):
    M = np.ascontiguousarray(M)
    return Weierstrass(np.dot(M, 0.005*(x-o))) + 600


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Griewank(x, M, o):
    M = np.ascontiguousarray(M)
    return Griewank(np.dot(M, 6*(x-o))) + 700


@numba.njit("f8(f8[:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rastrigin(x, o):
    return Rastrigin(0.0512*(x-o)) + 800


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Rastrigin(x, M, o):
    M = np.ascontiguousarray(M)
    return Rastrigin(np.dot(M, 0.0512*(x-o))) + 900


@numba.njit("f8(f8[:],f8[:])", nogil=True, fastmath=True)
def Shifted_Schwefel(x, o):
    return Modified_Schwefel(10*(x-o)) + 1000


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Schwefel(x, M, o):
    M = np.ascontiguousarray(M)
    return Modified_Schwefel(np.dot(M, 10*(x - o))) + 1100


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Katsuura(x, M, o):
    M = np.ascontiguousarray(M)
    return Katsuura(np.dot(M, 0.05*(x-o))) + 1200


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_HappyCat(x, M, o):
    M = np.ascontiguousarray(M)
    return HappyCat(np.dot(M, 0.05*(x-o))) + 1300


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_HGBat(x, M, o):
    M = np.ascontiguousarray(M)
    return HGBat(np.dot(M, 0.05*(x-o))) + 1400


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Expended_Griewank_plus_Rosenbrock(np.dot(M, 0.05*(x-o))+1) + 1500


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Expanded_Scaffer_F6(x, M, o):
    M = np.ascontiguousarray(M)
    return Expanded_Scaffer_F6(np.dot(M, x-o)+1) + 1600


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_1(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Modified_Schwefel(np.dot(np.ascontiguousarray(M[0:0.3*dim, 0:0.3*dim]), x[0:0.3*dim]-o[0:0.3*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.3*dim:0.6*dim, 0.3*dim:0.6*dim]), x[0.3*dim:0.6*dim]-o[0.3*dim:0.6*dim]))
    res += High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim]-o[0.6*dim:dim]))
    return  res + 1700


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_2(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Bent_Cigar(np.dot(np.ascontiguousarray(M[0:0.3*dim, 0:0.3*dim]), x[0:0.3*dim] - o[0:0.3*dim]))
    res += HGBat(np.dot(np.ascontiguousarray(M[0.3*dim:0.6*dim, 0.3*dim:0.6*dim]), x[0.3*dim:0.6*dim] - o[0.3*dim:0.6*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim] - o[0.6*dim:dim]))
    return res + 1800


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_3(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Griewank(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim] - o[0:0.2*dim]))
    res += Weierstrass(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim] - o[0.2*dim:0.4*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.4*dim:0.7*dim, 0.4*dim:0.7*dim]), x[0.4*dim:0.7*dim] - o[0.4*dim:0.7*dim]))
    res += Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim] - o[0.7*dim:dim]))
    return res + 1900


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_4(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = HGBat(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim] - o[0:0.2*dim]))
    res += Discus(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim] - o[0.2*dim:0.4*dim]))
    res += Expended_Griewank_plus_Rosenbrock(np.dot(np.ascontiguousarray(M[0.4*dim:0.7*dim, 0.4*dim:0.7*dim]), x[0.4*dim:0.7*dim] - o[0.4*dim:0.7*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim] - o[0.7*dim:dim]))
    return res + 2000


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_5(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[0:0.1*dim, 0:0.1*dim]), x[0:0.1*dim] - o[0:0.1*dim]))
    res += HGBat(np.dot(np.ascontiguousarray(M[0.1*dim:0.3*dim, 0.1*dim:0.3*dim]), x[0.1*dim:0.3*dim] - o[0.1*dim:0.3*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.3*dim:0.5*dim, 0.3*dim:0.5*dim]), x[0.3*dim:0.5*dim] - o[0.3*dim:0.5*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.5*dim:0.7*dim, 0.5*dim:0.7*dim]), x[0.5*dim:0.7*dim] - o[0.5*dim:0.7*dim]))
    res += High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim] - o[0.7*dim:dim]))
    return res + 2100


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_6(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Katsuura(np.dot(np.ascontiguousarray(M[0:0.1*dim, 0:0.1*dim]), x[0:0.1*dim] - o[0:0.1*dim]))
    res += HappyCat(np.dot(np.ascontiguousarray(M[0.1*dim:0.3*dim, 0.1*dim:0.3*dim]), x[0.1*dim:0.3*dim] - o[0.1*dim:0.3*dim]))
    res += Expended_Griewank_plus_Rosenbrock(np.dot(np.ascontiguousarray(M[0.3*dim:0.5*dim, 0.3*dim:0.5*dim]), x[0.3*dim:0.5*dim] - o[0.3*dim:0.5*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.5*dim:0.7*dim, 0.5*dim:0.7*dim]), x[0.5*dim:0.7*dim] - o[0.5*dim:0.7*dim]))
    res += Ackley(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim] - o[0.7*dim:dim]))
    return res + 2200


@numba.njit("f8[:](f8[:,:],int64[:])", nogil=True, fastmath=True)
def composition_omega(x_o, sig):
    n, dim = x_o.shape
    w = np.zeros(n)
    N = numba.prange(n)
    for i in N:
        c_x = np.ascontiguousarray(x_o[i, :])
        t = np.dot(c_x, c_x)
        w[i] = 1/np.sqrt(t)*np.exp(-t / (2 * dim * sig[i]))

    return w/np.sum(w)


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_1(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50])))
    res = omega[0] * (Shifted_Rotated_Rosenbrock(x, M[0:dim, :], o[0, :])-400)
    res += omega[1] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim:dim*2, :], o[1, :]) + 99.9999)
    res += omega[2] * (1e-26*Rotated_Bent_Cigar(x, M[dim*2:dim*3, :], o[2, :])-2e-24 + 200)
    res += omega[3] * (1e-6*Rotated_Discus(x, M[dim*3:dim*4, :], o[3, :])-3e-4+300)
    res += omega[4] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5, :], o[4, :]) + 399.9999)
    return res + 2300

@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_2(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([20, 20, 20])))
    res = omega[0] * (Shifted_Schwefel(x, o[0, :]) - 1000)
    res += omega[1] * (Shifted_Rotated_Rastrigin(x, M[0:dim, :], o[1, :]) - 800)
    res += omega[2] * (Shifted_Rotated_HGBat(x, M[dim:dim*2, :], o[2, :]) - 1200)
    return res + 2400


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_3(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (0.25*Shifted_Rotated_Schwefel(x, M[0:dim, :], o[0, :])-275)
    res += omega[1] * (Shifted_Rotated_Rastrigin(x, M[dim:dim*2, :], o[1, :])-800)
    res += omega[2] * (1e-7*Rotated_High_Conditioned_Elliptic(x, M[dim*2:dim*3, :], o[2, :])-1e-5 + 200)
    return res + 2500


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_4(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 10, 10, 10, 10])))
    res = omega[0] * (0.25*Shifted_Rotated_Schwefel(x, M[0:dim, :], o[0, :])-275)
    res += omega[1] * (Shifted_Rotated_HappyCat(x, M[dim:dim*2, :], o[1, :])-1200)
    res += omega[2] * (1e-7*Rotated_High_Conditioned_Elliptic(x, M[dim*2:dim*3, :], o[2, :])+199.99999)
    res += omega[3] * (2.5*Shifted_Rotated_Weierstrass(x, M[dim*3:dim*4, :], o[3, :])-1200)
    res += omega[4] * (10*Shifted_Rotated_Griewank(x, M[dim*4:dim*5, :], o[4, :])-6600)
    return res + 2600


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_5(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.array([10, 10, 10, 20, 20]))
    res = omega[0] * (10*Shifted_Rotated_HGBat(x, M[0:dim, :], o[0, :])-14000)
    res += omega[1] * (10*Shifted_Rotated_Rastrigin(x, M[dim:dim*2, :], o[1, :])-8900)
    res += omega[2] * (2.5*Shifted_Rotated_Schwefel(x, M[dim*2:dim*3, :], o[2, :])-2550)
    res += omega[3] * (25*Shifted_Rotated_Weierstrass(x, M[dim*3:dim*4, :], o[3, :])-14700)
    res += omega[4] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5, :], o[4, :]) + 399.9999)
    return res + 2700


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_6(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50])))
    res = omega[0] * (2.5*Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock(x, M[0:dim, :], o[0, :])-3750)
    res += omega[1] * (10*Shifted_Rotated_HappyCat(x, M[dim:dim*2, :], o[1, :])-12900)
    res += omega[2] * (2.5*Shifted_Rotated_Schwefel(x, M[dim*2:dim*3, :], o[2, :])-2550)
    res += omega[3] * (5e-4*Shifted_Rotated_Expanded_Scaffer_F6(x, M[dim*3:dim*4, :], o[3, :])+299.2)
    res += omega[4] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5, :], o[4, :])+399.9999)
    return res + 2800


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_7(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (Hybrid_1(x, M[0:dim, :], o[0, :])-1700)
    res += omega[1] * (Hybrid_2(x, M[dim:dim*2, :], o[1, :])-1700)
    res += omega[2] * (Hybrid_3(x, M[dim*2:dim*3, :], o[2, :])-1700)
    return res + 2900


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_8(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (Hybrid_4(x, M[0:dim, :], o[0, :])-2000)
    res += omega[1] * (Hybrid_5(x, M[dim:dim*2, :], o[1, :])-2000)
    res += omega[2] * (Hybrid_6(x, M[dim*2:dim*3, :], o[2, :])-2000)
    return res + 3000





















