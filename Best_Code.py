import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
import matplotlib.cm as cm
from scipy import interpolate,optimize,integrate
import mymodule as mm
import random
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import sys
import os
from scipy.stats import rayleigh
import pickle
import configparser

config = configparser.ConfigParser()

config.read(sys.argv[1])

#Constants
M_sun = 1.989e33
M_earth = 5.97e27
AU_to_cm = 1.496e13
year = 3.1536e7
G = 6.67e-8
k_B = 1.38e-16
m_p = 1.67e-24
M_jup = 1.898e30 #g
year = 365.25*24*3600
ELLINTCONST = 1.21106
rho_p = 3
C_D = 0.44

sigmae = config.getfloat('solid_distribution','sigmae')
sigmai = config.getfloat('solid_distribution','sigmai')

#Rate constants
c1 = 4.75
c2 = 22.6
e_star = sigmae*2
F_I = 17.3
G_I = 38.2

#Collision Constants
QG = .0001
PHI = 8.0
QS = 70000.0
MU = .5

#Disk
r_in = config.getfloat('disk','r_in')*AU_to_cm
r_out = config.getfloat('disk','r_out')*AU_to_cm
r_in_box = config.getfloat('disk','r_in_box')*AU_to_cm
r_out_box = config.getfloat('disk','r_out_box')*AU_to_cm
M_star = config.getfloat('disk','M_star')*M_sun
ratio=config.getfloat('disk','ratio')
gamma=config.getfloat('disk','gamma')
nr=config.getint('disk','nr')
t_max = config.getfloat('disk','t_max')*year

alpha_in = config.getfloat('disk','alpha_in')
alpha_out = config.getfloat('disk','alpha_out')
r_alpha = config.getfloat('disk','r_alpha') * AU_to_cm

NDG = config.getboolean('disk','NDG')

#Jupiter
M_j = config.getfloat('planet','M_j')
a_j = config.getfloat('planet','a_j') * AU_to_cm
e_j = config.getfloat('planet','e_j')
w_j = config.getfloat('planet','w_j')
planet_flag = config.getboolean('planet','planet_flag')

file_name = config.get('files','file_name')
disk_name = config.get('files','disk_name')

disk = mm.disk(r_in, r_out, r_in_box, r_out_box, M_star, ratio, gamma, nr, alpha_in, alpha_out, r_alpha)
disk.add_body(M_j*M_jup,a_j,e_j,w_j)

disk.load_dens(disk_name+'_dens')
disk.load_ad(disk_name+'_ad')


min_size = config.getfloat('resolution','min_size')
max_size = config.getfloat('resolution','max_size')
N_a_p = config.getint('resolution','N_a_p')
N_m_p = config.getint('resolution','N_m_p')
local_epsilon = config.getfloat('resolution','local_epsilon')
global_epsilon = config.getfloat('resolution','global_epsilon')

size_delta = config.getfloat('solid_distribution','size_delta')
a_min = config.getfloat('solid_distribution','a_min') * AU_to_cm
a_max = config.getfloat('solid_distribution','a_max') * AU_to_cm
#Step factor
prop = (10**((max_size+np.log10(2))-min_size))**(1/(N_m_p-1)) #Bigger than 1

epsilon = config.getfloat('cascade','epsilon')
b = config.getfloat('cascade','b')
m_min = 10.**min_size

save_state_vector = []

#Identities
delta_ij = np.transpose(np.reshape(np.tile(np.identity(N_m_p),N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))
delta_ik = np.reshape(np.tile(np.identity(N_m_p),N_m_p),(N_m_p,N_m_p,N_m_p))
delta_jk = np.transpose(np.reshape(np.tile(np.identity(N_m_p),N_m_p),(N_m_p,N_m_p,N_m_p)),(0,2,1))

def probabilistic_round(v):
    v_floor = np.floor(v)
    rand = np.random.rand(len(v))
    v_new = np.where((v-v_floor)>rand,1.,0.)
    v_new += v_floor
    return v_new

def probabilistic_round_array(v):
    v_floor = np.floor(v)
    rand = np.reshape(np.random.rand(len(v)*len(v[0])),(len(v),len(v[0])))
    v_new = np.where((v-v_floor)>rand,1.,0.)
    v_new += v_floor
    return v_new


def dfdv(v):
    return v**2/(ELLINTCONST*np.sqrt((1-v**2)*(v**2-.25)))


class state:
    def __init__(self,n_p,m_p,a_p,t,ul,dust):
            self.n_p = n_p
            self.m_p = m_p
            self.a_p = a_p
            self.ul = ul
            self.dust = dust
            self.t = t

def save_state(new_state, name):

    save_state_vector.append(new_state)
    with open(name+'.npy', 'wb') as f:
        np.save(f, save_state_vector)
    f.close()


m_p = np.logspace(min_size,np.log10(2)+max_size,N_m_p) #It can go up to double the maximum size
#Number of bins which are no longer evolved in simulation
n_frozen = np.sum(np.where(np.log10(m_p)>max_size,1,0))

r_p = (3*m_p/(4*np.pi*rho_p))**(1./3.)

#Radius for planetesimals
r_j = np.reshape(np.tile(r_p,N_m_p),(N_m_p,N_m_p))
r_k = r_j.T

#Mass for planetesimals
m_j = np.reshape(np.tile(m_p,N_m_p),(N_m_p,N_m_p))
m_k = m_j.T

time_bins = config.getint('output','time_bins')
time_min = config.getfloat('output','time_min')
time_max = config.getfloat('output','time_max')

saveTime = np.logspace(np.log10(time_min),np.log10(time_max),time_bins)#[0.1,1,10,100,1000,10000,1e5,1e6]
saveTimeIndex = 0

t_actual = 0
a_p_aux = np.logspace(np.log10(a_min),np.log10(a_max), N_a_p+1 )
a_p_array = np.sqrt(a_p_aux[:-1]*a_p_aux[1:])
width_array = a_p_aux[1:] - a_p_aux[:-1]

#Upper limit planetesimals
ul_planetesimals = np.zeros(N_a_p)
dust_mass = np.zeros(N_a_p)

n_p_array = np.zeros((N_a_p,N_m_p))
m_p_array = np.reshape(np.tile(m_p,N_a_p),np.shape(n_p_array))

ind_delta = int(np.floor(np.log(size_delta/r_p[0])/np.log(prop**(1./3.))))

solid_ratio = config.getfloat('solid_distribution','solid_ratio')

#Initial condition

fixed = int(N_m_p*0.4)

for a_index in range(len(a_p_array)):
    n_p_array[a_index] = 1e5*(m_p/m_p[-1])**(-5/6)#np.where(m_p>m_p[-fixed],(m_p/m_p[-1])**(-5/6),0)

'''
for a_index in range(len(a_p_array)):
    #Total mass in the ring
    mass_delta = solid_ratio * disk.dens(a_p_array[a_index]/AU_to_cm,0)*np.pi* ( a_p_aux[a_index+1]**2 - a_p_aux[a_index]**2)

    mass_delta_1 = mass_delta * m_p[ind_delta]/(m_p[ind_delta]+m_p[ind_delta+1])
    mass_delta_2 =mass_delta * m_p[ind_delta+1]/(m_p[ind_delta]+m_p[ind_delta+1])

    n_p_array[a_index][ind_delta] = int(mass_delta_1/m_p[ind_delta])
    n_p_array[a_index][ind_delta+1] = int(mass_delta_2/m_p[ind_delta+1])
'''




#Setting for the lambdas
randomLength = 100
lambdaVec = np.zeros(randomLength)
for i in range(0, randomLength):
    guess = .75
    move = 1.0/8
    for j in range(0, 25):
        val = integrate.quad(dfdv, 0.5, guess)[0]
        if val > (i+.5)/(randomLength):
            guess = guess - move
        else:
            guess = guess + move
        move = move/2
    lambdaVec[i] = guess


while t_actual < time_max:

    print(disk.M_j[0]/M_jup)
    disk.M_j = [M_j*M_jup]

    delta_array = []
    delta_n_array = []
    m_lost_array = []

    #Collisions
    for a_index in range(len(a_p_array)):

        a_p = a_p_array[a_index]
        n_p = n_p_array[a_index]

        v_k = np.sqrt(G*M_star/a_p)

        ring_width = width_array[a_index]

        #Numbers for planetesimals
        n_j = np.reshape(np.tile(n_p,N_m_p),(N_m_p,N_m_p))
        n_k = n_j.T

        #Eccentricities and varpis
        if planet_flag:

            e_p, w_p = disk.analytical_single_NDG(a_p/AU_to_cm,t_actual,r_p,NDG)

            h1 = np.reshape(np.tile(np.multiply(e_p,np.sin(w_p)),N_m_p),(N_m_p,N_m_p))
            k1 = np.reshape(np.tile(np.multiply(e_p,np.cos(w_p)),N_m_p),(N_m_p,N_m_p))

            h2 = h1.T
            k2 = k1.T

        else:

            e_p = np.reshape(rayleigh.rvs(size=N_m_p**2, scale = 1e-3),(N_m_p,N_m_p))
            w_p = 2*np.pi*np.random.rand(N_m_p,N_m_p)

            h1 = np.multiply(e_p,np.sin(w_p))
            k1 = np.multiply(e_p,np.cos(w_p))

            e_p = np.reshape(rayleigh.rvs(size=N_m_p**2, scale = 1e-3),(N_m_p,N_m_p))
            w_p = 2*np.pi*np.random.rand(N_m_p,N_m_p)

            h2 = np.multiply(e_p,np.sin(w_p))
            k2 = np.multiply(e_p,np.cos(w_p))


        e_error = np.reshape(rayleigh.rvs(size=N_m_p**2, scale = 1e-3),(N_m_p,N_m_p))
        w_error = 2*np.pi*np.random.rand(N_m_p,N_m_p)

        h3 = np.multiply(e_error,np.sin(w_error))
        k3 = np.multiply(e_error,np.cos(w_error))

        e_jk = np.sqrt((h1-h2+h3)**2 + (k1-k2+k3)**2)

        #This is for same size interactions
        h_j = np.diag(h1)
        k_j = np.diag(k1)

        #Errors in the diagonal
        h_e_j = np.diag(h3)
        k_e_j = np.diag(k3)

        e_jj = (np.sqrt((h_j[:-2] - h_j[2:] + h_e_j[1:-1] )**2 + (k_j[:-2] - k_j[2:] + k_e_j[1:-1])**2))/2

        e_jj_0 = np.sqrt((h_j[0] - h_j[1] + h_e_j[0] )**2 + (k_j[0] - k_j[1] + k_e_j[0])**2)
        e_jj_f = np.sqrt((h_j[-1] - h_j[-2] + h_e_j[-1] )**2 + (k_j[-1] - k_j[-2] + k_e_j[-1])**2)

        e_jj = np.concatenate(([e_jj_0],e_jj,[e_jj_f]))

        #Reintroducing it on the main array e_jk
        np.fill_diagonal(e_jk,0)
        e_jk = e_jk + np.diag(e_jj)

        lamb = np.random.randint(low=0, high=randomLength-1, size=(N_m_p,N_m_p))
        lamb = lambdaVec[lamb]

        v_col = lamb * v_k * e_jk

        #Rates
        v_esc = np.sqrt(2*G*(m_j+m_k)/(r_j+r_k))
        A_g = np.pi * (r_j + r_k)**2

        sigma_j = n_j /  (2 * np.pi * a_p * ring_width ) #Sigma_j over m_j
        e_H = ((r_j + r_k)/(3*M_sun))**(1./3.)
        sigmae_prime = c1*e_H/2 + sigmae

        R1 = sigma_j * A_g * v_k / (4 * np.pi**3 * a_p) * (F_I + (v_esc/v_k)**2 * G_I/(e_star+c1*e_H)**2 ) * (1+c2*e_H/e_star)
        R2 = ( ELLINTCONST *  sigma_j * A_g * v_k * e_jk ) / (2 * np.pi**(3/2) * sigmai * a_p ) * (1 + (v_esc/v_col)**2)
        K2 = ( ELLINTCONST *  sigma_j * A_g * v_k * e_jk ) / (2 * np.pi**(3/2) * sigmai * a_p )

        umax = e_jj/(np.pi**(0.5)*np.diag(sigmae_prime))

        R_ii = (2*np.diag(R1)/umax*(np.arctan(umax) - np.log(1+umax**2)/(2*umax)) + 2*np.diag(sigmae_prime)*np.sqrt(np.pi)*np.diag(K2)/umax*((umax**2 - np.log(1+umax**2))/2 + (1/umax)*(umax*(umax**2 - 3)/3 + np.arctan(umax)) + (np.diag(v_esc)*e_jj/(np.sqrt(np.pi)*np.diag(v_col)*np.diag(sigmae_prime)))**2*(np.log(1+umax**2)/2 + (1/umax)*(umax - np.arctan(umax)))))

        f = f = 1./(1.+( e_jk / (np.sqrt(np.pi) * sigmae_prime)**2))

        R_ij = (f*R1 + (1-f)*R2)

        np.fill_diagonal(R_ij,0)
        R_ij = R_ij + np.diag(R_ii)
        Z_ij = n_k * R_ij * year

        Z_ij[np.isnan(Z_ij)] = 0
        Z_ij[np.isinf(Z_ij)] = 0

        try:
            Z_ij = np.random.poisson(Z_ij)
        except:
            #To Do Mask large values
            Z_ij=Z_ij

        Z_jk = np.transpose(np.reshape(np.tile(Z_ij,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))

        #Largest remnant
        m_tot = m_j+m_k
        mu = m_j*m_k/(m_j+m_k)
        Q_R = 0.5*mu*v_col**2/m_tot
        r_tot = (3*m_tot/(4*np.pi*rho_p))**(1./3.)
        Q_star = QS*r_tot**(9*MU/(3 - 2*PHI))*v_col**(2.0-3*MU)+ QG*r_tot**(3.0*MU)*v_col**(2.0-3*MU)

        m_lr = (1-0.5*np.divide(Q_R,Q_star))*m_tot

        m_lr = np.where(m_lr<0,0,m_lr) #if mass is negative set to zero
        m_lr = np.where(m_lr<2*b*m_tot,0,m_lr)

        m_ref = np.transpose(np.reshape(np.tile(m_p,(N_m_p,N_m_p)),(N_m_p,N_m_p,N_m_p)),(2,1,0))
        m_ratio = np.divide(m_ref,m_lr)

        m_diff = abs(m_lr-m_ref)

        m_cut = np.where(m_lr/m_tot>=0.5,0.5*(m_tot-m_lr),np.maximum(m_tot*b,0.5*m_lr))
        m_cut =  np.where(m_cut>m_min,m_cut,m_min)

        m_lr = np.transpose(np.reshape(np.tile(m_lr,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))
        m_tot = np.transpose(np.reshape(np.tile(m_tot,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))

        m_cut = np.transpose(np.reshape(np.tile(m_cut,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))

        m_cut_ref = np.where((m_cut>m_ref) & (m_cut/prop<m_ref) , m_ref,0)
        m_cut_ref = np.sum(m_cut_ref,0)

        m_cut_ref = np.transpose(np.reshape(np.tile(m_cut_ref,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))

        m_cut_0 = np.where( (m_cut>m_ref) & (m_cut/prop<m_ref) , abs(m_cut-m_ref),0)
        m_cut_1 = np.where( (m_cut*prop>m_ref) & (m_cut<m_ref) , abs(m_cut-m_ref),0)

        m_cut_0 = np.sum(m_cut_0,0) / (np.sum(m_cut_0,0) + np.sum(m_cut_1,0))
        m_cut_0[np.isnan(m_cut_0)] = 0
        m_cut_0[np.isinf(m_cut_0)] = 0

        m_cut_0 = np.transpose(np.reshape(np.tile(m_cut_0,N_m_p),(N_m_p,N_m_p,N_m_p)),(1,0,2))
        m_cut_1 = 1 - m_cut_0

        #Largest remnants
        F_ijk = np.where(   (m_ratio<prop) & (m_ratio > 1), np.roll(m_diff,1,axis=0), 0)
        F_ijk += np.where(   (1/m_ratio<prop) & (1/m_ratio > 1), np.roll(m_diff,-1,axis=0), 0)
        #Normalized
        F_ijk = F_ijk/np.sum(F_ijk,0)

        #Cascade
        F_ijk += np.where(m_ref/prop/m_cut <= 1, m_ref**(1+epsilon) *  (1-prop**(-2-epsilon)) * (m_tot-m_lr)*m_cut_0 / (m_cut_ref*prop)**(2+epsilon) ,0)
        F_ijk += np.where(m_ref/m_cut <= 1, m_ref**(1+epsilon) * (1-prop**(-2-epsilon)) * (m_tot-m_lr)*m_cut_1 / (m_cut_ref)**(2+epsilon) , 0)

        cascade = np.where(m_ref/prop/m_cut <= 1, m_ref**(1+epsilon) *  (1-prop**(-2-epsilon)) * (m_tot-m_lr)*m_cut_0 / (m_cut_ref*prop)**(2+epsilon) ,0)
        cascade += np.where(m_ref/m_cut <= 1, m_ref**(1+epsilon) * (1-prop**(-2-epsilon)) * (m_tot-m_lr)*m_cut_1 / (m_cut_ref)**(2+epsilon) , 0)

        cascade[np.isnan(cascade)] = 0
        cascade[np.isinf(cascade)] = 0

        F_ijk[np.isnan(F_ijk)] = 0
        F_ijk[np.isinf(F_ijk)] = 0

        C_ijk = Z_jk/2 * (F_ijk-delta_ik-delta_jk)

        delta_n = np.sum(C_ijk,(1,2))

        delta_n = np.where(m_p>m_p[-fixed],0,delta_n)

        #np.min(np.where(eps1>eps2,eps1,eps2))
        #delta_t_aux = np.min((delta_t_aux,300)) #Maximum allowed timestep

        delta_n_array.append(delta_n)

        #Lost mass of the tail

        #m_lost1 = m_cut_0[0] * (m_tot[0] - m_lr[0]) * (m_cut_ref[0]*prop**2/m_min)**(-2-epsilon)
        #m_lost2 = m_cut_1[0] * (m_tot[0] - m_lr[0]) * (m_cut_ref[0]*prop/m_min)**(-2-epsilon)
        #m_lost = m_lost1 + m_lost2

        #m_lost = np.nansum(Z_ij/2 * (1 + np.identity(N_m_p) ) * m_lost)
        #m_lost = -np.sum(delta_n*m_p)/M_earth
        #m_lost_array.append(m_lost)

        #im = plt.imshow( (m_tot[0] - m_lr[0] - np.sum(cascade*m_ref,0) - m_lost2 - m_lost1) )
        #im = plt.imshow(Z_jk[0]-Z_ij)
        #plt.colorbar(im)
        #plt.show()
        #input('wait')
    delta_n_array = np.asarray(delta_n_array)



    #Inter ring interactions in one year
    e_p_array = []
    w_p_array = []
    delta_crossings = np.zeros(np.shape(delta_n_array))

    for a_p in a_p_array:

        e_p, w_p = disk.analytical_single_NDG(a_p/AU_to_cm,t_actual,r_p,NDG)

        e_p_array.append(e_p)
        w_p_array.append(w_p)

    #List of all eccentricities and varpis, grouped by semimajor axis first
    a = np.reshape(np.tile(a_p_array,(N_m_p,1)).T,(N_a_p*N_m_p))
    e = np.reshape(np.asarray(e_p_array),(N_a_p*N_m_p))
    w = np.reshape(np.asarray(w_p_array),(N_a_p*N_m_p))
    n = np.reshape(np.asarray(n_p_array),(N_a_p*N_m_p))
    r = np.reshape(np.tile(r_p,(N_a_p,1)),(N_a_p*N_m_p))

    #Prediction
    w1 = np.tile(w,(len(w),1))
    w2 = w1.T
    l1 = np.tile(a*(1-e**2),(len(w),1))
    l2 = l1.T
    e1 = np.tile(e,(len(e),1))
    e2 = e1.T

    delta_w = w1-w2

    A = l2-l1
    B = e1*l2*np.cos(delta_w) - e2*l1
    C = e1*l2*np.sin(delta_w)

    delta = C**2 + B**2 - A**2

    ind = np.where(delta>0)

    for i,j in zip(ind[0],ind[1]):
        #Not overcounting and actually in separate rings
        if i>j and i//N_m_p != j//N_m_p and n[i]>0 and n[j]>0:
            try:
                a1 = a[i]
                a2 = a[j]
                e1 = e[i]
                e2 = e[j]
                w1 = w[i]
                w2 = w[j]
                r1 = r[i]
                r2 = r[j]
                l1 = a1*(1-e1**2)
                l2 = a2*(1-e2**2)
                m1 = 4*np.pi/3 * r1**3
                m2 = 4*np.pi/3 * r2**3
                n1 = n[i]
                n2 = n[j]
                v_k = np.sqrt(G*M_star/np.sqrt(a1*a2))

                delta_w = w1-w2

                A = l2-l1
                B = e1*l2*np.cos(delta_w) - e2*l1
                C = e1*l2*np.sin(delta_w)

                delta = C**2 + B**2 - A**2

                #First angle
                cos = (-A*B+C*np.sqrt(delta))/(B**2+C**2)
                sin = (-A-B*cos)/C

                theta1 = np.arccos(cos)
                if sin<0:
                    theta1=-theta1

                #Second angle
                cos = (-A*B-C*np.sqrt(delta))/(B**2+C**2)
                sin = (-A-B*cos)/C

                theta2 = np.arccos(cos)
                if sin<0:
                    theta2=-theta2

                ri1 = l2/(1+e2*np.cos(theta1))
                ri2 = l2/(1+e2*np.cos(theta2))

                v_coll_1 = 3e6 * np.sqrt((e1*np.sin(theta1-delta_w)/np.sqrt(l1) - e2*np.sin(theta1)/np.sqrt(l2))**2 + (np.sqrt(l1)/ri1 - np.sqrt(l2)/ri1)**2)
                v_coll_2 = 3e6 * np.sqrt((e1*np.sin(theta2-delta_w)/np.sqrt(l1) - e2*np.sin(theta2)/np.sqrt(l2))**2 + (np.sqrt(l1)/ri2 - np.sqrt(l2)/ri2)**2)


                for v_col in [v_coll_1,v_coll_2]:

                    #Rates
                    v_esc = np.sqrt(2*G*(m1+m2)/(r1+r2))
                    A_g = np.pi * (r1 + r2)**2

                    ring_width = width_array[j//N_m_p]

                    sigma_j = n2 /  (2 * np.pi * np.sqrt(a1*a2) * ring_width )
                    e_H = ((r1 + r2)/(3*1.6*M_sun))**(1./3.)
                    sigmae_prime = c1*e_H/2 + sigmae

                    e_jk = v_col/v_k

                    R1 = sigma_j * A_g * v_k / (4 * np.pi**3 * np.sqrt(a1*a2)) * (F_I + (v_esc/v_k)**2 * G_I/(e_star+c1*e_H)**2 ) * (1+c2*e_H/e_star)
                    R2 = ( ELLINTCONST *  sigma_j * A_g * e_jk ) / (2 * np.pi**(3/2) * sigmai * np.sqrt(a1*a2) ) * (1 + (v_esc/v_col)**2)

                    f = f = 1./(1.+( e_jk / (np.sqrt(np.pi) * sigmae_prime)**2))

                    R = (f*R1 + (1-f)*R2)
                    Z = n1 * R * year

                    #Largest remnant
                    m_tot = m1+m2
                    mu = m1*m2/(m1+m2)
                    Q_R = 0.5*mu*v_col**2/m_tot
                    r_tot = (3*m_tot/(4*np.pi*rho_p))**(1./3.)
                    Q_star = QS*r_tot**(9*MU/(3 - 2*PHI))*v_col**(2.0-3*MU)+ QG*r_tot**(3.0*MU)*v_col**(2.0-3*MU)

                    m_lr = (1-0.5*np.divide(Q_R,Q_star))*m_tot

                    if m_lr<0 or m_lr<2*b*m_tot:
                        m_lr = 0

                    if m_lr/m_tot>=0.5:
                        m_cut = np.max([0.5*(m_tot-m_lr),m_min])
                    else:
                        m_cut = np.max([m_tot*b,0.5*m_lr,m_min])

                    m_ratio = m_p/m_lr
                    m_diff = abs(m_lr-m_p)
                    mass_dist = np.zeros(N_m_p)

                    ind_high = np.where((m_ratio<prop) & (m_ratio > 1))
                    ind_low = np.where((1/m_ratio<prop) & (1/m_ratio > 1))

                    lr_low = m_diff[ind_low]/(m_diff[ind_low] + m_diff[ind_high])
                    lr_high = 1 - lr_low

                    mass_dist[ind_low] = lr_low
                    mass_dist[ind_high] = lr_high

                    #Cascade
                    mass_dist += np.where(m_p/prop/m_cut <= 1, (1-prop**(-2-epsilon)) * (m_tot-m_lr)*lr_low / (m_cut*prop)**(2+epsilon) ,0)
                    mass_dist += np.where(m_p/m_cut <= 1, (1-prop**(-2-epsilon)) * (m_tot-m_lr)*lr_high / (m_cut)**(2+epsilon) , 0)

                    aux = np.sum(np.sum(delta_crossings,axis=0)*m_p)/M_earth

                    #New planetesimals
                    delta_crossings[j//N_m_p] += Z * mass_dist
                    #Merged planetesimals
                    delta_crossings[j//N_m_p,j%N_m_p] += -Z
                    delta_crossings[i//N_m_p,i%N_m_p] += -Z

                    #print('Mass change',aux - np.sum(np.sum(delta_crossings,axis=0)*m_p)/M_earth,(m_tot-m_lr)*(m_cut/m_min)**(-2-epsilon)/M_earth)

                    #print(delta_crossings[i//N_m_p])
                    #print(delta_crossings[j//N_m_p])
                    #print('Mass change',np.sum(np.sum(delta_crossings,axis=0)*m_p)/M_earth)
                    #input('wait')
            except:
                print('Error in crossing.')
                print(a[i],a[j],e[i],e[j],w[i],w[j],r[i],r[j],a1*(1-e1**2),a2*(1-e2**2),4*np.pi/3 * r1**3,4*np.pi/3 * r2**3,n[i],n[j],np.sqrt(G*M_star/np.sqrt(a1*a2)))

    delta_n_array += delta_crossings


    eps1 = np.abs(np.where(n_p_array>0,n_p_array,1)*local_epsilon/delta_n_array)

    eps_mass = np.reshape(np.tile(np.sum(n_p_array*m_p_array,axis=1),N_m_p),(N_m_p,N_a_p)).T
    eps2 = np.abs(1/np.divide(delta_n_array*m_p_array,global_epsilon*eps_mass))

    #eps1 = eps1[:-n_frozen]
    #eps2 = eps2[:-n_frozen]

    eps1 = np.where(np.isnan(eps1),1e30,eps1)
    eps1 = np.where(np.isinf(eps1),1e30,eps1)

    eps2 = np.where(np.isnan(eps2),1e30,eps2)
    eps2 = np.where(np.isinf(eps2),1e30,eps2)

    time_elapsed = np.min([np.min(np.max([np.min(eps1,axis=1),np.min(eps2,axis=1)],axis=0)),1e3])

    change_in_number = probabilistic_round_array(np.asarray(delta_n_array)*time_elapsed)

    #if np.sum(np.sum(change_in_number,axis=0)*m_p)/M_earth > 0:
    #    change_in_number = probabilistic_round_array(np.asarray(delta_n_array)*time_elapsed)
    #prob_error = np.sum((delta_n_array*time_elapsed - change_in_number)*m_p_array)/np.sum((delta_n_array*time_elapsed)*m_p_array)

    '''
    if abs(prob_error) > 0.5:
        print('Something went wrong with probabilistic round')
        print(prob_error)
        input('wait')
    '''

    #print(np.max(abs(change_in_number-delta_n_array)))
    n_p_array += change_in_number


    #Lost mass
    dust_mass += abs(np.sum(change_in_number*m_p_array,1)) #np.asarray(m_lost_array) * time_elapsed

    reset_mass = np.sum(np.sum(n_p_array,axis=0)*m_p)/M_earth
    #If any bin is below zero, set it back to zero. This may generate mass but it shouldn't be much because of the limit on the timestep.
    n_p_array = np.where(n_p_array<0,0,n_p_array)


    '''
    reset_mass -= np.sum(np.sum(n_p_array,axis=0)*m_p)/M_earth
    reset_mass = abs(reset_mass)

    if reset_mass > 1e-3:
        print(t_actual,time_elapsed,reset_mass)
        print('--------------Mass created-------------------')
    '''

    #Reseting bins above upper limit
    n_p_array = n_p_array.T

    #Going through each size
    for reset in range(n_frozen):
        ul_planetesimals += n_p_array[-(reset+1)]*m_p_array.T[-(reset+1)]
        n_p_array[-(reset+1)] = 0 * n_p_array[-(reset+1)]

    n_p_array = n_p_array.T

    total_mass = np.sum(np.sum(n_p_array,axis=0)*m_p)/M_earth

    t_actual += time_elapsed

    #change_in_mass_real = np.sum(np.sum(np.asarray(delta_n_array)*time_elapsed,axis=0)*m_p)/M_earth
    change_in_mass = np.sum(np.sum(change_in_number,axis=0)*m_p)/M_earth

    #if change_in_mass > 1e-3:
    #    print(t_actual,time_elapsed,change_in_mass,change_in_mass_real)
    #    print('--------------Mass prob-------------------')

    print(t_actual,time_elapsed,total_mass)
    #print(np.sum(np.sum(change_in_number,axis=0)*m_p)/M_earth)

    try:
        migrated_array = np.zeros(np.shape(n_p_array))

        #Migration
        for a_index in range(len(a_p_array)):

            a_p = a_p_array[a_index] #in cm
            n_p = n_p_array[a_index]

            e_p, w_p = disk.analytical_single_NDG(a_p/AU_to_cm,t_actual,r_p,NDG)

            dr = 1e-4 * a_p
            rho = lambda x: mm.one_planet(x,a_j,M_j/M_star,disk.alpha_dz,1/disk.h(x))*disk.dens(x/AU_to_cm,t_actual)/2/disk.H_g(x)
            drho = ( rho(a_p+dr) - rho(a_p-dr) ) /2/dr
            alpha = - drho/rho(a_p) * a_p

            beta = 3/7

            v_k = np.sqrt(G*M_star/a_p)
            tau_0 = 8/3/C_D * rho_p/rho(a_p) * r_p/v_k
            tau_0 *= 1/year
            eta = (alpha + beta)/2*(disk.H_g(a_p)/a_p)**2

            da = -2*a_p/tau_0*np.sqrt(5/8*e_p**2 + eta**2)*(eta + (alpha/4+5/16) * e_p**2 )

            new_a_p = a_p * np.ones(len(da)) + da*time_elapsed


            for new_a_p_i,k in zip(new_a_p,range(len(n_p))):

                new_a_p_i = np.clip(new_a_p_i,0.1*AU_to_cm,1.5*AU_to_cm)
                i=0
                while new_a_p_i>=a_p_array[i]:
                    i+=1

                a_1 = abs(new_a_p_i-a_p_array[i-1])
                a_2 = abs(a_p_array[i]-new_a_p_i)
                a_tot = a_1+a_2
                a_1 *= 1/a_tot
                a_2 *= 1/a_tot

                migrated_array[i-1][k] += n_p[k] * a_2
                migrated_array[i][k] += n_p[k] * a_1

        n_p_array = np.asarray(migrated_array) #Should this be rounded?
    except:
        pass

    f = open('timestep'+file_name+'.txt', 'a+')
    f.write(str(t_actual)+' , '+str(np.min(time_elapsed))+'\n')
    f.close()

    if t_actual > saveTime[saveTimeIndex]:

        #Save snapshot
        ul_instance = ul_planetesimals.copy()
        dust_instance = dust_mass.copy()
        new_state = state(n_p_array,m_p,a_p_array,t_actual,ul_instance,dust_instance)
        save_state(new_state,file_name)
        saveTimeIndex+=1
