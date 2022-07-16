#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as C
from time import time as t

def orderRandom(Z, fAlloy):
    f = fAlloy
    
    N = np.linspace(0, Z, Z+1)
    P = np.linspace(0, 0, Z+1)
    
    for i in range (Z+1):
        n = N[i]
        binom_coef = C(Z, n)
        P[i] = binom_coef * (f * (f**(Z-n)) * ((1-f)**n) + (1-f) * (f**n) * ((1-f)**(Z-n)))
    return N, P
#%%
def order2D(config):
    Z = 4
    nBox = len(config)
    N = np.linspace(0, Z, Z+1)
    P = np.linspace(0, 0, Z+1)
    
    bond = np.zeros((nBox, nBox))
    
    for i in range(nBox):
        for j in range(nBox):
            c = config[j][i]
            u, d, l, r = config[(j-1)%nBox][i], config[(j+1)%nBox][i], \
            config[j][(i-1)%nBox], config[j][(i+1)%nBox]
            neighbours = [u, d, l, r]
            
            for neighbour in neighbours:
                if neighbour != c:
                    bond[j][i] += 1
            
    for i in range(Z+1):
        P[i] = np.sum(bond == i)/nBox**2
    return N, P
#%%
def slice_energy(ixa, iya, ixb, iyb, config, Ematrix):
    #gives the energy of 3x4 or 4x3 slice 
    n = len(config)
    bonds = 0
    
    for i in list(set([ixa, ixb])):
        for j in list(set([iya, iyb])):
            c = config[j][i]
            u, d, l, r = config[(j-1)%n][i], config[(j+1)%n][i], config[j][(i-1)%n], \
            config[j][(i+1)%n]
            neighbours = [u, d, l, r]
            
            for neighbour in neighbours:
                if neighbour != c:
                    bonds += 1
            
    energy = bonds * Ematrix[1][0]
    energy -= Ematrix[config[iya][ixa]][config[iyb][ixb]]
    
    return energy
#%%
def getNeighbour(nBox, ix1, iy1, dab):
    direction = {1: [0, -1], 2: [0, 1], 3:[1, 0], 4:[-1, 0]}    
    
    initial = np.array([ix1, iy1])
    final = initial + np.array(direction[dab])
    
    final %= nBox
    
    ix2, iy2 = final[0], final[1]
    
    return ix2, iy2
#%%
def swapinfo(ixa, iya, dab, nBox, config, Ematrix):
    ixb, iyb = getNeighbour(nBox, ixa, iya, dab)
    
    E_i = slice_energy(ixa, iya, ixb, iyb, config, Ematrix)
      
    temp = np.copy(config)

    temp[iya, ixa], temp[iyb, ixb] = temp[iyb, ixb], temp[iya, ixa]
    
    E_f = slice_energy(ixa, iya, ixb, iyb, temp, Ematrix)
        
    dE = E_f-E_i
    
    return ixb, iyb, dE
#%%
def Energy(config, Ematrix):
    
    hor_E = 0.0
    ver_E = 0.0
    
    for row in config:
        for i in range(len(row)-1):
            hor_E += Ematrix[int(row[i])][int(row[i+1])]
    
    config_T = config.T
    for column in config_T:
        for i in range(len(column)-1):
            ver_E += Ematrix[int(column[i])][int(column[i+1])]
    
    total_E = hor_E + ver_E
    return total_E
#%%
def create_config(nBox, fAlloy):
    nAtoms = nBox**2
    nAlloy = int(nAtoms * fAlloy)
    onedim_config = [0 for _ in range(nAtoms)]
    
    for i in range(nAlloy):
        onedim_config[i] = 1
        
    onedim_config = np.random.permutation(onedim_config)
    
    config = np.reshape(onedim_config, [nBox, nBox])
    
    return config
#%%
def alloy2D(nBox, fAlloy, nSweeps, nEquil, T, Eam, job):
    Eaa = 0.0
    Emm = 0.0
    
    Ematrix = np.array([[Emm, Eam], [Eam, Eaa]])
    
    kB = 8.617332e-5 #eV/K
    kT = kB*T #in eV
    
    config = np.copy(create_config(nBox, fAlloy))
    
    direction = [1, 2, 3, 4]
    index = np.arange(0, nBox)
    random_x = np.random.choice(index, nSweeps)
    random_y = np.random.choice(index, nSweeps)
    random_dir = np.random.choice(direction, nSweeps)
    
    E_list = [Energy(config, Ematrix)]
    
    nBar_sample = 0
    for i in range(nSweeps):
        ixa, iya, dab = random_x[i], random_y[i], random_dir[i]
        ixb, iyb, dE = swapinfo(ixa, iya, dab, nBox, config, Ematrix)
        
        if i in range(nEquil,nSweeps):
            N1, P1 = order2D(config)
            nBar_sample += np.dot(N1,P1)
        
        if dE <= 0:
            E_list.append(E_list[-1]+dE)
            config[iya][ixa], config[iyb][ixb] = config[iyb][ixb], config[iya][ixa]
            
        else:
            R = np.random.random()
            
            if np.exp(-dE/kT) > R:
                E_list.append(E_list[-1]+dE)
                config[iya][ixa], config[iyb][ixb] = config[iyb][ixb], config[iya][ixa]
                
            else:
                E_list.append(E_list[-1])
                pass
    
    plt.figure(0)
    plt.title('Configuration for f={0}, T={1}$K$, Eam={2}$eV$'.format(fAlloy, T, Eam))
    plt.xticks(np.arange(0, nBox, 10))
    plt.yticks(np.arange(0, nBox, 10))
    plt.imshow(config, cmap='gray')
    plt.savefig('{0} Final config for f={1}, T={2}, Eam={3}.png'.format(job, fAlloy, T, Eam))
    plt.close(0)
    
    plt.figure(1)        
    plt.plot(E_list)
    plt.title('Energy Variation for f={0}, T={1}$K$, Eam={2}$eV$'.format(fAlloy, T, Eam))
    plt.xlabel('Iterations')
    plt.ylabel('Energy $(eV)$')
    plt.savefig('{0} Energy for f={1}, T={2}, Eam={3}.png'.format(job, fAlloy, T, Eam))
    plt.close(1)    
    
    N, P = order2D(config)
    N0,P0 = orderRandom(4, fAlloy)
    plt.figure(2)
    bar_width = 0.35
    plt.bar(N-bar_width/2, P, bar_width, label='Simulation')
    plt.bar(N0+bar_width/2, P0, bar_width, label='Random')
    plt.title('Distribution for f={0}, T={1}$K$, Eam={2}$eV$'.format(fAlloy, T, Eam))
    plt.xlabel('Number of unlike neighbours')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('{0} Distribution for f={1}, T={2}, Eam={3}.png'.format(job, fAlloy, T, Eam))
    plt.close(2)
    
    nBar = nBar_sample/(nSweeps-nEquil)
    Ebar = np.mean(E_list)
    C_v = np.var(E_list)/(kB*(T**2))    
    print('')
    print('C_v = {0:7.3f}'.format(C_v), 'kB')
    print('nbar = {0:7.3f}'.format(nBar))
    
    return nBar, Ebar, C_v
#%%
def alloy2D_plot(nBox, fAlloy, nSweeps, nEquil, T, Eam, job):
    Eaa = 0.0
    Emm = 0.0
    Ematrix = np.array([[Emm, Eam], [Eam, Eaa]])
    
    kB = 8.617332e-5 #eV/K
    kT = kB*T #in eV
    
    config = np.copy(create_config(nBox, fAlloy))
    
    direction = [1, 2, 3, 4]
    index = np.arange(0, nBox)
    random_x = np.random.choice(index, nSweeps)
    random_y = np.random.choice(index, nSweeps)
    random_dir = np.random.choice(direction, nSweeps)
    E_list = [Energy(config, Ematrix)]
    
    nBar_sample = 0    
    for i in range(nSweeps):
        ixa, iya, dab = random_x[i], random_y[i], random_dir[i]
        ixb, iyb, dE = swapinfo(ixa, iya, dab, nBox, config, Ematrix)
        
        if i in range(nEquil,nSweeps):
            N1, P1 = order2D(config)
            nBar_sample += np.dot(N1,P1)
        
        if dE <= 0:
            E_list.append(E_list[-1]+dE)
            config[iya][ixa], config[iyb][ixb] = config[iyb][ixb], config[iya][ixa]
        else:
            R = np.random.random()
            if np.exp(-dE/kT) > R:
                E_list.append(E_list[-1]+dE)
                config[iya][ixa], config[iyb][ixb] = config[iyb][ixb], config[iya][ixa]
            else:
                E_list.append(E_list[-1])
                pass
            
    N, P = order2D(config)    
    nBar = nBar_sample/(nSweeps-nEquil)
    Ebar = np.mean(E_list)
    C_v = np.var(E_list)/(kB*(T**2))    
    
    return nBar, Ebar, C_v
#%%
'''
start = t()
alloy2D(nBox = 10, fAlloy = 0.05, nSweeps = 100000, nEquil = 50000, T = 300, \
Eam = 0.1, job = 1)

end = t()
print(end-start)
'''
#%%
def transition_temp(T_low, T_high, nBox, fAlloy, nSweeps, nEquil, Eam, job):
    T_range = np.linspace(T_low, T_high, 150)
    C_list = []
    tt_list = []
    for T_i in T_range:
        nBar, Ebar, C = alloy2D_plot(nBox, fAlloy, nSweeps, nEquil, T_i, Eam, job)
        tt_list.append(nBar)
        C_list.append(C)
    
    plt.figure(3)
    plt.plot(T_range, C_list, 'ro')
    plt.title('Heat Capacity by Temperature for f={0}, Eam={1}$eV$'.format(fAlloy, Eam))
    plt.xlabel("Temperature / K")
    plt.ylabel("Heat Capacity / eV/K")
    plt.savefig('{0} Heat capacity for f={1}, Eam={2}.png'.format(job, fAlloy, Eam))
    plt.close(3)
    
    plt.figure(4)
    plt.plot(T_range, tt_list, 'ro')
    plt.title('Transition temperature for f={0}, Eam={1}$eV$'.format(fAlloy, Eam))
    plt.xlabel("Temperature / K")
    plt.ylabel("Bond Order Parameter")
    plt.savefig('{0} Transition temperature for f={1}, Eam={2}.png'.format(job, fAlloy, Eam))
    plt.close(4)
    return    
#%%
def main():
    nBox    = 100
    nEquil  = 995000
    nSweeps = 1000000
    fAlloy_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    T_list = [300, 1000, 2000]
    Eam_list = [-0.1, 0.0, 0.1]

    file = open('stats.csv', 'w')
    file.write('''Job number, Alloy fraction, Temperature (K), Unlike bond energy (eV), \
               Average number of unlike neighbours, Average energy (eV), Heat capacity \n''')
     
    count = 0
    for fAlloy in fAlloy_list:
        for Eam in Eam_list:
            for T in T_list:
                start = t()
                count += 1
                job = '{:04d}'.format(count)
                #echo the parameters back
                print("")
                print("Simulation ", job)
                print("---------------------------------------------")
                print("Cell size                      = ", nBox)
                print("Alloy fraction                 = ", fAlloy)
                print("Total number of moves          = ", nSweeps)
                print("Number of equilibration moves  = ", nEquil)
                print("Temperature                    = ", T, "K")
                print("Bond energy                    = ", Eam, "eV")
                
                #run the simulation
                nBar, Ebar, C_v = alloy2D(nBox, fAlloy, nSweeps, nEquil, T, Eam, job)
                
                file.write('{0:4d}, {1:6.4f}, {2:8.2f}, {3:5.2f}, {4:6.4f}, {5:14.7g}, \
                           {6:14.7g}\n'.format(count, fAlloy, T, Eam, nBar, Ebar, C_v))
                end = t()
                elap_time = end - start
                print("elapsed time : %.2f" %elap_time, 's')
    file.close() 
    print("\nAll Simulations Completed")           

if __name__ == "__main__":
    main()
#%%
def transition_temp2(T_low, T_high, nBox, fAlloy, nSweeps, nEquil, Eam, job):
    T_range = np.linspace(T_low, T_high, 100)
    C_list = []
    tt_list = []
    for T_i in T_range:
        nBar, Ebar, C = alloy2D_plot(nBox, fAlloy, nSweeps, nEquil, T_i, Eam, job)
        tt_list.append(nBar)
        C_list.append(C)
    return tt_list, C_list, T_range
#%%               
def main2():
    nBox    = 10
    nEquil  = 50000
    nSweeps = 100000
    fAlloy_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    Eam_list = [-0.1, 0.0, 0.1]
    T_low = 300
    T_high = 4300
     
    count = 0
    for fAlloy in fAlloy_list:
        for Eam in Eam_list:
            start = t()
            count += 1
            job = '{:04d}'.format(count)
            print("")
            print("Simulation ", job)
            print("---------------------------------------------")
            print("Cell size                      = ", nBox)
            print("Alloy fraction                 = ", fAlloy)
            print("Total number of moves          = ", nSweeps)
            print("Number of equilibration moves  = ", nEquil)
            print("Bond energy                    = ", Eam, "eV")
            transition_temp(T_low, T_high, nBox, fAlloy, nSweeps, nEquil, Eam, job)
            end = t()
            elap_time = end - start
            print("elapsed time : %.2f" %elap_time, 's')
    print("\nAll Simulations Completed")
  
if __name__ == "__main__":
    main2()    
#%%
def main3():
    nBox_list = [10, 50, 100]
    nEquil  = 50000
    nSweeps = 100000
    fAlloy = 0.1
    Eam = -0.1
    T_low = 300
    T_high = 4300
     
    count = 0
    start = t()
    print('start')
    for nBox in nBox_list:
        count += 1
        job = '{:04d}'.format(count)
        
        print('count before' + str(count))
        tt_list, C_list, T_range = \
        transition_temp2(T_low, T_high, nBox, fAlloy, nSweeps, nEquil, Eam, job)
        print('count after' + str(count))
        
        if count == 1:
            T1 = tt_list
            C1 = C_list
        elif count == 2:
            T2 = tt_list
            C2 = C_list
        elif count == 3:
            T3 = tt_list
            C3 = C_list     
            
    print('I am plotting')        
    
    plt.figure(5)
    plt.plot(T_range, C1, 'bo', label = '10', markersize = 3)
    plt.plot(T_range, C2, 'ro', label = '50', markersize = 3)
    plt.plot(T_range, C3, 'ko', label = '100', markersize = 3)
    plt.xlabel("Temperature / K")
    plt.ylabel("Heat Capacity / eV/K")
    plt.savefig('Heat Capacity with nBox variation.png')
    plt.close(5)
    
    plt.figure(6)
    plt.plot(T_range, T1, 'bo', label = '10', markersize = 3)
    plt.plot(T_range, T2, 'ro', label = '50', markersize = 3)
    plt.plot(T_range, T3, 'ko', label = '100', markersize = 3)
    plt.xlabel("Temperature / K")
    plt.ylabel("Bond Order Parameter")
    plt.savefig('Bond order with nBox variation.png')
    plt.close(6)
    
    end = t()
    elap_time = end - start
    print("elapsed time : %.2f" %elap_time, 's')
    print("\nAll Simulations Completed")
 
if __name__ == "__main__":
    main3()    

    
    
    
    
    
    
    
    
    
    
    