import numpy as np
import array
import os
import matplotlib.pyplot as plt


'''
 NOTES: 1) This code only works for single patch per mpi rank (i.e. ranks and patches equivalent).  
        Need to implement file read for multiple patches per rank.
        
        2) This code only works for a single AIO rank. Need to implenment file 
        read for multiple AIO ranks.
        
        3) Parameters such as nranks_x, patch_nx, etc. are hard-coded. 
        
        4) This code assumes the data laylout inside the dump files.
        
           The data in each dump file is assumed to be written as a simple sequence of bytes with the following bottom-up hierarchy:
           x-slices -> field_variable -> patches/ranks           
           
           (x-slices are always arranged contiguously regardless of number of dimensions)
           
'''


# use this routine to read from AIO dump (assuming single AIO rank)
# assumes single patch per mpi rank

def read_wombat_3d_aio(dump_num, fields, nfields):
    
    # copy data from file into a buffer                    
    filename = file_path + "Snapshots/compr-"+str(dump_num).zfill(4)+"-part00-000"  
    data_buffer = array.array(data_format)            
    data_buffer.fromfile(open(filename, 'rb'), os.path.getsize(filename) // data_buffer.itemsize)

    ix = 0
        
    # loop over ranks
    for ri in range(0,nranks_x):
        for rj in range(0,nranks_y):
            for rk in range(0,nranks_z):

                print('Reading data for mpi rank# ',ri,rj,rk)
                
                ilow = 1 - patch_nb + patch_nx*int(ri) 
                ihi  = ilow + patch_nx + 2*patch_nb - 1
                jlow = 1 - patch_nb + patch_ny*int(rj)
                jhi  = jlow + patch_ny + 2*patch_nb - 1
                klow = 1 - patch_nb + patch_nz*int(rk)
                khi  = klow + patch_nz + 2*patch_nb - 1
                                           
                izone_low = 1 + patch_nx*int(ri)
                jzone_low = 1 + patch_ny*int(rj)
                kzone_low = 1 + patch_nz*int(rk)
                
                izone_hi = izone_low + patch_nx - 1
                jzone_hi = jzone_low + patch_ny - 1
                kzone_hi = kzone_low + patch_nz - 1

                for n in range(0,nfields):                    
                    for k in range(klow,khi+1):                                                
                        for j in range(jlow,jhi+1):
                            for i in range(ilow,ihi+1):
                        
                                if i >= izone_low and i <= izone_hi and j >= jzone_low and j <= jzone_hi and k >= kzone_low and k <= kzone_hi:     
                                    fields[n][i-1][j-1][k-1] = data_buffer[ix]      
                                    
                                ix = ix + 1    
             


def read_wombat_1d_aio(dump_num, fields, nfields):
    
    # copy data from file into a buffer                    
    filename = file_path + "Snapshots/compr-"+str(dump_num).zfill(4)+"-part00-000"  
    data_buffer = array.array(data_format)            
    data_buffer.fromfile(open(filename, 'rb'), os.path.getsize(filename) // data_buffer.itemsize)


    #print("data_buffer = ",data_buffer)

    ix = 0
        
    # loop over ranks
    for ri in range(0,nranks_x):
        for rj in range(0,nranks_y):
            for rk in range(0,nranks_z):

                print('Reading data for mpi rank# ',ri,rj,rk)
                
                ilow = 1 - patch_nb + patch_nx*int(ri) 
                ihi  = ilow + patch_nx + 2*patch_nb - 1
                                           
                izone_low = 1 + patch_nx*int(ri)
                izone_hi = izone_low + patch_nx - 1
                
                for n in range(0,nfields):                    
                    for i in range(ilow,ihi+1):                        
                        if i >= izone_low and i <= izone_hi:     
                            fields[n][i-1] = data_buffer[ix]      
                            
                        ix = ix + 1    
             
            
                
# use this routine to read dumps when AIO turned off (i.e. each MPI rank outputs it's own file)       
# assumes single patch per mpi rank
def read_wombat_3d(ndump, fields, nfields):
    

    # loop over ranks
    for rk in range(0,nranks_z):
        for rj in range(0,nranks_y):
            for ri in range(0,nranks_x):

                nrank = ri + rj*nranks_x + rk*nranks_x*nranks_y
                
    
                # read meta data file to get rank coordinates                    
                filename = file_path + "Snapshots/compr-"+str(ndump).zfill(4)+"-"+str(nrank).zfill(3)  
                
                with open(filename, 'r') as file:
                   for line in file:
                       line = line.strip() 
                       #print(line)
                    
                       split_line = line.split() 
                       
                       if split_line:
                           
                           #if (split_line[0] == 'time'):
                           #     t = float(split_line[1])
                    
                           #elif (split_line[0] == 'dtime'):
                           #    dt = float(split_line[1])
                               
                           if(split_line[0] == 'boxb'):
                               my_coords_x = int(split_line[2])
                               my_coords_y = int(split_line[3])
                               my_coords_z = int(split_line[4])
            
            
                # copy data from dump file into a buffer    
                filename = "Snapshots/compr-"+str(ndump).zfill(4)+"-part00-"+str(nrank).zfill(3)  # name of the file
                data_buffer = array.array(data_format)            
                data_buffer.fromfile(open(filename, 'rb'), os.path.getsize(filename) // data_buffer.itemsize)
                
                #print(data)
                
                ilow = 1 - patch_nb + patch_nx*int(my_coords_x) 
                ihi  = ilow + patch_nx + 2*patch_nb - 1
                jlow = 1 - patch_nb + patch_ny*int(my_coords_y)
                jhi  = jlow + patch_ny + 2*patch_nb - 1
                klow = 1 - patch_nb + patch_nz*int(my_coords_z)
                khi  = klow + patch_nz + 2*patch_nb - 1
                
                           
                izone_low = 1 + patch_nx*int(my_coords_x)
                jzone_low = 1 + patch_ny*int(my_coords_y)
                kzone_low = 1 + patch_nz*int(my_coords_z)
                
                izone_hi = izone_low + patch_nx - 1
                jzone_hi = jzone_low + patch_ny - 1
                kzone_hi = kzone_low + patch_nz - 1
   
                ix = 0
                
                for n in range(0,nfields):                    
                    for k in range(klow,khi+1):                                                
                        for j in range(jlow,jhi+1):
                            for i in range(ilow,ihi+1):
                        
                                if i >= izone_low and i <= izone_hi and j >= jzone_low and j <= jzone_hi and k >= kzone_low and k <= kzone_hi:     
                                    fields[n][i-1][j-1][k-1] = data_buffer[ix]      
                                    
                                ix = ix + 1    
                         
def plot_1d(ndump,x,field,field_name,direction):
    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    FRAME_WIDTH = 25
    FRAME_HEIGHT = 16
    
    filename = "Frames/" + field_name + "_" + direction + "cut_dump="+str(ndump)+".png"
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
    plt.figure(figsize=(FRAME_WIDTH,FRAME_HEIGHT))
    plt.suptitle("Dump # " + str(ndump))

    plt.plot(x,field,'b-^')
    plt.xlabel(direction)    
    plt.ylabel(field_name)
    
    plt.savefig(filename)
    plt.close()

                            
###############################################################################

# number of space dimensions
ndims = 1

# number of patches per mpi rank
n_xpatches = 1
n_ypatches = 1
n_zpatches = 1

# number of mpi ranks
nranks_x = 1
nranks_y = 1
nranks_z = 1

# patch size
patch_nx = 64
patch_ny = 1
patch_nz = 1
patch_nb = 5                   

# world grid size
nxtot = patch_nx * nranks_x
nytot = patch_ny * nranks_y
nztot = patch_nz * nranks_z

# number of variables (rho, vx, vy, vz, ... etc.)
nfields = 7
fields = np.zeros(shape=(nfields,nxtot,nytot,nztot))
fields_1d = np.zeros(shape=(nfields,nxtot))

x = np.linspace(-0.5, 0.5, num=nxtot)

# file data format            
data_format = 'f'  # float = 'f', double = 'd' 

# output filepath
file_path = ''#'/data/uchu/tanzid/Wombat_Sprint_5/MASTER/wombat/build/'

###############################################################################



# Example: Read AIO dump# 1 and store the data in the 'fields_1d array'
read_wombat_1d_aio(1, fields_1d, nfields)

# Get rho and vx from the fields_1d array
fields_1d_rho = fields_1d[0,:]
fields_1d_vx = fields_1d[1,:]

# plot rho and vx 
#plot_1d(1,x,fields_1d_rho,'rho','x')
#plot_1d(1,x,fields_1d_vx,'vx','x')

  