import numpy as np
import matplotlib as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg',force=True) #To allow changing the colormap

start_time = time.time()

tensao = np.random.random((50, 50)) #Matriz tensões random

#Mesh density control 
num_points_upper = 50
num_points_lower = 50
num_points_cyl   = 50 

#num_points_upper = num_points_lower = num_points_cyl = 100 #Same elements for the 3 components

#Geometry data [mm]
#Cylinder
cyl_radius = 49.2
height_cyl = 100        # NEED ATTETION!!!!!!!!!!!!!!!!!!!!!!!!!  
#Upper Shell
up_u_radius=16.25       #Cylinder Connection       
lw_u_radius=49.4
height_u = 58.85534
#Lower Shell
up_l_radius= 16.25      #Upper shell connection    
lw_l_radius= 26.3
height_l = 23.3787

# Generating points in cylindrical coordinates for the upper conical shell
theta_upper = np.linspace(0, 2 * np.pi, num_points_upper)
z_upper = np.linspace(height_l, height_l + height_u, num_points_upper)
theta_upper, z_upper = np.meshgrid(theta_upper, z_upper)

# Convert to Cartesian coordinates with specified radii for the upper conical shell
r_upper_initial = up_u_radius     #radius in mm
r_upper_final = lw_u_radius       #radius in mm
r_upper = np.linspace(r_upper_initial, r_upper_final, num_points_upper)[:, np.newaxis]
x_upper = r_upper * np.cos(theta_upper)
y_upper = r_upper * np.sin(theta_upper)
z_upper = z_upper

# Plotting the upper conical shell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_upper, y_upper, z_upper, facecolors=plt.cm.plasma(tensao), rstride=1, cstride=1) #script that paints the surface with the tensile values 

# Adjusting the cylinder diameter
cylinder_diameter = cyl_radius*2

# Adding a cylinder on top
z_cylinder = np.linspace(height_l + height_u, height_l + height_u + height_cyl, num_points_cyl)   
theta_cylinder, z_cylinder = np.meshgrid(theta_upper, z_cylinder)
x_cylinder = (cylinder_diameter / 2) * np.cos(theta_cylinder)  # Adjusted the x-coordinates for the cylinder
y_cylinder = (cylinder_diameter / 2) * np.sin(theta_cylinder)  # Adjusted the y-coordinates for the cylinder
z_cylinder = z_cylinder
ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='r')

# Generating points in cylindrical coordinates for the lower conical shell
theta_lower = np.linspace(0, 2 * np.pi, num_points_lower)
z_lower = np.linspace(0, height_l, num_points_lower)
theta_lower, z_lower = np.meshgrid(theta_lower, z_lower)

# Convert to Cartesian coordinates with specified radii for the lower conical shell
r_lower_initial = lw_l_radius       #radius in mm
r_lower_final = up_l_radius        #radius in mm
r_lower = np.linspace(r_lower_initial, r_lower_final, num_points_lower)[:, np.newaxis]
x_lower = r_lower * np.cos(theta_lower)
y_lower = r_lower * np.sin(theta_lower)
z_lower = z_lower

x_lower = -x_lower

# Plotting the lower conical shell
ax.plot_surface(x_lower, y_lower, z_lower, color='g')

#Time taken calculation
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds.")

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

