import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import os
import time

start_time = time.time()

#Folder Names
main_folder = "FEM Analysis - Data"
stress_folder = "Stress Data"
strain_folder = "Strain Data"
natfreqs_folder = "Natural Frequencies Data"

points_list = [
    [0,0],
    [0, 48.59],
    [810, 48.59],
    [868.85, 11.91],
    [900,9],
    [950,10],
    [1001.9,23.3],
]

#Slice angle to be poltted
rev_degrees = 180

# Matrix of stress values for the surface (random values for example)
stress_matrix = np.random.rand(100, len(points_list) - 1)

# Matrix of strain values for the surface (random values for example)
strain_matrix = np.random.rand(100, len(points_list) - 1)

#Natural Frequencies generator (random values for example)
total_height = 1001.9
z_coordinates = np.linspace(1, 10, 10)
natural_freqs = natural_freqs = 0.03 + 0.1 * (z_coordinates/1000)**2 + 0.5 * (z_coordinates/1000)
# Writing data to a text file
with open('data_natfreqs.txt', 'w') as file:
    file.write("Z_coordinates, Natural_frequencies\n")
    for z, freq in zip(z_coordinates, natural_freqs):
        file.write(f"{z}, {freq}\n") 

#Functions

def folders_creator (main_folder,stress_folder,strain_folder,natfreqs_folder):
    # Check if the folder already exists; if not, create the folder
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
        print(f"Folder '{main_folder}' created successfully.")
    else:
        print(f"The folder '{main_folder}' already exists.")
    
    
    
    # Build the full path for the new folder inside the main folder
    stress_path = os.path.join(main_folder, stress_folder)
    strain_path = os.path.join(main_folder, strain_folder)
    natfreqs_path = os.path.join(main_folder, natfreqs_folder)

    # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(stress_path) : 
        os.mkdir(stress_path)
        #print(f"Folder '{stress_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{stress_path}' already exists inside '{main_folder}'.")
        
    # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(strain_path) : 
        os.mkdir(strain_path)
        #print(f"Folder '{strain_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{strain_path}' already exists inside '{main_folder}'.")

     # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(natfreqs_path) : 
        os.mkdir(natfreqs_path)
        #print(f"Folder '{natfreqs_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{natfreqs_path}' already exists inside '{main_folder}'.")
    
def geometry_plot(points, rev_degrees,main_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rev_angle = m.radians(rev_degrees)

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

  
    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder, "geometry_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def stress_plot(points, rev_degrees, stress_matrix,stress_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rev_angle = m.radians(rev_degrees)

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='blue', alpha=0.5)
        ax.plot(x2, y2, z2, color='blue', alpha=0.5)

        # Fill the space between revolutions with a surface and apply color based on stress values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), cmap='viridis_r', alpha=0.8)
        surf.set_array(stress_matrix[:, i])

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar with shrink to reduce its size
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label('Stress')

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,stress_folder, "stress_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def strain_plot(points, rev_degrees, strain_matrix,strain_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rev_angle = m.radians(rev_degrees)

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='blue', alpha=0.5)
        ax.plot(x2, y2, z2, color='blue', alpha=0.5)

        # Fill the space between revolutions with a surface and apply color based on strain values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), cmap='plasma', alpha=0.8)
        surf.set_array(strain_matrix[:, i])  # Apply strain values to the surface

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar with shrink to reduce its size
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label('Strain')

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,strain_folder, "strain_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def plot_frequencies(coordinates_z, natural_frequencies,main_folder,natfreqs_folder):
    plt.figure()
    plt.plot(coordinates_z, natural_frequencies, marker='o', linestyle='', color='b')
    plt.xlabel('Vibration Mode')
    plt.ylabel('Natural Frequency')
    plt.title('Natural Frequencies Graph')
    plt.grid(True)

     # Adding labels with coordinates (z, natural frequency)
    for z, freq in zip(coordinates_z, natural_frequencies):
        plt.text(z,freq, f'({z:.2f},{freq:.5f})', fontsize=8, ha='right')


    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,natfreqs_folder, "natural_frequencies_graph.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()

#Plots/Figures

#Creation of folders to add files from the analysis
folders_creator(main_folder,stress_folder,strain_folder,natfreqs_folder)

#Geometry Plot
geometry_plot(points_list,rev_degrees,main_folder)

#Stress
#stress_plot(points_list,rev_degrees,stress_matrix,stress_folder)

#Strain
#strain_plot(points_list,rev_degrees,strain_matrix,strain_folder)

#Natural Frequencies 
#plot_frequencies(z_coordinates,natural_freqs,main_folder,natfreqs_folder)

#Time taken calculation
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds.")



