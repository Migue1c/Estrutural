Python
• Python Version: 3.11
• Libraries to install:
– pip install numpy
– pip install pandas
– pip install scipy
– pip install matplotlib

Excel File Inputs
• The entered data must be previously converted to the International System of Units to ensure
their consistency and uniformity.
• Do not input data in cells with blank headers.
• The Material sheet aims to define the properties of materials that make up the structure.
• The Loading sheet is designed to receive data related to the pressure curve for dynamic analysis.
• The Input sheet constructs the geometry and mesh, and also defines applied pressure and
constraints for static analysis.
• Points → Numbering of points.
• z and r → Coordinates of characteristic points of the structure.
• thi → Thickness of the shell at that point.
• Discontinuity → Insert 1 if it is a point of discontinuity, for example, a vertex, and 0 otherwise.
• Material → Insert 1, 2, 3, ..., n, to define the material between the point defined in this line
and the next line.
• Conditions → Boundary conditions at the point, entering a value from 0 to 7, where:
0 → Fixed
1 → Vertical Displacement
2 → Horizontal Displacement
3 → Rotation
4 → Vertical Displacement + Horizontal Displacement
5 → Vertical Displacement + Rotation
6 → Horizontal Displacement + Rotation
7 → Free
• Conditions1 → Boundary conditions for the points that will be created, between the point
defined in this line and the next line, again entering a value from 0 to 7.
• NeTotal → Total number of elements to discretize the problem for analysis.
• Mesh Type → Insert 1 if you want to build a mesh with a higher number of elements near
points with discontinuity. Insert 0 if you want a mesh with evenly distributed elements.
• Distance → Calculation of the distance between the point of this line and the next line
• Nn → Number of nodes to add between the point defined in this line and the next line.
• Loading → Value of pressure at the respective point.
• Add → Number of time intervals to add between each time frame of the pressure function
defined by the user
• Delta t → Integration time step
Note: The cells to be filled by the user in Excel are only the colored cells; the remaining cells are
derived from calculations based on the previous ones

Code Inputs
The values displayed in the pop-up window are only suggestive.
• Show → Insert 1 if you want to visualize the graph resulting from modal analysis and 0 if you
only want to save the graph. (Note: if the choice is 1, the program is paused while viewing the
graph)
• Angle of revolution → Angle of revolution (in degrees) to be shown when opening files in
Tecplot.
• Number of revolution points → Number of revolution points along the angle of revolution.
• Deformation → Deformation factor applied to the geometry of static and dynamic analysis.
The maximum value corresponds to 5 as geometry becomes imperceptible beyond that. (Note:
The deformation factor is multiplied by 100 to obtain the deformation imposed on the geome-
try)
It is important to note that the elapsed time since the program’s initialization is entirely depen-
dent on the number of elements chosen.

