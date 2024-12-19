Specimen Context:
Specimens: A total of 15 specimens, each with a unique shape.
Volume Size Distribution: The volume size distribution curves of all specimens are identical, achieved by proportionally scaling each particle's semiaxis lengths (rx, ry, rz).
Standard Volume: Defined as the volume of spherical particles (Shape 5: AR = 1, epsilon_1 = 1, epsilon_2 = 1).
Packing Aspect Ratio (AR): Ratio of the semiaxis length in the x-direction to the semiaxis length in the y-direction (AR = rx / ry).

Header Clarifications:
id: Unique identifier for each object in the DEM (Discrete Element Method) packing. IDs 0–5 correspond to six boundary walls, while IDs 6–10,005 represent the 10,000 particles.
rx: Length of the semiaxis in the x-direction of the particle.
ry: Length of the semiaxis in the y-direction of the particle.
rz: Length of the semiaxis in the z-direction of the particle.
epsl1 (epsilon_1): Shape parameter for superellipsoids, affecting curvature along the x- and y-axes.
epsl2 (epsilon_2): Shape parameter for superellipsoids, affecting curvature along the z-axis.
V: Volume of the particle.
x: x-coordinate of the particle's center.
y: y-coordinate of the particle's center.
z: z-coordinate of the particle's center.
qw: Scalar term (w) of the quaternion representing the particle’s orientation, with the x-axis as the reference direction.
qx: x-component of the quaternion representing the particle’s orientation.
qy: y-component of the quaternion representing the particle’s orientation.
qz: z-component of the quaternion representing the particle’s orientation.
lx: Length of the boundary wall in the x-direction.
ly: Length of the boundary wall in the y-direction.
lz: Length of the boundary wall in the z-direction.

MATLAB code 
a1_plot_shape.m generates the plot for each shape
Orientation_and_quaternion.m verifies the quaternion and orientation for the particle