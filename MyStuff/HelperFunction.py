import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch, Circle, Arc, Patch, Arrow
from matplotlib.lines import Line2D 
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Line3D

outdir = f'/home/useradd/projects/bilby/MyStuff/images/'


def calculate_chirp_mass_and_ratio(m1, m2):
    chirp_mass = (m1 * m2) ** (3./5.) / (m1 + m2) ** (1./5.)
    mass_ratio = m2 / m1
    return chirp_mass, mass_ratio
def calculate_m1_m2_from_chirp_mass_and_ratio(chirp_mass, mass_ratio):
    # Since (m1 * m2) = (chirp_mass * (m1 + m2)^(1/5))^5 / (m1 * m2)^(3/5)
    # and mass_ratio = m2 / m1, we can solve the quadratic equation for m1:
    # m1^2 - (chirp_mass * (1 + mass_ratio)^(1/5))^5 / mass_ratio^(3/5) * m1 + chirp_mass^5 = 0
    A = 1
    B = -(chirp_mass * (1 + mass_ratio) ** (1./5.)) ** 5 / mass_ratio ** (3./5.)
    C = chirp_mass ** 5
    m1 = (-B + (B**2 - 4 * A * C)**0.5) / (2 * A)
    m2 = mass_ratio * m1
    return m1, m2






    def draw_orbit(self, center_of_mass, radius):
        p = np.linspace(0, 2 * np.pi, 100)
        x = center_of_mass[0] + radius * np.cos(p)
        y = center_of_mass[1] + radius * np.sin(p)
        z = center_of_mass[2] + radius * np.sin(p) * 0.1  # Small z variation for 3D effect
        self.ax.plot(x, y, z, color='black', linestyle='dashed')

    def draw_luminosity_distance(self, center_of_mass, distance_vector):
        # Update to add label for luminosity distance
        end_point = center_of_mass + distance_vector
        self.ax.plot([center_of_mass[0], end_point[0]],
                     [center_of_mass[1], end_point[1]],
                     [center_of_mass[2], end_point[2]],
                     color='purple', linestyle='-', linewidth=2, label='Luminosity Distance')
        self.ax.text(end_point[0], end_point[1], end_point[2], '$d_L$', fontsize=12, color='purple')

    def draw_black_holes(self, pos_bh1, pos_bh2, spin_bh1, spin_bh2,momentum_bh1, momentum_bh2, center_of_mass):
        # Draw black holes
        self.ax.scatter(*pos_bh1, color='blue', s=100, label='m1')
        self.ax.scatter(*pos_bh2, color='red', s=200, label='m2')

        # Draw spins as arrows
        self.draw_vector(pos_bh1, pos_bh1 + spin_bh1, 'blue', '$\\vec{a}_1$')
        self.draw_vector(pos_bh2, pos_bh2 + spin_bh2, 'red', '$\\vec{a}_2$')
        
        # Draw momentum vectors as arrows
        self.draw_vector(pos_bh1, pos_bh1 + momentum_bh1, 'blue', '$\\vec{p}_1$')
        self.draw_vector(pos_bh2, pos_bh2 + momentum_bh2, 'red', '$\\vec{p}_2$')
        # Draw the center of mass
        self.ax.scatter(*center_of_mass, color='black', s=50, label='CM')

        # Draw vectors R1 and R2 from the center of mass to each black hole
        self.draw_vector(center_of_mass, pos_bh1, 'blue', '$\\vec{R}_1$')
        self.draw_vector(center_of_mass, pos_bh2, 'red', '$\\vec{R}_2$')

        # Draw angle annotations for phi_jl (angle between total angular momentum and black hole spin)
        # this is a placeholde -  need to compute the actual angles:
        #self.draw_angle_arc(pos_bh1, 0.1, 0, 45, 'green', '$\\phi_{JL}$')

    def draw_system(self, pos_bh1, pos_bh2, spin_bh1, spin_bh2, momentum_bh1, momentum_bh2, center_of_mass, lum_dist_vector):
        # Call other methods to draw the black holes, spins, orbit, and luminosity distance
        self.draw_black_holes(pos_bh1, pos_bh2, spin_bh1, spin_bh2, momentum_bh1, momentum_bh2, center_of_mass)
        self.draw_orbit(center_of_mass, 0.35)
        self.draw_luminosity_distance(center_of_mass, lum_dist_vector)
        
        orbital_normal = np.array([0, 0, 1])  # the orbital plane is in the xy-plane

        self.draw_spin_angle(pos_bh1, spin_bh1, orbital_normal, 'green', 'θ1')
        self.draw_spin_angle(pos_bh2, spin_bh2, orbital_normal, 'green', 'θ2')

        
        
        
        spin_legend_1 = Line2D([0], [0], linestyle="none", marker=">", color='blue', label='$\\vec{a}_1$')
        spin_legend_2 = Line2D([0], [0], linestyle="none", marker=">", color='red', label='$\\vec{a}_2$')
        mom_legend_1 = Line2D([0], [0], linestyle="none", marker=">", color='blue', label='$\\vec{P}_1$')
        mom_legend_2 = Line2D([0], [0], linestyle="none", marker=">", color='red', label='$\\vec{P}_2$')
        
 
        ''' arc attampet
        ## Then when calling the draw_arc_3d method:
        theta_normal = np.array([0, 0, 1])  # Normal to the orbital plane
        phi_jl_normal = calculate_normal_to_plane(spin_bh1 + spin_bh2, [0, 0, 1])  # Normal to the total spin

        # Assuming start_angle and end_angle
        start_angle_theta = 0
        end_angle_theta = np.degrees(np.arccos(np.dot(spin_bh1, [0, 0, 1]) / np.linalg.norm(spin_bh1)))
        start_angle_phi_jl = 0
        end_angle_phi_jl = np.degrees(np.arccos(np.dot(spin_bh1 + spin_bh2, [0, 0, 1]) / np.linalg.norm(spin_bh1 + spin_bh2)))

        # Draw the arcs
        # self.draw_arc_3d(pos_bh1, theta_normal, 0.1, start_angle_theta, end_angle_theta, 'green', 'θ₁')
        # self.draw_arc_3d(pos_bh2, theta_normal, 0.1, start_angle_theta, end_angle_theta, 'green', 'θ₂')
        # self.draw_arc_3d(center_of_mass, phi_jl_normal, 0.2, start_angle_phi_jl, end_angle_phi_jl, 'orange', 'φ_JL')
       
        # Draw angles with viewing lines
        # self.draw_angle_with_viewing_line(pos_bh1, spin_bh1, [0, 0, 1], 0.1, 'green', '$\\theta_1$')
        # self.draw_angle_with_viewing_line(pos_bh2, spin_bh2, [0, 0, 1], 0.1, 'green', '$\\theta_2$')
        # self.draw_angle_with_viewing_line(center_of_mass, spin_bh1 + spin_bh2, [0, 0, 1], 0.2, 'orange', '$\\phi_{JL}$')
        '''
        
        
        
        # Update the legend to include all new elements
        self.ax.legend(loc='upper right', fontsize='small')
        handles, labels = self.ax.get_legend_handles_labels()
        handles.extend([spin_legend_1, spin_legend_2, mom_legend_1, mom_legend_2])  # Add custom handles for spins

        # Draw the legend
        self.ax.legend(handles=handles, loc='upper right', fontsize='small')


