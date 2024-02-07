import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch, Circle, Arc, Patch, Arrow
from matplotlib.lines import Line2D 
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
def plot_for_research_proposal():
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Black hole positions
    pos_bh1 = (0.3, 0.5)
    pos_bh2 = (0.7, 0.5)

    # Black hole circles
    bh1 = Circle(pos_bh1, 0.05, color='blue')
    bh2 = Circle(pos_bh2, 0.05, color='red')
    ax.add_patch(bh1)
    ax.add_patch(bh2)

    # Spin arrows
    ax.annotate('', xy=pos_bh1, xytext=(pos_bh1[0] + 0.1, pos_bh1[1]),
                arrowprops=dict(arrowstyle="->", color='blue'))
    ax.annotate('', xy=pos_bh2, xytext=(pos_bh2[0] - 0.1, pos_bh2[1]),
                arrowprops=dict(arrowstyle="->", color='red'))

    # Orbital arc
    orbit = Arc((0.5, 0.5), 0.4, 0.4, theta1=0, theta2=360, linestyle='dashed', color='black')
    ax.add_patch(orbit)

    # Center of mass
    ax.plot(0.5, 0.5, 'k+', markersize=10)

    # Parameter annotations
    # Luminosity distance
    ax.annotate('', xy=(0.5, 0.1), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="<->", color='purple'))
    ax.text(0.5, 0.3, '$d_L$', fontsize=12, color='purple')

    # Right ascension and declination
    ax.axhline(y=0.5, color='green', linestyle='--')
    ax.axvline(x=0.5, color='green', linestyle='--')
    ax.text(0.5, 0.6, 'ra/dec', fontsize=12, color='green')

    # Polarization angle psi
    psi_line = FancyArrowPatch((0.5, 0.7), (0.6, 0.7), arrowstyle='<->', color='orange')
    ax.add_patch(psi_line)
    ax.text(0.55, 0.72, '$\psi$', fontsize=12, color='orange')

    # Phase
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.6, 0.6),
                arrowprops=dict(arrowstyle="->", color='red'))
    ax.text(0.6, 0.6, '$\phi$', fontsize=12, color='red')

    # Additional parameters (e.g., mass ratio, chirp mass) can be annotated similarly

    # Setting limits and aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title of the plot
    ax.set_title('Binary Black Hole System')

    # Save the figure
    fig.savefig(outdir + 'binary_black_hole_system_complete.png', dpi=300)


def plot_binary_black_hole_system():
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define positions and properties of the black holes, spins, and orbits
    pos_bh1 = (0.3, 0.5)
    pos_bh2 = (0.7, 0.5)
    spin_bh1 = (0.1, 0.2)
    spin_bh2 = (-0.1, 0.2)
    momentum_bh1 = (-0.2, 0)
    momentum_bh2 = (0.2, 0)

    # Draw black holes as circles
    circle1 = Circle(pos_bh1, 0.05, color='blue')
    circle2 = Circle(pos_bh2, 0.05, color='red')
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Draw spins as arrows
    spin_arrow1 = FancyArrowPatch(pos_bh1, np.add(pos_bh1, spin_bh1), mutation_scale=20, color='blue')
    spin_arrow2 = FancyArrowPatch(pos_bh2, np.add(pos_bh2, spin_bh2), mutation_scale=20, color='red')
    ax.add_patch(spin_arrow1)
    ax.add_patch(spin_arrow2)

    # Draw momenta as dashed arrows
    momentum_arrow1 = FancyArrowPatch(pos_bh1, np.add(pos_bh1, momentum_bh1), 
                                      mutation_scale=20, linestyle='dashed', color='black')
    momentum_arrow2 = FancyArrowPatch(pos_bh2, np.add(pos_bh2, momentum_bh2), 
                                      mutation_scale=20, linestyle='dashed', color='black')
    ax.add_patch(momentum_arrow1)
    ax.add_patch(momentum_arrow2)

    # Draw orbits as dashed circles
    orbit1 = Arc((0.5, 0.5), 0.4, 0.4, theta1=0, theta2=180, linestyle='dashed', color='black')
    orbit2 = Arc((0.5, 0.5), 0.4, 0.4, theta1=180, theta2=360, linestyle='dashed', color='black')
    ax.add_patch(orbit1)
    ax.add_patch(orbit2)

    # Draw center of mass
    ax.plot(0.5, 0.5, 'kx', markersize=10)

    # Label the black holes
    ax.text(pos_bh1[0] - 0.05, pos_bh1[1], '$m_1$', ha='center', va='center', fontsize=12, color='white')
    ax.text(pos_bh2[0] + 0.05, pos_bh2[1], '$m_2$', ha='center', va='center', fontsize=12, color='white')

    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    fig.savefig(outdir + 'binary_black_hole_system_1.png', dpi=300)

    return fig, ax



class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        # The z-order is determined by the average of the z-coordinates of all the vertices
        return np.mean(zs)

class BlackHoleSystem3D:
    def __init__(self, ax):
        self.ax = ax

    def draw_vector(self, start, end, color, label):
        arrow = Arrow3D(*zip(start, end), mutation_scale=20, arrowstyle="-|>", color=color)
        self.ax.add_artist(arrow)
        mid_point = (start + end) / 2
        self.ax.text(*mid_point, label, fontsize=12, color=color)

    def draw_angle_arc(self, center, radius, start_angle, end_angle, color, label):
        # This is a placeholder for angle arc drawing implementation
        # You need to replace this with actual drawing code
        pass
        
    def rotation_matrix(self, axis, theta):

        """
        Returns the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
            
    def draw_arc_3d(self, center, normal, radius, start_angle, end_angle, color, label):
        # Generate the circle points in the plane
        t = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = center[2] + np.zeros_like(t)

        # Rotate the points according to the normal vector
        c = np.cross([0, 0, 1], normal)
        d = np.linalg.norm(c)
        if d != 0:
            c /= d  # Normalize the rotation vector
            cos_angle = np.dot([0, 0, 1], normal)
            angle = np.arccos(cos_angle)
            rot_matrix = self.rotation_matrix(c, angle)
            x, y, z = np.dot(rot_matrix, np.vstack([x - center[0], y - center[1], z - center[2]]))
            x += center[0]
            y += center[1]
            z += center[2]

        # Create a path in 3D space and add it to the plot
        points = np.array([x, y, z]).T.reshape(-1, 3)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segs, colors=color)
        self.ax.add_collection(lc)

        # Annotate the angle
        mid_idx = len(t) // 2  # Index at the midpoint of the arc
        self.ax.text(x[mid_idx], y[mid_idx], z[mid_idx], label, color=color)


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

    def draw_black_holes(self, pos_bh1, pos_bh2, spin_bh1, spin_bh2, center_of_mass):
        # Draw black holes
        self.ax.scatter(*pos_bh1, color='blue', s=100, label='m1')
        self.ax.scatter(*pos_bh2, color='red', s=100, label='m2')

        # Draw spins as arrows
        self.draw_vector(pos_bh1, pos_bh1 + spin_bh1, 'blue', '$\\vec{a}_1$')
        self.draw_vector(pos_bh2, pos_bh2 + spin_bh2, 'red', '$\\vec{a}_2$')

        # Draw the center of mass
        self.ax.scatter(*center_of_mass, color='black', s=50, label='CM')

        # Draw vectors R1 and R2 from the center of mass to each black hole
        self.draw_vector(center_of_mass, pos_bh1, 'blue', '$\\vec{R}_1$')
        self.draw_vector(center_of_mass, pos_bh2, 'red', '$\\vec{R}_2$')

        # Draw angle annotations for phi_jl (angle between total angular momentum and black hole spin)
        # this is a placeholde -  need to compute the actual angles:
        self.draw_angle_arc(pos_bh1, 0.1, 0, 45, 'green', '$\\phi_{JL}$')

    def draw_system(self, pos_bh1, pos_bh2, spin_bh1, spin_bh2, center_of_mass, lum_dist_vector):
        # Call other methods to draw the black holes, spins, orbit, and luminosity distance
        self.draw_black_holes(pos_bh1, pos_bh2, spin_bh1, spin_bh2, center_of_mass)
        self.draw_orbit(center_of_mass, 0.35)
        self.draw_luminosity_distance(center_of_mass, lum_dist_vector)
        
        spin_legend_1 = Line2D([0], [0], linestyle="none", marker=">", color='blue', label='$\\vec{a}_1$')
        spin_legend_2 = Line2D([0], [0], linestyle="none", marker=">", color='red', label='$\\vec{a}_2$')
        
        

        theta_normal = np.array([0.1, 0.1, 1])   # This is just an example and would need to be computed based on the system's actual parameters
        phi_jl_normal = np.array([-0.1, 0.1, 1])  #  This is just an example and would need to be computed based on the system's actual parameters
        # Assuming start_angle and end_angle 
        start_angle = 0
        end_angle_theta = 45  # Example end angle for θ
        end_angle_phi_jl = 30  # Example end angle for φ_JL
        # Draw the arcs for θ and φ_JL
        self.draw_arc_3d(pos_bh1, theta_normal, 0.1, start_angle, end_angle_theta, 'green', 'θ₁')
        self.draw_arc_3d(pos_bh2, theta_normal, 0.1, start_angle, end_angle_theta, 'green', 'θ₂')
        self.draw_arc_3d(center_of_mass, phi_jl_normal, 0.2, start_angle, end_angle_phi_jl, 'orange', 'φ_JL')

        # Update the legend to include all new elements
        self.ax.legend(loc='upper right', fontsize='small')
        handles, labels = self.ax.get_legend_handles_labels()
        handles.extend([spin_legend_1, spin_legend_2])  # Add custom handles for spins

        # Draw the legend
        self.ax.legend(handles=handles, loc='upper right', fontsize='small')



def draw_black_hole_system_new():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    system = BlackHoleSystem3D(ax)

    # Define black hole positions and properties
    pos_bh1 = np.array([0.3, 0.5, 0])
    pos_bh2 = np.array([0.7, 0.5, 0])
    spin_bh1 = np.array([-0.1, 0.2, 0.3])
    spin_bh2 = np.array([0.1, 0.2, -0.3])
    center_of_mass = (pos_bh1 + pos_bh2) / 2
    lum_dist_vector = np.array([0, 0, 1])  # Example vector for luminosity distance

    # Draw the complete black hole system
    system.draw_system(pos_bh1, pos_bh2, spin_bh1, spin_bh2, center_of_mass, lum_dist_vector)

    # Hide the axes and show the plot
    ax.axis('off')
    print(f"Saving figure to: {outdir}draw_black_hole_system_new.png")
    fig.savefig(outdir + 'draw_black_hole_system_new.png', dpi=300)


def draw_black_hole_system():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define black hole positions and properties
    pos_bh1 = np.array([0.3, 0.5, 0])
    pos_bh2 = np.array([0.7, 0.5, 0])
    spin_bh1 = np.array([-0.1, 0.2, 0.3])
    spin_bh2 = np.array([0.1, 0.2, -0.3])

    # Define luminosity distance line properties
    lum_dist_line = np.array([0, 0, 1])  # Example vector for luminosity distance

    # Draw black holes
    ax.scatter(*pos_bh1, color='blue', s=100, label='m1')
    ax.scatter(*pos_bh2, color='red', s=100, label='m2')

    # Draw spins as arrows
    ax.add_artist(Arrow3D(*zip(pos_bh1, pos_bh1 + spin_bh1), mutation_scale=20, arrowstyle="-|>", color='blue', label='Spin m1'))
    ax.add_artist(Arrow3D(*zip(pos_bh2, pos_bh2 + spin_bh2), mutation_scale=20, arrowstyle="-|>", color='red', label='Spin m2'))

    # Draw the center of mass
    center_of_mass = (pos_bh1 + pos_bh2) / 2
    ax.scatter(*center_of_mass, color='black', s=50, label='CM')

    # Draw dashed orbit paths
    p = np.linspace(0, 2 * np.pi, 100)
    r = 0.35
    x = center_of_mass[0] + r * np.cos(p)
    y = center_of_mass[1] + r * np.sin(p)
    z = center_of_mass[2] + r * np.sin(p) * 0.1  # small z variation for 3D effect
    ax.plot(x, y, z, color='black', linestyle='dashed', label='Orbit')

    # Draw luminosity distance
    ax.plot([center_of_mass[0], center_of_mass[0] + lum_dist_line[0]],
            [center_of_mass[1], center_of_mass[1] + lum_dist_line[1]],
            [center_of_mass[2], center_of_mass[2] + lum_dist_line[2]],
            color='purple', linestyle='-', linewidth=2, label='Luminosity Distance')
    # Labels
    ax.text(pos_bh1[0], pos_bh1[1], pos_bh1[2], 'm1', color='blue')
    ax.text(pos_bh2[0], pos_bh2[1], pos_bh2[2], 'm2', color='red')
    ax.text(center_of_mass[0], center_of_mass[1], center_of_mass[2] + 1, '$d_L$', color='purple')

    # Adjust the axes limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.5, 1.5)

    # Hide the axes
    ax.axis('off')

    # Legend
    ax.legend(loc='upper right')
    fig.savefig(outdir + 'draw_black_hole_system.png', dpi=300)

    return fig, ax



if __name__ == "__main__":
    
    
    draw_black_hole_system_new()
    draw_black_hole_system()
    
    
    plot_for_research_proposal()


    plot_binary_black_hole_system()

    parameter_order = [
        'psi', 'a_2', 'a_1', 'tilt_2', 'tilt_1', 'phi_12', 'phi_jl', 
        'luminosity_distance', 'theta_jn', 'phase', 'ra', 'mass_ratio', 
        'chirp_mass', 'dec'
    ]