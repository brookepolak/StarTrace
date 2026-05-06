import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt


class SubClusters():
    """
    A class that generates N equal sized sub-clusters
    for use as initial conditions in an N-body simulation.
    """
    
    def __init__(self, num_total,  num_subclusters, radius, subcluster_rho,
                 subcluster_virial_ratio=0.1, global_virial_ratio=None, seed=0, 
                 masses=1.0, sample_IMF=False, IMF_range=(0.08,20), G=0.0044983099795944):
        """
        
        All units are in Msun/pc/Myr.

        Args:
            num_total (_type_): Total number of stars in the cluster.
            num_subclusters (_type_): Number of subclusters.
            radius (_type_): Radius of the entire star cluster -- radius within which to initialize
                             subclusters.
            subcluster_rho (_type_): Stellar density of the subclusters.
            subcluster_virial_ratio (float, optional): Virial ratio (Ekin/Epot) of subclusters. 
                                                       Equilibrium = 0.5. Defaults to 0.1.
            global_virial_ratio (_type_, optional): Global virial ratio. Defaults to None.
            seed (int, optional): Random seed. Defaults to 0.
            masses (float, optional): Masses of the stars. Defaults to 1.0.
            sample_IMF (bool, optional): Whether to sample the IMF for star masses. NOT IMPLEMENTED. 
                                         Defaults to False.
            IMF_range (tuple, optional): Mass range of IMF sampling Defaults to (0.08,20).
            G (float, optional): Graviational constant. Defaults to pc^3/(Msun*Myr^2)

        Raises:
            ValueError: _description_
        """
        
        np.random.seed(seed)
        
        self.n_sc   = num_total//num_subclusters
        self.n      = num_subclusters * self.n_sc
        self.r      = radius
        self.rho_sc = subcluster_rho
        self.G      = G 
                
        # pre-allocate arrays, much faster than appending
        self.masses = np.empty(self.n)
        self.x      = np.empty(self.n)
        self.y      = np.empty(self.n)
        self.z      = np.empty(self.n)
        self.vx     = np.empty(self.n)
        self.vy     = np.empty(self.n)
        self.vz     = np.empty(self.n)
                
        # generate masses
        if not sample_IMF:
            self.masses = [masses]*self.n
        else:
            raise ValueError("OOPS! not implemented yet :)")
        
        # Generate subcluster centers randomly in a uniform sphere 
        subcluster_positions = self.sample_uniform_sphere(num_subclusters, self.r)
        
        # Get scaled plummer radius to achieve desired subcluster density
        self.r_sc = self.compute_plummer_radius(self.n_sc, masses, self.rho_sc)
            
        for i, subcluster_position in enumerate(subcluster_positions):
            # sc_pos, sc_vel = self.make_subcluster()
            # sc_pos = self.sample_uniform_sphere(self.n_sc, self.r_sc)
            
            # Plummer!
            
            x, y, z, vx, vy, vz = self.generate_plummer_model(self.n_sc, self.r_sc, 
                                                              virial_ratio=subcluster_virial_ratio, 
                                                              total_mass=masses*self.n_sc, 
                                                              G=self.G, mass_cutoff=0.95)
            

            # start and end indices of subcluster particles
            sc_start_idx = i*self.n_sc
            sc_end_idx   = sc_start_idx+self.n_sc
            print(np.std(vx),self.n_sc)
            self.x[sc_start_idx:sc_end_idx]  = x + subcluster_position[0]
            self.y[sc_start_idx:sc_end_idx]  = y + subcluster_position[1]
            self.z[sc_start_idx:sc_end_idx]  = z + subcluster_position[2]
            self.vx[sc_start_idx:sc_end_idx] = vx
            self.vy[sc_start_idx:sc_end_idx] = vy
            self.vz[sc_start_idx:sc_end_idx] = vz
            
        # Add bulk velocities if requested
        if global_virial_ratio is not None:
            print(f"\nAdding bulk velocities for global Q = {global_virial_ratio}")
            self.add_subcluster_bulk_velocities(subcluster_positions, global_virial_ratio)
        
        return
    
    def sample_uniform_sphere(self, n, r):
        """
        Return a list of x, y, z sampled in a uniform spherical volume of radius r.

        Args:
            n (_type_): _description_
            r (_type_): _description_

        Returns:
            _type_: _description_
        """
        u = np.random.rand(n)
        cos_theta = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        R = r * u**(1/3)
        sin_theta = np.sqrt(1 - cos_theta**2)

        x = R * sin_theta * np.cos(phi)
        y = R * sin_theta * np.sin(phi)
        z = R * cos_theta

        return np.column_stack((x, y, z))

    def make_subcluster(self):
        """
        Position generation:
            Generate a point on the surface of a unit sphere using Gaussian method.
            Then another random variable between 0-1 scales the point within the volume.
        
        Velocity generation:
            Generate a point of the surface of a unit sphere using Gaussian method.
            Scale by velocity dispersion. 
        
        Returns positions and velocities of all stars in subcluster. 
        """

        pos = np.random.standard_normal(size=(self.n_sc, 3))
        print(pos)
        norm = np.linalg.norm(pos, axis=1)
        scale_factor = self.r_sc * np.random.uniform(0, 1, size=self.n_sc) / norm
        pos = pos*scale_factor[:, None]
        
        vel = np.random.standard_normal(size=(self.n_sc, 3))
        norm = np.linalg.norm(vel, axis=1)
        vel = self.vdisp * vel / norm[:, None]
                
        return pos, vel     
    
    def generate_plummer_model(self, n_stars, radius, virial_ratio=0.25, 
                          total_mass=1.0, G=1.0, mass_cutoff=0.999, random_seed=None):
        """
        Generate a Plummer model with specified parameters.
        
        Parameters:
        -----------
        n_stars : int
            Number of stars to generate
        radius : float
            Characteristic radius of the cluster
        virial_ratio : float, optional
            Virial ratio Q = 2*KE/|PE|. Default is 0.5 (virial equilibrium)
        total_mass : float, optional
            Total mass of the system (default 1.0)
        G : float, optional
            Gravitational constant (default 1.0)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        x, y, z, vx, vy, vz : ndarray
            Arrays of shape (n_stars,) containing positions and velocities
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate radii from inverse cumulative mass distribution
        random_mass = np.random.uniform(0, mass_cutoff, n_stars)
        r = 1.0 / np.sqrt(np.power(random_mass, -2.0/3.0) - 1.0)
        
        # Generate positions in spherical coordinates
        theta = np.arccos(np.random.uniform(-1.0, 1.0, n_stars))
        phi = np.random.uniform(0.0, 2*pi, n_stars)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Generate velocity magnitudes using rejection sampling
        v_samples = []
        while len(v_samples) < n_stars:
            x_sample = np.random.uniform(0, 1.0, n_stars - len(v_samples))
            y_sample = np.random.uniform(0, 0.1, n_stars - len(v_samples))
            g = (x_sample**2) * np.power(1.0 - x_sample**2, 3.5)
            accepted = x_sample[y_sample <= g]
            v_samples.extend(accepted)
        
        v_mag = np.array(v_samples[:n_stars]) * sqrt(2.0) * np.power(1.0 + r**2, -0.25)
        
        # Generate velocity directions
        v_theta = np.arccos(np.random.uniform(-1.0, 1.0, n_stars))
        v_phi = np.random.uniform(0.0, 2*pi, n_stars)
        
        # Convert to Cartesian velocities
        vx = v_mag * np.sin(v_theta) * np.cos(v_phi)
        vy = v_mag * np.sin(v_theta) * np.sin(v_phi)
        vz = v_mag * np.cos(v_theta)
        
        # Scale positions and velocities (Plummer model normalization)
        scale_factor = 1.695
        x = x / scale_factor
        y = y / scale_factor
        z = z / scale_factor
        vx = vx / sqrt(1.0 / scale_factor)
        vy = vy / sqrt(1.0 / scale_factor)
        vz = vz / sqrt(1.0 / scale_factor)
        
        # Center the system
        x = x - np.mean(x)
        y = y - np.mean(y)
        z = z - np.mean(z)
        vx = vx - np.mean(vx)
        vy = vy - np.mean(vy)
        vz = vz - np.mean(vz)
        
        # Scale to desired radius
        current_radius = np.sqrt(np.mean(x**2 + y**2 + z**2))
        position_scale = radius / current_radius
        x = x * position_scale
        y = y * position_scale
        z = z * position_scale
        
        # Scale velocities for the new radius and mass
        # Velocity scale: sqrt(G * M / R)
        velocity_scale = sqrt(G * total_mass / radius)
        vx = vx * velocity_scale
        vy = vy * velocity_scale
        vz = vz * velocity_scale
        
        # Adjust for desired virial ratio
        # Standard Plummer has Q ≈ 0.5, scale to desired value
        virial_scale = sqrt(virial_ratio / 0.5)
        vx = vx * virial_scale
        vy = vy * virial_scale
        vz = vz * virial_scale
        
        return x, y, z, vx, vy, vz
    
    def compute_plummer_radius(self, n_stars, m_star, rho_sc):
        M = n_stars * m_star
        a = (3 * M / (4 * np.pi * rho_sc))**(1/3)
        return 1.3 * a  # convert to your generator radius

    def add_subcluster_bulk_velocities(self, subcluster_centers, global_virial_ratio=0.5):
        """
        Add bulk velocities to subclusters pointing toward the center of mass.
        Scales velocities so the entire system achieves the desired virial ratio.
        
        Parameters:
        -----------
        subcluster_centers : array-like, shape (num_subclusters, 3)
            Center positions of each subcluster
        global_virial_ratio : float
            Desired virial ratio Q = 2*KE/|PE| for the entire system
            
        Returns:
        --------
        None (modifies self.vx, self.vy, self.vz in place)
        """
        
        num_subclusters = len(subcluster_centers)
        
        # Calculate total center of mass
        total_mass = np.sum(self.masses)
        com = np.array([
            np.sum(self.masses * self.x) / total_mass,
            np.sum(self.masses * self.y) / total_mass,
            np.sum(self.masses * self.z) / total_mass
        ])
        
        print(f"Center of mass: {com}")
        print(f"Subcluster centers: {subcluster_centers}")
        
        # Calculate current kinetic energy (internal motions)
        KE_internal = 0.5 * np.sum(self.masses * (self.vx**2 + self.vy**2 + self.vz**2))
        
        # Calculate total potential energy
        phi, PE_total = self.compute_gravitational_potential()
        
        print(f"PE_total: {PE_total:.3e}, KE_internal: {KE_internal:.3e}")
        
        # Assign initial radial velocities toward CoM
        v_scale_initial = sqrt(abs(PE_total) / total_mass)
        
        # Store original velocities to calculate bulk component later
        vx_original = self.vx.copy()
        vy_original = self.vy.copy()
        vz_original = self.vz.copy()
        
        for i in range(num_subclusters):
            # Indices for this subcluster
            sc_start_idx = i * self.n_sc
            sc_end_idx = sc_start_idx + self.n_sc
            
            # Direction toward center of mass (FROM subcluster TO com)
            r_vec = com - subcluster_centers[i]  # Vector pointing from SC to CoM
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag > 0:
                direction = r_vec / r_mag  # Unit vector toward CoM
                
                # Initial bulk velocity (toward CoM = infall)
                v_bulk = direction * v_scale_initial
                
                print(f"Subcluster {i}: pos={subcluster_centers[i]}, direction={direction}, v_bulk={v_bulk}")
                
                # Add to all particles in this subcluster
                self.vx[sc_start_idx:sc_end_idx] += v_bulk[0]
                self.vy[sc_start_idx:sc_end_idx] += v_bulk[1]
                self.vz[sc_start_idx:sc_end_idx] += v_bulk[2]
        
        # Calculate total KE with bulk motions
        KE_total = 0.5 * np.sum(self.masses * (self.vx**2 + self.vy**2 + self.vz**2))
        
        # Target kinetic energy
        KE_target = global_virial_ratio * abs(PE_total) / 2
        KE_bulk_target = KE_target - KE_internal
        
        if KE_bulk_target < 0:
            print("Warning: Internal KE already exceeds target. Setting bulk velocities to zero.")
            self.vx = vx_original
            self.vy = vy_original
            self.vz = vz_original
            return
        
        # Current bulk KE
        KE_bulk_current = KE_total - KE_internal
        
        # Scale factor for bulk velocities
        if KE_bulk_current > 0:
            bulk_scale = sqrt(KE_bulk_target / KE_bulk_current)
        else:
            bulk_scale = 1.0
        
        print(f"Bulk velocity scale factor: {bulk_scale:.3f}")
        
        # Apply scaling to bulk component only
        for i in range(num_subclusters):
            sc_start_idx = i * self.n_sc
            sc_end_idx = sc_start_idx + self.n_sc
            
            # Calculate subcluster's bulk velocity (current mean)
            v_bulk_x = np.mean(self.vx[sc_start_idx:sc_end_idx])
            v_bulk_y = np.mean(self.vy[sc_start_idx:sc_end_idx])
            v_bulk_z = np.mean(self.vz[sc_start_idx:sc_end_idx])
            
            # Original internal velocities
            v_internal_x = vx_original[sc_start_idx:sc_end_idx]
            v_internal_y = vy_original[sc_start_idx:sc_end_idx]
            v_internal_z = vz_original[sc_start_idx:sc_end_idx]
            
            # Scale: v_new = v_internal + bulk_scale * v_bulk
            self.vx[sc_start_idx:sc_end_idx] = v_internal_x + bulk_scale * v_bulk_x
            self.vy[sc_start_idx:sc_end_idx] = v_internal_y + bulk_scale * v_bulk_y
            self.vz[sc_start_idx:sc_end_idx] = v_internal_z + bulk_scale * v_bulk_z
        
        # Verify final virial ratio
        KE_final = 0.5 * np.sum(self.masses * (self.vx**2 + self.vy**2 + self.vz**2))
        Q_final = 2 * KE_final / abs(PE_total)
        
        print(f"Target virial ratio: {global_virial_ratio:.4f}")
        print(f"Achieved virial ratio: {Q_final:.4f}")


    def save_ics(self, filename='ics.txt'):
        """
        Save ics to file. Columns are m, pos, vel.

        Args:
            filename (str, optional): _description_. Defaults to 'ics.txt'.
        """
        
        out_array = np.column_stack((self.masses, 
                              self.x, self.y, self.z,
                              self.vx, self.vy, self.vz))
        
        np.savetxt(filename, out_array)
        
        print(f"ICs for {len(out_array)} particles saved to {filename}. Columns are: mass, position, velocity.")
    
    
    
    def compute_gravitational_potential(self, softening=0.0):
        """
        Compute gravitational potential per particle and total PE
        equivalent to AMUSE's particles.potential().

        Parameters
        ----------
        softening : float
            Plummer softening length (same units as positions, e.g. pc)

        Returns
        -------
        phi : ndarray (N,)
            Potential at each particle (Phi_i)
        PE_total : float
            Total potential energy
        """

        x = self.x
        y = self.y
        z = self.z
        m = np.array(self.masses)

        N = len(m)

        # pairwise separations
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dz = z[:, None] - z[None, :]

        r = np.sqrt(dx**2 + dy**2 + dz**2 + softening**2)

        # avoid self-interaction
        np.fill_diagonal(r, np.inf)

        # potential at each particle
        phi = -self.G * np.sum(m[None, :] / r, axis=1)

        # total potential energy
        PE_total = 0.5 * np.sum(m * phi)

        return phi, PE_total
    
    def plot_ics(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Scatter plot (white particles)
        ax.scatter(self.x, self.y, self.z, c='white', edgecolors='none')

        # Remove grid and axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')


        # ---- Sphere overlay ----
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        x = self.r * np.outer(np.cos(u), np.sin(v)) 
        y = self.r * np.outer(np.sin(u), np.sin(v)) 
        z = self.r * np.outer(np.ones_like(u), np.cos(v)) 

        ax.plot_surface(x, y, z,
                        color='cyan',
                        alpha=0.15,   # translucency
                        linewidth=0,
                        shade=True)

        # Equal aspect ratio
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        plt.show()
        # plt.savefig("ics.png", dpi=200)
        
    
    def plot_vel_ics(self, subsample=None, arrow_scale=5.0, color_by='direction', show_com=True):
        """
        Plot the 3D velocity vectors of particles using quiver.
        
        Parameters:
        -----------
        subsample : int or None
            Number of particles to plot (randomly sampled). 
            If None, plots all particles (can be slow for large N).
        arrow_scale : float
            Scale factor for arrow lengths
        color_by : str
            How to color the arrows:
            - 'magnitude': color by velocity magnitude
            - 'direction': color by direction toward/away from COM
            - 'uniform': single color (cyan)
        show_com : bool
            Whether to show the center of mass as a large marker
        """
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Calculate center of mass
        total_mass = np.sum(self.masses)
        com = np.array([
            np.sum(self.masses * self.x) / total_mass,
            np.sum(self.masses * self.y) / total_mass,
            np.sum(self.masses * self.z) / total_mass
        ])
        
        # Subsample particles if requested
        if subsample is not None and subsample < self.n:
            indices = np.random.choice(self.n, size=subsample, replace=False)
        else:
            indices = np.arange(self.n)
        
        x_sub = self.x[indices]
        y_sub = self.y[indices]
        z_sub = self.z[indices]
        vx_sub = self.vx[indices]
        vy_sub = self.vy[indices]
        vz_sub = self.vz[indices]
        
        # Calculate velocity magnitudes
        v_mag = np.sqrt(vx_sub**2 + vy_sub**2 + vz_sub**2)
        
        # Determine arrow colors
        if color_by == 'magnitude':
            colors = v_mag
            cmap = 'plasma'
        elif color_by == 'direction':
            # Radial velocity relative to CENTER OF MASS (positive = outward, negative = inward)
            r_vec_x = x_sub - com[0]
            r_vec_y = y_sub - com[1]
            r_vec_z = z_sub - com[2]
            r = np.sqrt(r_vec_x**2 + r_vec_y**2 + r_vec_z**2 + 1e-10)
            v_radial = (r_vec_x * vx_sub + r_vec_y * vy_sub + r_vec_z * vz_sub) / r
            colors = v_radial
            cmap = 'coolwarm'
        else:  # uniform
            colors = 'cyan'
            cmap = None
        
        # Plot velocity vectors with quiver
        quiver = ax.quiver(x_sub, y_sub, z_sub,
                        vx_sub, vy_sub, vz_sub,
                        length=arrow_scale,
                        normalize=False,
                        arrow_length_ratio=0.3,
                        linewidth=1.5,
                        alpha=0.8,
                        cmap=cmap,
                        colors=colors if color_by == 'uniform' else None)
        
        # Set color array if using colormap
        if color_by != 'uniform':
            quiver.set_array(colors)
            quiver.set_cmap(cmap)
        
        # Plot particle positions as small dots
        ax.scatter(x_sub, y_sub, z_sub, c='white', s=5, alpha=0.5, edgecolors='none')
        
        # Plot center of mass
        if show_com:
            ax.scatter([com[0]], [com[1]], [com[2]], 
                    c='red', s=200, marker='*', 
                    edgecolors='white', linewidths=2,
                    label='Center of Mass', zorder=1000)
        
        # Add colorbar if using color mapping
        if color_by != 'uniform':
            cbar = plt.colorbar(quiver, ax=ax, pad=0.1, shrink=0.8)
            if color_by == 'magnitude':
                cbar.set_label('Velocity Magnitude (pc/Myr)', color='white', fontsize=12)
            else:
                cbar.set_label('Radial Velocity: Blue=Infall, Red=Outflow (pc/Myr)', 
                            color='white', fontsize=12)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Sphere overlay
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.r * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.r * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.r * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere,
                        color='cyan',
                        alpha=0.1,
                        linewidth=0,
                        shade=True)
        
        # Remove grid and axes ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # White labels
        ax.set_xlabel('X (pc)', color='white', fontsize=12)
        ax.set_ylabel('Y (pc)', color='white', fontsize=12)
        ax.set_zlabel('Z (pc)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        if show_com:
            ax.legend(loc='upper left', facecolor='black', edgecolor='white', 
                    labelcolor='white', fontsize=10)
        
        # Equal aspect ratio
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        # Title
        title = f'Velocity Vectors (n={len(indices)}'
        if subsample is not None and subsample < self.n:
            title += f' of {self.n}'
        title += f', v_rms={np.sqrt(np.mean(v_mag**2)):.3f} pc/Myr)'
        ax.set_title(title, color='white', fontsize=14, pad=20)
        
        print(f"Center of Mass: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
        
        plt.tight_layout()
        plt.show()
        
    def check_virial(self):
        """
        Compute virial ratio globally and per subcluster using internal potential.
        """

        # ---- total KE ----
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        KE_total = 0.5 * np.sum(self.masses * v2)

        # ---- total PE ----
        _, PE_total = self.compute_gravitational_potential()

        Q_total = 2 * KE_total / abs(PE_total)

        print(f"\n=== GLOBAL VIRIAL ===")
        print(f"Q = {Q_total:.4f}")

        print(f"\n=== SUBCLUSTER VIRIALS ===")

        for i in range(self.n // self.n_sc):
            start = i * self.n_sc
            end   = start + self.n_sc

            m = self.masses[start:end]
            vx = self.vx[start:end]
            vy = self.vy[start:end]
            vz = self.vz[start:end]

            KE = 0.5 * np.sum(m * (vx**2 + vy**2 + vz**2))

            # temporary object for reuse
            temp = SubClusters(1,1,1,1)  # dummy
            temp.x = self.x[start:end]
            temp.y = self.y[start:end]
            temp.z = self.z[start:end]
            temp.masses = m
            temp.G = self.G

            _, PE = temp.compute_gravitational_potential()

            Q = 2 * KE / abs(PE)

            print(f"Subcluster {i}: Q = {Q:.4f}")


    def plot_radial_density_profile(self, n_bins=50):
        """
        Plots radial density profiles for each subcluster and overlays Plummer model.
        """

        fig, ax = plt.subplots(figsize=(7,5))

        cmap = plt.cm.viridis
        n_sc_total = self.n // self.n_sc

        for i in range(n_sc_total):
            start = i * self.n_sc
            end   = start + self.n_sc

            x = self.x[start:end]
            y = self.y[start:end]
            z = self.z[start:end]

            # ---- center of subcluster ----
            com = np.array([np.mean(x), np.mean(y), np.mean(z)])

            r = np.sqrt((x - com[0])**2 +
                        (y - com[1])**2 +
                        (z - com[2])**2)

            # ---- radial bins ----
            bins = np.logspace(np.log10(min(r)+1e-5), np.log10(max(r)), n_bins)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            counts, _ = np.histogram(r, bins=bins)

            # shell volumes
            shell_vol = (4/3)*pi*(bins[1:]**3 - bins[:-1]**3)

            density = counts / shell_vol

            color = cmap(i / n_sc_total)
            ax.plot(bin_centers, density, drawstyle='steps-mid',
                    color=color, label=f'SC {i}')

            # ---- Plummer model overlay ----
            M = np.sum(self.masses[start:end])
            a = self.r_sc / 1.3  # approximate scale radius

            r_model = np.logspace(np.log10(min(r)), np.log10(max(r)), 200)
            rho_model = (3*M)/(4*pi*a**3) * (1 + (r_model/a)**2)**(-2.5)

            ax.plot(r_model, rho_model, linestyle='--', color=color, alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel("r")
        ax.set_ylabel("Density")
        ax.set_title("Radial Density Profiles (Subclusters)")

        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()