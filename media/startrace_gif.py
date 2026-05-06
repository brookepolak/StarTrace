"""
StarTrace GitHub Visualization - With Real k-NN Graph
=====================================================

Enhanced version showing the actual k-NN graph in 3D velocity space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from pathlib import Path
import os

# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

class Config:
    """Animation configuration."""
    
    # Data paths - UPDATE THIS TO YOUR DATA PATH
    DATA_PATH = "/gpfs/work3/0/ulc15220/bp/StarTrace/sims/"
    
    # Simulations to visualize
    SIMULATIONS = [
        {'nsc': 1, 'seed': 0, 'label': 'MONOLITHIC', 'class_idx': 0},
        {'nsc': 2, 'seed': 0, 'label': 'MERGER', 'class_idx': 1},
        {'nsc': 3, 'seed': 0, 'label': 'SUBCLUSTERS', 'class_idx': 2},
        {'nsc': 7, 'seed': 0, 'label': 'SUBCLUSTERS', 'class_idx': 2},
    ]
    
    # Animation parameters
    SNAPSHOTS = list(range(0, 11))
    FPS = 10
    DPI = 100
    
    # Timing (in frames)
    FRAMES_PER_SNAPSHOT = 3
    FRAMES_PAUSE = 10
    FRAMES_GRAPH_BUILD = 20     # Graph fades in
    FRAMES_GRAPH_ROTATE = 30    # Rotate to show structure
    FRAMES_PREDICTION = 15      # Dotted line appears
    FRAMES_LABEL = 20
    FRAMES_TRANSITION = 10
    
    # k-NN Graph parameters
    K_NEIGHBORS = 8  # Fewer for cleaner visualization
    N_GRAPH_NODES = 150  # Subsample for graph display
    
    # Visual parameters
    N_DISPLAY_STARS = 500
    STAR_SIZE = 1.0
    STAR_ALPHA = 0.9
    
    # Colors
    COLOR_STARS = '#000000'  # Black stars
    COLOR_GRAPH_NODES = 'orangered'  # Will use gradient
    COLOR_GRAPH_EDGES = '#000000'  # Black edges
    COLOR_OUTPUT_NODES = ['#FFDD00', '#FFDD00', '#FFDD00']  # Yellow for all output nodes
    COLOR_LABEL_ACTIVE = '#FFDD00'  # Yellow when active
    COLOR_LABEL_INACTIVE = '#000000'  # Black when inactive
    
    OUTPUT_PATH = "startrace_demo.gif"


# ═════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════

def load_simulation_snapshot(data_path: str, nsc: int, seed: int, snapshot: int):
    """Load a single simulation snapshot."""
    sim_name = f"NSC{nsc}SEED{seed}"
    filename = f"{data_path}{sim_name}/data.{snapshot}"
    
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return None
    
    # Load data: [mass, x, y, z, vx, vy, vz]
    data = np.loadtxt(filename, skiprows=1)
    coords = data[:, 1:4]      # x, y, z positions
    velocities = data[:, 4:7]  # vx, vy, vz velocities
    
    return coords, velocities


def load_simulation_sequence(data_path: str, nsc: int, seed: int, snapshots: list):
    """Load entire simulation sequence."""
    sequence = []
    for snap in snapshots:
        result = load_simulation_snapshot(data_path, nsc, seed, snap)
        if result is not None:
            sequence.append(result)
    return sequence


def subsample_stars(coords: np.ndarray, n_target: int = 500, seed: int = 42):
    """Subsample stars for visualization."""
    if len(coords) <= n_target:
        return coords
    
    np.random.seed(seed)
    indices = np.random.choice(len(coords), n_target, replace=False)
    return coords[indices]


# ═════════════════════════════════════════════════════════════════════
# k-NN GRAPH CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════

def build_knn_graph(velocities: np.ndarray, k: int = 8, n_subsample: int = 150):
    """
    Build k-NN graph in velocity space.
    
    Returns:
        node_positions: (N, 3) velocity coordinates
        edges: list of (i, j) tuples
        node_colors: colors based on degree/density
    """
    # Subsample for cleaner visualization
    np.random.seed(42)
    if len(velocities) > n_subsample:
        indices = np.random.choice(len(velocities), n_subsample, replace=False)
        velocities = velocities[indices]
    
    # Standardize velocities
    velocities = (velocities - velocities.mean(axis=0)) / (velocities.std(axis=0) + 1e-8)
    
    # Build k-NN graph using KDTree
    tree = cKDTree(velocities)
    edges = []
    degrees = np.zeros(len(velocities))
    
    for i in range(len(velocities)):
        # Find k nearest neighbors (excluding self)
        distances, neighbors = tree.query(velocities[i], k=k+1)
        neighbors = neighbors[1:]  # Exclude self
        
        for j in neighbors:
            if i < j:  # Avoid duplicate edges
                edges.append((i, j))
                degrees[i] += 1
                degrees[j] += 1
    
    # Color nodes by degree (connectivity)
    node_colors = degrees / degrees.max()
    
    return velocities, edges, node_colors


# ═════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION
# ═════════════════════════════════════════════════════════════════════

class GraphVisualizer:
    """Visualize k-NN graph in 3D velocity space."""
    
    def __init__(self, ax):
        self.ax = ax
        self.nodes = None
        self.edges = None
        self.node_colors = None
        self.output_nodes = None
        self.cmap = plt.cm.YlOrRd  # Yellow to Red colormap
    
    def set_graph(self, nodes, edges, node_colors):
        """Set the graph data."""
        self.nodes = nodes
        self.edges = edges
        self.node_colors = node_colors
        
        # Position output nodes to the right of the graph
        self.output_nodes = np.array([
            [3.5, 1.0, 0.0],   # Top - Class 0
            [3.5, 0.0, 0.0],   # Middle - Class 1
            [3.5, -1.0, 0.0],  # Bottom - Class 2
        ])
    
    def draw(self, alpha=1.0, azim=45, show_prediction=False, 
             predicted_class=None, prediction_alpha=1.0):
        """Draw the graph."""
        self.ax.clear()
        
        if self.nodes is None:
            self.ax.axis('off')
            return
        
        # Draw edges first (background)
        for i, j in self.edges:
            x = [self.nodes[i, 0], self.nodes[j, 0]]
            y = [self.nodes[i, 1], self.nodes[j, 1]]
            z = [self.nodes[i, 2], self.nodes[j, 2]]
            self.ax.plot(x, y, z, color=Config.COLOR_GRAPH_EDGES, 
                        alpha=0.4*alpha, linewidth=0.5, zorder=1)
        
        # Draw nodes
        colors = self.cmap(self.node_colors)
        self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2],
                       c=colors, s=50*alpha, alpha=alpha, edgecolors='white',
                       linewidths=0.5, zorder=2)
        
        # No dotted line in graph - it will be in the labels panel instead
        
        # Clean 3D plot
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('none')
        self.ax.yaxis.pane.set_edgecolor('none')
        self.ax.zaxis.pane.set_edgecolor('none')
        
        # Set viewing angle
        self.ax.view_init(elev=20, azim=azim)
        
        # Equal aspect
        max_range = 3.0
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)
        
        self.ax.set_title('')  # No title


# ═════════════════════════════════════════════════════════════════════
# MAIN ANIMATOR
# ═════════════════════════════════════════════════════════════════════

class StarTraceAnimator:
    """Main animator with real k-NN graph."""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        
        # Load all simulation data
        print("Loading simulation data...")
        self.simulations = []
        for sim_config in Config.SIMULATIONS:
            sequence = load_simulation_sequence(
                data_path, sim_config['nsc'], sim_config['seed'], Config.SNAPSHOTS
            )
            
            # Build k-NN graph for final snapshot
            if len(sequence) > 0:
                final_coords, final_vels = sequence[-1]
                graph_nodes, graph_edges, graph_colors = build_knn_graph(
                    final_vels, 
                    k=Config.K_NEIGHBORS,
                    n_subsample=Config.N_GRAPH_NODES
                )
            else:
                graph_nodes = graph_edges = graph_colors = None
            
            self.simulations.append({
                'config': sim_config,
                'sequence': sequence,
                'graph': {
                    'nodes': graph_nodes,
                    'edges': graph_edges,
                    'colors': graph_colors
                }
            })
            print(f"  Loaded NSC{sim_config['nsc']}SEED{sim_config['seed']}: "
                  f"{len(sequence)} snapshots, {len(graph_edges) if graph_edges else 0} edges")
        
        # Calculate frame plan
        self._calculate_frame_plan()
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 6), facecolor='white')
        # No title - user will add manually
        
        # Create subplots - all 3D now
        self.ax_sim = self.fig.add_subplot(131, projection='3d')
        self.ax_graph = self.fig.add_subplot(132, projection='3d')
        self.ax_labels = self.fig.add_subplot(133)
        
        # Initialize graph visualizer
        self.graph_viz = GraphVisualizer(self.ax_graph)
        
        print(f"Total frames: {self.total_frames}")
        print(f"Estimated duration: {self.total_frames / Config.FPS:.1f} seconds")
    
    def _calculate_frame_plan(self):
        """Calculate frame timeline."""
        self.frame_plan = []
        
        for sim_idx, sim in enumerate(self.simulations):
            sim_frames = []
            
            # Evolution
            for snap_idx in range(len(Config.SNAPSHOTS)):
                for _ in range(Config.FRAMES_PER_SNAPSHOT):
                    sim_frames.append({
                        'stage': 'evolution',
                        'snapshot': snap_idx,
                        'sim_idx': sim_idx
                    })
            
            # Pause at final snapshot
            for _ in range(Config.FRAMES_PAUSE):
                sim_frames.append({
                    'stage': 'pause',
                    'snapshot': len(Config.SNAPSHOTS) - 1,
                    'sim_idx': sim_idx
                })
            
            # Graph builds
            for i in range(Config.FRAMES_GRAPH_BUILD):
                progress = i / Config.FRAMES_GRAPH_BUILD
                sim_frames.append({
                    'stage': 'graph_build',
                    'progress': progress,
                    'sim_idx': sim_idx
                })
            
            # Graph rotates
            for i in range(Config.FRAMES_GRAPH_ROTATE):
                angle = 45 + (i / Config.FRAMES_GRAPH_ROTATE) * 360
                sim_frames.append({
                    'stage': 'graph_rotate',
                    'azim': angle,
                    'sim_idx': sim_idx
                })
            
            # Prediction appears
            for i in range(Config.FRAMES_PREDICTION):
                progress = i / Config.FRAMES_PREDICTION
                sim_frames.append({
                    'stage': 'prediction',
                    'progress': progress,
                    'class_idx': sim['config']['class_idx'],
                    'sim_idx': sim_idx
                })
            
            # Show label
            for _ in range(Config.FRAMES_LABEL):
                sim_frames.append({
                    'stage': 'label',
                    'class_idx': sim['config']['class_idx'],
                    'sim_idx': sim_idx
                })
            
            # Transition
            if sim_idx < len(self.simulations) - 1:
                for _ in range(Config.FRAMES_TRANSITION):
                    sim_frames.append({
                        'stage': 'transition',
                        'sim_idx': sim_idx
                    })
            
            self.frame_plan.extend(sim_frames)
        
        self.total_frames = len(self.frame_plan)
    
    def _draw_simulation(self, coords, ax):
        """Draw 3D star cluster."""
        ax.clear()
        
        if coords is None or len(coords) == 0:
            return
        
        coords_display = subsample_stars(coords, Config.N_DISPLAY_STARS)
        
        ax.scatter(coords_display[:, 0], coords_display[:, 1], coords_display[:, 2],
                  s=Config.STAR_SIZE, c=Config.COLOR_STARS, alpha=Config.STAR_ALPHA,
                  edgecolors='none')
        
        # Clean plot
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.view_init(elev=20, azim=45)
        
        # Equal aspect
        max_range = np.array([coords_display[:, 0].max() - coords_display[:, 0].min(),
                             coords_display[:, 1].max() - coords_display[:, 1].min(),
                             coords_display[:, 2].max() - coords_display[:, 2].min()]).max() / 2.0
        
        mid = coords_display.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        # No title
    
    def _draw_labels(self, active_class=None, alpha=1.0, node_active=False, prediction_progress=0.0):
        """Draw classification labels with output nodes and dotted line."""
        self.ax_labels.clear()
        self.ax_labels.set_xlim(0, 1)
        self.ax_labels.set_ylim(0, 1)
        self.ax_labels.axis('off')
        
        # No header - just the class labels
        labels = ['MONOLITHIC', 'MERGER', 'SUBCLUSTERS']
        y_positions = [0.65, 0.45, 0.25]
        
        for class_idx, (label, y_pos) in enumerate(zip(labels, y_positions)):
            is_active = (class_idx == active_class)
            
            # Draw output node (circle) to the left of label
            node_x = 0.2
            node_y = y_pos
            
            # Draw dotted line pointing RIGHT toward the node (only for active class)
            if is_active and prediction_progress > 0:
                # Line goes from left edge (x=0) to the node (x=0.2)
                # Animate from left to right
                line_start_x = 0.0
                line_end_x = node_x
                
                # Current end point based on animation progress
                current_end_x = line_start_x + (line_end_x - line_start_x) * prediction_progress
                
                # Draw dotted line
                n_dots = 15
                for i in range(n_dots):
                    dot_x = line_start_x + (current_end_x - line_start_x) * (i / n_dots)
                    if i % 2 == 0 and dot_x <= current_end_x:  # Dotted effect
                        next_dot_x = line_start_x + (current_end_x - line_start_x) * ((i+1) / n_dots)
                        next_dot_x = min(next_dot_x, current_end_x)
                        self.ax_labels.plot([dot_x, next_dot_x], [node_y, node_y], 
                                          'k-', linewidth=3, solid_capstyle='round',
                                          zorder=2)
            
            if is_active and node_active:
                # Filled yellow circle
                circle = plt.Circle((node_x, node_y), 0.05, 
                                   color=Config.COLOR_LABEL_ACTIVE, 
                                   fill=True, zorder=3)
                self.ax_labels.add_patch(circle)
                # Black outline
                circle_outline = plt.Circle((node_x, node_y), 0.05, 
                                           color='black', fill=False, 
                                           linewidth=2.5, zorder=4)
                self.ax_labels.add_patch(circle_outline)
            else:
                # Empty circle (just outline)
                circle = plt.Circle((node_x, node_y), 0.05, 
                                   color='black', fill=False, 
                                   linewidth=2.5, zorder=3)
                self.ax_labels.add_patch(circle)
            
            # Draw label text - bigger font (18pt)
            if is_active and alpha > 0.5:
                # Active: Yellow text with black outline
                text_color = Config.COLOR_LABEL_ACTIVE
                fontweight = 'bold'
                fontsize = 22
                
                # Draw text with outline effect
                import matplotlib.patheffects as path_effects
                text = self.ax_labels.text(0.45, y_pos, label,
                                          ha='left', va='center',
                                          fontsize=fontsize, fontweight=fontweight,
                                          color=text_color, alpha=alpha, zorder=5,
                                          family='sans-serif')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=4, foreground='black'),
                    path_effects.Normal()
                ])
            else:
                # Inactive: Black text
                text_color = Config.COLOR_LABEL_INACTIVE
                fontweight = 'normal'
                fontsize = 18
                
                self.ax_labels.text(0.45, y_pos, label,
                                   ha='left', va='center',
                                   fontsize=fontsize, fontweight=fontweight,
                                   color=text_color, alpha=1.0, zorder=5,
                                   family='sans-serif')
    
    def update(self, frame_num):
        """Update function for animation."""
        if frame_num >= len(self.frame_plan):
            return
        
        frame_info = self.frame_plan[frame_num]
        stage = frame_info['stage']
        sim_idx = frame_info['sim_idx']
        sim_data = self.simulations[sim_idx]
        
        if frame_num % 50 == 0:
            print(f"Frame {frame_num}/{self.total_frames} "
                  f"({100*frame_num/self.total_frames:.1f}%)")
        
        # Draw based on stage
        if stage in ['evolution', 'pause']:
            snapshot_idx = frame_info['snapshot']
            coords, _ = sim_data['sequence'][snapshot_idx]
            self._draw_simulation(coords, self.ax_sim)
            self.graph_viz.draw(alpha=0)  # Hidden
            self._draw_labels()  # No active class yet
        
        elif stage == 'graph_build':
            coords, _ = sim_data['sequence'][-1]
            self._draw_simulation(coords, self.ax_sim)
            
            # Fade in graph
            self.graph_viz.set_graph(
                sim_data['graph']['nodes'],
                sim_data['graph']['edges'],
                sim_data['graph']['colors']
            )
            self.graph_viz.draw(alpha=frame_info['progress'])
            self._draw_labels()  # No active class yet
        
        elif stage == 'graph_rotate':
            coords, _ = sim_data['sequence'][-1]
            self._draw_simulation(coords, self.ax_sim)
            
            self.graph_viz.draw(alpha=1.0, azim=frame_info['azim'])
            self._draw_labels()  # No active class yet
        
        elif stage == 'prediction':
            coords, _ = sim_data['sequence'][-1]
            self._draw_simulation(coords, self.ax_sim)
            
            progress = frame_info['progress']
            
            # Show graph without dotted line
            self.graph_viz.draw(
                alpha=1.0,
                azim=45,
                show_prediction=False,
                predicted_class=frame_info['class_idx'],
                prediction_alpha=progress
            )
            
            # Activate node first (when line reaches it), then label
            node_active = progress > 0.5  # Node fills after line is halfway
            label_alpha = max(0, (progress - 0.5) * 2)  # Label lights up after node
            
            self._draw_labels(
                active_class=frame_info['class_idx'],
                alpha=label_alpha,
                node_active=node_active,
                prediction_progress=progress
            )
        
        elif stage == 'label':
            coords, _ = sim_data['sequence'][-1]
            self._draw_simulation(coords, self.ax_sim)
            
            self.graph_viz.draw(
                alpha=1.0,
                azim=45,
                show_prediction=False,
                predicted_class=frame_info['class_idx'],
                prediction_alpha=1.0
            )
            
            self._draw_labels(
                active_class=frame_info['class_idx'],
                alpha=1.0,
                node_active=True,
                prediction_progress=1.0
            )
        
        elif stage == 'transition':
            self.ax_sim.clear()
            self.graph_viz.draw(alpha=0)
            self._draw_labels()  # Reset to no active class
        
        plt.tight_layout()
    
    def create_animation(self):
        """Create and save the animation."""
        print("\nCreating animation...")
        
        anim = FuncAnimation(
            self.fig, 
            self.update,
            frames=self.total_frames,
            interval=1000/Config.FPS,
            repeat=True
        )
        
        print(f"\nSaving to {self.output_path}...")
        writer = PillowWriter(fps=Config.FPS)
        anim.save(self.output_path, writer=writer, dpi=Config.DPI)
        
        print(f"✓ Animation saved!")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH)
    parser.add_argument('--output', type=str, default=Config.OUTPUT_PATH)
    parser.add_argument('--fps', type=int, default=Config.FPS)
    parser.add_argument('--dpi', type=int, default=Config.DPI)
    
    args = parser.parse_args()
    
    Config.DATA_PATH = args.data_path
    Config.OUTPUT_PATH = args.output
    Config.FPS = args.fps
    Config.DPI = args.dpi
    
    if not os.path.exists(Config.DATA_PATH):
        print(f"Error: Data path not found: {Config.DATA_PATH}")
        return
    
    animator = StarTraceAnimator(Config.DATA_PATH, Config.OUTPUT_PATH)
    animator.create_animation()
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
