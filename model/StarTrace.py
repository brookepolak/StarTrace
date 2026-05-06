"""
StarTrace: Core Library for Star Cluster Subcluster Classification
===================================================================

A modular library for training, validating, and predicting the number of
subclusters in star cluster merger simulations using Graph Neural Networks.

Classes:
    Config: Configuration and hyperparameters
    StarClusterGraphDataset: PyTorch Geometric dataset for star clusters
    GNNBlock: Graph neural network building block
    StarClusterGNN: Main GNN model architecture
    UncertaintyQuantifier: MC Dropout for uncertainty estimation
    ValidationPlots: Publication-quality validation plots
    Trainer: End-to-end training pipeline
    Validator: Model validation and evaluation
    Predictor: Inference on new data

Functions:
    map_nsc_to_class: Map NSC count to class index
    class_to_label: Convert class index to readable label
    load_simulation_data: Load simulation snapshot files
    derive_physical_features: Compute per-star features
    compute_global_features: Compute cluster-level features

Author: Brooke Polak
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.transforms import KNNGraph

# Matplotlib styling
try:
    import cmasher as cmr
    CMAP = cmr.ocean
except ImportError:
    CMAP = plt.cm.viridis

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})


# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

class Config:
    """
    Centralized configuration for StarTrace training and inference.
    
    All hyperparameters and settings are defined here. Can be overridden
    programmatically or via command-line arguments in CLI scripts.
    
    Attributes:
        N_CLASSES (int): Number of classification classes (e.g., 3 for [1, 2, 3+])
        SNAPSHOT (int): Which simulation snapshot to load (timestep)
        K_NEIGHBORS (int): Number of nearest neighbors for k-NN graph construction
        HIDDEN_DIM (int): Hidden layer dimensionality in GNN
        USE_GLOBAL_FEATURES (bool): Whether to include cluster-level features
        BATCH_SIZE (int): Training batch size
        EPOCHS (int): Maximum number of training epochs
        LR (float): Learning rate for Adam optimizer
        WEIGHT_DECAY (float): L2 regularization strength
        VAL_FRACTION (float): Fraction of data used for validation
        DROPOUT (float): Dropout probability in classifier layers
        PATIENCE (int): Early stopping patience (epochs without improvement)
        MIN_DELTA (float): Minimum change to qualify as improvement
        DEVICE (torch.device): Training device (CPU/GPU)
        OUTPUT_DIR (Path): Directory for saving outputs
        MODEL_PATH (Path): Path to save/load model checkpoint
        PLOT_DIR (Path): Directory for saving plots
    """
    
    # Data parameters
    N_CLASSES = 3           # Number of classes: 1, 2, 3+ subclusters
    SNAPSHOT = 15           # Which simulation snapshot to use
    
    # Model architecture
    K_NEIGHBORS = 32        # k-NN graph connectivity in phase space
    HIDDEN_DIM = 128        # Hidden layer dimensionality
    USE_GLOBAL_FEATURES = False  # Include cluster-level features
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    VAL_FRACTION = 0.15
    
    # Regularization
    DROPOUT = 0.5
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Output paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "StarTrace_best_model.pt"
    PLOT_DIR = OUTPUT_DIR / "plots"
    
    @classmethod
    def update(cls, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Example:
            Config.update(N_CLASSES=4, HIDDEN_DIM=256)
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


# ═════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def map_nsc_to_class(n_subclusters: int, n_classes: int = None) -> int:
    """
    Map original NSC (number of subclusters) to class index.
    
    The mapping depends on the number of classes:
    - For 3 classes: [1] → 0, [2] → 1, [3+] → 2
    - For 4 classes: [1] → 0, [2] → 1, [3] → 2, [4+] → 3
    
    Args:
        n_subclusters: Original number of subclusters from simulation (1-8)
        n_classes: Number of classification classes (default: Config.N_CLASSES)
    
    Returns:
        Class index (0 to n_classes-1)
        
    Examples:
        >>> map_nsc_to_class(1, n_classes=3)
        0
        >>> map_nsc_to_class(5, n_classes=3)
        2
        >>> map_nsc_to_class(3, n_classes=4)
        2
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    if n_classes == 3:
        # 3-class system: 1, 2, 3+
        if n_subclusters == 1:
            return 0
        elif n_subclusters == 2:
            return 1
        else:  # 3+
            return 2
    elif n_classes == 4:
        # 4-class system: 1, 2, 3, 4+
        if n_subclusters == 1:
            return 0
        elif n_subclusters == 2:
            return 1
        elif n_subclusters == 3:
            return 2
        else:  # 4+
            return 3
    else:
        # General case: directly map up to n_classes-1, then group the rest
        return min(n_subclusters - 1, n_classes - 1)


def class_to_label(class_idx: int, n_classes: int = None) -> str:
    """
    Convert class index to human-readable label.
    
    Args:
        class_idx: Class index (0 to n_classes-1)
        n_classes: Number of classification classes (default: Config.N_CLASSES)
    
    Returns:
        Human-readable string label
        
    Examples:
        >>> class_to_label(0, n_classes=3)
        '1 subcluster'
        >>> class_to_label(2, n_classes=3)
        '3+ subclusters'
        >>> class_to_label(3, n_classes=4)
        '4+ subclusters'
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    if class_idx < n_classes - 1:
        return f"{class_idx + 1} subcluster{'s' if class_idx > 0 else ''}"
    else:
        return f"{n_classes}+ subclusters"


def get_class_labels(n_classes: int = None) -> List[str]:
    """
    Get list of all class labels for plotting.
    
    Args:
        n_classes: Number of classification classes (default: Config.N_CLASSES)
    
    Returns:
        List of short label strings for each class
        
    Examples:
        >>> get_class_labels(3)
        ['1', '2', '3+']
        >>> get_class_labels(4)
        ['1', '2', '3', '4+']
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    labels = []
    for i in range(n_classes):
        if i < n_classes - 1:
            labels.append(str(i + 1))
        else:
            labels.append(f"{n_classes}+")
    return labels


def get_nsc_plot_values(n_classes: int = None) -> np.ndarray:
    """
    Get NSC values for plotting (handling the "+" class).
    
    For visualization purposes, the highest class (e.g., "3+" or "4+") 
    needs a numeric value. We use n_classes + 3 to visually separate it.
    
    Args:
        n_classes: Number of classification classes (default: Config.N_CLASSES)
    
    Returns:
        Array of NSC values for x-axis plotting
        
    Examples:
        >>> get_nsc_plot_values(3)
        array([1, 2, 6])
        >>> get_nsc_plot_values(4)
        array([1, 2, 3, 7])
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    values = list(range(1, n_classes))  # [1, 2, ..., n_classes-1]
    values.append(n_classes + 3)  # e.g., 6 for 3-class, 7 for 4-class
    return np.array(values)


# ═════════════════════════════════════════════════════════════════════
# DATA LOADING AND PREPROCESSING
# ═════════════════════════════════════════════════════════════════════

def load_simulation_data(data_path: str, sim_name: str, snapshot: int) -> Tuple[bool, Optional[np.ndarray], Optional[int]]:
    """
    Load a single simulation snapshot from disk.
    
    Args:
        data_path: Base directory containing simulation folders
        sim_name: Simulation name (e.g., "NSC3SEED42")
        snapshot: Snapshot number to load
    
    Returns:
        Tuple of (success, coordinates, n_subclusters):
            - success (bool): True if file exists and loaded successfully
            - coordinates (np.ndarray or None): (N, 6) array of [x, y, z, vx, vy, vz]
            - n_subclusters (int or None): True number of subclusters from filename
    
    File Format:
        Text file with columns: [mass, x, y, z, vx, vy, vz]
        First row is skipped (header)
    """
    filename = f"{data_path}{sim_name}/data.{snapshot}"
    
    if not os.path.exists(filename):
        return False, None, None
    
    # Load data: columns are [mass, x, y, z, vx, vy, vz]
    data = np.loadtxt(filename, skiprows=1)
    coords = data[:, 1:7]  # Drop mass, keep position + velocity
    
    # Extract true label from filename (e.g., NSC4SEED123 → 4)
    n_subclusters = int(re.findall(r"NSC(\d+)", sim_name)[0])
    
    # Standardize: zero mean, unit variance per feature
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    
    return True, coords.astype(np.float32), n_subclusters


def compute_global_features(coords: np.ndarray) -> np.ndarray:
    """
    Compute global cluster-level features.
    
    These are statistical properties of the entire cluster, broadcast to
    each star node. Note: experiments showed these did not significantly
    improve accuracy, so USE_GLOBAL_FEATURES defaults to False.
    
    Args:
        coords: (N, 6) array of standardized [x, y, z, vx, vy, vz]
    
    Returns:
        (7,) array of global properties:
            [0] Half-mass radius
            [1] Position dispersion
            [2] Velocity dispersion
            [3] Velocity anisotropy
            [4] Virial ratio (KE/|PE|)
            [5] Total angular momentum magnitude
            [6] Log number of stars
    """
    pos = coords[:, :3]
    vel = coords[:, 3:]
    
    # 1. Half-mass radius
    r = np.linalg.norm(pos, axis=1)
    r_sorted = np.sort(r)
    r_half = r_sorted[len(r_sorted) // 2]
    
    # 2. Position dispersion
    pos_dispersion = np.sqrt(np.mean(pos ** 2))
    
    # 3. Velocity dispersion
    vel_dispersion = np.sqrt(np.mean(vel ** 2))
    
    # 4. Velocity anisotropy
    v_radial = np.abs((pos * vel).sum(axis=1) / (r + 1e-8))
    v_tangential = np.sqrt(np.sum(vel**2, axis=1) - v_radial**2 + 1e-8)
    anisotropy = (v_radial.mean() - v_tangential.mean()) / (v_radial.mean() + v_tangential.mean() + 1e-8)
    
    # 5. Virial ratio (approximate PE with sampling for speed)
    KE_total = 0.5 * np.sum(vel ** 2)
    n_sample = min(200, len(pos))
    idx = np.random.choice(len(pos), n_sample, replace=False)
    pos_sample = pos[idx]
    PE_approx = 0.0
    for i in range(len(pos_sample)):
        dists = np.linalg.norm(pos_sample[i+1:] - pos_sample[i], axis=1)
        PE_approx -= np.sum(1.0 / (dists + 0.1))
    PE_approx *= (len(pos) / n_sample)**2
    virial_ratio = KE_total / (abs(PE_approx) + 1e-8)
    
    # 6. Total angular momentum
    L_total = np.cross(pos, vel).sum(axis=0)
    L_total_mag = np.linalg.norm(L_total)
    
    # 7. Number of stars (log-scaled)
    n_stars_log = np.log10(len(coords) + 1)
    
    return np.array([
        r_half, pos_dispersion, vel_dispersion, anisotropy,
        virial_ratio, L_total_mag, n_stars_log
    ], dtype=np.float32)


def derive_physical_features(coords: np.ndarray, include_global: bool = False) -> np.ndarray:
    """
    Compute per-star physical features from phase-space coordinates.
    
    Transforms raw [x, y, z, vx, vy, vz] into physically meaningful features
    that help the GNN identify dynamical substructure.
    
    Args:
        coords: (N, 6) array of standardized [x, y, z, vx, vy, vz]
        include_global: If True, append global cluster features to each star
    
    Returns:
        (N, 13) array if include_global=False, or (N, 20) if True
        
        Per-star features (13):
            [0:3]   Position (x, y, z)
            [3:6]   Velocity (vx, vy, vz)
            [6]     Radial distance
            [7]     Speed
            [8:11]  Angular momentum vector (Lx, Ly, Lz)
            [11]    Angular momentum magnitude
            [12]    Kinetic energy
        
        Global features (7, if included):
            See compute_global_features() docstring
    """
    pos = coords[:, :3]
    vel = coords[:, 3:]
    
    # Per-star derived quantities
    r_mag = np.linalg.norm(pos, axis=1, keepdims=True)
    v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
    L = np.cross(pos, vel)
    L_mag = np.linalg.norm(L, axis=1, keepdims=True)
    KE = 0.5 * (vel ** 2).sum(axis=1, keepdims=True)
    
    per_star_features = np.hstack([coords, r_mag, v_mag, L, L_mag, KE])  # (N, 13)
    
    if include_global:
        global_features = compute_global_features(coords)  # (7,)
        global_features_broadcast = np.tile(global_features, (len(coords), 1))  # (N, 7)
        features = np.hstack([per_star_features, global_features_broadcast])  # (N, 20)
    else:
        features = per_star_features  # (N, 13)
    
    return features.astype(np.float32)


def create_labels(class_idx: int, n_classes: int = None) -> torch.Tensor:
    """
    Create one-hot encoded labels for classification.
    
    Args:
        class_idx: Class index (0 to n_classes-1)
        n_classes: Number of classes (default: Config.N_CLASSES)
    
    Returns:
        One-hot tensor of shape (n_classes,)
        
    Example:
        >>> create_labels(1, n_classes=3)
        tensor([0., 1., 0.])
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    y = torch.zeros(n_classes)
    y[class_idx] = 1.0
    
    return y


# ═════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════

class StarClusterGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for star cluster classification.
    
    Loads simulation snapshots, constructs k-NN graphs in phase space,
    and prepares graph data objects for GNN training/inference.
    
    Each graph represents one star cluster with:
        - Node features: per-star physical properties (13 or 20 dimensions)
        - Edge structure: k-nearest neighbors in 6D phase space
        - Graph label: number of subclusters (mapped to class index)
    
    Args:
        data_path: Base directory containing simulation folders
        sim_names: List of simulation names to load (e.g., ["NSC1SEED0", ...])
        snapshot: Which snapshot timestep to load
        k: Number of nearest neighbors for graph construction
        use_global: Whether to include global cluster features
        n_classes: Number of classification classes (default: Config.N_CLASSES)
    
    Attributes:
        graphs: List of PyTorch Geometric Data objects
        k: Number of neighbors in k-NN graph
        use_global: Whether global features are included
    """
    
    def __init__(self, 
                 data_path: str, 
                 sim_names: List[str], 
                 snapshot: int, 
                 k: int = None,
                 use_global: bool = None,
                 n_classes: int = None):
        super().__init__()
        
        if k is None:
            k = Config.K_NEIGHBORS
        if use_global is None:
            use_global = Config.USE_GLOBAL_FEATURES
        if n_classes is None:
            n_classes = Config.N_CLASSES
        
        self.k = k
        self.use_global = use_global
        self.n_classes = n_classes
        self.knn_transform = KNNGraph(k=k, loop=False, cosine=False)
        self.graphs = []
        
        print(f"Loading {n_classes}-class dataset from {data_path}")
        print(f"Class mapping: {', '.join([f'{i+1}→{i}' if i < n_classes-1 else f'{i+1}+→{i}' for i in range(n_classes)])}")
        print(f"k-NN graph with k={k}")
        print(f"Global features: {'ENABLED' if use_global else 'DISABLED'}")
        
        # Load all simulations
        loaded_count = 0
        for sim_name in sim_names:
            exists, coords, n_subclusters = load_simulation_data(
                data_path, sim_name, snapshot
            )
            
            if not exists:
                continue
            
            # Map to class system
            class_idx = map_nsc_to_class(n_subclusters, n_classes)
            
            # Derive features
            features = derive_physical_features(coords, include_global=use_global)
            label = create_labels(class_idx, n_classes)
            
            # Create graph data object
            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                pos=torch.tensor(coords, dtype=torch.float),
                y=label.unsqueeze(0),  # (1, n_classes)
            )
            
            # Build k-NN graph
            data = self.knn_transform(data)
            
            self.graphs.append(data)
            loaded_count += 1
        
        print(f"\nLoaded {loaded_count} simulations successfully")
        if loaded_count > 0:
            print(f"Features per star: {self.graphs[0].x.shape[1]}")
    
    def len(self):
        """Return number of graphs in dataset."""
        return len(self.graphs)
    
    def get(self, idx):
        """Get graph at index idx."""
        return self.graphs[idx]


def print_dataset_statistics(dataset: StarClusterGraphDataset) -> List[int]:
    """
    Print class distribution statistics for a dataset.
    
    Args:
        dataset: StarClusterGraphDataset instance
    
    Returns:
        List of sample counts per class
    """
    labels = [dataset.get(i).y.argmax().item() for i in range(len(dataset))]
    labels = np.array(labels)
    
    n_classes = dataset.n_classes
    class_labels = [class_to_label(i, n_classes) for i in range(n_classes)]
    
    print(f"\n{'─'*50}")
    print(f"Dataset Statistics ({n_classes}-Class System)")
    print(f"{'─'*50}")
    print(f"Total samples: {len(labels)}")
    print(f"\n{'Class':>6} {'Label':>18} {'Count':>8} {'%':>8}")
    print(f"{'-'*44}")
    
    n_per_class = []
    for cls in range(n_classes):
        n = (labels == cls).sum()
        n_per_class.append(n)
        pct = n / len(labels) * 100 if len(labels) > 0 else 0
        print(f"{cls:>6} {class_labels[cls]:>18} {n:>8} {pct:>7.1f}%")
    
    print(f"{'─'*50}\n")
    return n_per_class


# ═════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════

class GNNBlock(nn.Module):
    """
    Graph neural network block with skip connection.
    
    Consists of:
        - SAGEConv layer (graph convolution)
        - Batch normalization
        - ReLU activation
        - Residual/skip connection
    
    Args:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels 
            else nn.Identity()
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass with skip connection.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
        
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        return out + self.skip(x)


class StarClusterGNN(nn.Module):
    """
    Graph neural network for star cluster subcluster classification.
    
    Architecture:
        1. Input projection to hidden dimension
        2. Three GNN blocks with skip connections
        3. Global pooling (mean + max)
        4. Two-layer MLP classifier with dropout
    
    Args:
        in_channels: Input feature dimensionality (13 or 20)
        n_classes: Number of output classes (default: Config.N_CLASSES)
        hidden: Hidden layer dimensionality (default: Config.HIDDEN_DIM)
    """
    
    def __init__(self, 
                 in_channels: int, 
                 n_classes: int = None, 
                 hidden: int = None):
        super().__init__()
        
        if n_classes is None:
            n_classes = Config.N_CLASSES
        if hidden is None:
            hidden = Config.HIDDEN_DIM
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        
        self.gnn1 = GNNBlock(hidden, hidden)
        self.gnn2 = GNNBlock(hidden, hidden)
        self.gnn3 = GNNBlock(hidden, hidden * 2)
        
        pool_dim = hidden * 4  # Concatenation of mean and max pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(p=Config.DROPOUT),
            
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=Config.DROPOUT),
            
            nn.Linear(hidden, n_classes),
        )
    
    def forward(self, data, return_embedding=False):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data/Batch object with:
                - x: Node features
                - edge_index: Graph connectivity
                - batch: Batch assignment vector
            return_embedding: If True, also return graph embeddings
        
        Returns:
            logits: (batch_size, n_classes) class logits
            embedding: (batch_size, pool_dim) graph embeddings (if requested)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)
        x = self.gnn3(x, edge_index)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        embedding = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        logits = self.classifier(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits


# ═════════════════════════════════════════════════════════════════════
# UNCERTAINTY QUANTIFICATION
# ═════════════════════════════════════════════════════════════════════

class UncertaintyQuantifier:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses multiple forward passes with dropout enabled at test time
    to estimate prediction uncertainty via variance across samples.
    
    Args:
        model: Trained StarClusterGNN model
        n_samples: Number of MC samples to draw (default: 30)
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 30):
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict_with_uncertainty(self, data, device):
        """
        Make predictions with uncertainty quantification.
        
        Args:
            data: PyTorch Geometric Data/Batch object
            device: torch device (CPU/GPU)
        
        Returns:
            Tuple of (mean_probs, std_probs, entropy, predicted_class):
                - mean_probs: (batch_size, n_classes) mean probabilities
                - std_probs: (batch_size, n_classes) standard deviations
                - entropy: (batch_size,) predictive entropy
                - predicted_class: (batch_size,) predicted class indices
        """
        self.model.eval()
        data = data.to(device)
        
        # Collect predictions from multiple MC samples
        predictions = []
        for _ in range(self.n_samples):
            self.enable_dropout()
            logits = self.model(data)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions, axis=0)  # (n_samples, batch_size, n_classes)
        
        # Compute statistics
        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        predicted_class = mean_probs.argmax(axis=1)
        
        return mean_probs, std_probs, entropy, predicted_class


# ═════════════════════════════════════════════════════════════════════
# VALIDATION PLOTS
# ═════════════════════════════════════════════════════════════════════

class ValidationPlots:
    """
    Publication-quality validation plots.
    
    Generates a suite of diagnostic plots for model evaluation:
        - Validation summary (predicted vs true NSC)
        - Confusion matrix
        - Per-class accuracy
        - Error distribution
        - Confidence calibration
    
    Args:
        output_dir: Directory to save plots
        n_classes: Number of classes (default: Config.N_CLASSES)
    """
    
    def __init__(self, output_dir: Path, n_classes: int = None):
        if n_classes is None:
            n_classes = Config.N_CLASSES
        
        self.n_classes = n_classes
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.colors = [CMAP(i / (n_classes - 1)) for i in range(n_classes)]
        self.primary_color = CMAP(0.5)
    
    def plot_validation_summary(self, results: Dict, save_path: Path):
        """
        Plot mean predicted NSC vs true NSC with 90% confidence intervals.
        
        Args:
            results: Dictionary with 'probs' and 'true_labels' keys
            save_path: Path to save the plot
        """
        probs = results['probs']
        true_labels = results['true_labels']
        
        mean_predictions = []
        ci_lower = []
        ci_upper = []
        
        # NSC values for plotting (e.g., [1, 2, 6] for 3-class)
        nsc_values = get_nsc_plot_values(self.n_classes)
        
        for true_cls in range(self.n_classes):
            mask = true_labels == true_cls
            if mask.sum() == 0:
                continue
            
            class_probs = probs[mask]
            predicted_nsc = (class_probs * nsc_values).sum(axis=1)
            
            mean_pred = predicted_nsc.mean()
            ci_90 = np.percentile(predicted_nsc, [5, 95])
            
            mean_predictions.append(mean_pred)
            ci_lower.append(ci_90[0])
            ci_upper.append(ci_90[1])
        
        true_nsc_plot = nsc_values
        mean_predictions = np.array(mean_predictions)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Perfect prediction line
        ax.plot([1, nsc_values[-1] + 1], [1, nsc_values[-1] + 1], '--', 
                color='k', linewidth=2.5, zorder=1, label='Perfect prediction')
        
        # Error bars
        ax.errorbar(true_nsc_plot, mean_predictions, 
                   yerr=[mean_predictions - ci_lower, ci_upper - mean_predictions],
                   fmt='o', markersize=10, capsize=8, capthick=2.5, 
                   elinewidth=2.5, color=self.primary_color, 
                   markeredgecolor='white', markeredgewidth=1.5,
                   label='Mean ± 90% CI', zorder=2)
        
        ax.set_xlabel(r'True $N_{\rm sc}$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'Mean Predicted $N_{\rm sc}$', fontsize=14, fontweight='bold')
        ax.set_title('Validation Summary', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(nsc_values)
        ax.set_xticklabels(get_class_labels(self.n_classes))
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved validation summary to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, results: Dict, save_path: Path):
        """
        Plot normalized confusion matrix heatmap.
        
        Args:
            results: Dictionary with 'predictions' and 'true_labels' keys
            save_path: Path to save the plot
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        # Build confusion matrix
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(true_labels, predictions):
            cm[true, pred] += 1
        
        # Normalize by row
        cm_normalized = cm.astype(float)
        for i in range(self.n_classes):
            if cm[i, :].sum() > 0:
                cm_normalized[i, :] = cm[i, :] / cm[i, :].sum() * 100
        
        fig, ax = plt.subplots(figsize=(9, 8))
        
        im = ax.imshow(cm_normalized, cmap=CMAP, vmin=0, vmax=100, aspect='auto')
        
        labels = get_class_labels(self.n_classes)
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel(r'Predicted $N_{\rm sc}$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'True $N_{\rm sc}$', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        
        # Annotate cells
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                val = cm_normalized[i, j]
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       color=color, fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()


# ═════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training when it stops improving
    for a specified number of epochs (patience).
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(self, patience: int = None, min_delta: float = None):
        if patience is None:
            patience = Config.PATIENCE
        if min_delta is None:
            min_delta = Config.MIN_DELTA
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def plot_training_curves(train_losses: List[float], 
                         val_losses: List[float], 
                         train_accs: List[float], 
                         val_accs: List[float], 
                         save_path: Path,
                         n_classes: int = None):
    """
    Plot and save training curves (loss and accuracy).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the plot
        n_classes: Number of classes for title (default: Config.N_CLASSES)
    """
    if n_classes is None:
        n_classes = Config.N_CLASSES
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(val_losses, label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title(f"Training Loss ({n_classes}-Class)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.plot([a * 100 for a in train_accs], label="Train", linewidth=2)
    ax2.plot([a * 100 for a in val_accs], label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title(f"Classification Accuracy ({n_classes}-Class)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ═════════════════════════════════════════════════════════════════════

class Trainer:
    """
    End-to-end training pipeline for StarTrace GNN.
    
    Handles dataset loading, model initialization, training loop,
    validation, checkpointing, and logging.
    
    Args:
        data_path: Path to simulation data directory
        n_seeds: Number of random seeds per NSC class
        n_scs: Maximum NSC value in dataset (e.g., 8 for NSC1-NSC8)
        config: Optional Config object (uses global Config if None)
    """
    
    def __init__(self, 
                 data_path: str,
                 n_seeds: int = 300,
                 n_scs: int = 8,
                 config = None):
        
        self.config = config if config is not None else Config
        self.data_path = data_path
        self.n_seeds = n_seeds
        self.n_scs = n_scs
        
        # Create output directories
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.config.PLOT_DIR.mkdir(exist_ok=True, parents=True)
        
        # Will be populated during training
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
    def build_dataset(self):
        """Load and split dataset into train/validation."""
        print("Building dataset...")
        
        # Generate simulation names
        sim_names = []
        for nsc in range(1, self.n_scs + 1):
            for seed in range(self.n_seeds):
                sim_names.append(f"NSC{nsc}SEED{seed}")
        
        # Load dataset
        self.dataset = StarClusterGraphDataset(
            data_path=self.data_path,
            sim_names=sim_names,
            snapshot=self.config.SNAPSHOT,
            k=self.config.K_NEIGHBORS,
            use_global=self.config.USE_GLOBAL_FEATURES,
            n_classes=self.config.N_CLASSES
        )
        
        # Print statistics
        n_per_class = print_dataset_statistics(self.dataset)
        
        # Train/val split
        n_val = int(len(self.dataset) * self.config.VAL_FRACTION)
        n_train = len(self.dataset) - n_val
        
        train_set, val_set = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        print(f"Train samples: {n_train}")
        print(f"Validation samples: {n_val}")
        
        return n_per_class
    
    def setup_model(self, n_features: int, n_per_class: List[int]):
        """Initialize model, optimizer, loss, and early stopping."""
        print(f"\nFeatures per star: {n_features}")
        
        # Model
        self.model = StarClusterGNN(
            in_channels=n_features,
            n_classes=self.config.N_CLASSES,
            hidden=self.config.HIDDEN_DIM
        ).to(self.config.DEVICE)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS
        )
        
        # Loss with class weighting to handle imbalance
        weights = 1.0 / torch.tensor(n_per_class, dtype=torch.float)
        weights = weights / weights.sum() * self.config.N_CLASSES
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.config.DEVICE))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=self.config.MIN_DELTA
        )
    
    def train_one_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pred_class = logits.argmax(dim=1)
            true_class = batch.y.argmax(dim=1)
            total_correct += (pred_class == true_class).sum().item()
            total_samples += batch.num_graphs
        
        return total_loss / total_samples, total_correct / total_samples
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in self.val_loader:
            batch = batch.to(self.config.DEVICE)
            
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            pred_class = logits.argmax(dim=1)
            true_class = batch.y.argmax(dim=1)
            total_correct += (pred_class == true_class).sum().item()
            total_samples += batch.num_graphs
        
        return total_loss / total_samples, total_correct / total_samples
    
    def save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, n_features: int):
        """Save model checkpoint."""
        config_dict = {
            'N_CLASSES': self.config.N_CLASSES,
            'K_NEIGHBORS': self.config.K_NEIGHBORS,
            'HIDDEN_DIM': self.config.HIDDEN_DIM,
            'USE_GLOBAL_FEATURES': self.config.USE_GLOBAL_FEATURES,
            'SNAPSHOT': self.config.SNAPSHOT,
        }
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': config_dict,
            'n_features': n_features,
            'use_global_features': self.config.USE_GLOBAL_FEATURES,
            'n_classes': self.config.N_CLASSES,
        }, self.config.MODEL_PATH)
    
    def train(self) -> nn.Module:
        """
        Run full training loop.
        
        Returns:
            Trained model
        """
        print(f"\n{'═'*60}")
        print(f"StarTrace: {self.config.N_CLASSES}-Class Star Cluster GNN Training")
        print(f"Classes: {', '.join([class_to_label(i, self.config.N_CLASSES) for i in range(self.config.N_CLASSES)])}")
        if self.config.USE_GLOBAL_FEATURES:
            print("(WITH GLOBAL CLUSTER FEATURES)")
        print(f"{'═'*60}")
        print(f"Device: {self.config.DEVICE}")
        print(f"Data path: {self.data_path}")
        print(f"{'═'*60}\n")
        
        # Build dataset
        n_per_class = self.build_dataset()
        
        # Setup model
        n_features = self.dataset.get(0).x.shape[1]
        self.setup_model(n_features, n_per_class)
        
        # Training loop
        print(f"\n{'═'*60}")
        print(f"Training for {self.config.EPOCHS} epochs")
        print(f"{'═'*60}\n")
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        
        for epoch in range(1, self.config.EPOCHS + 1):
            tr_loss, tr_acc = self.train_one_epoch()
            vl_loss, vl_acc = self.evaluate()
            self.scheduler.step()
            
            train_losses.append(tr_loss)
            train_accs.append(tr_acc)
            val_losses.append(vl_loss)
            val_accs.append(vl_acc)
            
            # Save best model
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                self.save_checkpoint(epoch, vl_acc, vl_loss, n_features)
            
            # Logging
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{self.config.EPOCHS} | "
                    f"Train: loss={tr_loss:.4f}, acc={tr_acc*100:5.1f}% | "
                    f"Val: loss={vl_loss:.4f}, acc={vl_acc*100:5.1f}%"
                )
            
            # Early stopping
            if self.early_stopping(vl_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n{'═'*60}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
        print(f"Model saved to: {self.config.MODEL_PATH}")
        print(f"{'═'*60}\n")
        
        # Plot training curves
        plot_path = self.config.PLOT_DIR / f"training_curves_{self.config.N_CLASSES}class.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                           plot_path, self.config.N_CLASSES)
        
        return self.model


# ═════════════════════════════════════════════════════════════════════
# VALIDATOR CLASS
# ═════════════════════════════════════════════════════════════════════

class Validator:
    """
    Model validation and evaluation pipeline.
    
    Loads a trained model, runs inference on validation data with
    uncertainty quantification, and generates diagnostic plots.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to simulation data directory
        output_dir: Directory to save validation results
        n_seeds: Number of random seeds per NSC class
        n_scs: Maximum NSC value in dataset
    """
    
    def __init__(self,
                 model_path: str,
                 data_path: str,
                 output_dir: str = "outputs/validation",
                 n_seeds: int = 300,
                 n_scs: int = 8):
        
        self.model_path = Path(model_path)
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.n_seeds = n_seeds
        self.n_scs = n_scs
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Will be populated during loading
        self.model = None
        self.val_loader = None
        self.device = Config.DEVICE
        self.uncertainty_quantifier = None
        self.plotter = None
        self.n_classes = None
    
    def load_model_and_data(self):
        """Load trained model and validation dataset."""
        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            use_global = checkpoint.get('use_global_features', False)
            self.n_classes = checkpoint.get('n_classes', Config.N_CLASSES)
        else:
            state_dict = checkpoint
            use_global = False
            self.n_classes = Config.N_CLASSES
        
        print(f"Loading dataset (use_global_features={use_global})...")
        
        # Generate simulation names
        sim_names = []
        for nsc in range(1, self.n_scs + 1):
            for seed in range(self.n_seeds):
                sim_names.append(f"NSC{nsc}SEED{seed}")
        
        # Load dataset
        dataset = StarClusterGraphDataset(
            data_path=self.data_path,
            sim_names=sim_names,
            snapshot=Config.SNAPSHOT,
            k=Config.K_NEIGHBORS,
            use_global=use_global,
            n_classes=self.n_classes
        )
        
        # Split dataset (same split as training)
        n_val = int(len(dataset) * Config.VAL_FRACTION)
        n_train = len(dataset) - n_val
        
        _, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        
        # Initialize model
        n_features = dataset.get(0).x.shape[1]
        self.model = StarClusterGNN(
            in_channels=n_features,
            n_classes=self.n_classes,
            hidden=Config.HIDDEN_DIM
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Model loaded successfully ({n_features} features per star)")
        
        # Initialize uncertainty quantifier and plotter
        self.uncertainty_quantifier = UncertaintyQuantifier(self.model, n_samples=30)
        self.plotter = ValidationPlots(self.output_dir, self.n_classes)
    
    def collect_predictions(self) -> Dict:
        """Collect predictions with uncertainty on validation set."""
        all_probs = []
        all_stds = []
        all_entropies = []
        all_true_labels = []
        
        print("Collecting predictions with uncertainty quantification...")
        
        for batch in self.val_loader:
            probs, stds, entropy, _ = self.uncertainty_quantifier.predict_with_uncertainty(
                batch, self.device
            )
            all_probs.append(probs)
            all_stds.append(stds)
            all_entropies.append(entropy)
            all_true_labels.append(batch.y.argmax(dim=1).cpu().numpy())
        
        return {
            'probs': np.vstack(all_probs),
            'stds': np.vstack(all_stds),
            'entropies': np.concatenate(all_entropies),
            'true_labels': np.concatenate(all_true_labels),
            'predictions': np.vstack(all_probs).argmax(axis=1)
        }
    
    def print_summary(self, results: Dict):
        """Print validation summary statistics."""
        predictions = results['predictions']
        true_labels = results['true_labels']
        accuracy = (predictions == true_labels).mean()
        
        labels = get_class_labels(self.n_classes)
        
        print(f"\n{'═'*60}")
        print(f"VALIDATION SUMMARY ({self.n_classes}-Class Model)")
        print(f"{'═'*60}")
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        print(f"\nPer-Class Accuracy:")
        print(f"{'─'*40}")
        
        for cls in range(self.n_classes):
            mask = true_labels == cls
            if mask.sum() > 0:
                acc = (predictions[mask] == cls).mean()
                n = mask.sum()
                print(f"  N_sc = {labels[cls]:>3s}: {acc*100:5.1f}%  (n={n})")
        
        print(f"{'═'*60}\n")
    
    def run(self) -> Dict:
        """
        Run complete validation pipeline.
        
        Returns:
            Dictionary of validation results
        """
        print("\n" + "="*60)
        print("Running Validation")
        print("="*60 + "\n")
        
        # Load model and data
        self.load_model_and_data()
        
        # Collect predictions
        results = self.collect_predictions()
        
        # Print summary
        self.print_summary(results)
        
        # Generate plots
        print("\nGenerating validation plots...")
        
        self.plotter.plot_validation_summary(
            results, self.output_dir / "validation_summary.png"
        )
        self.plotter.plot_confusion_matrix(
            results, self.output_dir / "confusion_matrix.png"
        )
        
        print(f"\n✓ All plots saved to {self.output_dir}")
        print(f"{'='*60}\n")
        
        return results


# ═════════════════════════════════════════════════════════════════════
# PREDICTOR CLASS
# ═════════════════════════════════════════════════════════════════════

class Predictor:
    """
    Inference interface for new star cluster data.
    
    Loads a trained model and provides methods for predicting the
    number of subclusters in new clusters with uncertainty quantification.
    
    Args:
        model_path: Path to trained model checkpoint
        device: torch device (auto-detected if None)
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or Config.DEVICE
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract configuration
        if isinstance(checkpoint, dict):
            n_features = checkpoint.get('n_features', 13)
            self.n_classes = checkpoint.get('n_classes', Config.N_CLASSES)
            self.use_global = checkpoint.get('use_global_features', False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            n_features = 13
            self.n_classes = Config.N_CLASSES
            self.use_global = False
            state_dict = checkpoint
        
        # Initialize model
        self.model = StarClusterGNN(
            in_channels=n_features,
            n_classes=self.n_classes,
            hidden=Config.HIDDEN_DIM
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Uncertainty quantifier
        self.uncertainty_quantifier = UncertaintyQuantifier(self.model, n_samples=30)
        
        # k-NN transform
        self.knn_transform = KNNGraph(k=Config.K_NEIGHBORS, loop=False, cosine=False)
        
        print("Model loaded successfully!")
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"  Trained epoch: {checkpoint['epoch']}")
            print(f"  Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    def preprocess_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Preprocess raw coordinates to standardized format.
        
        Args:
            coords: (N, 6) array of [x, y, z, vx, vy, vz]
        
        Returns:
            Preprocessed (N, 6) array
        """
        n_stars = len(coords)
        print(f"  Processing cluster with {n_stars} stars")
        
        # Standardize: zero mean, unit variance per feature
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        
        return coords.astype(np.float32)
    
    def predict(self, coords: np.ndarray, return_full_distribution: bool = True) -> Dict:
        """
        Predict number of subclusters with uncertainty.
        
        Args:
            coords: (N, 6) array of [x, y, z, vx, vy, vz]
            return_full_distribution: Include full probability distribution
        
        Returns:
            Dictionary with prediction results and uncertainties
        """
        # Preprocess
        coords = self.preprocess_coordinates(coords)
        features = derive_physical_features(coords, include_global=self.use_global)
        
        # Create graph
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            pos=torch.tensor(coords, dtype=torch.float),
        )
        data = self.knn_transform(data)
        data = data.to(self.device)
        
        # Predict with uncertainty
        with torch.no_grad():
            probs, stds, entropy, pred_class = self.uncertainty_quantifier.predict_with_uncertainty(
                data, self.device
            )
        
        # Extract results
        probs = probs[0]
        stds = stds[0]
        entropy = entropy[0]
        pred_class = pred_class[0]
        
        # Top predictions
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            {
                'nsc': class_to_label(int(idx), self.n_classes),
                'probability': float(probs[idx]),
                'uncertainty': float(stds[idx])
            }
            for idx in top_indices
        ]
        
        # Build result
        result = {
            'predicted_nsc': class_to_label(int(pred_class), self.n_classes),
            'predicted_class': int(pred_class),
            'confidence': float(probs[pred_class] * 100),
            'uncertainty': float(entropy),
            'top_predictions': top_predictions,
        }
        
        if return_full_distribution:
            result['distribution'] = {
                class_to_label(i, self.n_classes): {
                    'probability': float(probs[i]),
                    'std': float(stds[i])
                }
                for i in range(self.n_classes)
            }
        
        return result
    
    def predict_from_file(self, filepath: str, **kwargs) -> Dict:
        """
        Predict from a file containing coordinates.
        
        Args:
            filepath: Path to coordinate file
            **kwargs: Additional arguments for predict()
        
        Returns:
            Prediction dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"\nLoading cluster data from {filepath}...")
        
        # Load data
        try:
            coords = np.loadtxt(filepath, comments='#')
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
        
        # Handle different formats
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        # Check if first column is mass (7 columns total)
        if coords.shape[1] == 7:
            print("  Detected mass column, dropping it...")
            coords = coords[:, 1:7]
        elif coords.shape[1] != 6:
            raise ValueError(f"Expected 6 or 7 columns, got {coords.shape[1]}")
        
        print(f"  Loaded {len(coords)} stars")
        
        return self.predict(coords, **kwargs)
    
    def print_summary(self, result: Dict):
        """Print human-readable prediction summary."""
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70 + "\n")
        
        print(f"Predicted: {result['predicted_nsc']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Uncertainty (entropy): {result['uncertainty']:.3f}")
        
        print(f"\nTop 3 predictions:")
        print(f"{'Rank':>5} {'NSC':>15} {'Probability':>12} {'±Std':>10}")
        print("-" * 45)
        
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"{i:>5} {pred['nsc']:>15} {pred['probability']*100:>10.1f}% "
                  f"{pred['uncertainty']*100:>9.1f}%")
        
        print("\n" + "="*70 + "\n")
