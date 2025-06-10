from ansys.dpf import core as dpf
from ansys.dpf.core.operators.invariant import von_mises_eqv_fc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from matplotlib.colors import Normalize
import json
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum



@dataclass
class StressLayerData:
    """Data structure for stress-specific layer geometry information."""
    layer_height: float
    thickness: float
    regions: Dict[str, List[List[List[float]]]]  # stress_level -> list of polygons -> list of points -> [x, y]
    node_count: int
    stress_statistics: Dict[str, float]
    nodes : Optional[np.ndarray] = None  # Array of nodes in the 
    
    def to_summary_dict(self) -> dict:
        keys = ["layer_height", "thickness", "regions", "node_count", "stress_statistics"]
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in asdict(self).items() if k in keys}

@dataclass
class StressContour:
    """Data structure for stress-specific contour information."""
    stress_level: str
    polygons: List[List[List[float]]]  # list of polygons -> list of points -> [x, y]
    area: float
    node_count: int

@dataclass
class StressLayer:
    """Data structure for complete stress layer information."""
    z_height: float
    thickness: float
    contours: List[StressContour]
    total_area: float
    total_nodes: int
    stress_range: Tuple[float, float]

class FEMAnalysis:
    """
    A comprehensive class for FEM stress analysis and slicing operations.
    
    This class encapsulates the entire workflow from loading ANSYS .rst files
    to computing von Mises stress, classifying stress levels, and extracting
    nodes in specific slices for further analysis.
    """
    
    def __init__(self, rst_path: str, num_classes: int = 3,
                 eps: float = 1, min_samples: int = 5, min_cluster_area: float = 1, 
                 min_nodes: int = 150, max_allowed: float = 0.5, rotation_matrix: Optional[np.ndarray] = None):
        """
        Initialize the FEM Stress Analyzer.
        
        Parameters:
            rst_path (str): Path to the ANSYS .rst result file
            num_classes (int): Number of stress classification classes (default: 3)
            classification_strategy (ClassificationStrategy or str): Strategy for stress classification 
                (default: ClassificationStrategy.QUANTILE)
            eps (float): DBSCAN clustering radius parameter (default: 0.5)
            min_samples (int): DBSCAN minimum samples per cluster (default: 5)
            min_cluster_area (float): Minimum area threshold for clusters (default: 1e-6)
            min_nodes (int): Minimum nodes required in slice (default: 150)
            max_allowed (float): Maximum tolerance for slice extraction (default: 0.5)
        """
        self.rst_path = rst_path
        self.num_classes = num_classes
        
        # Clustering parameters
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_area = min_cluster_area
        self.min_nodes = min_nodes
        self.max_allowed = max_allowed
        
        # Instance attributes for intermediate state
        self.model = None
        self.mesh = None
        self.nodes = None
        self.unit = None
        self.unit_scale = None
        self.stress_bins = None
        
        # OPTIMIZATION: Add caching for slice operations
        self._slice_cache = {}
        self._z_coords_cache = None

        # Rotation
        self.rotation_matrix = rotation_matrix  # Default: Identität

        # Stress threshold
        self.stress_thresshold = None
        
        # Initialize the model and prepare data
        self._initialize()
    
    def _initialize(self):
        """Initialize the model and prepare FEM data."""
        self.model = self.load_model()
        self.mesh = self.model.metadata.meshed_region
        self.unit, self.unit_scale = self.get_length_unit_and_scale()
        self.get_node_data_with_stress()
        self.calculate_stress_thresholds()

        #self.unit_scale = self.get_unit_scale(self.unit)
        #self.node_array, self.stress_bins = self.prepare_fem_data()
        #self.stress_thresshold = self.calculate_stress_thresholds()

    def load_model(self):
        """
        Load the ANSYS model from the .rst file.
        
        Returns:
            dpf.Model: The loaded ANSYS DPF model
            
        Raises:
            ValueError: If file is not a .rst file
            FileNotFoundError: If file does not exist
        """
        path_obj = Path(self.rst_path)

        if path_obj.suffix != ".rst":
            raise ValueError("Only .rst files are supported.")

        if not path_obj.exists():
            raise FileNotFoundError(f"Result file not found: {path_obj}")
        
        return dpf.Model(self.rst_path)
    
    def get_length_unit_and_scale(self) -> str:
        """
        Extract the unit system from the model metadata.

        Returns:
            Tuple[str, str]: (length_unit, stress_unit)
        """
        metadata = self.model.metadata
        units_str = metadata.result_info.unit_system

        parts = units_str.split(":")[1].strip().split(", ")
        length_unit = parts[0]

       
        unit_scales = {
            "m": 1000.0,      # meters to millimeters
            "mm": 1.0,        # already in millimeters
            "cm": 10.0        # centimeters to millimeters
        }

        if length_unit not in unit_scales:
            raise ValueError(f"Unsupported unit '{length_unit}'. Add it to unit_scales if needed.")
        
        scale = unit_scales.get(length_unit, 1.0)  # default = no scaling

        return length_unit, scale
        
    def compute_von_mises_per_node(self, target_unit: str = "MPa") -> np.ndarray:
        """
        Compute the von Mises stress per node from the FEM model.
        
        Parameters:
            to_unit (str): Desired output unit. Supports "Pa" or "MPa". Default: "MPa"

        Returns:
            np.ndarray: Array of shape (n_nodes, 2), with [NodeID, VonMises stress]
        """

        # Step 1: Get elemental stress result
        stress_fc = self.model.results.stress().eval()

        # Step 2: Compute von Mises using DPF operator
        vm_op = von_mises_eqv_fc()
        vm_op.inputs.fields_container.connect(stress_fc)
        vm_fc = vm_op.outputs.fields_container()
        vm_field = vm_fc[0]  # usually ElementalNodal

        # Step 3: Convert to nodal stress
        vm_nodal = vm_field.to_nodal()

        # Step 4: Extract data
        node_ids = vm_nodal.scoping.ids
        von_mises_values = vm_nodal.data

        # Step 5: Unit conversion if needed
        current_unit = vm_nodal.unit  # typically 'Pa'
        
        if target_unit == "MPa" and current_unit == "Pa":
            von_mises_values = von_mises_values / 1e6
        elif target_unit == "Pa" and current_unit == "MPa":
            von_mises_values = von_mises_values * 1e6

        # Step 6: Combine node IDs and stress values
        vm_array = np.column_stack((node_ids, von_mises_values))

        return vm_array
    
    def extract_node_coordinates(self):
        """
        Extract node coordinates from the mesh.
        
        Returns:
            np.ndarray: Array of shape (n_nodes, 4) containing [NodeID, X, Y, Z]
        """
        node_ids = np.array(self.mesh.nodes.scoping.ids)  # shape: (n_nodes,)
        coordinates_field = self.mesh.nodes.coordinates_field
        coordinates = coordinates_field.data
        coordinates *= self.unit_scale  # Scale coordinates to mm if needed
        node_info = np.concatenate([node_ids.reshape(-1, 1), coordinates], axis=1)  # [NodeID, X, Y, Z]
        return node_info
    
    def get_node_data_with_stress(self):
        """
        Build combined array with node coordinates and von Mises stress.
        
        Returns:
            np.ndarray: Array of shape (n_nodes, 5) containing [NodeID, X, Y, Z, VonMises]
        """
        node_info = self.extract_node_coordinates()  # [NodeID, X, Y, Z]
        vm_array = self.compute_von_mises_per_node()  # [NodeID, VonMises (Pa)]

        # Match node order by ID
        sort_idx = np.argsort(node_info[:, 0])
        node_info_sorted = node_info[sort_idx]
        sort_vm_idx = np.argsort(vm_array[:, 0])
        vm_array_sorted = vm_array[sort_vm_idx]

        self.nodes = np.concatenate([node_info_sorted, vm_array_sorted[:, 1].reshape(-1, 1)], axis=1)
        return self.nodes # [NodeID, X, Y, Z, VonMises]
    
    def calculate_stress_thresholds(self, streckgrenze: float = 70, sicherheitsfaktor: float = 1.5) -> tuple[float, float]:
        """
        Berechnet die von-Mises-Grenzwerte für Spannungszonen (low, moderate, high)
        unter Berücksichtigung eines Sicherheitsfaktors.

        Args:
            streckgrenze (float): Technische Streckgrenze (Rp0.2) des Materials in MPa. 520
            sicherheitsfaktor (float): Sicherheitsfaktor zur konservativen Auslegung.

        Returns:
            tuple: (low_max, mod_max) Spannungsgrenzen in MPa.
        """
        sigma_zul = streckgrenze / sicherheitsfaktor
        low_max = 0.3 * sigma_zul
        mod_max = 0.7 * sigma_zul

        print(f"[INFO] Calculated stress thresholds: Low = {low_max:.2f} MPa, Moderate = {mod_max:.2f} MPa")

        self.stress_thresshold = (low_max, mod_max)
        return self.stress_thresshold
    
    def get_nodes_in_slice(self, z: float, thickness: float) -> np.ndarray:
        """
        Extract nodes within a Z-slice, adaptively increasing tolerance if needed.

        Args:
            z (float): Z height
            thickness (float): Initial slice thickness ; Default = 0.25 mm).

        Returns:
            np.ndarray: Array of shape (n_nodes_in_slice, 4) → [X, Y, Z, VonMises]
        """

        if self.nodes is None:
            self.get_node_data_with_stress()
        
        # Basic checks
        if not isinstance(z, (int, float)) or not isinstance(thickness, (int, float)):
            raise TypeError("z and thickness must be numeric.")
        if thickness <= 0.001:
            raise ValueError("thickness must be greater than 0.")

        z_coords = self.nodes[:, 3]  # Extract Z coordinates from nodes
        fem_nodes = self.nodes[:, 1:5] # Extract X, Y, Z coordinates from nodes
        tolerance = thickness
        z_diffs = np.abs(z_coords - z)  # [|z1 - z|, |z2 - z|, ...]

        MAX_TRIES = 5
        tries = 0

        slice_mask = z_diffs <= tolerance # mask : [True, False, True, ...]
        slice_nodes = fem_nodes[slice_mask]

        while slice_nodes.shape[0] < self.min_nodes and tolerance < self.max_allowed and tries < MAX_TRIES:
            tolerance *= 1.5
            slice_mask = z_diffs <= tolerance
            slice_nodes = fem_nodes[slice_mask]
            tries += 1

        return slice_nodes

    def classify_and_cluster_stress_regions(self, xy_points: np.ndarray, stress_values: np.ndarray) -> dict:
        """
        Classify XY FEM node coordinates into stress zones and apply clustering
        to generate geometric regions.

        Args:
            xy_points (np.ndarray): Array of shape (n, 2) containing X, Y node coordinates.
            stress_values (np.ndarray): Array of shape (n,) with von Mises stress values.

        Returns:
            dict: {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...],
                "high": [Polygon, ...]
            }
        """
        low_max, mod_max = self.stress_thresshold

        # 1. Build masks
        low_mask = stress_values < low_max
        moderate_mask = (stress_values >= low_max) & (stress_values < mod_max)
        high_mask = stress_values >= mod_max

        stress_masks = {
            "low": low_mask,
            "moderate": moderate_mask,
            "high": high_mask
        }
        # print(f"[INFO] Stress masks created: low={np.sum(low_mask)}, moderate={np.sum(moderate_mask)}, high={np.sum(high_mask)}") for z = 0.0
        

        regions = {"low": [], "moderate": [], "high": []}

        # OPTIMIZATION 4: Process each stress level with vectorized operations
        for label, mask in stress_masks.items():
            if not np.any(mask):
                continue
                
            points = xy_points[mask]
            if len(points) == 0:
                continue

            # OPTIMIZATION 5: Use more efficient clustering with early exit
            if len(points) < self.min_samples:
                # Skip clustering for very small point sets
                if len(points) >= 3:
                    try:
                        poly = Polygon(points).convex_hull
                        if poly.area >= self.min_cluster_area:
                            regions[label].append(poly)
                    except:
                        pass  # Skip invalid polygons
                continue

            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = clustering.labels_
            
            # OPTIMIZATION 6: Vectorized cluster processing
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_points = points[cluster_mask]
                
                if len(cluster_points) >= 3:
                    try:
                        poly = Polygon(cluster_points).convex_hull
                        if poly.area >= self.min_cluster_area:
                            regions[label].append(poly)
                    except:
                        pass  # Skip invalid polygons
            
            # OPTIMIZATION 7: Improved noise handling with vectorized distance computation
            noise_mask = labels == -1
            if np.any(noise_mask):
                noise_points = points[noise_mask]
                non_noise_points = points[~noise_mask]
                
                if len(non_noise_points) > 0:
                    # Vectorized distance computation using broadcasting
                    distances = np.linalg.norm(
                        noise_points[:, np.newaxis, :] - non_noise_points[np.newaxis, :, :], 
                        axis=2
                    )
                    min_distances = np.min(distances, axis=1)
                    
                    # Process noise points that are close to clusters
                    close_noise_mask = min_distances < 1.5 * self.eps
                    close_noise_points = noise_points[close_noise_mask]
                    
                    for pt in close_noise_points:
                        try:
                            poly = Point(pt).buffer(self.eps / 2)
                            if poly.area >= self.min_cluster_area:
                                regions[label].append(poly)
                        except:
                            pass  # Skip invalid polygons

        

        return regions

    def generate_slice_stress_regions(self, z: float, thickness: float = 0.25, zone_per_slice: int = 3) -> dict | None:
        """
        Generate stress-based regions for a specific Z-slice by classifying and clustering FEM nodes.

        Args:
            z (float): Target Z height of the slice.
            thickness (float): Slice thickness tolerance (default = 0.01 mm).

        Returns:
            dict | None: Dictionary with stress regions and slice nodes, or None if insufficient nodes.
                        {
                            "regions": {"low": [Polygon], "moderate": [...], "high": [...]},
                            "slice_nodes": np.ndarray of shape (n, 4)
                        }
        """
        # Step 1: Get nodes in the slice
        slice_nodes = self.get_nodes_in_slice(z, thickness)
        
        # Step 2: Check if slice has enough nodes
        if slice_nodes.shape[0] < self.min_nodes:
            print(f"[INFO] Not enough nodes at z={z:.3f}. Found: {slice_nodes.shape[0]}, required: {self.min_nodes}")
            return None

        # Step 3: Classify and cluster stress regions
        regions = self.classify_and_cluster_stress_regions(
            xy_points=slice_nodes[:, :2],                     # X, Y
            stress_values=slice_nodes[:, 3],                  # Von Mises stress
        )

        # Step 4: Clean overlaps
        cleaned_regions = self._clean_overlap_regions(regions)


        # lets print data only for z = 0.0
        if z == 0.0:
            print("==" * 50)
            print(f"[INFO] Processing slice at z={z:.3f} with thickness {thickness:.3f} mm")
            #regions
            print("Without cleaning:")
            print(f"[INFO] Slice at z={z:.3f} has {len(regions['low'])} low, "
                  f"{len(regions['moderate'])} moderate, and "
                  f"{len(regions['high'])} high stress regions.")
            print("After cleaning:")
            print(f"[INFO] Slice at z={z:.3f} has {len(cleaned_regions['low'])} low, "
                  f"{len(cleaned_regions['moderate'])} moderate, and "
                  f"{len(cleaned_regions['high'])} high stress regions.")
            print(f"[INFO] Slice nodes count: {slice_nodes.shape[0]}")

        # Step 5: Return result
        # NEW: Zone limiting functionality
        if zone_per_slice is not None and zone_per_slice > 0:
            cleaned_regions = self._limit_zones_per_slice(cleaned_regions, zone_per_slice)

        return {
            "regions": regions,
            "slice_nodes": slice_nodes
        }

    def _limit_zones_per_slice(self, regions: Dict[str, List], zone_per_slice: int) -> Dict[str, List]:
        """
        Limit the number of zones per slice by sorting by area and merging smaller zones.
        
        Parameters:
            regions (Dict[str, List]): Dictionary of stress regions with polygons
            zone_per_slice (int): Maximum number of zones allowed per slice. Defaults to 3
            
        Returns:
            Dict[str, List]: Limited regions with merged smaller zones
        """
        # Collect all zones with their areas and stress levels
        all_zones = []
        for stress_level, polygons in regions.items():
            for poly in polygons:
                if hasattr(poly, 'area') and poly.area > 0:
                    all_zones.append({
                        'polygon': poly,
                        'area': poly.area,
                        'stress_level': stress_level
                    })
        
        # If we have fewer zones than the limit, return as is
        if len(all_zones) <= zone_per_slice:
            return regions
        
        # Sort zones by area (largest first)
        all_zones.sort(key=lambda x: x['area'], reverse=True)
        
        # Keep the largest zones
        kept_zones = all_zones[:zone_per_slice]
        remaining_zones = all_zones[zone_per_slice:]
        
        # Create new regions dict with kept zones
        limited_regions = {"low": [], "moderate": [], "high": []}
        for zone in kept_zones:
            limited_regions[zone['stress_level']].append(zone['polygon'])
        
        # Merge remaining smaller zones into the nearest kept zones
        for remaining_zone in remaining_zones:
            best_match = None
            min_distance = float('inf')
            
            # Find the closest kept zone by centroid distance
            remaining_centroid = remaining_zone['polygon'].centroid
            
            for kept_zone in kept_zones:
                kept_centroid = kept_zone['polygon'].centroid
                distance = remaining_centroid.distance(kept_centroid)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = kept_zone
            
            # Merge the remaining zone with the best match
            if best_match is not None:
                try:
                    # Find the polygon in the limited_regions and merge
                    stress_level = best_match['stress_level']
                    for i, poly in enumerate(limited_regions[stress_level]):
                        if poly == best_match['polygon']:
                            # Merge the polygons
                            merged_poly = unary_union([poly, remaining_zone['polygon']])
                            limited_regions[stress_level][i] = merged_poly
                            break
                except Exception as e:
                    # If merging fails, just skip this zone
                    print(f"Warning: Could not merge zone: {e}")
                    continue
        
        return limited_regions



    def _clean_overlap_regions(self, regions: dict) -> dict:
        """
        Clean overlapping regions between high, moderate, and low zones.
        Higher stress zones take precedence.

        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [...],
                "high": [...]
            }

        Returns:
            dict: Same format, but with overlaps removed between zones.
        """
        cleaned = {"low": [], "moderate": [], "high": []}

        # Union per level (skip if empty)
        high_union = (
            unary_union(regions["high"]) if len(regions["high"]) > 1 else
            regions["high"][0] if regions["high"] else None
        )

        moderate_union = (
            unary_union(regions["moderate"]) if len(regions["moderate"]) > 1 else
            regions["moderate"][0] if regions["moderate"] else None
        )
        if moderate_union and high_union:
            moderate_union = moderate_union.difference(high_union)

        low_union = (
            unary_union(regions["low"]) if len(regions["low"]) > 1 else
            regions["low"][0] if regions["low"] else None
        )
        if low_union and high_union:
            low_union = low_union.difference(high_union)
        if low_union and moderate_union:
            low_union = low_union.difference(moderate_union)

        # Add only non-empty geometries
        if high_union and not high_union.is_empty:
            cleaned["high"].append(high_union)

        if moderate_union and not moderate_union.is_empty:
            cleaned["moderate"].append(moderate_union)

        if low_union and not low_union.is_empty:
            cleaned["low"].append(low_union)

        return cleaned

    

    def slice_with_stress_analysis(self, z_heights: List[float], thickness: float = 0.1, 
                                 zone_per_slice: Optional[int] = 3) -> List[Dict[str, Any]]:
        """
        Enhanced slicing function that processes multiple Z heights with zone limiting.
        
        Parameters:
            z_heights (List[float]): List of Z heights to process
            thickness (float): Layer thickness (default: 0.1)
            zone_per_slice (int, optional): Maximum number of zones per slice (default: 3)
            
        Returns:
            List[Dict[str, Any]]: List of processed layer results
        """
        results = []
        
        if self.nodes is None:
            self.get_node_data_with_stress()
            
        # Calculate stress thresholds based on bins
        
            
        stress_thresholds = self.stress_thresshold

    
        for z in z_heights:
            # Extract nodes and regions with zone limiting
            slice_result = self.generate_slice_stress_regions(z=z, thickness=thickness)
            
            # Convert to serializable format
            regions_serializable = {}
            total_area = 0.0
            
            for stress_level, polygons in slice_result["regions"].items():
                regions_serializable[stress_level] = []
                for poly in polygons:
                    if hasattr(poly, 'exterior'):
                        # Single polygon
                        coords = list(poly.exterior.coords[:-1])  # Remove duplicate last point
                        regions_serializable[stress_level].append(coords)
                        total_area += poly.area
                    elif hasattr(poly, 'geoms'):
                        # MultiPolygon
                        for sub_poly in poly.geoms:
                            coords = list(sub_poly.exterior.coords[:-1])
                            regions_serializable[stress_level].append(coords)
                            total_area += sub_poly.area
            
            # Get stress statistics for this slice
            slice_nodes = slice_result["slice_nodes"]
            if len(slice_nodes) > 0:
                slice_stress_values = slice_nodes[:, 3]  # VonMises column
                stress_stats = {
                    "min_stress": float(np.min(slice_stress_values)),
                    "max_stress": float(np.max(slice_stress_values)),
                    "mean_stress": float(np.mean(slice_stress_values)),
                    "std_stress": float(np.std(slice_stress_values)),
                    "median_stress": float(np.median(slice_stress_values)),
                    "low_threshold": float(stress_thresholds[0]),
                    "moderate_threshold": float(stress_thresholds[1])
                }
            else:
                stress_stats = {
                    "min_stress": 0.0,
                    "max_stress": 0.0,
                    "mean_stress": 0.0,
                    "std_stress": 0.0,
                    "median_stress": 0.0,
                    "low_threshold": 0.0,
                    "moderate_threshold": 0.0
                }
            
            layer_data = StressLayerData(
                layer_height=float(z),
                thickness=float(thickness),
                regions=regions_serializable,
                node_count=len(slice_nodes),
                stress_statistics=stress_stats,
                nodes=slice_nodes
            )
            
            results.append({
                "layer_data": layer_data,
                "regions": slice_result["regions"],
                "slice_nodes": slice_nodes,
                "stress_thresholds": stress_thresholds,
                "zone_count": sum(len(polys) for polys in slice_result["regions"].values()),
                "zone_limit_applied": zone_per_slice is not None
            })
        
        return results

    def get_fem_model_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the spatial bounds of the FEM model.
        
        Returns:
            dict: Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples
        """
        if self.nodes is None:
            self.get_node_data_with_stress()
            
        coords = self.nodes[:, 1:4]  # Extract X, Y, Z coordinates

        

        print("**" * 50)
        print(f"[INFO] FEM model bounds: X({np.min(coords[:, 0])}, {np.max(coords[:, 0])}), "
              f"Y({np.min(coords[:, 1])}, {np.max(coords[:, 1])}), "
              f"Z({np.min(coords[:, 2])}, {np.max(coords[:, 2])})")
        print("**" * 50)

        return {
            'x': (float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))),
            'y': (float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))),
            'z': (float(np.min(coords[:, 2])), float(np.max(coords[:, 2])))
        }
    
    def get_stress_statistics(self):
        """
        Get statistical information about the stress data.
        
        Returns:
            dict: Dictionary containing stress statistics
        """
        if self.nodes is None:
            self.get_node_data_with_stress()
            
        stress_values = self.nodes[:, 4]  # VonMises column
        
        return {
            "min_stress": np.min(stress_values),
            "max_stress": np.max(stress_values),
            "mean_stress": np.mean(stress_values),
            "std_stress": np.std(stress_values),
            "median_stress": np.median(stress_values),
            "stress_bins": self.stress_bins,
            "num_nodes": len(stress_values),
            "low_threshold": self.stress_thresshold[0],
            "moderate_threshold": self.stress_thresshold[1]
        }


    def export_fem_nodes_to_csv(self, output_path: str):
        """
        Export the FEM node data with von Mises stress to a CSV file.
        
        Parameters:
            output_path (str): Path to save the CSV file
        """
        if self.nodes is None:
            raise ValueError("Node data not available. Run get_node_data_with_stress() first.")
        
        df = pd.DataFrame(self.nodes, columns=["NodeID", "X", "Y", "Z", "VonMises(MPa)"])
        df.to_csv(output_path, index=False)
        print(f"Node data exported to {output_path}")





    


        