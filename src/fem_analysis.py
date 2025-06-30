from ansys.dpf import core as dpf
from ansys.dpf.core.operators.invariant import von_mises_eqv_fc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPolygon, Point, mapping
from shapely.ops import unary_union
from matplotlib.colors import Normalize
import json
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import alphashape
from collections import Counter
from alphashape import optimizealpha
from scipy.spatial import Delaunay, ConvexHull
from shapely import wkt
from shapely.validation import explain_validity

# Calculate average nearest neighbor distance
from scipy.spatial import distance_matrix


def estimate_alpha( points, k=5):
    
        if len(points) < 4:
            return 0.0  # fall back to convex hull
        
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                p1 = tuple(points[simplex[i]])
                p2 = tuple(points[simplex[(i + 1) % 3]])
                edges.add(frozenset((p1, p2)))

        lengths = [np.linalg.norm(np.array(list(e)[0]) - np.array(list(e)[1])) for e in edges]
        return np.mean(lengths) * 1.5  # scale as needed



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
        self.stress_threshold = None
        
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
        #self.stress_threshold = self.calculate_stress_thresholds()

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
    
    def calculate_stress_thresholds(self, streckgrenze: float = 620, sicherheitsfaktor: float = 1.5) -> tuple[float, float]:
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

        self.stress_threshold = (low_max, mod_max)
        return self.stress_threshold
    
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
        tolerance = 0.25 #thickness   #0.1
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
        
        stress_masks = self.generate_fallback_masks(xy_points, stress_values)

        print(f"[INFO] Stress masks created: "
            f"{np.sum(stress_masks['low'])} low, "
            f"{np.sum(stress_masks['moderate'])} moderate, "
            f"{np.sum(stress_masks['high'])} high")
        
        regions = {"low": [], "moderate": [], "high": []}

        # OPTIMIZATION 4: Process each stress level with vectorized operations
        for label, mask in stress_masks.items():
            if not np.any(mask):
                continue
                
            points = xy_points[mask]
            if len(points) == 0:
                continue

            # 3. DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = clustering.labels_

            label_counts = Counter(labels)
            num_clusters = len([lbl for lbl in label_counts if lbl != -1])
            num_noise = label_counts.get(-1, 0)

            print(f"[DBSCAN Debug] {label.upper()} zone: {num_clusters} clusters, {num_noise} noise points")
            for cid, count in label_counts.items():
                print(f"  - {'Noise' if cid == -1 else 'Cluster'} {cid}: {count} points")


            
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

                    except Exception as e:
                        print(f"[Warning] Failed alphashape for {label}-cluster: {e}")
            
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
    
    def generate_fallback_masks(self, xy_points: np.ndarray, stress_values: np.ndarray) -> dict:
        """
        Generate adjusted stress masks with fallback logic.
        Uses DBSCAN to find real clusters. If all clusters are too small to create valid polygons,
        the points are reassigned to the next-lower stress class.

        Returns:
            dict: Final masks → { "low": mask, "moderate": mask, "high": mask }
        """

        print(f"[INFO] XY-Ponits: {len(xy_points)}")
        low_max, mod_max = self.stress_threshold

        masks = {
            "low": stress_values < low_max,
            "moderate": (stress_values >= low_max) & (stress_values < mod_max),
            "high": stress_values >= mod_max
        }

        fallback_order = ["high", "moderate"]
        fallback_target = {"high": "moderate", "moderate": "low"}

        adjusted_masks = {k: v.copy() for k, v in masks.items()}

        for label in fallback_order:
            current_mask = adjusted_masks[label]
            indices = np.where(current_mask)[0]
            if len(indices) < 3:
                # Not enough for clustering
                adjusted_masks[fallback_target[label]][indices] = True
                adjusted_masks[label][indices] = False
                print(f"[↘️] {label.upper()}: too few points → fallback to {fallback_target[label]}")
                continue

            points = xy_points[indices]

            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = clustering.labels_
            unique_clusters = np.unique(labels)
            unique_clusters = unique_clusters[unique_clusters != -1]

            valid_cluster_found = False
            for cid in unique_clusters:
                cluster_points = points[labels == cid]
                if len(cluster_points) >= 3:
                    try:
                        poly = Polygon(cluster_points).convex_hull
                        if poly.area >= self.min_cluster_area:
                            valid_cluster_found = True
                            
                    except Exception as e:
                        continue

            if not valid_cluster_found:
                # All clusters too small → fallback
                adjusted_masks[fallback_target[label]][indices] = True
                adjusted_masks[label][indices] = False
                print(f"[↘️] {label.upper()}: no valid clusters → fallback to {fallback_target[label]}")

        return adjusted_masks

    
    # New method to generate stress regions for a specific Z-slice

    def generate_slice_stress_regions(self, z: float, thickness: float = 0.25, 
                                 zone_per_slice: int = 3, ensure_no_overlap: bool = False, outer_shape: Polygon = None) -> dict | None:
        """
        Generiert stress-basierte Regionen für eine spezifische Z-Schicht.
        
        Args:
            z (float): Ziel-Z-Höhe der Schicht
            thickness (float): Schichtdicke (Standard = 0.25 mm)
            zone_per_slice (int): Max. Anzahl Zonen pro Schicht (Standard = 3)
            ensure_no_overlap (bool): Stelle sicher, dass keine Überlappungen existieren
            
        Returns:
            dict | None: Dictionary mit Stress-Regionen und Schicht-Knoten
        """

        # Schritt 1: Hole Knoten in der Schicht
        slice_nodes = self.get_nodes_in_slice(z, thickness)
        
        # Schritt 2: Prüfe ob genug Knoten vorhanden sind
        if slice_nodes.shape[0] < self.min_nodes:
            print(f"[INFO] Nicht genug Knoten bei z={z:.3f}. Gefunden: {slice_nodes.shape[0]}")
            return None
        
        # Schritt 3: Klassifiziere und clustere Stress-Regionen
        regions = self.classify_and_cluster_stress_regions(
            xy_points=slice_nodes[:, :2],
            stress_values=slice_nodes[:, 3]
        )

        # Schritt 4: Bereinige Überlappungen (ERWEITERT)
        if ensure_no_overlap:
            cleaned_regions = self._clean_overlap_regions_v5(regions)
            #cleaned_regions = self.trim_and_validate_regions(cleaned_regions, outer_shape)
            print(f"[Cleaned regions with no overlapping is called]")
            
        else:
            cleaned_regions = regions

        # Step 4b: Fill uncovered areas if slice is not fully covered
        if outer_shape and ensure_no_overlap:
            cleaned_regions, _ = self.fill_uncovered_gaps(
                cleaned_regions, outer_shape
            )
            

        
        # Schritt 5: Überprüfung auf Überschneidungen (nur Debug-Zweck)
        def check_zone_overlaps(zone_a: list, zone_b: list, name_a='Zone A', name_b='Zone B') -> list:
            
            overlaps = []
            for i, poly_a in enumerate(zone_a):
                for j, poly_b in enumerate(zone_b):
                    if poly_a.intersects(poly_b):
                        intersection = poly_a.intersection(poly_b)
                        if not intersection.is_empty and intersection.area > 1e-6:
                            print(f"[⚠️] {name_a}[{i}] überschneidet sich mit {name_b}[{j}] → Fläche: {intersection.area:.4f} mm²")
                            overlaps.append((i, j, intersection.area))
            if not overlaps:
                print(f"[✅] Keine Überschneidungen zwischen {name_a} und {name_b} gefunden.")
            return overlaps

        check_zone_overlaps(cleaned_regions.get("low", []), cleaned_regions.get("moderate", []), "LOW", "MODERATE")
        check_zone_overlaps(cleaned_regions.get("moderate", []), cleaned_regions.get("high", []), "MODERATE", "HIGH")
        check_zone_overlaps(cleaned_regions.get("low", []), cleaned_regions.get("high", []), "LOW", "HIGH")
            
        # Schritt 6: Post-Processing für Hatching-Kompatibilität
        #cleaned_regions = self._prepare_for_hatching(cleaned_regions, outer_shape)
        
        # Debug-Ausgabe für z=0.0
        self._print_zone_statistics(z, thickness, regions, cleaned_regions)
        
        return {
            "regions": cleaned_regions,
            "slice_nodes": slice_nodes,
            "coverage": self._calculate_coverage(cleaned_regions)
        }

    

    # Fill Uncovered Gaps Method
    # ---------------------------------------------
    def fill_uncovered_gaps(self, regions, outer_shape, buffer_dist=0):
        """
        Füllt nicht abgedeckte Bereiche in den Stresszonen, indem sie der nächstgelegenen Zone zugewiesen werden.
        Ziel: Die gesamte outer_shape-Fläche soll mit Stresszonen abgedeckt sein.

        Args:
            regions (dict): dict mit "low", "moderate", "high", jeweils Liste von Polygons.
            outer_shape (Polygon): Die vollständige Kontur (Slice) der Schicht.
            buffer_dist (float): Abstandsschwelle für Zuordnung.

        Returns:
            (dict, list): Aktualisierte Zonen (dict) und Zuordnungsprotokoll (list of (zone, area)).
        """
        print("[DEBUG] Starting fill_uncovered_gaps")

        if not outer_shape or outer_shape.is_empty or not outer_shape.is_valid:
            print("[❌] Outer shape is invalid")
            return regions, []

        print(f"[DEBUG] Outer shape area: {outer_shape.area:.4f} mm²")

        assignment_log = []
        cleaned = {"low": [], "moderate": [], "high": []}

        # Vereinheitlichte Zonen erzeugen
        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            print(f"  [DEBUG FILL GAPS] {category.upper()} zone: {len(polys)} polygons")

            valid_polys = []
            for p in polys:
                if p and not p.is_empty:
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_valid:
                        valid_polys.append(p)

            unified_zones[category] = unary_union(valid_polys) if valid_polys else None
            cleaned[category] = valid_polys

        # Gesamtabdeckung berechnen
        all_union_parts = [g for g in unified_zones.values() if g and not g.is_empty]
        if not all_union_parts:
            cleaned["moderate"].append(outer_shape)
            assignment_log.append(("moderate", outer_shape.area))
            return cleaned, assignment_log
        

        print("[DEBUG] ➕ Initial polygon counts BEFORE gap fill:")
        for k, v in cleaned.items():
            print(f"  {k.upper()} = {len(v)} polygons")


        # Gaps finden
        merged_union = unary_union(all_union_parts)
        missing_area = outer_shape.difference(merged_union)
        if missing_area.is_empty:
            print("[✅] No uncovered gaps to fill.")
            return cleaned, assignment_log

        print(f"[🧩] Uncovered area: {missing_area.area:.4f} mm²")
        gap_polys = self._split_into_polygons(missing_area)
        print(f"[INFO] Total uncovered gaps: {len(gap_polys)}")


        # 🔍 Entferne Gaps mit zu kleiner Fläche (numerisches Rauschen)
        MIN_GAP_AREA = 1e-4  # = 0.0001 mm²
        gap_polys = [g for g in gap_polys if g.area > MIN_GAP_AREA]


        for i, gap in enumerate(gap_polys):
            # Calculate distances to all zones for THIS gap
            
            distances = {}
            for level in ["low", "moderate", "high"]:
            
                if unified_zones[level] and not unified_zones[level].is_empty:
                    distances[level] = gap.distance(unified_zones[level])

            # Find nearest zone (only if we have valid zones)
            

            if distances:
                nearest_zone = min(distances, key=distances.get)

                
                
                # Assign gap to nearest zone
                #cleaned[nearest_zone].append(gap)  # Simple assignment - no clipping needed

                combined = unary_union(cleaned[nearest_zone] + [gap])
                cleaned[nearest_zone] = self._split_into_polygons(combined)
                print(f"    [DEBUG] ⬆️ {nearest_zone.upper()} now has {len(cleaned[nearest_zone])} polygons after gap {i}")


                assignment_log.append((nearest_zone, gap.area))
                print(f"[🩹] Gap {i} (area={gap.area:.4f}) assigned to {nearest_zone.upper()} (distance={distances[nearest_zone]:.3f})")
            else:
                # Fallback if no zones exist (shouldn't happen)
                cleaned["moderate"].append(gap)
                assignment_log.append(("moderate", gap.area))
                print(f"[🩹] Gap {i} (area={gap.area:.4f}) fallback-assigned to MODERATE")

        # Keep the merge step
        cleaned = self._merge_adjacent_zones(cleaned, buffer_distance=buffer_dist)

        print("[DEBUG] 🔄 After merge_adjacent_zones:")
        for k, v in cleaned.items():
            print(f"  {k.upper()} = {len(v)} polygons")

        # Nach dem finalen cleaned[level] = trimmed
        #for level in cleaned:
        #    cleaned[level] = [p for p in cleaned[level] if p.area > 1e-4]


        # Validierung: Nur innerhalb der outer_shape
        for level in cleaned:
            trimmed = []
            for poly in cleaned[level]:
                inter = poly.intersection(outer_shape)
                if inter and not inter.is_empty:
                    trimmed.extend(self._split_into_polygons(inter))
            cleaned[level] = trimmed

        total_area = sum(p.area for zone in cleaned.values() for p in zone)
        print(f"[CHECK] Total assigned area: {total_area:.4f} / {outer_shape.area:.4f} mm² ({100 * total_area / outer_shape.area:.1f}%)")

        return cleaned, assignment_log

    def _split_into_polygons(self, geometry):
        """
        Converts a geometry into a list of polygons.
        """
        if geometry is None or geometry.is_empty:
            return []
            
        if isinstance(geometry, Polygon):
            return [geometry] if geometry.is_valid and not geometry.is_empty else []
        elif isinstance(geometry, MultiPolygon):
            return [geom for geom in geometry.geoms if geom.is_valid and not geom.is_empty]
        else:
            return []

    def _merge_adjacent_zones(self, regions: dict, buffer_distance: float = 0) -> dict:
        """
        Merges nearby or touching polygons within each stress level to form cleaner zones.
        Used after adding small patches or filling gaps.
        """
      

        merged = {"low": [], "moderate": [], "high": []}

        for stress_level, polygons in regions.items():
            if not polygons:
                continue

            valid = [p for p in polygons if p and not p.is_empty and p.is_valid]
            if not valid:
                continue

            buffered = [p.buffer(buffer_distance) for p in valid]
            merged_shape = unary_union(buffered).buffer(-buffer_distance)

            merged[stress_level] = self._split_into_polygons(merged_shape)

        return merged
    

    

    # Hatching Preparation Methods  

    # new method to prepare zones for hatching
    def _prepare_for_hatching(self, regions: dict, outer_shape) -> dict:
        """
        Prepares zones for optimal hatching by detecting and trimming thin protrusions.
        
        Args:
            regions (dict): Cleaned regions
            outer_shape: Boundary shape for validation
            
        Returns:
            dict: Hatching-optimized regions
        """
        print("[DEBUG] =================== HATCHING PREPARATION ===================")
        
        hatching_ready = {"low": [], "moderate": [], "high": []}
        problem_areas = []  # Track areas that need neck trimming
        
        # Step 1: Analyze each zone and identify problems
        for stress_level, polygons in regions.items():
            print(f"\n[DEBUG] Analyzing {stress_level.upper()} zone: {len(polygons)} polygons")
            
            for i, poly in enumerate(polygons):
                if not poly or poly.is_empty:
                    continue
                    
                print(f"  [DEBUG] Polygon {i}: Area = {poly.area:.3f} mm²")
                
                # Check if polygon is hatchable
                hatchable, problem_type, neck_info = self._analyze_polygon_for_hatching(poly, poly_id=f"{stress_level}_{i}")
                
                if hatchable:
                    # Keep good polygons as-is
                    hatching_ready[stress_level].append(poly)
                    print(f"    ✅ Polygon {i} is hatchable")
                else:
                    # Mark problematic areas for neck trimming
                    problem_areas.append({
                        'polygon': poly,
                        'original_zone': stress_level,
                        'problem_type': problem_type,
                        'id': f"{stress_level}_{i}",
                        'neck_info': neck_info
                    })
                    print(f"    ❌ Polygon {i} has problem: {problem_type}")
        
        # Step 2: Process problem areas with neck trimming
        if problem_areas:
            print(f"\n[DEBUG] Processing {len(problem_areas)} problematic polygons...")
            hatching_ready = self._trim_necks_and_redistribute(hatching_ready, problem_areas)
        
        # Step 3: Validate final result
        self._validate_hatching_zones(hatching_ready, outer_shape)
        
        return hatching_ready


    def _detect_necks_and_protrusions(self, polygon, hatch_spacing, min_width):
        """
        Enhanced detection that also identifies WHERE the neck is located.
        
        Args:
            polygon: Shapely Polygon to analyze
            hatch_spacing: Hatch spacing in mm
            min_width: Minimum acceptable width
        
        Returns:
            tuple: (neck_detected: bool, reason: str, neck_info: dict)
        """
        # Method 1: Progressive erosion to detect thin connections
        erosion_scales = [0.8, 1.2, 1.6, 2.0]  # multiples of hatch_spacing  
        
        for scale in erosion_scales:
            erosion_distance = hatch_spacing * scale
            eroded = polygon.buffer(-erosion_distance)
            
            if eroded.is_empty:
                continue
                
            # Check if erosion breaks the polygon into multiple pieces
            if hasattr(eroded, 'geoms'):
                num_pieces = len(list(eroded.geoms))
                if num_pieces > 1:
                    # Found a neck - store information about where to cut
                    neck_info = {
                        'erosion_scale': scale,
                        'erosion_distance': erosion_distance,
                        'pieces': list(eroded.geoms),
                        'original_polygon': polygon
                    }
                    return True, f"Neck detected: erosion at {scale:.1f}x hatch spacing breaks into {num_pieces} pieces", neck_info
            
            # Check progressive area loss
            area_loss = (polygon.area - eroded.area) / polygon.area
            expected_loss = 0.15 + 0.08 * scale  # Progressive threshold
            
            if area_loss > expected_loss:
                # Found thin protrusion - store erosion info for trimming
                neck_info = {
                    'erosion_scale': scale,
                    'erosion_distance': erosion_distance,
                    'area_loss': area_loss,
                    'eroded_shape': eroded,
                    'original_polygon': polygon
                }
                return True, f"Thin protrusion detected: {area_loss*100:.1f}% area loss at {scale:.1f}x erosion", neck_info
        
        return False, "No necks or thin protrusions detected", None


    def _analyze_polygon_for_hatching(self, polygon, poly_id="", hatch_spacing=0.08, safety_factor=4.0):
        """
        Enhanced polygon analysis that returns neck information for trimming.
        
        Args:
            polygon: Shapely Polygon to analyze
            poly_id: Identifier for debugging
            hatch_spacing: Hatch spacing in mm
            safety_factor: Safety multiplier for minimum width
        
        Returns:
            tuple: (is_hatchable: bool, problem_type: str, neck_info: dict)
        """
        # Calculate PBF-LB specific requirements
        min_width = hatch_spacing * safety_factor
        min_length = 0.75
        min_area = min_width * min_length * 0.5
        
        print(f"    [ANALYZE] Checking polygon {poly_id}...")
        print(f"      PBF-LB Requirements: hatch={hatch_spacing}mm, min_width={min_width:.4f}mm, min_length={min_length}mm")
        
        # Check 1: Minimum area for meaningful hatching
        if polygon.area < min_area:
            print(f"      - Area too small for hatching: {polygon.area:.3f} < {min_area:.3f} mm²")
            return False, "too_small", None
        
        # Check 2: Basic width analysis using erosion
        eroded = polygon.buffer(-min_width/2)
        if eroded.is_empty:
            print(f"      - Too narrow for {safety_factor}x hatch spacing: disappears with {min_width:.4f}mm erosion")
            return False, "too_narrow", None
        
        # Check 3: Minimum length for efficient laser operation
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        
        if max_dimension < min_length:
            print(f"      - Too short for efficient laser operation: {max_dimension:.3f} < {min_length} mm")
            return False, "too_short", None
        
        # Check 4: Aspect ratio
        aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else float('inf')
        
        if aspect_ratio > 20:
            print(f"      - Extreme aspect ratio: {aspect_ratio:.1f} (problematic for laser scanning)")
            return False, "elongated", None
        
        # Check 5: ENHANCED NECK DETECTION with location info
        neck_detected, neck_reason, neck_info = self._detect_necks_and_protrusions(polygon, hatch_spacing, min_width)
        if neck_detected:
            print(f"      - {neck_reason}")
            return False, "thin_protrusions", neck_info
        
        print(f"      ✅ Polygon passes all PBF-LB hatching checks")
        print(f"         Area: {polygon.area:.3f} mm², Dimensions: {width:.4f}×{height:.4f} mm, Aspect: {aspect_ratio:.1f}")
        return True, "good", None


    def _trim_necks_and_redistribute(self, hatching_ready, problem_areas):
        """
        Trims necks from problematic polygons and redistributes only the neck parts.
        
        Args:
            hatching_ready: Current good zones
            problem_areas: List of problematic polygons with neck information
        
        Returns:
            dict: Updated zones with trimmed polygons
        """
        print(f"\n[TRIM_NECKS] Processing {len(problem_areas)} problematic polygons...")
        
        for problem in problem_areas:
            poly = problem['polygon']
            original_zone = problem['original_zone']
            problem_type = problem['problem_type']
            poly_id = problem['id']
            neck_info = problem.get('neck_info')
            
            print(f"  [TRIM_NECKS] Handling {poly_id} ({problem_type})...")
            
            if problem_type == "thin_protrusions" and neck_info:
                # Try to trim the neck and keep the main body
                main_body, neck_part = self._separate_neck_from_body(poly, neck_info)
                
                if main_body and not main_body.is_empty:
                    # Keep main body in original zone
                    hatching_ready[original_zone].append(main_body)
                    print(f"    → Main body KEPT in {original_zone.upper()} zone")
                    print(f"      Main body area: {main_body.area:.3f} mm² ({main_body.area/poly.area*100:.1f}% of original)")
                    
                    if neck_part and not neck_part.is_empty:
                        # Assign neck to nearest moderate zone
                        nearest_moderate = self._find_nearest_moderate_zone(neck_part, hatching_ready)
                        hatching_ready["moderate"] = self._merge_with_nearest_moderate(neck_part, hatching_ready["moderate"], nearest_moderate)
                        print(f"    → Neck part assigned to MODERATE zone")
                        print(f"      Neck area: {neck_part.area:.3f} mm² ({neck_part.area/poly.area*100:.1f}% of original)")
                else:
                    # Fallback: assign entire polygon to moderate
                    nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                    hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                    print(f"    → Could not separate neck, entire polygon assigned to MODERATE")
            
            else:
                # For other problem types, assign entire polygon to moderate
                nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                print(f"    → {problem_type} assigned to MODERATE zone")
        
        return hatching_ready


    def _separate_neck_from_body(self, polygon, neck_info):
        """
        Separates the neck from the main body using erosion information.
        
        Args:
            polygon: Original polygon with neck
            neck_info: Information about where the neck is located
        
        Returns:
            tuple: (main_body_polygon, neck_polygon)
        """
        try:
            if 'pieces' in neck_info:
                # Case 1: Erosion broke polygon into pieces - identify main body
                pieces = neck_info['pieces']
                erosion_distance = neck_info['erosion_distance']
                
                # Find the largest piece as main body
                main_piece = max(pieces, key=lambda p: p.area)
                
                # Reconstruct main body by dilating back
                main_body = main_piece.buffer(erosion_distance * 0.9)
                main_body = main_body.intersection(polygon)  # Clip to original bounds
                
                # Neck is the remainder
                neck_part = polygon.difference(main_body)
                
                return main_body, neck_part
                
            elif 'eroded_shape' in neck_info:
                # Case 2: Use eroded shape as main body
                eroded = neck_info['eroded_shape']
                erosion_distance = neck_info['erosion_distance']
                
                # Reconstruct main body
                main_body = eroded.buffer(erosion_distance * 0.8)
                main_body = main_body.intersection(polygon)
                
                # Neck is the remainder
                neck_part = polygon.difference(main_body)
                
                return main_body, neck_part
                
        except Exception as e:
            print(f"    [WARNING] Neck separation failed: {e}")
        
        return None, None


    def _find_nearest_moderate_zone(self, polygon, hatching_ready):
        """Find the nearest MODERATE zone polygon to merge with."""
        if "moderate" not in hatching_ready or not hatching_ready["moderate"]:
            return 0
        
        centroid = polygon.centroid
        min_distance = float('inf')
        nearest_index = 0
        
        for i, moderate_poly in enumerate(hatching_ready["moderate"]):
            distance = centroid.distance(moderate_poly.centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index


    def _merge_with_nearest_moderate(self, problem_polygon, moderate_list, nearest_index):
        """Merge the problem polygon with the nearest MODERATE zone."""
        from shapely.ops import unary_union
        
        if not moderate_list:
            return [problem_polygon]
        
        if nearest_index >= len(moderate_list):
            nearest_index = 0
        
        try:
            nearest_moderate = moderate_list[nearest_index]
            merged = unary_union([nearest_moderate, problem_polygon])
            
            updated_list = moderate_list.copy()
            updated_list[nearest_index] = merged
            
            return updated_list
            
        except Exception as e:
            print(f"    [WARNING] Merge failed: {e}, appending separately")
            return moderate_list + [problem_polygon]
    # ---------------------------------------------
    def _prepare_for_hatching_(self, regions: dict, outer_shape) -> dict:
        """
        Prepares zones for optimal hatching by detecting and fixing thin protrusions.
        
        Args:
            regions (dict): Cleaned regions
            outer_shape: Boundary shape for validation
            
        Returns:
            dict: Hatching-optimized regions
        """
        print("[DEBUG] =================== HATCHING PREPARATION ===================")
        
        hatching_ready = {"low": [], "moderate": [], "high": []}
        problem_areas = []  # Track areas that need redistribution
        
        # Step 1: Analyze each zone and identify problems
        for stress_level, polygons in regions.items():
            print(f"\n[DEBUG] Analyzing {stress_level.upper()} zone: {len(polygons)} polygons")
            
            for i, poly in enumerate(polygons):
                if not poly or poly.is_empty:
                    continue
                    
                print(f"  [DEBUG] Polygon {i}: Area = {poly.area:.3f} mm²")
                
                # Check if polygon is hatchable
                hatchable, problem_type = self._analyze_polygon_for_hatching(poly, poly_id=f"{stress_level}_{i}")
                
                if hatchable:
                    # Keep good polygons as-is
                    hatching_ready[stress_level].append(poly)
                    print(f"    ✅ Polygon {i} is hatchable")
                else:
                    # Mark problematic areas for redistribution
                    problem_areas.append({
                        'polygon': poly,
                        'original_zone': stress_level,
                        'problem_type': problem_type,
                        'id': f"{stress_level}_{i}"
                    })
                    print(f"    ❌ Polygon {i} has problem: {problem_type}")
        
        # Step 2: Process problem areas
        if problem_areas:
            print(f"\n[DEBUG] Processing {len(problem_areas)} problematic polygons...")
            hatching_ready = self._redistribute_problem_areas(hatching_ready, problem_areas)
        
        # Step 3: Validate final result

        self._validate_hatching_zones(hatching_ready, outer_shape)
        
        return hatching_ready

    def _analyze_polygon_for_hatching_(self, polygon, poly_id="", hatch_spacing=0.08, safety_factor=4.0):
        """
        Enhanced polygon analysis with improved neck detection for PBF-LB requirements.
        
        Args:
            polygon: Shapely Polygon to analyze
            poly_id: Identifier for debugging
            hatch_spacing: Hatch spacing in mm (typically 0.06-0.10 mm for PBF-LB)
            safety_factor: Safety multiplier for minimum width (3-4x hatch spacing)
        
        Returns:
            tuple: (is_hatchable: bool, problem_type: str)
        """
        
        
        # Calculate PBF-LB specific requirements
        min_width = hatch_spacing * safety_factor  # 0.32 mm for 0.08mm spacing, 4x safety
        min_length = 0.75  # mm - avoid inefficient laser start/stops
        min_area = min_width * min_length * 0.5  # ~0.12 mm² minimum meaningful area
        
        print(f"    [ANALYZE] Checking polygon {poly_id}...")
        print(f"      PBF-LB Requirements: hatch={hatch_spacing}mm, min_width={min_width:.4f}mm, min_length={min_length}mm")
        
        # Check 1: Minimum area for meaningful hatching
        if polygon.area < min_area:
            print(f"      - Area too small for hatching: {polygon.area:.3f} < {min_area:.3f} mm²")
            return False, "too_small"
        
        # Check 2: Basic width analysis using erosion
        eroded = polygon.buffer(-min_width/2)
        if eroded.is_empty:
            print(f"      - Too narrow for {safety_factor}x hatch spacing: disappears with {min_width:.4f}mm erosion")
            return False, "too_narrow"
        
        # Check 3: Minimum length for efficient laser operation
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]  # max_x - min_x
        height = bounds[3] - bounds[1]  # max_y - min_y
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        
        if max_dimension < min_length:
            print(f"      - Too short for efficient laser operation: {max_dimension:.3f} < {min_length} mm")
            return False, "too_short"
        
        # Check 4: Aspect ratio (very elongated shapes cause laser problems)
        aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else float('inf')
        
        if aspect_ratio > 20:  # Very elongated - problematic for hatching
            print(f"      - Extreme aspect ratio: {aspect_ratio:.1f} (problematic for laser scanning)")
            return False, "elongated"
        
        # Check 5: ENHANCED NECK DETECTION - Multi-scale erosion analysis
        neck_detected, neck_reason = self._detect_necks_and_protrusions(polygon, hatch_spacing, min_width)
        if neck_detected:
            print(f"      - {neck_reason}")
            return False, "thin_protrusions"
        
        # Check 6: Hatch efficiency (polygon should be "solid" enough for efficient scanning)
        #convex_hull = polygon.convex_hull
        #convexity = polygon.area / convex_hull.area if convex_hull.area > 0 else 0
        #if convexity < 0.4:  # Very non-convex - inefficient hatching
        #    print(f"      - Low convexity: {convexity:.4f} (inefficient for laser scanning)")
        #    return False, "complex_shape"
        
        # Check 7: Minimum effective hatch area
        #hatchable_core = polygon.buffer(-hatch_spacing/2)
        #if not hatchable_core.is_empty:
        #    hatch_efficiency = hatchable_core.area / polygon.area
        #    if hatch_efficiency < 0.3:  # Less than 30% can be effectively hatched
        #        print(f"      - Low hatch efficiency: {hatch_efficiency*100:.1f}% effective hatch coverage")
        #        return False, "poor_hatch_efficiency"
        
        print(f"      ✅ Polygon passes all PBF-LB hatching checks")
        print(f"         Area: {polygon.area:.3f} mm², Dimensions: {width:.4f}×{height:.4f} mm, Aspect: {aspect_ratio:.1f}")
        return True, "good"

    def _detect_necks_and_protrusions_(self, polygon, hatch_spacing, min_width):
        """
        Simple detection for problematic necks and thin protrusions.
        Uses the working area loss method but with relaxed thresholds.
        
        Args:
            polygon: Shapely Polygon to analyze
            hatch_spacing: Hatch spacing in mm
            min_width: Minimum acceptable width
        
        Returns:
            tuple: (neck_detected: bool, reason: str)
        """
        
        
        # Method 1: Progressive erosion to detect thin connections
        erosion_scales = [0.8, 1.2, 1.6, 2.0]  # multiples of hatch_spacing  
        
        for scale in erosion_scales:
            erosion_distance = hatch_spacing * scale
            eroded = polygon.buffer(-erosion_distance)
            
            if eroded.is_empty:
                continue
                
            # Check if erosion breaks the polygon into multiple pieces
            if hasattr(eroded, 'geoms'):
                num_pieces = len(list(eroded.geoms))
                if num_pieces > 1:
                    return True, f"Neck detected: erosion at {scale:.1f}x hatch spacing breaks into {num_pieces} pieces"
            
            # Check progressive area loss
            area_loss = (polygon.area - eroded.area) / polygon.area
            expected_loss = 0.15 + 0.08 * scale  # Progressive threshold
            #expected_loss = 0.35 + 0.05 * scale 
            
            if area_loss > expected_loss:
                return True, f"Thin protrusion detected: {area_loss*100:.1f}% area loss at {scale:.1f}x erosion"
        
        return False, "No necks or thin protrusions detected"

    def _detect_necks_and_protrusions_old(self, polygon, hatch_spacing, min_width):
        """
        Enhanced neck and thin protrusion detection using multiple methods.
        
        Args:
            polygon: Shapely Polygon to analyze
            hatch_spacing: Hatch spacing in mm
            min_width: Minimum acceptable width
        
        Returns:
            tuple: (neck_detected: bool, reason: str)
        """
        
        
        # Method 1: Progressive erosion to detect thin connections
        erosion_scales = [0.8, 1.2, 1.6, 2.0]  # multiples of hatch_spacing  
        
        for scale in erosion_scales:
            erosion_distance = hatch_spacing * scale
            eroded = polygon.buffer(-erosion_distance)
            
            if eroded.is_empty:
                continue
                
            # Check if erosion breaks the polygon into multiple pieces
            if hasattr(eroded, 'geoms'):
                num_pieces = len(list(eroded.geoms))
                if num_pieces > 1:
                    return True, f"Neck detected: erosion at {scale:.1f}x hatch spacing breaks into {num_pieces} pieces"
            
            # Check progressive area loss
            area_loss = (polygon.area - eroded.area) / polygon.area
            #expected_loss = 0.15 + 0.08 * scale  # Progressive threshold
            expected_loss = 0.35 + 0.05 * scale 
            
            if area_loss > expected_loss:
                return True, f"Thin protrusion detected: {area_loss*100:.1f}% area loss at {scale:.1f}x erosion"
        
        # Method 2: Interior point sampling to find narrow regions
        bounds = polygon.bounds
        width_samples = max(15, int((bounds[2] - bounds[0]) / hatch_spacing))
        height_samples = max(15, int((bounds[3] - bounds[1]) / hatch_spacing))
        
        narrow_points = 0
        total_interior_points = 0
        
        for i in range(width_samples):
            for j in range(height_samples):
                # Create sample point
                x = bounds[0] + (bounds[2] - bounds[0]) * i / (width_samples - 1)
                y = bounds[1] + (bounds[3] - bounds[1]) * j / (height_samples - 1)
                point = Point(x, y)
                
                if polygon.contains(point):
                    total_interior_points += 1
                    
                    # Find distance to polygon boundary
                    distance_to_boundary = point.distance(polygon.boundary)
                    
                    # If distance to boundary is less than half minimum width, it's a narrow region
                    if distance_to_boundary < min_width / 2.2:  # Slightly more tolerant than min_width/2
                        narrow_points += 1
        
        if total_interior_points > 0:
            narrow_ratio = narrow_points / total_interior_points
            if narrow_ratio > 0.25:  # More than 25% of interior points are in narrow regions
                return True, f"Neck detected: {narrow_ratio*100:.1f}% of interior points in regions narrower than {min_width/2.2:.3f}mm"
        
        # Method 3: Skeleton-based width analysis using medial axis approximation
        # Create a simplified medial axis using erosion/dilation
        test_distances = np.linspace(hatch_spacing * 0.5, min_width * 0.8, 8)
        
        for test_dist in test_distances:
            eroded = polygon.buffer(-test_dist)
            if eroded.is_empty:
                continue
                
            # Dilate back to approximate medial axis
            medial_approx = eroded.buffer(test_dist * 0.9)
            
            if not medial_approx.is_empty:
                # Check if medial axis approximation captures main shape
                overlap_ratio = medial_approx.intersection(polygon).area / polygon.area
                
                if overlap_ratio < 0.7:  # Lost significant area in medial axis approximation
                    return True, f"Neck detected: medial axis analysis shows narrow connection at {test_dist:.3f}mm"
        
        # Method 4: Convex hull analysis for protrusions
        convex_hull = polygon.convex_hull
        hull_difference = convex_hull.difference(polygon)
        
        if not hull_difference.is_empty:
            # Check if the missing area from convex hull suggests thin protrusions
            concave_ratio = hull_difference.area / convex_hull.area
            
            if concave_ratio > 0.3:  # Significant concave regions
                # Test if these concave regions indicate thin protrusions
                buffered_original = polygon.buffer(hatch_spacing * 0.5)
                hull_coverage = buffered_original.intersection(convex_hull).area / convex_hull.area
                
                if hull_coverage < 0.85:  # Even with buffering, doesn't fill convex hull well
                    return True, f"Thin protrusion detected: complex concave shape with {concave_ratio*100:.1f}% hull difference"
        
        # Method 5: Boundary complexity analysis
        # Calculate boundary length vs area ratio
        boundary_length = polygon.boundary.length
        area_sqrt = np.sqrt(polygon.area)
        complexity_ratio = boundary_length / (4 * area_sqrt)  # Normalized by square perimeter
        
        if complexity_ratio > 2.5:  # Very complex boundary relative to area
            # This suggests thin protrusions or highly irregular shape
            return True, f"Complex boundary detected: boundary complexity ratio {complexity_ratio:.2f} suggests thin features"
        
        return False, "No necks or thin protrusions detected"

    def _analyze_polygon_for_hatching_old(self, polygon, poly_id="", hatch_spacing=0.08, safety_factor=4.0):
        """
        Analyzes a polygon to detect hatching problems based on PBF-LB requirements.
        
        Args:
            polygon: Shapely Polygon to analyze
            poly_id: Identifier for debugging
            hatch_spacing: Hatch spacing in mm (typically 0.06-0.10 mm for PBF-LB)
            safety_factor: Safety multiplier for minimum width (3-4x hatch spacing)
        
        Returns:
            tuple: (is_hatchable: bool, problem_type: str)
        """
        # Calculate PBF-LB specific requirements
        min_width = hatch_spacing * safety_factor  # 0.32 mm for 0.08mm spacing, 4x safety
        min_length = 0.75  # mm - avoid inefficient laser start/stops
        min_area = min_width * min_length * 0.5  # ~0.12 mm² minimum meaningful area
        
        print(f"    [ANALYZE] Checking polygon {poly_id}...")
        print(f"      PBF-LB Requirements: hatch={hatch_spacing}mm, min_width={min_width:.4f}mm, min_length={min_length}mm")
        
        # Check 1: Minimum area for meaningful hatching
        if polygon.area < min_area:
            print(f"      - Area too small for hatching: {polygon.area:.3f} < {min_area:.3f} mm²")
            return False, "too_small"
        
        # Check 2: Width analysis using erosion (critical for hatch lines)
        eroded = polygon.buffer(-min_width/2)
        if eroded.is_empty:
            print(f"      - Too narrow for {safety_factor}x hatch spacing: disappears with {min_width:.4f}mm erosion")
            return False, "too_narrow"
        
        # Check 3: Minimum length for efficient laser operation
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]  # max_x - min_x
        height = bounds[3] - bounds[1]  # max_y - min_y
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        
        if max_dimension < min_length:
            print(f"      - Too short for efficient laser operation: {max_dimension:.3f} < {min_length} mm")
            return False, "too_short"
        
        # Check 4: Aspect ratio (very elongated shapes cause laser problems)
        aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else float('inf')
        
        if aspect_ratio > 20:  # Very elongated - problematic for hatching
            print(f"      - Extreme aspect ratio: {aspect_ratio:.1f} (problematic for laser scanning)")
            return False, "elongated"
        
        # Check 5: Thin protrusions detection (critical for hatch quality)
        # Compare original area with buffered area using hatch-spacing based erosion
        hatch_erosion = polygon.buffer(-hatch_spacing)  # Erosion by one hatch spacing
        if not hatch_erosion.is_empty:
            area_loss = (polygon.area - hatch_erosion.area) / polygon.area
            if area_loss > 0.4:  # Lost more than 40% area with hatch-spacing erosion
                print(f"      - Thin protrusions detected: {area_loss*100:.1f}% area loss with hatch-spacing erosion")
                return False, "thin_protrusions"
        
        # Check 6: Hatch efficiency (polygon should be "solid" enough for efficient scanning)
        convex_hull = polygon.convex_hull
        convexity = polygon.area / convex_hull.area if convex_hull.area > 0 else 0
        if convexity < 0.4:  # Very non-convex - inefficient hatching
            print(f"      - Low convexity: {convexity:.4f} (inefficient for laser scanning)")
            return False, "complex_shape"
        
        # Check 7: Minimum effective hatch area
        # After accounting for edge effects, how much area can actually be hatched?
        hatchable_core = polygon.buffer(-hatch_spacing/2)  # Area that gets full hatch coverage
        if not hatchable_core.is_empty:
            hatch_efficiency = hatchable_core.area / polygon.area
            if hatch_efficiency < 0.3:  # Less than 30% can be effectively hatched
                print(f"      - Low hatch efficiency: {hatch_efficiency*100:.1f}% effective hatch coverage")
                return False, "poor_hatch_efficiency"
        
        print(f"      ✅ Polygon passes all PBF-LB hatching checks")
        print(f"         Area: {polygon.area:.3f} mm², Dimensions: {width:.4f}×{height:.4f} mm, Aspect: {aspect_ratio:.1f}")
        return True, "good"

    # Ditrubution and Redistribution Methods

    def _redistribute_problem_areas_new(self, hatching_ready, problem_areas):
        """
        Redistributes problematic areas to MODERATE zone.
        Keeps good areas in their original zones.
        
        Args:
            hatching_ready: Current good zones
            problem_areas: List of problematic polygons with metadata
        
        Returns:
            dict: Updated zones with redistributed areas
        """
        print(f"\n[REDISTRIBUTE] Processing {len(problem_areas)} problem areas...")
        
        for problem in problem_areas:
            poly = problem['polygon']
            original_zone = problem['original_zone']
            problem_type = problem['problem_type']
            poly_id = problem['id']
            
            print(f"  [REDISTRIBUTE] Handling {poly_id} ({problem_type})...")
            
            # Strategy: Assign ALL problematic areas to NEAREST MODERATE zone
            if problem_type in ["thin_protrusions", "complex_shape"]:
                # Try light simplification first
                simplified = self._simplify_for_hatching(poly)
                
                if simplified and not simplified.is_empty and simplified.area > poly.area * 0.8:
                    # Use simplified version but assign to nearest MODERATE
                    nearest_moderate = self._find_nearest_moderate_zone(simplified, hatching_ready)
                    hatching_ready["moderate"] = self._merge_with_nearest_moderate(simplified, hatching_ready["moderate"], nearest_moderate)
                    print(f"    → Simplified and merged with nearest MODERATE zone")
                    print(f"      Area: {poly.area:.3f} → {simplified.area:.3f} mm² ({simplified.area/poly.area*100:.1f}% preserved)")
                else:
                    # Use original shape but assign to nearest MODERATE
                    nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                    hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                    print(f"    → Original shape merged with nearest MODERATE zone")
                    print(f"      Note: {problem_type} will be handled with MODERATE zone parameters")
            
            # Strategy 2: Small/narrow areas go to nearest MODERATE
            elif problem_type in ["too_small", "too_narrow"]:
                nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                print(f"    → {problem_type} merged with nearest MODERATE zone (area {poly.area:.3f} mm²)")
            
            # Strategy 3: Elongated shapes go to nearest MODERATE
            elif problem_type == "elongated":
                nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                print(f"    → Elongated shape merged with nearest MODERATE zone")
            
            # Strategy 4: Poor efficiency goes to nearest MODERATE
            elif problem_type in ["poor_hatch_efficiency"]:
                nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                print(f"    → Poor efficiency merged with nearest MODERATE zone")
            
            # Fallback: any other problem type goes to nearest MODERATE
            else:
                nearest_moderate = self._find_nearest_moderate_zone(poly, hatching_ready)
                hatching_ready["moderate"] = self._merge_with_nearest_moderate(poly, hatching_ready["moderate"], nearest_moderate)
                print(f"    → {problem_type} merged with nearest MODERATE zone (fallback)")
        
        return hatching_ready

    def _find_nearest_moderate_zone_(self, polygon, hatching_ready):
        """
        Find the nearest MODERATE zone polygon to merge with.
        
        Args:
            polygon: The problematic polygon to assign
            hatching_ready: Current zones with polygons
        
        Returns:
            int: Index of nearest MODERATE polygon, or 0 if none found
        """
        if "moderate" not in hatching_ready or not hatching_ready["moderate"]:
            return 0  # No moderate zones available
        
        centroid = polygon.centroid
        min_distance = float('inf')
        nearest_index = 0
        
        for i, moderate_poly in enumerate(hatching_ready["moderate"]):
            distance = centroid.distance(moderate_poly.centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index

    def _merge_with_nearest_moderate_(self, problem_polygon, moderate_list, nearest_index):
        """
        Merge the problem polygon with the nearest MODERATE zone.
        
        Args:
            problem_polygon: The polygon to merge
            moderate_list: List of current MODERATE polygons
            nearest_index: Index of the nearest MODERATE polygon
        
        Returns:
            list: Updated MODERATE polygon list
        """
        from shapely.ops import unary_union
        
        if not moderate_list:
            return [problem_polygon]
        
        if nearest_index >= len(moderate_list):
            nearest_index = 0
        
        try:
            # Merge with the nearest MODERATE polygon
            nearest_moderate = moderate_list[nearest_index]
            merged = unary_union([nearest_moderate, problem_polygon])
            
            # Replace the nearest polygon with the merged result
            updated_list = moderate_list.copy()
            updated_list[nearest_index] = merged
            
            return updated_list
            
        except Exception as e:
            print(f"    [WARNING] Merge failed: {e}, appending separately")
            # Fallback: just append to the list
            return moderate_list + [problem_polygon]
        """
        Light simplification to smooth small irregularities while preserving main shape.
        
        Args:
            polygon: Input polygon
            tolerance_factor: Simplification aggressiveness
        
        Returns:
            Simplified polygon or None if failed
        """
        try:
            # Very conservative simplification - just smooth tiny irregularities
            hatch_spacing = 0.08  # mm
            tolerance = hatch_spacing * tolerance_factor  # 0.04mm tolerance
            
            # Light morphological smoothing
            smoothed = polygon.buffer(tolerance/2).buffer(-tolerance/2)
            
            if smoothed and not smoothed.is_empty and smoothed.area > polygon.area * 0.8:
                return smoothed
            else:
                # Fallback: minimal vertex reduction
                simplified = polygon.simplify(tolerance/2, preserve_topology=True)
                if simplified and not simplified.is_empty:
                    return simplified
                
        except Exception as e:
            print(f"    [WARNING] Simplification failed: {e}")
        
        return polygon  # Return original if simplification fails
    # ---------------------------------------------
    def _redistribute_problem_areas_(self, hatching_ready, problem_areas):
        """
        Redistributes problematic areas to nearby suitable zones.
        
        Args:
            hatching_ready: Current good zones
            problem_areas: List of problematic polygons with metadata
            outer_shape: Boundary for validation
        
        Returns:
            dict: Updated zones with redistributed areas
        """
        print(f"\n[REDISTRIBUTE] Processing {len(problem_areas)} problem areas...")
        
        for problem in problem_areas:
            poly = problem['polygon']
            original_zone = problem['original_zone']
            problem_type = problem['problem_type']
            poly_id = problem['id']
            
            print(f"  [REDISTRIBUTE] Handling {poly_id} ({problem_type})...")
            
            # Strategy depends on problem type
            if problem_type in ["too_small", "too_narrow"]:
                # Assign to nearest larger zone
                target_zone = self._find_nearest_suitable_zone(poly, hatching_ready)
                if target_zone:
                    hatching_ready[target_zone].append(poly)
                    print(f"    → Assigned to {target_zone.upper()} (nearest suitable zone)")
                else:
                    # Fallback: assign to moderate
                    hatching_ready["moderate"].append(poly)
                    print(f"    → Fallback assigned to MODERATE")
                    
            elif problem_type in ["thin_protrusions", "complex_shape"]:
                # Try to simplify/smooth the shape
                simplified = self._simplify_for_hatching(poly)
                if simplified and not simplified.is_empty:
                    # Re-check if simplified version is hatchable
                    is_good, _ = self._analyze_polygon_for_hatching(simplified, f"{poly_id}_simplified")
                    if is_good:
                        hatching_ready[original_zone].append(simplified)
                        print(f"    → Simplified and kept in {original_zone.upper()}")
                    else:
                        # Still problematic, assign to nearest zone
                        target_zone = self._find_nearest_suitable_zone(simplified, hatching_ready)
                        hatching_ready[target_zone or "moderate"].append(simplified)
                        print(f"    → Simplified and assigned to {(target_zone or 'moderate').upper()}")
                else:
                    # Simplification failed, assign as-is to nearest zone
                    target_zone = self._find_nearest_suitable_zone(poly, hatching_ready)
                    hatching_ready[target_zone or "moderate"].append(poly)
                    print(f"    → Simplification failed, assigned to {(target_zone or 'moderate').upper()}")
                    
            elif problem_type == "elongated":
                # For elongated shapes, try to split or assign to moderate zone
                hatching_ready["moderate"].append(poly)
                print(f"    → Elongated shape assigned to MODERATE")
        
        return hatching_ready


    def _find_nearest_suitable_zone_(self, problem_poly, hatching_ready):
        """
        Finds the nearest zone that has good hatchable polygons.
        
        Args:
            problem_poly: Polygon to relocate
            hatching_ready: Current good zones
        
        Returns:
            str: Zone name or None
        """
        min_distance = float('inf')
        best_zone = None
        
        for zone_name, zone_polys in hatching_ready.items():
            if not zone_polys:  # Skip empty zones
                continue
                
            # Calculate distance to this zone
            zone_union = unary_union(zone_polys)
            distance = problem_poly.distance(zone_union)
            
            if distance < min_distance:
                min_distance = distance
                best_zone = zone_name
        
        print(f"      Nearest suitable zone: {best_zone} (distance: {min_distance:.3f})")
        return best_zone


    def _simplify_for_hatching_(self, polygon, hatch_spacing=0.08):
        """
        Applies PBF-LB specific simplification techniques to make polygon more hatchable.
        
        Args:
            polygon: Input polygon
            hatch_spacing: Hatch spacing for erosion/dilation operations
        
        Returns:
            Polygon: Simplified polygon
        """
        try:
            # Method 1: Hatch-spacing based erosion-dilation to remove sub-hatch features
            erosion_dist = hatch_spacing * 1.5  # Remove features smaller than 1.5x hatch spacing
            eroded = polygon.buffer(-erosion_dist)
            if not eroded.is_empty:
                restored = eroded.buffer(erosion_dist)
                if not restored.is_empty and restored.area > polygon.area * 0.6:  # Don't lose too much area
                    print(f"      Applied hatch-based morphology: {polygon.area:.3f} → {restored.area:.3f} mm²")
                    return restored
            
            # Method 2: Remove thin protrusions using smaller erosion
            small_erosion = polygon.buffer(-hatch_spacing)
            if not small_erosion.is_empty:
                restored_small = small_erosion.buffer(hatch_spacing)
                if not restored_small.is_empty and restored_small.area > polygon.area * 0.7:
                    print(f"      Applied protrusion removal: {polygon.area:.3f} → {restored_small.area:.3f} mm²")
                    return restored_small
            
            # Method 3: Polygon simplification (remove unnecessary vertices)
            simplified = polygon.simplify(hatch_spacing/4, preserve_topology=True)  # Simplify to hatch precision
            if not simplified.is_empty and simplified.area > polygon.area * 0.8:
                print(f"      Applied vertex simplification: {polygon.area:.3f} → {simplified.area:.3f} mm²")
                return simplified
            
            # Method 4: Convex hull for very complex shapes (last resort)
            hull = polygon.convex_hull
            if hull.area < polygon.area * 1.3:  # Don't add too much area
                print(f"      Applied convex hull (last resort): {polygon.area:.3f} → {hull.area:.3f} mm²")
                return hull
            
        except Exception as e:
            print(f"      Simplification error: {e}")
        
        # Return original if all methods fail
        print(f"      No effective simplification possible")
        return polygon



    def _validate_hatching_zones(self, hatching_ready, outer_shape):
        """
        Validates the final hatching zones.
        """
        print(f"\n[VALIDATION] Final hatching zone summary:")
        

        if not outer_shape or outer_shape.is_empty or not outer_shape.is_valid:
            print("[❌] Outer shape is invalid")
        
        
        total_area = 0
        total_polygons = 0
        
        for zone_name, zone_polys in hatching_ready.items():
            zone_area = sum(p.area for p in zone_polys if p and not p.is_empty)
            poly_count = len([p for p in zone_polys if p and not p.is_empty])
            total_area += zone_area
            total_polygons += poly_count
            
            print(f"  {zone_name.upper()}: {poly_count} polygons, {zone_area:.4f} mm²")
        
        coverage = (total_area / outer_shape.area * 100) if outer_shape.area > 0 else 0
        print(f"  TOTAL: {total_polygons} polygons, {total_area:.4f} mm² ({coverage:.1f}% coverage)")
        print("[DEBUG] =================== HATCHING PREPARATION COMPLETE ===================\n")

    # Helper Methods
    def _calculate_coverage(self, regions: dict) -> dict:
        """
        Berechnet die Abdeckung pro Schicht.
        
        Args:
            regions (dict): Regionen-Dictionary
            
        Returns:
            dict: Abdeckungsstatistiken pro Schicht
        """
        area_by_stress = {"low": 0, "moderate": 0, "high": 0}

        for level in ["low", "moderate", "high"]:
            polygons = regions.get(level, [])
            unioned = unary_union(polygons)
            area_by_stress[level] = unioned.area if unioned else 0
        
        total_area = sum(area_by_stress.values())

        return {
            "total_area": total_area,
            "area_by_stress": area_by_stress,
            "percentages": {
                level: (area / total_area * 100 if total_area > 0 else 0)
                for level, area in area_by_stress.items()
            }
        }

    def _print_zone_statistics(self, z: float, thickness: float, 
                            original_regions: dict, cleaned_regions: dict):
        """
        Gibt detaillierte Statistiken über die Zonenerstellung aus.
        """
        print(f"[INFO] Schicht bei z={z:.3f} mm (Dicke: {thickness:.3f} mm)")

        print("\nOriginal Regionen (mit Überlappungen):")
        for level in ["high", "moderate", "low"]:
            polygons = original_regions.get(level, [])
            count = len(polygons)
            area = sum(p.area for p in polygons)
            print(f"  {level:8}: {count} Polygone, Fläche: {area:.4f} mm²")
            for i, p in enumerate(polygons):
                print(f"    - [{level.upper()} #{i}] Fläche = {p.area:.4f} mm²")

        print("\nBereinigte Regionen (ohne Überlappungen):")
        for level in ["high", "moderate", "low"]:
            polygons = cleaned_regions.get(level, [])
            count = len(polygons)
            area = sum(p.area for p in polygons)
            print(f"  {level:8}: {count} Polygone, Fläche: {area:.4f} mm²")
            for i, p in enumerate(polygons):
                print(f"    - [{level.upper()} #{i}] Fläche = {p.area:.4f} mm²")

        coverage = self._calculate_coverage(cleaned_regions)
        print(f"\nGesamtabdeckung: {coverage['total_area']:.4f} mm²")
        print("Prozentuale Verteilung:")
        for level, pct in coverage['percentages'].items():
            print(f"  {level:8}: {pct:.1f}%")

        

    def _export_cleaned_zones_to_json(self, cleaned_regions: dict, z: float):
        output = {}
        for label, poly_list in cleaned_regions.items():
            output[label] = [mapping(poly) for poly in poly_list if poly and not poly.is_empty]

        filename = f"cleaned_zones_z{z:.3f}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[✅] Cleaned zones saved to: {filename}")

    def _clean_overlap_regions_v5(self, regions: dict) -> dict:
        """
        Bereinigt überlappende Regionen mit intelligenter Zonenerhaltung.
        
        Strategie:
        - Erkennt automatisch ob Zonen verschachtelt oder teilweise überlappend sind
        - Bei verschachtelten Zonen: Erstellt Ringe (Donuts) für äußere Zonen
        - Bei teilweisen Überlappungen: Schneidet Überlappungen heraus
        - Erhält die visuelle Struktur der Stress-Verteilung
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Vereinige alle Polygone pro Kategorie
        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            if polys:
                valid_polys = [p for p in polys if p and not p.is_empty]
                if valid_polys:
                    unified_zones[category] = unary_union(valid_polys)
                else:
                    unified_zones[category] = None
            else:
                unified_zones[category] = None
        
        # Vereinige alle Polygone pro Kategorie
        


        print("[DEBUG] Unified Zones Content:")
        for key, value in unified_zones.items():
            if value is None:
                print(f"  {key.upper()}: None")
            elif isinstance(value, Polygon):
                print(f"  {key.upper()}: Polygon, Area = {value.area:.4f}, Bounds = {value.bounds}")
            elif isinstance(value, MultiPolygon):
                print(f"  {key.upper()}: MultiPolygon with {len(value.geoms)} parts, Total Area = {value.area:.4f}")
                for i, poly in enumerate(value.geoms):
                    print(f"    Part {i+1}: Area = {poly.area:.4f}, Bounds = {poly.bounds}")
            else:
                print(f"  {key.upper()}: Unknown geometry type: {type(value)}")
        
        # Prüfe ob die Zonen verschachtelt sind ODER sich überlappen
        is_nested = self._check_if_nested(unified_zones)

        is_nested = True
        
        if is_nested:
            # Verschachtelte oder überlappende Zonen: Verwende dynamische Reihenfolge
            
            nesting_order = self._get_nesting_order(unified_zones)
            print(f"[NESTED] = {is_nested}")
            print(f"[PROCESSING ORDER] = {nesting_order}")
            
            # Verarbeite Zonen in der korrekten Reihenfolge (von innen nach außen)
            processed_zones = {}
            
            for i, zone_name in enumerate(nesting_order):
                if i == 0:
                    # Innerste Zone bleibt unverändert
                    processed_zones[zone_name] = unified_zones[zone_name]
                else:
                    # Äußere Zonen: Entferne ALLE Überlappungen mit inneren Zonen
                    if unified_zones[zone_name]:
                        outer_zone = unified_zones[zone_name]
                        
                        # Entferne alle Überlappungen mit inneren Zonen (nicht nur vollständige Verschachtelungen)
                        for inner_zone_name in nesting_order[:i]:
                            print(f"[Outer zone] : {zone_name}")
                            print(f"[inner zone] : {inner_zone_name}")
                            if unified_zones[inner_zone_name]:
                                # Prüfe ob es eine Überlappung gibt
                                intersection = outer_zone.intersection(unified_zones[inner_zone_name])
                                if not intersection.is_empty:
                                    # Entferne die Überlappung von der äußeren Zone
                                    outer_zone = outer_zone.difference(unified_zones[inner_zone_name])
                                    print(f"[NESTED] Removed overlap: {inner_zone_name} from {zone_name}")
                                    
                        
                        processed_zones[zone_name] = outer_zone if not outer_zone.is_empty else None
                    else:
                        processed_zones[zone_name] = None
            
            # Füge die verarbeiteten Zonen zu cleaned hinzu
            for zone_name in nesting_order:
                if processed_zones[zone_name] and not processed_zones[zone_name].is_empty:
                    self._add_zone_to_cleaned(processed_zones[zone_name], zone_name, cleaned)
        
        else:
            print("[NOT NESTED] - Using priority-based subtraction")

            priority_order = ["high", "moderate", "low"]
            processed = {}

            for i, category in enumerate(priority_order):
                zone = unified_zones.get(category)
                if not zone or zone.is_empty:
                    continue

                # Subtrahiere alle höherpriorisierten (inneren) Zonen
                higher = [processed[p] for p in priority_order[:i] if p in processed and processed[p] and not processed[p].is_empty]
                if higher:
                    subtraction = unary_union(higher)
                    cleaned_zone = zone.difference(subtraction)
                else:
                    cleaned_zone = zone

                if cleaned_zone and not cleaned_zone.is_empty:
                    processed[category] = cleaned_zone
                    self._add_zone_to_cleaned(cleaned_zone, category, cleaned)
                    print(f"[✅] {category.upper()} kept with area: {cleaned_zone.area:.4f} mm²")
                else:
                    print(f"[⚠️] {category.upper()} was removed after overlap cleaning")


        
        # Verschmelze fragmentierte Teile
        #cleaned = self._merge_nearby_fragments(cleaned)
        
        return cleaned


    def trim_and_validate_regions(self, cleaned_regions: dict, outer_shape: Polygon) -> dict:
        """
        Schneidet alle Zonen innerhalb der outer_shape zu und gibt validierte Regionen zurück.

        Args:
            cleaned_regions (dict): dict mit "low", "moderate", "high" → Liste von Polygone
            outer_shape (Polygon): Die Gesamtfläche der Schicht
        
        Returns:
            dict: Bereinigte und getrimmte Regionen, zugeschnitten auf outer_shape
        """
        print("\n[🔍] Finale Validierung und Trim der Regionen nach dem Cleaning")

        trimmed_regions = {"low": [], "moderate": [], "high": []}
        total_area = 0.0

        for level in cleaned_regions:
            for poly in cleaned_regions[level]:
                if not poly or poly.is_empty:
                    continue
                clipped = poly.intersection(outer_shape)
                if clipped and not clipped.is_empty:
                    parts = self._split_into_polygons(clipped)
                    trimmed_regions[level].extend(parts)
                    total_area += sum(p.area for p in parts)

        print(f"[CHECK] Final assigned area: {total_area:.4f} / {outer_shape.area:.4f} mm² ({100 * total_area / outer_shape.area:.1f}%)")

        return trimmed_regions


    def _add_zone_to_cleaned(self, geometry, zone_name: str, cleaned: dict):
        """
        Hilfsfunktion um Geometrie zur cleaned dict hinzuzufügen
        """
        if geometry and not geometry.is_empty:
            if isinstance(geometry, MultiPolygon):
                for geom in geometry.geoms:
                    if geom.area >= self.min_cluster_area:
                        cleaned[zone_name].append(geom)
            elif geometry.area >= self.min_cluster_area:
                cleaned[zone_name].append(geometry)


    def _get_nesting_order(self, unified_zones: dict) -> list:
        """
        Bestimmt die Reihenfolge der Verschachtelung für korrekte Ringbildung.
        
        Returns:
            list: Reihenfolge der Zonen von innen nach außen
        """
        threshold = 0.95
        
        # Berechne Enthaltensein-Matrix
        containment = {
            "low": {"in": [], "contains": []},
            "moderate": {"in": [], "contains": []},
            "high": {"in": [], "contains": []}
        }
        
        # Prüfe welche Zone in welcher enthalten ist
        for inner in ["low", "moderate", "high"]:
            for outer in ["low", "moderate", "high"]:
                if inner != outer and unified_zones[inner] and unified_zones[outer]:
                    intersection = unified_zones[inner].intersection(unified_zones[outer])
                    if not intersection.is_empty:
                        overlap_ratio = intersection.area / unified_zones[inner].area
                        if overlap_ratio > threshold:
                            containment[inner]["in"].append(outer)
                            containment[outer]["contains"].append(inner)
        
        # Sortiere Zonen nach Verschachtelungstiefe
        # Mehr enthaltende Zonen = weiter innen (innerste zuerst)
        zone_depth = []
        for zone in ["low", "moderate", "high"]:
            if unified_zones[zone]:
                # Tiefe = Anzahl der Zonen die diese Zone enthalten
                depth = len(containment[zone]["in"])
                zone_depth.append((zone, depth))
        
        # Sortiere absteigend nach Tiefe (innerste zuerst)
        zone_depth.sort(key=lambda x: x[1], reverse=True)
        
        # Zusätzliche Sortierung bei gleicher Tiefe nach Größe (kleinere zuerst)
        if len(zone_depth) > 1:
            # Gruppiere nach Tiefe
            depth_groups = {}
            for zone, depth in zone_depth:
                if depth not in depth_groups:
                    depth_groups[depth] = []
                depth_groups[depth].append(zone)
            
            # Sortiere innerhalb jeder Tiefe nach Größe
            sorted_zones = []
            for depth in sorted(depth_groups.keys(), reverse=True):
                zones_at_depth = depth_groups[depth]
                if len(zones_at_depth) > 1:
                    # Sortiere nach Flächengröße (kleinere zuerst)
                    zones_at_depth.sort(key=lambda z: unified_zones[z].area if unified_zones[z] else 0)
                sorted_zones.extend(zones_at_depth)
            
            return sorted_zones
        
        return [zone[0] for zone in zone_depth]


    def _debug_nesting_info(self, unified_zones: dict, nesting_order: list):
        """
        Debug-Funktion um Verschachtelungsinfo auszugeben
        """
        if hasattr(self, 'debug') and self.debug:
            print(f"[DEBUG] Nesting Order: {nesting_order}")
            
            for i, zone in enumerate(nesting_order):
                if unified_zones[zone]:
                    area = unified_zones[zone].area
                    print(f"  {i+1}. {zone}: {area:.4f} area")
                    
                    # Zeige Enthaltensein
                    for other_zone in nesting_order:
                        if other_zone != zone and unified_zones[other_zone]:
                            intersection = unified_zones[zone].intersection(unified_zones[other_zone])
                            if not intersection.is_empty:
                                overlap_ratio = intersection.area / unified_zones[zone].area
                                if overlap_ratio > 0.95:
                                    print(f"    -> {zone} is contained in {other_zone}")
                                elif overlap_ratio > 0.1:
                                    print(f"    -> {zone} overlaps with {other_zone} ({overlap_ratio:.1%})")
            
    def _clean_overlap_regions_v4(self, regions: dict) -> dict:
        """
        Bereinigt überlappende Regionen mit intelligenter Zonenerhaltung.
        
        Strategie:
        - Erkennt automatisch ob Zonen verschachtelt oder teilweise überlappend sind
        - Bei verschachtelten Zonen: Erstellt Ringe (Donuts) für äußere Zonen
        - Bei teilweisen Überlappungen: Schneidet Überlappungen heraus
        - Erhält die visuelle Struktur der Stress-Verteilung
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
        
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Vereinige alle Polygone pro Kategorie (mit Buffer)
        buffer_amount = 0.5  # z. B. 0.1 mm Vergrößerung

        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            if polys:
                valid_polys = []
                for p in polys:
                    if p and not p.is_empty:
                        if not p.is_valid:
                            p = p.buffer(0)
                        if p.is_valid:
                            # Optional: leicht vergrößern
                            p = p.buffer(buffer_amount)
                            valid_polys.append(p)
                if valid_polys:
                    unified_zones[category] = unary_union(valid_polys)
                else:
                    unified_zones[category] = None
            else:
                unified_zones[category] = None
        
        # Prüfe ob die Zonen verschachtelt sind (nested)
        is_nested = self._check_if_nested(unified_zones)
        
        if is_nested:
            # Verschachtelte Zonen: Erstelle Ringe/Donuts
            # Reihenfolge: Von innen nach außen arbeiten

            print(f"[NESTED] = {is_nested}")
            
            # Low bleibt wie es ist (innerste Zone)
            if unified_zones["low"]:
                if isinstance(unified_zones["low"], MultiPolygon):
                    for geom in unified_zones["low"].geoms:
                        if geom.area > self.min_cluster_area:
                            cleaned["low"].append(geom)
                elif unified_zones["low"].area > self.min_cluster_area:
                    cleaned["low"].append(unified_zones["low"])
            
            # Moderate wird ein Ring (moderate minus low)
            if unified_zones["moderate"]:
                moderate_ring = unified_zones["moderate"]
                if unified_zones["low"]:
                    moderate_ring = moderate_ring.difference(unified_zones["low"])
                
                if moderate_ring and not moderate_ring.is_empty:
                    if isinstance(moderate_ring, MultiPolygon):
                        for geom in moderate_ring.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned["moderate"].append(geom)
                    elif moderate_ring.area > self.min_cluster_area:
                        cleaned["moderate"].append(moderate_ring)
            
            # High wird ein Ring (high minus moderate UND low)
            # WICHTIG: Wir müssen BEIDE inneren Zonen entfernen!
            if unified_zones["high"]:
                high_ring = unified_zones["high"]
                
                # Entferne die ORIGINALE moderate zone (nicht den Ring!)
                if unified_zones["moderate"]:
                    high_ring = high_ring.difference(unified_zones["moderate"])
                # ODER alternativ: Entferne sowohl moderate als auch low einzeln
                if unified_zones["moderate"]:
                    high_ring = high_ring.difference(unified_zones["moderate"])
                if unified_zones["low"]:
                     high_ring = high_ring.difference(unified_zones["low"])
                
                if high_ring and not high_ring.is_empty:
                    if isinstance(high_ring, MultiPolygon):
                        for geom in high_ring.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned["high"].append(geom)
                    elif high_ring.area > self.min_cluster_area:
                        cleaned["high"].append(high_ring)
        
        else:
            # Nicht verschachtelt: Verwende die gleiche Behandlung für alle
            # Entferne nur die Überlappungsbereiche
            
            all_overlaps = []
            
            # Finde alle paarweisen Überlappungen
            if unified_zones["high"] and unified_zones["moderate"]:
                overlap = unified_zones["high"].intersection(unified_zones["moderate"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            if unified_zones["high"] and unified_zones["low"]:
                overlap = unified_zones["high"].intersection(unified_zones["low"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            if unified_zones["moderate"] and unified_zones["low"]:
                overlap = unified_zones["moderate"].intersection(unified_zones["low"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            # Vereinige alle Überlappungen
            if all_overlaps:
                combined_overlaps = unary_union(all_overlaps)
            else:
                combined_overlaps = Polygon()
            
            # Entferne Überlappungen aus allen Zonen
            for category in ["low", "moderate", "high"]:
                if unified_zones[category] and not unified_zones[category].is_empty:
                    if not combined_overlaps.is_empty:
                        result = unified_zones[category].difference(combined_overlaps)
                    else:
                        result = unified_zones[category]
                    
                    if result and not result.is_empty:
                        if isinstance(result, MultiPolygon):
                            for geom in result.geoms:
                                if geom.area > self.min_cluster_area:
                                    cleaned[category].append(geom)
                        elif result.area > self.min_cluster_area:
                            cleaned[category].append(result)
        
        # Verschmelze fragmentierte Teile
        cleaned = self._merge_nearby_fragments(cleaned)
        
        return cleaned

    def _clean_overlap_regions_v3(self, regions: dict) -> dict:
        """
        last method
        Bereinigt überlappende Regionen mit intelligenter Zonenerhaltung.
        
        Strategie:
        - Erkennt automatisch ob Zonen verschachtelt oder teilweise überlappend sind
        - Bei verschachtelten Zonen: Erstellt Ringe (Donuts) für äußere Zonen
        - Bei teilweisen Überlappungen: Schneidet Überlappungen heraus
        - Erhält die visuelle Struktur der Stress-Verteilung
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
        
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Vereinige alle Polygone pro Kategorie
        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            if polys:
                valid_polys = [p for p in polys if p and not p.is_empty]
                if valid_polys:
                    unified_zones[category] = unary_union(valid_polys)
                else:
                    unified_zones[category] = None
            else:
                unified_zones[category] = None
        
        # Prüfe ob die Zonen verschachtelt sind (nested)
        is_nested = self._check_if_nested(unified_zones)
        
        if is_nested:
            # Verschachtelte Zonen: Erstelle Ringe/Donuts
            # Reihenfolge: Von innen nach außen arbeiten
            
            # Low bleibt wie es ist (innerste Zone)
            if unified_zones["low"]:
                if isinstance(unified_zones["low"], MultiPolygon):
                    for geom in unified_zones["low"].geoms:
                        if geom.area > self.min_cluster_area:
                            cleaned["low"].append(geom)
                elif unified_zones["low"].area > self.min_cluster_area:
                    cleaned["low"].append(unified_zones["low"])
            
            # Moderate wird ein Ring (moderate minus low)
            if unified_zones["moderate"]:
                moderate_ring = unified_zones["moderate"]
                if unified_zones["low"]:
                    moderate_ring = moderate_ring.difference(unified_zones["low"])
                
                if moderate_ring and not moderate_ring.is_empty:
                    if isinstance(moderate_ring, MultiPolygon):
                        for geom in moderate_ring.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned["moderate"].append(geom)
                    elif moderate_ring.area > self.min_cluster_area:
                        cleaned["moderate"].append(moderate_ring)
            
            # High wird ein Ring (high minus moderate)
            if unified_zones["high"]:
                high_ring = unified_zones["high"]
                if unified_zones["moderate"]:
                    high_ring = high_ring.difference(unified_zones["moderate"])
                
                if high_ring and not high_ring.is_empty:
                    if isinstance(high_ring, MultiPolygon):
                        for geom in high_ring.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned["high"].append(geom)
                    elif high_ring.area > self.min_cluster_area:
                        cleaned["high"].append(high_ring)
        
        else:
            # Nicht verschachtelt: Verwende die gleiche Behandlung für alle
            # Entferne nur die Überlappungsbereiche
            
            all_overlaps = []
            
            # Finde alle paarweisen Überlappungen
            if unified_zones["high"] and unified_zones["moderate"]:
                overlap = unified_zones["high"].intersection(unified_zones["moderate"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            if unified_zones["high"] and unified_zones["low"]:
                overlap = unified_zones["high"].intersection(unified_zones["low"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            if unified_zones["moderate"] and unified_zones["low"]:
                overlap = unified_zones["moderate"].intersection(unified_zones["low"])
                if not overlap.is_empty:
                    all_overlaps.append(overlap)
            
            # Vereinige alle Überlappungen
            if all_overlaps:
                combined_overlaps = unary_union(all_overlaps)
            else:
                combined_overlaps = Polygon()
            
            # Entferne Überlappungen aus allen Zonen
            for category in ["low", "moderate", "high"]:
                if unified_zones[category] and not unified_zones[category].is_empty:
                    if not combined_overlaps.is_empty:
                        result = unified_zones[category].difference(combined_overlaps)
                    else:
                        result = unified_zones[category]
                    
                    if result and not result.is_empty:
                        if isinstance(result, MultiPolygon):
                            for geom in result.geoms:
                                if geom.area > self.min_cluster_area:
                                    cleaned[category].append(geom)
                        elif result.area > self.min_cluster_area:
                            cleaned[category].append(result)
        
        # Verschmelze fragmentierte Teile
        cleaned = self._merge_nearby_fragments(cleaned)
        
        return cleaned

    def _check_if_nested_old(self, unified_zones: dict) -> bool:
        """
        Prüft ob die Zonen verschachtelt sind (eine Zone enthält die andere vollständig).
        
        Returns:
            bool: True wenn verschachtelt, False sonst
        """
        # Prüfe ob low vollständig in moderate enthalten ist
        low_in_moderate = False
        if unified_zones["low"] and unified_zones["moderate"]:
            intersection = unified_zones["low"].intersection(unified_zones["moderate"])
            if not intersection.is_empty:
                # Wenn die Schnittmenge fast gleich low ist, dann ist low in moderate
                low_in_moderate = intersection.area / unified_zones["low"].area > 0.95
        
        # Prüfe ob moderate vollständig in high enthalten ist
        moderate_in_high = False
        if unified_zones["moderate"] and unified_zones["high"]:
            intersection = unified_zones["moderate"].intersection(unified_zones["high"])
            if not intersection.is_empty:
                # Wenn die Schnittmenge fast gleich moderate ist, dann ist moderate in high
                moderate_in_high = intersection.area / unified_zones["moderate"].area > 0.95
        
        # Als verschachtelt betrachten wenn eine der Bedingungen erfüllt ist
        # oder wenn low in high enthalten ist (für den Fall dass moderate fehlt)
        low_in_high = False
        if unified_zones["low"] and unified_zones["high"] and not unified_zones["moderate"]:
            intersection = unified_zones["low"].intersection(unified_zones["high"])
            if not intersection.is_empty:
                low_in_high = intersection.area / unified_zones["low"].area > 0.95
        
        return low_in_moderate or moderate_in_high or low_in_high

    def _check_if_nested(self, unified_zones: dict) -> bool:
        """
        Prüft ob die Zonen verschachtelt sind (eine Zone enthält die andere vollständig).
        Deckt alle möglichen Verschachtelungsszenarien ab.
        
        Mögliche Fälle:
        1. Vollständige Verschachtelung: low ⊂ moderate ⊂ high
        2. Teilweise Verschachtelung: low ⊂ moderate, moderate ⊂ high (aber low nicht in high)
        3. Nur moderate in high (kein low)
        4. Nur low in high (kein moderate)
        5. Nur low in moderate (kein high)
        6. Beliebige Kombination wo eine Zone eine andere enthält
        
        Returns:
            bool: True wenn mindestens eine Zone vollständig in einer anderen enthalten ist
        """
        threshold = 0.80  # 95% Überlappung gilt als "enthalten"
        
        # Prüfe alle möglichen Verschachtelungen
        nesting_checks = []
        
        # Low in Moderate
        if unified_zones["low"] and unified_zones["moderate"]:
            intersection = unified_zones["low"].intersection(unified_zones["moderate"])
            if not intersection.is_empty:
                low_in_moderate = intersection.area / unified_zones["low"].area > threshold
                nesting_checks.append(low_in_moderate)
        
        # Low in High
        if unified_zones["low"] and unified_zones["high"]:
            intersection = unified_zones["low"].intersection(unified_zones["high"])
            if not intersection.is_empty:
                low_in_high = intersection.area / unified_zones["low"].area > threshold
                nesting_checks.append(low_in_high)
        
        # Moderate in High
        if unified_zones["moderate"] and unified_zones["high"]:
            intersection = unified_zones["moderate"].intersection(unified_zones["high"])
            if not intersection.is_empty:
                moderate_in_high = intersection.area / unified_zones["moderate"].area > threshold
                nesting_checks.append(moderate_in_high)
        
        # Zusätzliche Checks für umgekehrte Verschachtelungen (falls Daten fehlerhaft sind)
        
        # High in Moderate 
        if unified_zones["high"] and unified_zones["moderate"]:
            intersection = unified_zones["high"].intersection(unified_zones["moderate"])
            if not intersection.is_empty and unified_zones["high"].area > 0:
                high_in_moderate = intersection.area / unified_zones["high"].area > threshold
                nesting_checks.append(high_in_moderate)
        
        # Moderate in Low 
        if unified_zones["moderate"] and unified_zones["low"]:
            intersection = unified_zones["moderate"].intersection(unified_zones["low"])
            if not intersection.is_empty and unified_zones["moderate"].area > 0:
                moderate_in_low = intersection.area / unified_zones["moderate"].area > threshold
                nesting_checks.append(moderate_in_low)
        
        # High in Low 
        if unified_zones["high"] and unified_zones["low"]:
            intersection = unified_zones["high"].intersection(unified_zones["low"])
            if not intersection.is_empty and unified_zones["high"].area > 0:
                high_in_low = intersection.area / unified_zones["high"].area > threshold
                nesting_checks.append(high_in_low)
        
        # Debug-Ausgabe (optional)
        if hasattr(self, 'debug') and self.debug and nesting_checks:
            print(f"[DEBUG] Nesting detected: {any(nesting_checks)}")
            if unified_zones["low"] and unified_zones["moderate"]:
                print(f"  Low in Moderate: {low_in_moderate if 'low_in_moderate' in locals() else 'N/A'}")
            if unified_zones["moderate"] and unified_zones["high"]:
                print(f"  Moderate in High: {moderate_in_high if 'moderate_in_high' in locals() else 'N/A'}")
        
        # Rückgabe: True wenn irgendeine Verschachtelung gefunden wurde
        return any(nesting_checks) if nesting_checks else False


    def _get_nesting_order_old(self, unified_zones: dict) -> list:
        """
        Bestimmt die Reihenfolge der Verschachtelung für korrekte Ringbildung.
        
        Returns:
            list: Reihenfolge der Zonen von innen nach außen
        """
        threshold = 0.95
        
        # Berechne Enthaltensein-Matrix
        containment = {
            "low": {"in": []},
            "moderate": {"in": []},
            "high": {"in": []}
        }
        
        # Prüfe welche Zone in welcher enthalten ist
        for inner in ["low", "moderate", "high"]:
            for outer in ["low", "moderate", "high"]:
                if inner != outer and unified_zones[inner] and unified_zones[outer]:
                    intersection = unified_zones[inner].intersection(unified_zones[outer])
                    if not intersection.is_empty:
                        if intersection.area / unified_zones[inner].area > threshold:
                            containment[inner]["in"].append(outer)
        
        # Sortiere Zonen nach Anzahl der sie enthaltenden Zonen
        # Mehr enthaltende Zonen = weiter innen
        zone_order = []
        for zone in ["low", "moderate", "high"]:
            if unified_zones[zone]:
                zone_order.append((zone, len(containment[zone]["in"])))
        
        # Sortiere absteigend nach Anzahl der Container (innerste zuerst)
        zone_order.sort(key=lambda x: x[1], reverse=True)
        
        return [zone[0] for zone in zone_order]

    def _clean_overlap_regions_old(self, regions: dict) -> dict:
        """
        Bereinigt überlappende Regionen zwischen High-, Moderate- und Low-Stress-Zonen.
        Alle Zonen werden gleichberechtigt behandelt - überlappende Bereiche werden 
        aus allen beteiligten Zonen entfernt.
        
        Strategie:
        - Keine Zone hat Priorität
        - Bei Überlappungen wird der überlappende Bereich aus allen Zonen entfernt
        - Zonen behalten ihre nicht-überlappenden Bereiche
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
       
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Sammle alle Polygone mit ihrer Kategorie
        all_polygons = []
        for category in ["low", "moderate", "high"]:
            for poly in regions.get(category, []):
                if poly and not poly.is_empty:
                    all_polygons.append((poly, category))
        
        # Finde alle Überlappungen zwischen verschiedenen Kategorien
        overlaps_to_remove = []
        
        for i in range(len(all_polygons)):
            poly1, cat1 = all_polygons[i]
            
            for j in range(i + 1, len(all_polygons)):
                poly2, cat2 = all_polygons[j]
                
                # Nur Überlappungen zwischen verschiedenen Kategorien bearbeiten
                if cat1 != cat2:
                    intersection = poly1.intersection(poly2)
                    if not intersection.is_empty:
                        overlaps_to_remove.append(intersection)
        
        # Vereinige alle Überlappungen zu einer einzigen Geometrie
        if overlaps_to_remove:
            all_overlaps = unary_union(overlaps_to_remove)
        else:
            all_overlaps = None
        
        # Entferne die Überlappungen aus allen Zonen
        for category in ["low", "moderate", "high"]:
            for poly in regions.get(category, []):
                if not poly or poly.is_empty:
                    continue
                
                # Entferne alle Überlappungen
                if all_overlaps:
                    result_poly = poly.difference(all_overlaps)
                else:
                    result_poly = poly
                
                # Füge die bereinigte Zone hinzu, wenn sie noch existiert
                if result_poly and not result_poly.is_empty:
                    # Handle MultiPolygon results
                    if isinstance(result_poly, MultiPolygon):
                        for geom in result_poly.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned[category].append(geom)
                    elif result_poly.area > self.min_cluster_area:
                        cleaned[category].append(result_poly)
        
        # Optional: Verschmelze fragmentierte Teile derselben Stress-Kategorie
        cleaned = self._merge_nearby_fragments(cleaned)
    
        return cleaned
    
    def _clean_overlap_regions_v2(self, regions: dict) -> dict:
        """
        Robuste Bereinigung von Überlappungen zwischen Stress-Zonen.
        Diese einfachere Methode funktioniert für ALLE möglichen Überlappungsszenarien.
        
        Strategie:
        - Identifiziere alle Überlappungsbereiche zwischen verschiedenen Zonen
        - Entferne diese Bereiche aus ALLEN beteiligten Zonen
        - Einfach, robust und funktioniert für alle Fälle
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
        
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Schritt 1: Vereinige alle Polygone pro Kategorie
        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            if polys:
                valid_polys = [p for p in polys if p and not p.is_empty]
                if valid_polys:
                    unified_zones[category] = unary_union(valid_polys)
                else:
                    unified_zones[category] = None
            else:
                unified_zones[category] = None
        
        # Schritt 2: Finde ALLE Überlappungsbereiche
        all_overlaps = []
        
        # High-Moderate Überlappung
        if unified_zones["high"] and unified_zones["moderate"]:
            overlap = unified_zones["high"].intersection(unified_zones["moderate"])
            if not overlap.is_empty:
                all_overlaps.append(overlap)
        
        # High-Low Überlappung
        if unified_zones["high"] and unified_zones["low"]:
            overlap = unified_zones["high"].intersection(unified_zones["low"])
            if not overlap.is_empty:
                all_overlaps.append(overlap)
        
        # Moderate-Low Überlappung
        if unified_zones["moderate"] and unified_zones["low"]:
            overlap = unified_zones["moderate"].intersection(unified_zones["low"])
            if not overlap.is_empty:
                all_overlaps.append(overlap)
        
        # Schritt 3: Vereinige alle Überlappungen zu einer Geometrie
        if all_overlaps:
            combined_overlaps = unary_union(all_overlaps)
        else:
            combined_overlaps = Polygon()  # Empty polygon
        
        # Schritt 4: Entferne alle Überlappungen aus jeder Zone
        for category in ["low", "moderate", "high"]:
            if unified_zones[category] and not unified_zones[category].is_empty:
                # Entferne die Überlappungen
                if not combined_overlaps.is_empty:
                    result = unified_zones[category].difference(combined_overlaps)
                else:
                    result = unified_zones[category]
                
                # Verarbeite das Ergebnis
                if result and not result.is_empty:
                    if isinstance(result, MultiPolygon):
                        for geom in result.geoms:
                            if geom.area > self.min_cluster_area:
                                cleaned[category].append(geom)
                    elif result.area > self.min_cluster_area:
                        cleaned[category].append(result)
        
        # Optional: Verschmelze fragmentierte Teile
        cleaned = self._merge_nearby_fragments(cleaned)
        
        return cleaned
    
    def _clean_overlap_regions_old1(self, regions: dict) -> dict:
        """
        Bereinigt überlappende Regionen zwischen High-, Moderate- und Low-Stress-Zonen.
        Speziell für verschachtelte Zonen entwickelt - schneidet überlappende Bereiche
        aus allen beteiligten Zonen heraus.
        
        Strategie:
        - Keine Zone hat Priorität
        - Bei Überlappungen zwischen verschiedenen Zonen wird der überlappende 
        Bereich aus ALLEN beteiligten Zonen entfernt
        - Funktioniert auch bei vollständig verschachtelten Zonen
        
        Args:
            regions (dict): {
                "low": [Polygon, ...],
                "moderate": [Polygon, ...], 
                "high": [Polygon, ...]
            }
            
        Returns:
            dict: Bereinigte Regionen ohne Überlappungen
        """
        from shapely.geometry import MultiPolygon
        from shapely.ops import unary_union
        
        cleaned = {"low": [], "moderate": [], "high": []}
        
        # Vereinige alle Polygone pro Kategorie zu einer einzigen Geometrie
        unified_zones = {}
        for category in ["low", "moderate", "high"]:
            polys = regions.get(category, [])
            if polys:
                valid_polys = [p for p in polys if p and not p.is_empty]
                if valid_polys:
                    unified_zones[category] = unary_union(valid_polys)
                else:
                    unified_zones[category] = None
            else:
                unified_zones[category] = None
        
        # Berechne die exklusiven Bereiche für jede Zone
        # (Bereiche, die NUR zu dieser Zone gehören)
        
        # High exclusive: high - (high ∩ moderate) - (high ∩ low)
        if unified_zones["high"]:
            high_exclusive = unified_zones["high"]
            
            if unified_zones["moderate"]:
                high_moderate_overlap = unified_zones["high"].intersection(unified_zones["moderate"])
                if not high_moderate_overlap.is_empty:
                    high_exclusive = high_exclusive.difference(high_moderate_overlap)
            
            if unified_zones["low"]:
                high_low_overlap = unified_zones["high"].intersection(unified_zones["low"])
                if not high_low_overlap.is_empty:
                    high_exclusive = high_exclusive.difference(high_low_overlap)
            
            if high_exclusive and not high_exclusive.is_empty:
                if isinstance(high_exclusive, MultiPolygon):
                    for geom in high_exclusive.geoms:
                        if geom.area > self.min_cluster_area:
                            cleaned["high"].append(geom)
                elif high_exclusive.area > self.min_cluster_area:
                    cleaned["high"].append(high_exclusive)
        
        # Moderate exclusive: moderate - (moderate ∩ high) - (moderate ∩ low)
        if unified_zones["moderate"]:
            moderate_exclusive = unified_zones["moderate"]
            
            if unified_zones["high"]:
                moderate_high_overlap = unified_zones["moderate"].intersection(unified_zones["high"])
                if not moderate_high_overlap.is_empty:
                    moderate_exclusive = moderate_exclusive.difference(moderate_high_overlap)
            
            if unified_zones["low"]:
                moderate_low_overlap = unified_zones["moderate"].intersection(unified_zones["low"])
                if not moderate_low_overlap.is_empty:
                    moderate_exclusive = moderate_exclusive.difference(moderate_low_overlap)
            
            if moderate_exclusive and not moderate_exclusive.is_empty:
                if isinstance(moderate_exclusive, MultiPolygon):
                    for geom in moderate_exclusive.geoms:
                        if geom.area > self.min_cluster_area:
                            cleaned["moderate"].append(geom)
                elif moderate_exclusive.area > self.min_cluster_area:
                    cleaned["moderate"].append(moderate_exclusive)
        
        # Low exclusive: low - (low ∩ high) - (low ∩ moderate)
        if unified_zones["low"]:
            low_exclusive = unified_zones["low"]
            
            if unified_zones["high"]:
                low_high_overlap = unified_zones["low"].intersection(unified_zones["high"])
                if not low_high_overlap.is_empty:
                    low_exclusive = low_exclusive.difference(low_high_overlap)
            
            if unified_zones["moderate"]:
                low_moderate_overlap = unified_zones["low"].intersection(unified_zones["moderate"])
                if not low_moderate_overlap.is_empty:
                    low_exclusive = low_exclusive.difference(low_moderate_overlap)
            
            if low_exclusive and not low_exclusive.is_empty:
                if isinstance(low_exclusive, MultiPolygon):
                    for geom in low_exclusive.geoms:
                        if geom.area > self.min_cluster_area:
                            cleaned["low"].append(geom)
                elif low_exclusive.area > self.min_cluster_area:
                    cleaned["low"].append(low_exclusive)
        
        # Optional: Verschmelze fragmentierte Teile derselben Stress-Kategorie
        cleaned = self._merge_nearby_fragments(cleaned)
        
        return cleaned

    def _merge_nearby_fragments(self, regions: dict, buffer_dist: float = 0.1) -> dict:
        """
        Verschmilzt nahe beieinander liegende Fragmente derselben Stress-Kategorie.
        Dies hilft, wenn eine Zone durch Überlappungsentfernung fragmentiert wurde.
        
        Args:
            regions (dict): Bereinigte Regionen
            buffer_dist (float): Distanz für Verschmelzung in mm
            
        Returns:
            dict: Regionen mit verschmolzenen Fragmenten
        """
        
        
        merged = {"low": [], "moderate": [], "high": []}
        
        for stress_level, polygons in regions.items():
            if not polygons:
                continue
            
            # Verwende einen iterativen Ansatz für transitive Verschmelzung
            remaining = list(polygons)
            final_groups = []
            
            while remaining:
                # Starte mit dem ersten verbleibenden Polygon
                current_group = [remaining.pop(0)]
                
                # Suche iterativ nach allen verbundenen Polygonen
                changed = True
                while changed:
                    changed = False
                    new_remaining = []
                    
                    for poly in remaining:
                        # Prüfe ob das Polygon zu irgendeinem in der aktuellen Gruppe nah ist
                        is_close = False
                        for group_poly in current_group:
                            if poly.distance(group_poly) < buffer_dist:
                                is_close = True
                                break
                        
                        if is_close:
                            current_group.append(poly)
                            changed = True
                        else:
                            new_remaining.append(poly)
                    
                    remaining = new_remaining
                
                final_groups.append(current_group)
            
            # Verschmelze jede Gruppe
            for group in final_groups:
                if len(group) == 1:
                    merged[stress_level].append(group[0])
                else:
                    # Verschmelze mit kleinem Buffer
                    buffered = [p.buffer(buffer_dist/2) for p in group]
                    union = unary_union(buffered)
                    # Zurück zur Originalgröße
                    result = union.buffer(-buffer_dist/2)
                    
                    # Prüfe minimale Fläche
                    if isinstance(result, MultiPolygon):
                        for geom in result.geoms:
                            if geom.area > self.min_cluster_area:
                                merged[stress_level].append(geom)
                    elif result.area > self.min_cluster_area:
                        merged[stress_level].append(result)
        
        return merged

    def _split_into_polygons(self, geometry):
        """
        Konvertiert eine Geometrie in eine Liste von Polygonen.
        
        Args:
            geometry: Shapely Geometry (Polygon oder MultiPolygon)
            
        Returns:
            list: Liste von Polygon-Objekten
        """
        from shapely.geometry import Polygon, MultiPolygon
        
        if isinstance(geometry, Polygon):
            return [geometry]
        elif isinstance(geometry, MultiPolygon):
            return list(geometry.geoms)
        else:
            return []

    def _fill_gaps_with_base_stress(self, cleaned: dict, original_regions: dict) -> dict:
        """
        Füllt Lücken in der Zonenabdeckung mit der niedrigsten Stress-Kategorie.
        
        Args:
            cleaned (dict): Bereinigte Regionen
            original_regions (dict): Original-Regionen vor Bereinigung
            
        Returns:
            dict: Regionen mit gefüllten Lücken
        """
        from shapely.ops import unary_union
        from shapely.geometry import Polygon
        
        # Berechne die gesamte abgedeckte Fläche
        all_covered = []
        for stress_level in ["high", "moderate", "low"]:
            all_covered.extend(cleaned.get(stress_level, []))
        
        if not all_covered:
            return cleaned
        
        total_covered = unary_union(all_covered)
        
        # Berechne die ursprüngliche Gesamtfläche
        all_original = []
        for stress_level in ["high", "moderate", "low"]:
            all_original.extend(original_regions.get(stress_level, []))
        
        if not all_original:
            return cleaned
        
        total_original = unary_union(all_original)
        
        # Finde Lücken
        gaps = total_original.difference(total_covered)
        
        # Füge Lücken zur Low-Stress-Zone hinzu
        if gaps and not gaps.is_empty:
            gap_polygons = self._split_into_polygons(gaps)
            cleaned["low"].extend(gap_polygons)
        
        return cleaned

    def _filter_small_fragments(self, regions: dict, min_area: float = 0.1) -> dict:
        """
        Entfernt sehr kleine Polygonfragmente.
        
        Args:
            regions (dict): Regionen-Dictionary
            min_area (float): Minimale Fläche in mm²
            
        Returns:
            dict: Gefilterte Regionen
        """
        filtered = {"low": [], "moderate": [], "high": []}
        
        for stress_level, polygons in regions.items():
            for poly in polygons:
                if poly.area >= min_area:
                    filtered[stress_level].append(poly)
        
        return filtered

    def _merge_adjacent_zones_old(self, regions: dict, buffer_distance: float = 0.05) -> dict:
        """
        Optional: Verschmilzt benachbarte Zonen des gleichen Stress-Levels.
        
        Args:
            regions (dict): Regionen-Dictionary
            buffer_distance (float): Buffer-Distanz für Verschmelzung in mm
            
        Returns:
            dict: Regionen mit verschmolzenen benachbarten Zonen
        """
        from shapely.ops import unary_union
        
        merged = {"low": [], "moderate": [], "high": []}
        
        for stress_level, polygons in regions.items():
            if not polygons:
                continue
                
            # Buffer und dann negative Buffer für Verschmelzung
            buffered = [poly.buffer(buffer_distance) for poly in polygons]
            merged_buffered = unary_union(buffered)
            
            # Zurück zum Original
            final = merged_buffered.buffer(-buffer_distance)
            
            merged[stress_level] = self._split_into_polygons(final)
        
        return merged

    # Endregion: FEM Stress Slicing

    # method to output stress analysis results for multiple Z heights

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
        
            
        stress_thresholds = self.stress_threshold

        wkt_string = """POLYGON ((0 5.5, 0 4.949999999999999, 0 -5.5, 7.219999909400939 -5.5, 7.599999904632568 -5.5, 7.976207065582275 -4.214638066291809, 7.996007442474365 -4.146987438201904, 8.719320893287659 -3.0198212027549745, 8.757390022277832 -2.960496664047241, 9.769132375717163 -2.082963907718659, 9.822381973266602 -2.036777973175049, 11.04047908782959 -1.4800667941570282, 11.104589462280273 -1.4507662057876587, 12.430224943161011 -1.2600383102893828, 12.499995231628418 -1.25, 26.74992184638977 -1.25, 27.49991798400879 -1.25, 28.825565242767333 -1.4407271027565003, 28.895336151123047 -1.4507653713226318, 30.113444137573243 -2.007474625110626, 30.177555084228516 -2.0367751121520996, 31.18931465148926 -2.914305830001831, 31.242565155029297 -2.96049165725708, 31.965897178649904 -4.08765857219696, 32.00396728515625 -4.1469831466674805, 32.38019981384277 -5.432349157333374, 32.400001525878906 -5.5, 39.62000007629395 -5.5, 40 -5.5, 40 4.949999999999999, 40 5.5, 32.78000144958496 5.5, 32.400001525878906 5.5, 32.023768997192384 4.214633989334106, 32.00396728515625 4.1469831466674805, 31.280635261535643 3.0198162317276003, 31.242565155029297 2.96049165725708, 30.230805587768554 2.0829609394073487, 30.177555084228516 2.0367751121520996, 28.95944709777832 1.4800658583641053, 28.895336151123047 1.4507653713226318, 27.569688892364503 1.2600382685661315, 27.49991798400879 1.25, 13.249991369247436 1.25, 12.499995231628418 1.25, 11.17435975074768 1.4407278954982758, 11.104589462280273 1.4507662057876587, 9.886492347717285 2.0074773848056795, 9.822381973266602 2.036777973175049, 8.81063961982727 2.9143107295036312, 8.757390022277832 2.960496664047241, 8.034076571464539 4.0876628994941715, 7.996007442474365 4.146987438201904, 7.619800281524658 5.432349371910095, 7.599999904632568 5.5, 0.3799999952316293 5.5, 0 5.5))"""
        outer_shape = wkt.loads(wkt_string)

    
        for z in z_heights:
            # Extract nodes and regions with zone limiting
            slice_result = self.generate_slice_stress_regions(z=z, thickness=thickness,outer_shape= outer_shape)
            
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

    def slice_with_stress_analysis_new(self, z: float, thickness: float = 0.25,
                                     zone_per_slice: int = 3, ensure_no_overlap: bool = False,
                                     outer_shape: Polygon = None) -> Dict[str, Any]:
        """
        Enhanced slicing function for a single Z height.

        Parameters:
            z (float): Z height to process
            thickness (float): Layer thickness
            zone_per_slice (int): Max number of zones per slice
            ensure_no_overlap (bool): Apply overlap cleaning
            outer_shape (Polygon): Outer contour for trimming/filling

        Returns:
            Dict[str, Any]: Processed layer result
        """
        if self.nodes is None:
            self.get_node_data_with_stress()

        stress_thresholds = self.stress_threshold

        slice_result = self.generate_slice_stress_regions(
            z=z, thickness=thickness,
            ensure_no_overlap=ensure_no_overlap,
            outer_shape=outer_shape
        )

        regions_serializable = {}
        total_area = 0.0
        for stress_level, polygons in slice_result["regions"].items():
            regions_serializable[stress_level] = []
            for poly in polygons:
                if hasattr(poly, 'exterior'):
                    coords = list(poly.exterior.coords[:-1])
                    regions_serializable[stress_level].append(coords)
                    total_area += poly.area
                elif hasattr(poly, 'geoms'):
                    for sub_poly in poly.geoms:
                        coords = list(sub_poly.exterior.coords[:-1])
                        regions_serializable[stress_level].append(coords)
                        total_area += sub_poly.area

        slice_nodes = slice_result["slice_nodes"]
        if len(slice_nodes) > 0:
            slice_stress_values = slice_nodes[:, 3]
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
            stress_stats = {k: 0.0 for k in [
                "min_stress", "max_stress", "mean_stress", "std_stress", "median_stress",
                "low_threshold", "moderate_threshold"]}

        layer_data = StressLayerData(
            layer_height=float(z),
            thickness=float(thickness),
            regions=regions_serializable,
            node_count=len(slice_nodes),
            stress_statistics=stress_stats,
            nodes=slice_nodes
        )

        return {
            "layer_data": layer_data,
            "regions": slice_result["regions"],
            "slice_nodes": slice_nodes,
            "stress_thresholds": stress_thresholds,
            "zone_count": sum(len(polys) for polys in slice_result["regions"].values()),
            "zone_limit_applied": zone_per_slice is not None
        }
    
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
            "low_threshold": self.stress_threshold[0],
            "moderate_threshold": self.stress_threshold[1]
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


# NEW Methods

    

    def create_stress_regions(self, xy_points: np.ndarray, stress_values: np.ndarray, 
                            min_cluster_area: float = 1,
                            clustering_params: dict = None):
        """
        Create stress regions from point data using clustering and polygon wrapping.
        
        Args:
            xy_points: Array of shape (n, 2) with x,y coordinates
            stress_values: Array of shape (n,) with stress values
            stress_threshold: Tuple (low_max, moderate_max) for stress classification
            min_cluster_area: Minimum area for a valid stress region
            clustering_params: Dict with DBSCAN parameters per stress level
            
        Returns:
            dict: Polygons for each stress level {"low": [...], "moderate": [...], "high": [...]}
        """
        low_max, mod_max = self.stress_threshold
        
        # 1. Build masks
        low_mask = stress_values < low_max
        moderate_mask = (stress_values >= low_max) & (stress_values < mod_max)
        high_mask = stress_values >= mod_max
        
        stress_masks = {
            "low": low_mask,
            "moderate": moderate_mask,
            "high": high_mask
        }
        
        # Default clustering parameters - different for each stress level
        if clustering_params is None:
            clustering_params = {
                "low": {"eps": 0.5, "min_samples": 10},      # Larger eps for connected regions
                "moderate": {"eps": 0.5, "min_samples": 8},  # Medium connectivity
                "high": {"eps": 0.5, "min_samples": 5}       # Smaller eps for hotspots
            }
        
        regions = {"low": [], "moderate": [], "high": []}
        
        # 2. Process each stress level
        for stress_level, mask in stress_masks.items():
            if not np.any(mask):
                continue
                
            # Extract points for this stress level
            level_points = xy_points[mask]
            
            # 3. Cluster points using DBSCAN
            params = clustering_params.get(stress_level, {"eps": 0.5, "min_samples": 5})
            clustering = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels = clustering.fit_predict(level_points)
            
            # 4. Process each cluster
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:
                    # Handle outliers/noise
                    outlier_points = level_points[labels == -1]
                    outlier_polygons = self.handle_outliers(outlier_points, buffer_radius=params["eps"]/2)
                    regions[stress_level].extend(outlier_polygons)
                    continue
                
                # Get cluster points
                cluster_mask = labels == label
                cluster_points = level_points[cluster_mask]
                
                # 5. Create polygon from cluster
                polygon = self.create_polygon_from_points(cluster_points, method="alpha_shape")
                
                if polygon and polygon.area > min_cluster_area:
                    regions[stress_level].append(polygon)
        
        # 6. Post-process: merge nearby regions if needed
        regions = self.merge_nearby_regions(regions, merge_distance=0.5)
        
        return regions


    def create_polygon_from_points(self, points: np.ndarray, method: str = "alpha_shape") -> Polygon:
        """
        Create a polygon from a set of points using various methods.
        
        Args:
            points: Array of shape (n, 2) with x,y coordinates
            method: Method to use: "alpha_shape", "concave_hull", "buffer_union", "delaunay"
            
        Returns:
            Polygon or None if creation fails
        """
        if len(points) < 3:
            return None
        
        if method == "alpha_shape":
            # Use alpha shape for better concave hull
            try:
                # Alpha parameter: 0 = convex hull, larger = more detailed
                alpha = 2.0 / np.sqrt(len(points))  # Adaptive alpha based on point density
                polygon = alphashape.alphashape(points, alpha)
                return polygon if polygon.is_valid else polygon.buffer(0)
            except:
                # Fallback to buffer union if alphashape fails
                method = "buffer_union"
        
        if method == "buffer_union":
            # Create buffers around points and union them
            point_density = self.estimate_point_density(points)
            buffer_radius = point_density * 1.5  # Adaptive buffer based on density
            
            # Create buffered points
            buffered_points = [Point(p).buffer(buffer_radius) for p in points]
            
            # Union all buffers
            union = unary_union(buffered_points)
            
            # Simplify to reduce vertices
            return union.simplify(buffer_radius * 0.1)
        
        elif method == "concave_hull":
            # Use concave hull algorithm
            return self.create_concave_hull(points)
        
        elif method == "delaunay":
            # Use Delaunay triangulation with edge filtering
            return self.create_delaunay_polygon(points)
        
        else:
            # Fallback to convex hull
            try:
                hull = ConvexHull(points)
                return Polygon(points[hull.vertices])
            except:
                return None


    def estimate_point_density(self, points: np.ndarray) -> float:
        """
        Estimate the average spacing between points.
        """
        if len(points) < 2:
            return 1.0
        
        
        
        # Sample points if too many (for performance)
        if len(points) > 100:
            indices = np.random.choice(len(points), 100, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        # Calculate distances
        dist_matrix = distance_matrix(sample_points, sample_points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find nearest neighbor for each point
        min_distances = np.min(dist_matrix, axis=1)
        
        # Return median of minimum distances
        return np.median(min_distances)


    def handle_outliers(self, outlier_points: np.ndarray, buffer_radius: float = 0.5) -> list:
        """
        Handle outlier points by creating small polygons or merging nearby outliers.
        """
        if len(outlier_points) == 0:
            return []
        
        polygons = []
        
        # Try to cluster outliers with smaller parameters
        if len(outlier_points) > 1:
            mini_clustering = DBSCAN(eps=buffer_radius*2, min_samples=2)
            mini_labels = mini_clustering.fit_predict(outlier_points)
            
            for label in set(mini_labels):
                if label == -1:
                    # True single outliers
                    single_outliers = outlier_points[mini_labels == -1]
                    for point in single_outliers:
                        # Create small circular polygon
                        poly = Point(point).buffer(buffer_radius)
                        polygons.append(poly)
                else:
                    # Mini cluster of outliers
                    cluster_points = outlier_points[mini_labels == label]
                    poly = self.create_polygon_from_points(cluster_points, method="buffer_union")
                    if poly:
                        polygons.append(poly)
        else:
            # Single outlier
            poly = Point(outlier_points[0]).buffer(buffer_radius)
            polygons.append(poly)
        
        return polygons


    def create_concave_hull(self, points: np.ndarray, alpha: float = 0.1) -> Polygon:
        """
        Create concave hull using gift wrapping algorithm variant.
        """
        from scipy.spatial import Delaunay
        
        if len(points) < 3:
            return None
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Get all edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edges.add(edge)
        
        # Calculate edge lengths
        edge_lengths = {}
        for i, j in edges:
            length = np.linalg.norm(points[i] - points[j])
            edge_lengths[(i, j)] = length
        
        # Remove long edges (concavity)
        threshold = np.percentile(list(edge_lengths.values()), 100 * (1 - alpha))
        filtered_edges = [(i, j) for (i, j), length in edge_lengths.items() if length <= threshold]
        
        # Build boundary polygon from remaining edges
        # This is simplified - a full implementation would trace the boundary
        try:
            # For now, return convex hull as fallback
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
        except:
            return None


    def create_delaunay_polygon(self, points: np.ndarray, alpha_percent: float = 90) -> Polygon:
        """
        Create polygon using Delaunay triangulation with edge filtering.
        """
        from scipy.spatial import Delaunay
        
        tri = Delaunay(points)
        
        # Calculate all edge lengths
        edge_lengths = []
        for simplex in tri.simplices:
            for i in range(3):
                p1 = points[simplex[i]]
                p2 = points[simplex[(i+1)%3]]
                length = np.linalg.norm(p1 - p2)
                edge_lengths.append(length)
        
        # Set threshold for edge length
        threshold = np.percentile(edge_lengths, alpha_percent)
        
        # Create polygon from triangles with all edges below threshold
        
        valid_triangles = []
        
        for simplex in tri.simplices:
            triangle_points = points[simplex]
            
            # Check if all edges are below threshold
            valid = True
            for i in range(3):
                p1 = triangle_points[i]
                p2 = triangle_points[(i+1)%3]
                if np.linalg.norm(p1 - p2) > threshold:
                    valid = False
                    break
            
            if valid:
                valid_triangles.append(Polygon(triangle_points))
        
        if valid_triangles:
            return unary_union(valid_triangles).buffer(threshold * 0.1)
        else:
            # Fallback to convex hull
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])


    def merge_nearby_regions(self, regions: dict, merge_distance: float = 0.5) -> dict:
        """
        Merge regions that are very close to each other within the same stress level.
        """
        merged = {}
        
        for stress_level, polygons in regions.items():
            if not polygons:
                merged[stress_level] = []
                continue
            
            # Buffer and merge nearby polygons
            buffered = [p.buffer(merge_distance/2) for p in polygons]
            merged_poly = unary_union(buffered)
            
            # Erode back to original size
            result = merged_poly.buffer(-merge_distance/2)
            
            # Convert back to list of polygons
            if hasattr(result, 'geoms'):
                merged[stress_level] = list(result.geoms)
            else:
                merged[stress_level] = [result] if not result.is_empty else []
        
        return merged


    


        