import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from typing import List, Tuple, Optional, Union
import time
from concurrent.futures import ThreadPoolExecutor
from layer_geometry import Layer, Contour, LayerGeometryData, LayerGeometryHandler
from fem_analysis import FEMAnalysis


class STLSlicer:
    """
    A class-based STL slicer that converts 3D mesh data into structured layer geometry.
    
    This slicer processes STL files and generates LayerGeometryData objects with proper
    hierarchy of contours, including outer boundaries and holes.
    """
    
    def __init__(self, layer_height: float = 0.2, tolerance: float = 1e-6):
        """
        Initialize the STL slicer.
        
        Args:
            layer_height (float): Height of each layer in mm
            tolerance (float): Geometric tolerance for operations
        """
        self.layer_height = layer_height
        self.tolerance = tolerance
        self.mesh = None
        self.z_bounds = None
        self.processing_time = 0.0
        self.rotation_matrix = np.eye(4)  # Default: I

        # LayerGeometryData instance to hold results
        self.layer_geometry_data = LayerGeometryData(layers=[])
         
    def load_stl(self, file_path: str) -> None:
        """
        Load an STL file using trimesh.
        
        Args:
            file_path (str): Path to the STL file
            
        Raises:
            FileNotFoundError: If the STL file doesn't exist
            ValueError: If the file cannot be loaded as a valid mesh
        """
        try:
            self.mesh = trimesh.load_mesh(file_path)
            if not isinstance(self.mesh, trimesh.Trimesh):
                raise ValueError("Loaded file is not a valid triangular mesh")
            
            # Debug: Print raw bounds before rotation
            print(f"Mesh bounds: {self.mesh.bounds}")

            # Get Z bounds for slicing
            self.z_bounds = (self.mesh.bounds[0][2], self.mesh.bounds[1][2])
            print(f"Loaded mesh with Z bounds: {self.z_bounds[0]:.3f} to {self.z_bounds[1]:.3f}")
            
        except Exception as e:
            raise ValueError(f"Failed to load STL file '{file_path}': {str(e)}")
        
    def generate_z_levels(self, start_z: Optional[float] = None, end_z: Optional[float] = None) -> np.ndarray:
        """
        Generate Z levels for slicing based on layer height.
        
        Args:
            start_z (float, optional): Starting Z coordinate. If None, uses mesh minimum Z
            end_z (float, optional): Ending Z coordinate. If None, uses mesh maximum Z
            
        Returns:
            np.ndarray: Array of Z coordinates for slicing
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_stl() first.")
        
        if start_z is None:
            start_z = self.z_bounds[0] #+ self.layer_height / 2
        if end_z is None:
            end_z = self.z_bounds[1]
        
        # Generate Z levels
        z_levels = np.arange(start_z, end_z + self.layer_height, self.layer_height) # shape (N,)
        return z_levels 
    
    def slice_mesh(self, merge_overlapping: bool = True) -> LayerGeometryData:
        """
        Slice the entire mesh and return structured LayerGeometryData.
        
        Args:
            merge_overlapping (bool): Whether to merge overlapping polygons in each layer
            
        Returns:
            LayerGeometryData: Structured layer geometry data
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_stl() first.")
        
        # Generate Z levels
        z_levels = self.generate_z_levels()
        print(f"Slicing at {len(z_levels)} levels from Z={z_levels[0]:.3f} to Z={z_levels[-1]:.3f}")
        
        layers = []
        
        for layer_id, z_height in enumerate(z_levels, 1):
            # Slice at this Z height
            polygons = self.slice_at_z(z_height)
            
            if not polygons:
                continue
            
            # Optionally merge overlapping polygons
            if merge_overlapping:
                polygons = self.merge_overlapping_polygons(polygons)
            
            # Convert polygons to structured contours
            contours = self.polygons_to_contours(polygons, z_height, layer_id)
            
            if contours:
                layer = Layer(z_height=float(z_height), contours=contours)
                layers.append(layer)
                print(f"Layer {layer_id}: Z={z_height:.3f}, {len(contours)} contours")
        
        print(f"Slicing complete: {len(layers)} layers generated")
        return LayerGeometryData(layers=layers)
    

    def read_json(self, json_file_path: str) -> LayerGeometryData:
        """
        Read LayerGeometryData from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            LayerGeometryData: Parsed layer geometry data
        """
        if not json_file_path:
            raise ValueError("JSON file path must be provided")
        
        self.layer_geometry_data = LayerGeometryHandler.read_from_file(json_file_path)
        return self.layer_geometry_data
    
    def slice_at_z(self, z_height: float) -> List[Polygon]:
        """
        Slice the mesh at a specific Z height and return Shapely polygons.
        
        Args:
            z_height (float): Z coordinate to slice at
            
        Returns:
            List[Polygon]: List of polygons representing the cross-section
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_stl() first.")
        
        try:
            # Create a plane at the specified Z height
            plane_origin = [0, 0, z_height]
            plane_normal = [0, 0, 1]
            
            # Slice the mesh
            slice_2d = self.mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if slice_2d is None:
                return []
            
            # Convert Path3D to polygons
            polygons = []
            
            # Process each entity in the Path3D object
            for entity in slice_2d.entities:
                try:
                    # Get discrete points for this entity
                    discrete_points = entity.discrete(slice_2d.vertices)
                    
                    if len(discrete_points) >= 3:
                        # Convert to 2D by removing Z coordinate
                        points_2d = discrete_points[:, :2]
                        
                        # Create Shapely polygon
                        poly = Polygon(points_2d)
                        if poly.is_valid and poly.area > self.tolerance:
                            polygons.append(poly)
                            
                except Exception as e:
                    print(f"Warning: Failed to process entity: {str(e)}")
                    continue
            
            return polygons
            
        except Exception as e:
            print(f"Warning: Failed to slice at Z={z_height}: {str(e)}")
            return []
        
    
    def polygons_to_contours(self, polygons: List[Polygon], z_height: float, layer_id: int) -> List[Contour]:
        """
        Convert Shapely polygons to structured Contour objects with proper hierarchy.
        
        Args:
            polygons (List[Polygon]): List of Shapely polygons
            z_height (float): Z height of the layer
            layer_id (int): Layer identifier for contour naming
            
        Returns:
            List[Contour]: List of structured contour objects
        """
        if not polygons:
            return []
        
        contours = []
        contour_counter = 1
        
        # Process each polygon
        for i, polygon in enumerate(polygons):
            if not polygon.is_valid or polygon.area <= self.tolerance:
                continue
            
            # Extract exterior boundary
            exterior_coords = list(polygon.exterior.coords)
            if len(exterior_coords) < 4:  # Need at least 3 unique points + closing point
                continue
            
            # Create contour ID
            contour_id = f"L{layer_id}_C{contour_counter}"
            contour_counter += 1
            
            # Convert coordinates to Point objects for consistency with template
            #exterior_points = [Point(coord[0], coord[1]) for coord in exterior_coords]
            exterior_points = np.array(exterior_coords, dtype=float)

            
            # Create children contours for holes
            children = []
            for j, interior in enumerate(polygon.interiors):
                interior_coords = list(interior.coords)
                if len(interior_coords) >= 4:
                    hole_id = f"L{layer_id}_C{contour_counter}"
                    contour_counter += 1
                    
                    #interior_points = [Point(coord[0], coord[1]) for coord in interior_coords]
                    interior_points = np.array(interior_coords, dtype=float)

                    hole_contour = Contour(
                        id=hole_id,
                        type="hole",
                        points=interior_points
                    )
                    children.append(hole_contour)
            
            # Create main contour
            main_contour = Contour(
                id=contour_id,
                type="outer",
                points=exterior_points,
                children=children
            )
            
            contours.append(main_contour)
        
        return contours
    


    def slice(self, dt: LayerGeometryData, fem_analyzer: FEMAnalysis,
            layer_thickness: float = 0.1, zone_per_slice: Optional[int] = 3,
            merge_overlapping: bool = True
            ) -> LayerGeometryData:
        """
        Parallele Version: Enrich existing geometry (from JSON) with stress-based contours.
        """

        start_time = time.time()

        def process_layer(args):
            layer_id, layer = args
            z_height = layer.z_height
            stl_polygons = []

            for contour in layer.contours:
                try:
                    poly = Polygon(contour.points)
                    if poly.is_valid and poly.area > 1e-6:
                        stl_polygons.append(poly)
                except Exception as e:
                    print(f"Warning: Invalid contour in Layer {layer_id}: {e}")

            if not stl_polygons:
                return None

            if merge_overlapping:
                stl_polygons_merged = self.merge_overlapping_polygons(stl_polygons)
            else:
                stl_polygons_merged = stl_polygons

            print(f"Processing Layer {layer_id} at Z={z_height:.3f}")

            try:
                stress_result = fem_analyzer.generate_slice_stress_regions(
                    z=z_height, thickness=layer_thickness,
                )
                stress_regions = stress_result["regions"]
            except Exception as e:
                print(f"Warning: Failed to extract stress regions at Z={z_height:.3f}: {str(e)}")
                stress_regions = {"low": [], "moderate": [], "high": []}

            layer_contours = self._create_hierarchical_contours(
                stl_polygons_merged, stress_regions, z_height, layer_id, zone_per_slice
            )

            if layer_contours:
                layer_obj = Layer(z_height=float(z_height), contours=layer_contours)
                total_stress_children = sum(len(contour.children) for contour in layer_contours)
                print(f"Layer {layer_id}: Z={z_height:.3f}, {len(layer_contours)} main contours, "
                    f"{total_stress_children} stress child contours")
                return layer_obj
            return None

        #updated_layers = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_layer, enumerate(dt.layers, 1)))
            updated_layers = [layer for layer in results if layer is not None]

        self.processing_time = time.time() - start_time
        print(f"Parallel stress-integrated slicing complete in {self.processing_time:.2f} seconds: ")
        print(f"Stress-integrated slicing complete: {len(updated_layers)} layers generated")

        return LayerGeometryData(layers=updated_layers)
    
    def _create_hierarchical_contours(self, stl_polygons: List[Polygon], 
                                    stress_regions: dict, z_height: float, 
                                    layer_id: int, zone_per_slice: Optional[int] = None) -> List[Contour]:
        """
        Create hierarchical contours where stress regions become children of STL contours.
        
        Args:
            stl_polygons (List[Polygon]): Main geometry polygons from STL slicing
            stress_regions (dict): Stress-classified regions {"low": [...], "moderate": [...], "high": [...]}
            z_height (float): Z height of the layer
            layer_id (int): Layer identifier for contour naming
            zone_per_slice (int, optional): Maximum number of stress zones per slice
            
        Returns:
            List[Contour]: List of structured contour objects with stress children
        """
        if not stl_polygons:
            return []
        
        contours = []
        contour_counter = 1
        
        # Process each STL polygon as a main contour
        for i, stl_polygon in enumerate(stl_polygons):
            if not stl_polygon.is_valid or stl_polygon.area <= self.tolerance:
                continue
            
            # Extract exterior boundary
            exterior_coords = list(stl_polygon.exterior.coords)
            if len(exterior_coords) < 4:  # Need at least 3 unique points + closing point
                continue
            
            # Create main contour ID
            main_contour_id = f"L{layer_id}_C{contour_counter}"
            contour_counter += 1
            
            # Convert coordinates to Point objects
            #exterior_points = [Point(coord[0], coord[1]) for coord in exterior_coords]
            exterior_points = np.array(exterior_coords)
            
            # Create children contours for STL holes (original functionality)
            stl_children = []
            for j, interior in enumerate(stl_polygon.interiors):
                interior_coords = list(interior.coords)
                if len(interior_coords) >= 4:
                    hole_id = f"L{layer_id}_C{contour_counter}"
                    contour_counter += 1
                    
                    #interior_points = [Point(coord[0], coord[1]) for coord in interior_coords]
                    interior_points = np.array(interior_coords)
                    hole_contour = Contour(
                        id=hole_id,
                        type="inner",
                        points=interior_points
                    )
                    stl_children.append(hole_contour)
            
            # Create stress-based child contours
            stress_children = self._create_stress_child_contours(
                stress_regions, stl_polygon, z_height, layer_id, contour_counter, zone_per_slice
            )
            
            # Update contour counter
            contour_counter += len(stress_children)
            
            # Combine all children (STL holes + stress regions)
            all_children = stl_children + stress_children
            
            # Create main contour with all children
            main_contour = Contour(
                id=main_contour_id,
                type="outer",
                points=exterior_points,
                children=all_children
            )
            
            contours.append(main_contour)
        
        return contours
    
    def _create_stress_child_contours(self, stress_regions: dict, parent_polygon: Polygon,
                                    z_height: float, layer_id: int, 
                                    start_counter: int, zone_per_slice: Optional[int] = None) -> List[Contour]:
        """
        Create child contours for stress regions that intersect with the parent polygon.
        
        Args:
            stress_regions (dict): Stress-classified regions {"low": [...], "moderate": [...], "high": [...]}
            parent_polygon (Polygon): Parent STL polygon to intersect with
            z_height (float): Z height of the layer
            layer_id (int): Layer identifier
            start_counter (int): Starting counter for contour IDs
            zone_per_slice (int, optional): Maximum number of stress zones per slice
            
        Returns:
            List[Contour]: List of stress-based child contours
        """
        stress_children = []
        contour_counter = start_counter
        
        # Collect all stress contours with their areas for zone limiting
        all_stress_contours = []
        
        # Process each stress level
        for stress_level in ["high", "moderate", "low"]:  # Process high stress first (priority)
            if stress_level not in stress_regions:
                continue
                
            stress_polygons = stress_regions[stress_level]
            
            for stress_poly in stress_polygons:
                try:
                    # Find intersection with parent polygon
                    intersection = parent_polygon.intersection(stress_poly)
                    
                    if intersection.is_empty:
                        continue
                    
                    # Handle different intersection types

                    
                    intersected_polygons = []

                    
                    if isinstance(intersection, Polygon):
                        intersected_polygons.append(intersection)
                    elif isinstance(intersection, MultiPolygon):
                        intersected_polygons.extend(intersection.geoms)
                                    
                    # Create contours for intersected stress regions
                    for intersected_poly in intersected_polygons:
                        if hasattr(intersected_poly, 'exterior'):
                            exterior_coords = list(intersected_poly.exterior.coords)
                            if len(exterior_coords) >= 4:
                                stress_contour_id = f"L{layer_id}_S{stress_level.upper()}{contour_counter}"
                                contour_counter += 1
                                
                                #stress_points = [Point(coord[0], coord[1]) for coord in exterior_coords]
                                stress_points = np.array(exterior_coords)
                                stress_contour = Contour(
                                    id=stress_contour_id,
                                    type=f"zone",  # Use "zone" type to match template
                                    points=stress_points,
                                    properties={"stress_class": stress_level}
                                )
                                
                                # Store contour with area and priority for zone limiting
                                priority = {"high": 3, "moderate": 2, "low": 1}[stress_level]
                                all_stress_contours.append({
                                    "contour": stress_contour,
                                    "area": intersected_poly.area,
                                    "priority": priority,
                                    "stress_level": stress_level
                                })
                                
                except Exception as e:
                    print(f"Warning: Failed to process stress region intersection: {str(e)}")
                    continue
        
        # Apply zone limiting if specified
        if zone_per_slice is not None and len(all_stress_contours) > zone_per_slice:
            # Sort by priority (high stress first) then by area (largest first)
            all_stress_contours.sort(key=lambda x: (-x["priority"], -x["area"]))
            
            # Keep only the top zones
            selected_contours = all_stress_contours[:zone_per_slice]
            stress_children = [item["contour"] for item in selected_contours]
            
            # Log zone limiting action
            print(f"Zone limiting applied: reduced from {len(all_stress_contours)} to {zone_per_slice} stress zones at Z={z_height:.3f}")
        else:
            # No zone limiting, use all contours
            stress_children = [item["contour"] for item in all_stress_contours]
        
        return stress_children
    
    
    def merge_overlapping_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Merge overlapping polygons using Shapely's unary_union.
        
        Args:
            polygons (List[Polygon]): List of potentially overlapping polygons
            
        Returns:
            List[Polygon]: List of merged polygons
        """
        if not polygons:
            return []
        
        try:
            # Use unary_union to merge overlapping polygons
            merged = unary_union(polygons)
            
            # Handle different return types
            if isinstance(merged, Polygon):
                return [merged] if merged.is_valid and merged.area > self.tolerance else []
            elif isinstance(merged, MultiPolygon):
                return [poly for poly in merged.geoms if poly.is_valid and poly.area > self.tolerance]
            else:
                return []
                
        except Exception as e:
            print(f"Warning: Failed to merge polygons: {str(e)}")
            return polygons



def save_stl_to_json():
    # Write JSON
    print("Save STL in JSON")
    geo = LayerGeometryHandler()
    slicer = STLSlicer(
                layer_height=0.1,
                tolerance=1e-6
            )
    
    stl_file: str = r"data/geo_test3.stl"
    # STL-only slicing
    stl_only_file = r"data/geometry.json"
    slicer.load_stl(stl_file)   # Read STL File
    stl_only_data = slicer.slice_mesh(merge_overlapping=False)
    geo.write_to_file(stl_only_data, str(stl_only_file))



if __name__ == "__main__":
    #save_stl_to_json()
    print("STL slicing and JSON saving completed.")
    print("Run tests to verify functionality.")
    
    