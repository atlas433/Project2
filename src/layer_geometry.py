import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class Contour:
    """Represents a contour with points and optional children."""
    id: str
    type: str
    points: np.ndarray
    properties: Optional[Dict[str, Any]] = None
    children: Optional[List['Contour']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contour to dictionary format for JSON serialization."""
        result = {
            "id": self.id,
            "type": self.type,
            "points": self.points.tolist()

        }
        
        if self.properties:
            result["properties"] = self.properties
            
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contour':
        """Create Contour from dictionary format."""
        points = np.array(data["points"], dtype=float)
        
        children = None
        if "children" in data:
            children = [cls.from_dict(child_data) for child_data in data["children"]]
        
        return cls(
            id=data["id"],
            type=data["type"],
            points=points,
            properties=data.get("properties"),
            children=children
        )

@dataclass
class Layer:
    """Represents a layer with z-height and contours."""
    z_height: float
    contours: List[Contour]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary format for JSON serialization."""
        return {
            "z_height": self.z_height,
            "contours": [contour.to_dict() for contour in self.contours]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Create Layer from dictionary format."""
        contours = [Contour.from_dict(contour_data) for contour_data in data["contours"]]
        return cls(z_height=data["z_height"], contours=contours)
    
   


@dataclass
class LayerGeometryData:
    """Main class representing the complete layer geometry data structure."""
    layers: List[Layer]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "layers": [layer.to_dict() for layer in self.layers]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerGeometryData':
        """Create LayerGeometryData from dictionary format."""
        layers = [Layer.from_dict(layer_data) for layer_data in data["layers"]]
        return cls(layers=layers)

class LayerGeometryHandler:
    """Handler class for reading and writing LayerGeometryData JSON files."""
    
    @staticmethod
    def read_from_file(file_path: str) -> LayerGeometryData:
        """
        Read LayerGeometryData from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            LayerGeometryData object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If required fields are missing
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return LayerGeometryData.from_dict(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
    
    @staticmethod
    def write_to_file(geometry_data: LayerGeometryData, file_path: str, indent: int = 2) -> None:
        """
        Write LayerGeometryData to a JSON file.
        
        Args:
            geometry_data: LayerGeometryData object to write
            file_path: Path where to save the JSON file
            indent: JSON indentation level (default: 2)
        """
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(geometry_data.to_dict(), file, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def validate_structure(data: Dict[str, Any]) -> bool:
        """
        Validate that the data structure matches the expected LayerGeometryData format.
        
        Args:
            data: Dictionary to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check if main structure exists
            if "layers" not in data or not isinstance(data["layers"], list):
                return False
            
            for layer in data["layers"]:
                # Check layer structure
                if not isinstance(layer, dict):
                    return False
                if "z_height" not in layer or "contours" not in layer:
                    return False
                if not isinstance(layer["contours"], list):
                    return False
                
                # Check contour structure
                for contour in layer["contours"]:
                    if not isinstance(contour, dict):
                        return False
                    required_fields = ["id", "type", "points"]
                    if not all(field in contour for field in required_fields):
                        return False
                    if not isinstance(contour["points"], list):
                        return False
                    
                    # Check points structure
                    for point in contour["points"]:
                        if not isinstance(point, list) or len(point) != 2:
                            return False
                        if not all(isinstance(coord, (int, float)) for coord in point):
                            return False
            
            return True
        except Exception:
            return False
    
   
    
    @staticmethod
    def add_layer(geometry_data: LayerGeometryData, z_height: float, contours: List[Contour] = None) -> None:
        """
        Add a new layer to the geometry data.
        
        Args:
            geometry_data: LayerGeometryData object to modify
            z_height: Z-height of the new layer
            contours: List of contours for the layer (default: empty list)
        """
        if contours is None:
            contours = []
        
        new_layer = Layer(z_height=z_height, contours=contours)
        geometry_data.layers.append(new_layer)
    
    @staticmethod
    def get_layer_by_height(geometry_data: LayerGeometryData, z_height: float, tolerance: float = 1e-6) -> Optional[Layer]:
        """
        Get a layer by its z-height.
        
        Args:
            geometry_data: LayerGeometryData object to search
            z_height: Z-height to search for
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Layer object if found, None otherwise
        """
        for layer in geometry_data.layers:
            if abs(layer.z_height - z_height) <= tolerance:
                return layer
        return None
    
    @staticmethod
    def get_contour_by_id(layer: Layer, contour_id: str) -> Optional[Contour]:
        """
        Get a contour by its ID within a layer.
        
        Args:
            layer: Layer object to search
            contour_id: ID of the contour to find
            
        Returns:
            Contour object if found, None otherwise
        """
        def search_contour(contours: List[Contour], target_id: str) -> Optional[Contour]:
            for contour in contours:
                if contour.id == target_id:
                    return contour
                if contour.children:
                    result = search_contour(contour.children, target_id)
                    if result:
                        return result
            return None
        
        return search_contour(layer.contours, contour_id)
