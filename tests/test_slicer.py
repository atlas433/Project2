import pytest
import numpy as np
from shapely.geometry import Polygon
import trimesh
# Import the FEMAnalysis class from the module
from slicer import STLSlicer
from layer_geometry import LayerGeometryData

@pytest.fixture
def stl_path():
    return r"C:\DEV\Project2\data\geo_test3.stl"

@pytest.fixture
def json_path():
    return r"C:\DEV\Project2\data\geometry.json"

@pytest.fixture
def slicer(stl_path):
    slicer = STLSlicer(layer_height=0.1, tolerance=1e-6)
    return slicer

@pytest.fixture
def mesh(slicer, stl_path):
    """
    Fixture to load the mesh from the STL file.
    """
    mesh = slicer.load_stl(stl_path)
    return mesh


def test_load_stl(slicer, stl_path):
    """
    Test loading an STL file.
    """
    slicer.load_stl(stl_path)
    assert slicer.mesh is not None, "Mesh should be loaded successfully"
    assert isinstance(slicer.mesh, trimesh.Trimesh), "Loaded mesh should be a valid trimesh object"
    assert slicer.z_bounds is not None, "Z bounds should be set after loading the mesh"
    assert len(slicer.z_bounds) == 2, "Z bounds should contain two values (min and max)"
    print(f"Z bounds: {slicer.z_bounds}")


def test_generate_z_levels(slicer, stl_path):	
    """
    Test generating Z levels for slicing.
    """
    slicer.load_stl(stl_path)
    z_levels = slicer.generate_z_levels()
    assert isinstance(z_levels, np.ndarray), "Z levels should be a numpy array"
    assert len(z_levels) > 0, "Z levels array should not be empty"
    print(f"Generated Z levels: {len(z_levels)}")

def test_slice_at_z_level(slicer, stl_path):
    """
    Test slicing the mesh at a specific Z level.
    """
    slicer.load_stl(stl_path)
    z_levels = slicer.generate_z_levels()
    
    # Test slicing at the first Z level
    first_z_level = z_levels[0]
    polygons = slicer.slice_at_z(first_z_level)
    
    assert isinstance(polygons, list), "Sliced polygons should be returned as a list"
    assert len(polygons) > 0, "There should be at least one polygon for the first Z level"
    
    # Check if the polygons are valid
    for poly in polygons:
        assert isinstance(poly, Polygon), "Each sliced polygon should be a shapely Polygon"
    
    print(f"Number of polygons at Z level {first_z_level}: {len(polygons)}")


def test_read_json(slicer, stl_path, json_path):
    """
    Test reading a JSON file.
    """
    slicer.load_stl(stl_path)
    json_data = slicer.read_json(json_path)
    
    # json_data has LayerGeometryData structure
    # Example structure:
    # {
    #     "layers": [
    #         {
    #             "z_height": 0.1,
    #             "contours": [
    #                 {


    assert hasattr(json_data, "layers"), "JSON data should have a 'layers' attribute"
    assert isinstance(json_data, LayerGeometryData), "Deserialized object is not of type LayerGeometryData"
    assert len(json_data.layers) > 0, "No layers found in the loaded JSON data"

    
   
