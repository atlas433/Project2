import pytest
import numpy as np
from shapely.geometry import Polygon
# Import the FEMAnalysis class from the module
from fem_analysis import FEMAnalysis

@pytest.fixture
def rst_path():
    return r"C:\DEV\Project2\data\file_test3.rst"

@pytest.fixture
def fem_analysis(rst_path):
    return FEMAnalysis(rst_path=rst_path)


def test_initialization(fem_analysis):
    assert fem_analysis.num_classes == 3
    assert fem_analysis.eps == 1
    assert fem_analysis.min_samples == 5
    assert fem_analysis.min_cluster_area == 1
    assert fem_analysis.min_nodes == 150
    assert fem_analysis.max_allowed == 0.5
    
def test_load_model(fem_analysis):
    model = fem_analysis.load_model()
    assert model is not None # Ensure model is loaded
    assert hasattr(model, 'metadata') # Ensure model has metadata
    assert hasattr(model.metadata, 'meshed_region') # Ensure metadata has meshed_region

def test_get_length_unit_and_scale(fem_analysis):
    unit, scale = fem_analysis.get_length_unit_and_scale()
    assert unit is not None  # Ensure unit is retrieved
    assert isinstance(unit, str)  # Ensure unit is a string
    assert unit in ['mm', 'cm', 'm']  # Check against expected units
    assert isinstance(scale, float)  # Ensure scale is a float
    assert scale > 0  # Ensure scale is positive


def test_compute_von_mises_per_node(fem_analysis):
    vm_array = fem_analysis.compute_von_mises_per_node()
    assert vm_array is not None  # Ensure VM stress array is computed
    assert vm_array.ndim == 2  # Ensure VM stress array is 2D: [nodes, stress]
    assert vm_array.shape[0] == len(fem_analysis.model.metadata.meshed_region.nodes)  # Ensure VM stress array matches node count
    print("Von Mises stress computation test passed ✅")

def test_extract_node_coordinates(fem_analysis):
    nodes = fem_analysis.extract_node_coordinates()
    assert nodes is not None  # Ensure nodes are extracted
    assert nodes.shape[1] == 4  # Ensure nodes have id, 3 coordinates (x, y, z)
    assert nodes.shape[0] == len(fem_analysis.model.metadata.meshed_region.nodes)  # Ensure at least one node is extracted
    print("Node extraction test passed ✅")

def test_get_node_data_with_stress(fem_analysis):
    combined_array = fem_analysis.get_node_data_with_stress()
    assert combined_array is not None  # Ensure combined array is built
    assert combined_array.shape[1] == 5  # Ensure combined array has id, x, y, z, stress 
    assert combined_array.shape[0] == len(fem_analysis.model.metadata.meshed_region.nodes)  # Ensure at least one entry in the combined array
    print("Combined array test passed ✅")

    # Examine the first node 
    # [[0, 0.0, 0.0, 0.1, 50.0], ...]

def test_calculate_stress_thresholds(fem_analysis):

    # Use known inputs
    streckgrenze = 600  # MPa
    sicherheitsfaktor = 2.0

    # Expected result
    sigma_zul = streckgrenze / sicherheitsfaktor  # 300
    expected_low = 0.3 * sigma_zul                # 90.0
    expected_mod = 0.7 * sigma_zul                # 210.0

    thresholds = fem_analysis.calculate_stress_thresholds(streckgrenze=streckgrenze, sicherheitsfaktor=sicherheitsfaktor)
    assert thresholds is not None  # Ensure thresholds are calculated
    assert isinstance(thresholds, tuple)
    assert len(thresholds) == 2
    assert thresholds[0] == pytest.approx(expected_low, rel=1e-2)  # Check low threshold
    assert thresholds[1] == pytest.approx(expected_mod, rel=1e-2)  # Check moderate threshold


# Testing the clustering functionality

def test_get_nodes_in_slice(fem_analysis):

    # Output example: [X, Y, Z, VonMises]
    # [[0.0, 0.0, 0.1, 50.0],
    #  [0.5, 0.5, 0.1, 60.0], ...]
    

    z_target = 0.1      # Target z-coordinate for the slice
    tolerence = 0.25    # Elemnent size is 0.5, so tolerance is half of that, 0.25 mm

    slice_nodes = fem_analysis.get_nodes_in_slice(z=z_target, thickness=tolerence)

   

    assert slice_nodes is not None  # Ensure nodes are retrieved
    # 1. Check output shape
    assert isinstance(slice_nodes, np.ndarray)
    assert slice_nodes.shape[1] == 4  # [X, Y, Z, VonMises]

    # 2. Check all Z-values are within allowed tolerance
    z_coords = slice_nodes[:, 2]
    z_diff = np.abs(z_coords - z_target)
    assert np.all(z_diff <= fem_analysis.max_allowed)

    assert len(slice_nodes) > 0  # Ensure at least one node is found
    print("Nodes in slice test passed ✅")



@pytest.fixture
def slice_nodes(fem_analysis):
    z_target = 0.1      # Target z-coordinate for the slice
    tolerence = 0.25    # Element size is 0.5, so tolerance is half of that, 0.25 mm
    return fem_analysis.get_nodes_in_slice(z=z_target, thickness=tolerence)


def test_classify_and_cluster_stress_regions(fem_analysis, slice_nodes):
    # Extract XY coordinates and stress values from the slice nodes
    xy_points = slice_nodes[:, :2]  # Extract X and Y coordinates
    stress_values = slice_nodes[:, 3]  # Extract Von Mises stress values

    regions = fem_analysis.classify_and_cluster_stress_regions(xy_points, stress_values)

    assert isinstance(regions, dict)
    assert all(k in regions for k in ["low", "moderate", "high"])
    total_regions = sum(len(polys) for polys in regions.values())
    print(f"Total polygons created: {total_regions}")
    assert total_regions >= 0  # Could be 0 if data is too uniform


def test_stress_region_extraction(fem_analysis, slice_nodes):
    """
    Test the full region extraction pipeline:
    1. Get nodes in a slice
    2. Classify & cluster based on stress
    3. Clean overlap
    4. Return regions and slice nodes
    """

    z = 0.1  # Slice height
    thickness = 0.25

    result = fem_analysis.generate_slice_stress_regions(z=z, thickness=thickness)

    assert result is not None, "No regions returned (possibly too few nodes in slice)."

    regions = result["regions"]
    slice_nodes = result["slice_nodes"]

    # Basic shape checks
    assert isinstance(regions, dict)
    assert isinstance(slice_nodes, np.ndarray)
    assert slice_nodes.shape[1] == 4  # Should contain X, Y, Z, VonMises

    # Ensure at least one zone has a polygon
    assert any(len(polys) > 0 for polys in regions.values()), "No stress zones were detected."

    print(f"[✅] Stress region extraction passed. Regions: { {k: len(v) for k, v in regions.items()} }")


