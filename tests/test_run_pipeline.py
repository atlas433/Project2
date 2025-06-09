import pytest
from unittest.mock import patch
from pathlib import Path
from run_pipeline import IntegratedSlicingWorkflow, ProductionConfig, LayerGeometryData



def test_workflow_initialization():
    config = ProductionConfig(
        stl_path="data/geo_test3.stl",
        rst_path="data/file_test3.rst",
        output_dir="production_output",
        layer_height=0.1,
        simplify_factor=0.1
    )
    
    workflow = IntegratedSlicingWorkflow(config)
    
    # Assertions
    assert workflow.config == config
    assert isinstance(workflow.slicer, STLSlicer)
    assert isinstance(workflow.fem_analyzer, FEMAnalysis)
    assert isinstance(workflow.geometry_data, LayerGeometryData)
    assert Path(config.output_dir).exists()

    print("[✅] Workflow initialized successfully")
