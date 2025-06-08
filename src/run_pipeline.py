import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

# Import our custom modules
from slicer import STLSlicer
from fem_analysis import FEMAnalysis, StressLayerData
from layer_geometry import LayerGeometryHandler, LayerGeometryData


@dataclass
class ProductionConfig:
    """Configuration class for production parameters."""
    # STL Processing
    stl_file: str = r"C:\DEV\Project2\data\geo_test3.stl"
    json_file: str = r"C:\DEV\Project2\data\geometry.json"
    layer_height: float = 0.1
    merge_overlapping: bool = True
    
    # FEM Analysis
    rst_file: str = r"C:\DEV\Project2\data\file_test3.rst"
    num_stress_classes: int = 3
    
    # Clustering Parameters
    eps: float = 0.3                   # DBSCAN radius [mm] (Search radius)
    min_samples: int = 3             # Cluster forms with at least 3 points
    min_cluster_area: float = 3   # Minimum polygon area = 1 mm² --> it filters out tiny polygons
    min_nodes: int = 150
    max_allowed: float = 0.5
    
    
    
    # Production Limits
    zone_per_slice: int = 3
    max_layers: Optional[int] = None
    z_range: Optional[Tuple[float, float]] = None
    
    # Output Settings
    output_dir: str = r"C:\DEV\Project2\data\production_output"
    save_intermediate: bool = False
    export_json: bool = True
    export_visualization: bool = False


@dataclass
class ProductionResults:
    """Results container for production workflow."""
    config: ProductionConfig
    stl_bounds: Dict[str, Tuple[float, float]]
    fem_bounds: Dict[str, Tuple[float, float]]
    stress_statistics: Dict[str, Any]
    layer_count: int
    total_processing_time: float
    layer_geometry_data:List[LayerGeometryData]
    stress_layer_data: List[StressLayerData]
    output_files: List[str]
    warnings: List[str]
    errors: List[str]


class IntegratedSlicingWorkflow:
    """
    Main production workflow class that orchestrates STL slicing and stress analysis.
    
    This class provides a complete production-ready workflow for processing
    engineering data with integrated stress analysis and geometric slicing.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize the production workflow.
        
        Args:
            config (ProductionConfig): Production configuration parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        self.slicer = None
        self.fem_analyzer = None
        self.geometry_handler = LayerGeometryHandler()
        self.warnings = []
        self.errors = []
        self.geometry_data = LayerGeometryData(layers=[])
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)


    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the production workflow."""
        logger = logging.getLogger("ProductionWorkflow")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (create after output_dir is set)
        if hasattr(self, 'output_dir'):
            log_file = self.output_dir / "production.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def initialize_modules(self) -> bool:
        """
        Initialize the STL slicer and FEM analyzer modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing production modules...")
            
            # Initialize STL Slicer
            self.logger.info(f"Loading JSON file: {self.config.json_file}")
            self.slicer = STLSlicer(
                layer_height=self.config.layer_height,
                tolerance=1e-6
            )

            if not os.path.exists(self.config.json_file):
                raise FileNotFoundError(f"JSON file not found: {self.config.json_file}")
            
            self.geometry_data = self.geometry_handler.read_from_file(self.config.json_file) # Read JSON File

            self.logger.info("STL file loaded successfully")
            
            # Initialize FEM Analyzer
            self.logger.info(f"Loading FEM result file: {self.config.rst_file}")
            if not os.path.exists(self.config.rst_file):
                raise FileNotFoundError(f"RST file not found: {self.config.rst_file}")
            
            self.fem_analyzer = FEMAnalysis(
                rst_path=self.config.rst_file,
                num_classes=self.config.num_stress_classes,
                eps=self.config.eps,
                min_samples=self.config.min_samples,
                min_cluster_area=self.config.min_cluster_area,
                min_nodes=self.config.min_nodes,
                max_allowed=self.config.max_allowed
            )
            self.logger.info("FEM analyzer initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize modules: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            return False
     
    def process_integrated_slicing(self) -> LayerGeometryData:
        """
        Perform integrated STL slicing with stress analysis.
        
        Returns:
            LayerGeometryData: Complete layer geometry with stress integration
        """
        self.logger.info("Starting integrated slicing with stress analysis...")
        
        try:
            # Perform stress-integrated slicing
            layer_geometry_data = self.slicer.slice(
                        dt=self.geometry_data,
                        fem_analyzer=self.fem_analyzer,
                        merge_overlapping=self.config.merge_overlapping,
                        layer_thickness=self.config.layer_height,
                        zone_per_slice=self.config.zone_per_slice
                    )        
            
            self.logger.info(f"Integrated slicing completed: {len(layer_geometry_data.layers)} layers generated")
            return layer_geometry_data
            
        except Exception as e:
            error_msg = f"Integrated slicing failed: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            raise

    def export_results(self, layer_geometry_data: LayerGeometryData, 
                      stress_layer_data: List[StressLayerData]) -> List[str]:
        """
        Export all results to various formats.
        
        Args:
            layer_geometry_data (LayerGeometryData): Layer geometry data
            stress_layer_data (List[StressLayerData]): Stress analysis data
            
        Returns:
            List[str]: List of created output files
        """
        output_files = []
        


        try:
            # Export integrated layer geometry
            if self.config.export_json:
                geometry_file = self.output_dir / "production_layer_geometry.json"
                self.geometry_handler.write_to_file(layer_geometry_data, str(geometry_file))
                output_files.append(str(geometry_file))
                self.logger.info(f"Layer geometry exported to: {geometry_file}")
            
            
            # Export stress analysis data
            if self.config.export_json and stress_layer_data:
                stress_file = self.output_dir / "production_stress_analysis.json"

                stress_data = [layer_data.to_summary_dict() for layer_data in stress_layer_data]


                with open(stress_file, 'w') as f:
                    json.dump(stress_data, f, indent=2, default=str)



                output_files.append(str(stress_file))
                self.logger.info(f"Stress analysis exported to: {stress_file}")
            
            return output_files
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            return output_files
    
    def get_stl_bounds(self, geometry_data: LayerGeometryData) -> Dict[str, Tuple[float, float]]:
        """
        Compute STL bounds (min/max for x, y, z) from LayerGeometryData.
        
        Args:
            geometry_data (LayerGeometryData): The geometry data with layers and contours.
        
        Returns:
            Dict[str, Tuple[float, float]]: Bounds dictionary with keys 'x', 'y', 'z'.
        """
        all_points = []

        for layer in geometry_data.layers:
            z = layer.z_height
            for contour in layer.contours:
                # Assume contour.points is a list of (x, y) tuples
                for x, y in contour.points:
                    all_points.append((x, y, z))

        if not all_points:
            raise ValueError("No geometry points found in layer data.")

        all_points_np = np.array(all_points)
        bounds = {
            'x': (float(np.min(all_points_np[:, 0])), float(np.max(all_points_np[:, 0]))),
            'y': (float(np.min(all_points_np[:, 1])), float(np.max(all_points_np[:, 1]))),
            'z': (float(np.min(all_points_np[:, 2])), float(np.max(all_points_np[:, 2])))
        }
        print("==" * 50)
        print(f"STL Bounds: {bounds}")
        print("==" * 50)


        return bounds
    
    def process_stress_analysis(self, z_heights: List[float]) -> List[StressLayerData]:
        """
        Perform detailed stress analysis for specified Z heights.
        
        Args:
            z_heights (List[float]): List of Z heights to analyze
            
        Returns:
            List[StressLayerData]: Stress analysis results for each layer
        """
        self.logger.info(f"Processing stress analysis for {len(z_heights)} layers...")
        
        try:
            stress_results = self.fem_analyzer.slice_with_stress_analysis(
                z_heights=z_heights,
                thickness=self.config.layer_height,
                zone_per_slice=self.config.zone_per_slice
            )
            
            stress_layer_data = []
            for result in stress_results:
                stress_layer_data.append(result["layer_data"])
            
            self.logger.info("Stress analysis completed successfully")
            return stress_layer_data
            
        except Exception as e:
            error_msg = f"Stress analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            raise
    
    def run_production_workflow(self) -> ProductionResults:
        """
        Execute the complete production workflow.
        
        Returns:
            ProductionResults: Complete results of the production workflow
        """
        start_time = time.time()
        self.logger.info("Starting production workflow...")
        
        try:
            # Step 1: Initialize modules
            if not self.initialize_modules():
                raise RuntimeError("Module initialization failed")
            
            # Step 2: Process integrated slicing
            layer_geometry_data = self.process_integrated_slicing()
            
            # Step 3: Process detailed stress analysis
            z_heights = [layer.z_height for layer in layer_geometry_data.layers]
            stress_layer_data = self.process_stress_analysis(z_heights)
            

            
            # Step 4: Export results
            output_files = self.export_results(layer_geometry_data, stress_layer_data)

            # Step 4: Get spatial bounds and stress statistics
            

            stl_bounds_xyz = self.get_stl_bounds(layer_geometry_data)
            stl_bounds = {
                'z': stl_bounds_xyz['z'],
                'x': stl_bounds_xyz['x'],
                'y': stl_bounds_xyz['y']
            }

            fem_bounds = self.fem_analyzer.get_fem_model_bounds()
            stress_statistics = self.fem_analyzer.get_stress_statistics()

            print("X" * 50)
            #print(f"Stressthreshold: {self.fem_analyzer.stress_thresshold}")

            # Step 6: Create results object
            
            total_time = time.time() - start_time
            results = ProductionResults(
                config=self.config,
                stl_bounds=stl_bounds,
                fem_bounds=fem_bounds,
                stress_statistics=stress_statistics,
                layer_count=len(layer_geometry_data.layers),
                total_processing_time=total_time,
                layer_geometry_data=layer_geometry_data,
                stress_layer_data=stress_layer_data,
                output_files=output_files,
                warnings=self.warnings,
                errors=self.errors
            )

            
            
            
            # Step 7: Generate production report
            report_file = self.generate_production_report(results)
            if report_file:
                results.output_files.append(report_file)
            
            self.logger.info(f"Production workflow completed successfully in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            error_msg = f"Production workflow failed: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            raise

    def generate_production_report(self, results: ProductionResults) -> str:
        """
        Generate a comprehensive production report.
        
        Args:
            results (ProductionResults): Production results
            
        Returns:
            str: Path to the generated report file
        """
        report_file = self.output_dir / "production_report.md"
        
        try:
            with open(report_file, 'w') as f:
                f.write("# Production Workflow Report\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Configuration
                f.write("## Configuration\n\n")
                f.write(f"- **STL File:** {results.config.stl_file}\n")
                f.write(f"- **RST File:** {results.config.rst_file}\n")
                f.write(f"- **Layer Height:** {results.config.layer_height} mm\n")
                f.write(f"- **Stress Classes:** {results.config.num_stress_classes}\n")
                f.write(f"- **Zone Limit per Slice:** {results.config.zone_per_slice}\n\n")
                
                # Results Summary
                f.write("## Results Summary\n\n")
                f.write(f"- **Total Layers:** {results.layer_count}\n")
                f.write(f"- **Processing Time:** {results.total_processing_time:.2f} seconds\n")
                f.write(f"- **Output Files:** {len(results.output_files)}\n")
                f.write(f"- **Warnings:** {len(results.warnings)}\n")
                f.write(f"- **Errors:** {len(results.errors)}\n\n")
                
                # Spatial Bounds
                f.write("## Spatial Bounds\n\n")
                f.write("### STL Bounds\n")
                f.write(f"- **Z Range:** {results.stl_bounds['z'][0]:.3f} to {results.stl_bounds['z'][1]:.3f} mm\n\n")
                
                f.write("### FEM Bounds\n")
                for axis, (min_val, max_val) in results.fem_bounds.items():
                    f.write(f"- **{axis.upper()} Range:** {min_val:.3f} to {max_val:.3f} mm\n")
                f.write("\n")
                
                # Stress Statistics
                f.write("## Stress Statistics\n\n")
                stats = results.stress_statistics
                f.write(f"- **Min Stress:** {stats['min_stress']:.2f} MPa\n")
                f.write(f"- **Max Stress:** {stats['max_stress']:.2f} MPa\n")
                f.write(f"- **Mean Stress:** {stats['mean_stress']:.2f} MPa\n")
                f.write(f"- **Std Deviation:** {stats['std_stress']:.2f} MPa\n")
                f.write(f"- **Total Nodes:** {stats['num_nodes']}\n\n")
                
                # Output Files
                f.write("## Output Files\n\n")
                for file_path in results.output_files:
                    f.write(f"- `{file_path}`\n")
                f.write("\n")
                
                # Warnings and Errors
                if results.warnings:
                    f.write("## Warnings\n\n")
                    for warning in results.warnings:
                        f.write(f"- {warning}\n")
                    f.write("\n")
                
                if results.errors:
                    f.write("## Errors\n\n")
                    for error in results.errors:
                        f.write(f"- {error}\n")
                    f.write("\n")
                
                # Layer Details
                f.write("## Layer Details\n\n")
                for i, layer in enumerate(results.layer_geometry_data.layers[:10]):  # Show first 10 layers
                    total_contours = len(layer.contours)
                    total_children = sum(len(contour.children) for contour in layer.contours)
                    stress_children = sum(
                        len([child for child in contour.children 
                            if child.properties and 'stress_class' in child.properties])
                        for contour in layer.contours
                    )
                    
                    f.write(f"### Layer {i+1} (Z={layer.z_height:.3f})\n")
                    f.write(f"- **Main Contours:** {total_contours}\n")
                    f.write(f"- **Total Children:** {total_children}\n")
                    f.write(f"- **Stress Children:** {stress_children}\n")
                    f.write(f"- **Holes:** {total_children - stress_children}\n\n")
                
                if len(results.layer_geometry_data.layers) > 10:
                    f.write(f"... and {len(results.layer_geometry_data.layers) - 10} more layers\n\n")
            
            self.logger.info(f"Production report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
            return ""



def run_standard_production() -> ProductionResults:
    """
    Run the standard production workflow with default settings.
    
    Returns:
        ProductionResults: Results of the production workflow
    """
    config = ProductionConfig()
    workflow = IntegratedSlicingWorkflow(config)
    return workflow.run_production_workflow()
        

        
    






# Example usage and testing
if __name__ == "__main__":
    
    print("=== Production Module for Integrated STL Slicing and Stress Analysis ===\n")
    
    # Check if required files exist
    #required_files = ["Konstruktion1.stl", "test2.rst", "layer_geometry_handler.py"]
    required_files = [
    "data/geo_test3.stl",
    "data/file_test3.rst",
    "src/layer_geometry.py"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please ensure all required files are in the current directory.")
        exit(1)
    
    try:
        print("1. Running Standard Production Workflow...")
        print("-" * 50)
        
        # Run standard production
        results = run_standard_production()
        
        print(f"\n✅ Production completed successfully!")
        print(f"📊 Processed {results.layer_count} layers in {results.total_processing_time:.2f} seconds")
        print(f"📁 Generated {len(results.output_files)} output files")
        
        if results.warnings:
            print(f"⚠️  {len(results.warnings)} warnings generated")
        
        if results.errors:
            print(f"❌ {len(results.errors)} errors encountered")
        
        print(f"\n📋 Output files:")
        for file_path in results.output_files:
            print(f"   - {file_path}")
        
        print(f"\n📈 Stress Statistics:")
        stats = results.stress_statistics
        print(f"   - Stress Range: {stats['min_stress']:.2f} - {stats['max_stress']:.2f} MPa")
        print(f"   - Mean Stress: {stats['mean_stress']:.2f} ± {stats['std_stress']:.2f} MPa")
        print(f"   - Total Nodes: {stats['num_nodes']:,}")


    except Exception as e:
        print(f"\n❌ Production workflow failed: {str(e)}")
        print("\nPlease check:")
        print("1. All required files are present")
        print("2. Required Python packages are installed:")
        print("   - trimesh")
        print("   - shapely")
        print("   - ansys-dpf-core")
        print("   - scikit-learn")
        print("   - numpy")
        print("3. The STL and RST files are valid")

