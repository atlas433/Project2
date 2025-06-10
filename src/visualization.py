import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from typing import List, Dict, Optional, Tuple, Any
import warnings
from matplotlib.widgets import TextBox


# Import our data structures
from run_pipeline import ProductionResults
from layer_geometry import LayerGeometryData, Layer, Contour
from fem_analysis import StressLayerData

class VisualizationWidget:
    def __init__(self, figsize=(16, 8)):
        self.figsize = figsize
        self.fig = None
        self.ax_geometry = None
        self._geometry_patches = []
        self.ax_stress = None
        self.slider_ax = None
        self.slider = None
        
        
        # Data storage
        self.layer_geometry_data = None
        self.stress_layer_data = None
        self.current_layer_index = 0
        self.z_heights = [0.1, 0.2, 0.3]  # Example heights for layers
      
      
        # Default limits for empty plots
        self.global_limits = None 
    
        # Visualization settings
        #self.stress_colormap = 'viridis'  # Default colormap for stress visualization
        self.stress_colormap = 'plasma'  # Enhanced colormap for better visibility

        self.stress_alpha = 0.7

        # Enhanced color scheme for stress levels
        self.contour_colors = {
            'outer': '#2E86AB',           # Blue for outer contours
            'hole': '#A23B72',            # Purple for holes
            'zone': '#888888',            # Default gray for generic zones
            'low': '#4CAF50',             # Green for low stress
            'moderate': '#FF9800',        # Orange for moderate stress  
            'high': '#F44336'             # Red for high stress
        }
        
        self.contour_alphas = {
            'outer': 0.3,
            'hole': 0.5,
            'zone': 0.6,
            'low': 0.6,
            'moderate': 0.7,
            'high': 0.8
        }

        # Stress visualization cache
        self._stress_cache = {}
        self._colorbar = None
        self.stress_norm = None

    def setup_figure(self):
        
        self._set_axis_limits()
        """Set up empty figure with two panels."""
        self.fig = plt.figure(figsize=self.figsize)
        xlim, ylim = self.global_limits
        
        gs = self.fig.add_gridspec(4, 2, height_ratios=[1, 0.05, 0.05, 0.05], width_ratios=[1, 1],
                                   hspace=0.3, wspace=0.3)

        # Geometry panel (left)
        self.ax_geometry = self.fig.add_subplot(gs[0, 0])
        self.ax_geometry.set_title('Layer Geometry & Stress Zones', fontsize=14, fontweight='bold')
        self.ax_geometry.set_xlabel('X (mm)')
        self.ax_geometry.set_ylabel('Y (mm)')
        self.ax_geometry.grid(True, alpha=0.3)
        self.ax_geometry.set_aspect('equal')
        self.ax_geometry.set_xlim(xlim)
        self.ax_geometry.set_ylim(ylim)

       
        # Stress heatmap panel (right)
        self.ax_stress = self.fig.add_subplot(gs[0, 1])
        self.ax_stress.set_title('Von Mises Stress Heatmap', fontsize=14, fontweight='bold')
        self.ax_stress.set_xlabel('X (mm)')
        self.ax_stress.set_ylabel('Y (mm)')
        self.ax_stress.grid(True, alpha=0.3)
        self.ax_stress.set_aspect('equal')
        self.ax_stress.set_xlim(xlim)
        self.ax_stress.set_ylim(ylim)

        
        # Slider for layer navigation (spans both columns)
        self.slider_ax = self.fig.add_subplot(gs[1, :])
        
        if self.layer_geometry_data and len(self.z_heights) > 1:
            self.slider = Slider(
                self.slider_ax, 'Layer Z-Height', 
                min(self.z_heights), max(self.z_heights),
                valinit=self.z_heights[0], 
                valfmt='%.3f mm'
            )
            self.slider.on_changed(self.on_slider_change)

        # GridSpec: row 2, column 0 (label) and column 1 (textbox)
        self.label_ax = self.fig.add_subplot(gs[2, 0])
        self.label_ax.axis('off')  # Hide frame
        self.label_ax.text(1.0, 0.5, 'Layer Index:', ha='right', va='center', fontsize=10, transform=self.label_ax.transAxes)

        self.textbox_ax = self.fig.add_subplot(gs[2, 1])
        self.layer_input_box = TextBox(
            self.textbox_ax,
            label='',  # empty so no extra label space
            initial=str(self.current_layer_index + 1),
            hovercolor='lightblue',
            color='lightgray'
        )
        self.layer_input_box.on_submit(self.on_textbox_submit)
        
        # Add two buttons for navigation
        # zurück button row 3, column 0
        self.back_button_ax = self.fig.add_subplot(gs[3, 0])
        self.back_button_ax.axis('on') #
        self.back_button = plt.Button(
            self.back_button_ax, 'Back', color='lightgray', hovercolor='lightblue'
        )
        self.back_button.on_clicked(lambda x: self.on_textbox_submit(str(self.current_layer_index - 1 + 1)))
        # next button row 3, column 1
        self.next_button_ax = self.fig.add_subplot(gs[3, 1])
        self.next_button_ax.axis('on')
        self.next_button = plt.Button(
            self.next_button_ax, 'Next', color='lightgray', hovercolor='lightblue'
        )
        self.next_button.on_clicked(lambda x: self.on_textbox_submit(str(self.current_layer_index + 1 + 1)))



        self._add_legend()

    def _set_axis_limits(self):
        """Set axis limits based on layer geometry."""

        all_points = []

        for layer in self.layer_geometry_data.layers:
            for contour in layer.contours:
                if len(contour.points) > 0:
                    # Convert Point objects to coordinate arrays
                    coords = contour.points
                    all_points.extend(coords)
                for child in contour.children:
                    if len(child.points) > 0:
                        # Convert Point objects to coordinate arrays
                        child_coords = child.points
                        all_points.extend(child_coords)

        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # Add some padding
            x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
            y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            
           

            self.global_limits = ((x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding))

        print(f"Global limits X = {self.global_limits[0]}, Y = {self.global_limits[1]}")
        


    def _add_legend(self):
        """Add legend for contour types and stress levels."""
        legend_elements = []
        
        # Add main contour types
        for contour_type in ['outer', 'inner']:
            if contour_type in self.contour_colors:
                color = self.contour_colors[contour_type]
                alpha = self.contour_alphas.get(contour_type, 0.7)
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, 
                                label=contour_type.replace('_', ' ').title())
                )
        
        # Add stress level types
        for stress_level in ['low', 'moderate', 'high']:
            if stress_level in self.contour_colors:
                color = self.contour_colors[stress_level]
                alpha = self.contour_alphas.get(stress_level, 0.7)
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, 
                                label=f'{stress_level.title()} Stress')
                )
        
        self.ax_geometry.legend(handles=legend_elements, loc='upper right', 
                               bbox_to_anchor=(1.0, 1.0), fontsize=9)
        
    def on_slider_change(self, val):
        """Handle slider value changes."""
        # Find closest layer to slider value
        z_target = val
        distances = [abs(z - z_target) for z in self.z_heights]
        self.current_layer_index = distances.index(min(distances))
        
        # Update both panelsplea
        self.update_visualization()

    def on_textbox_submit(self, text):
        """Handle layer index input from textbox."""
        try:
            index = int(text) - 1  # Convert to zero-based index
            if 0 <= index < len(self.z_heights):
                self.current_layer_index = index
                self.slider.set_val(self.z_heights[index])
                self.update_visualization()
            else:
                print(f"Invalid layer index: {text}. Must be between 1 and {len(self.z_heights)}.")
        except ValueError:
            print(f"Invalid input: {text}. Please enter a valid integer layer index.")

    


    def load_production_data(self, production_results: ProductionResults):
        """
        Load data directly from production workflow results.
        
        Args:
            production_results (ProductionResults): Results from production workflow
        """
        self.layer_geometry_data = production_results.layer_geometry_data
        self.stress_layer_data = production_results.stress_layer_data
        self.z_heights = [layer.z_height for layer in self.layer_geometry_data.layers]
        self.current_layer_index = 0
        

        # get min and max stress
        min_stress = production_results.stress_statistics.get('min_stress', 0)
        max_stress = production_results.stress_statistics.get('max_stress', 0)
        self.stress_norm = Normalize(vmin=min_stress, vmax=max_stress)

        # print some basic info stress statistics
        print("++" * 100)
        print(f"[Info] Stress statistics: Min = {min_stress:.2f}, Max = {max_stress:.2f}")
        print("++" * 100)

        # Clear stress cache when new data is loaded
        #self._stress_cache.clear()
        
        print(f"Loaded {len(self.layer_geometry_data.layers)} layers from production results")
        print(f"Z-height range: {min(self.z_heights):.3f} to {max(self.z_heights):.3f}")
        

    def update_visualization(self):
        """Update both geometry and stress panels for current layer."""
        if not self.layer_geometry_data or self.current_layer_index >= len(self.layer_geometry_data.layers):
            return
        
        # Check if figure is set up
        if self.fig is None or self.ax_geometry is None or self.ax_stress is None:
            return
            
        current_layer = self.layer_geometry_data.layers[self.current_layer_index]
        
        # Update geometry panel
        self._update_geometry_panel(current_layer)
        
        # Update stress panel

        current_stress_layer = self.stress_layer_data[self.current_layer_index]
        self._update_stress_panel(current_stress_layer)
        
        # Update titles with current layer info
        self.ax_geometry.set_title(
            f'Layer Geometry & Stress Zones\nZ = {current_layer.z_height:.3f} mm '
            f'(Layer {self.current_layer_index + 1}/{len(self.layer_geometry_data.layers)})',
            fontsize=14, fontweight='bold'
        )
        
        self.ax_stress.set_title(
            f'Von Mises Stress Heatmap\nZ = {current_layer.z_height:.3f} mm',
            fontsize=14, fontweight='bold'
        )
        
        # Refresh the display
        self.fig.canvas.draw()

    
    
    
    def _update_geometry_panel(self, layer: Layer):
        """Update the geometry panel with contours and stress regions."""
        # Remove previous patches
        for coll in self._geometry_patches:
            coll.remove()
        self._geometry_patches.clear()

        
        all_patches = []
        all_colors = []
        all_alphas = []
        
        # Process each contour in the layer
        for contour in layer.contours:
            # Add main contour
            if len(contour.points) >= 3:
                # Convert Point objects to coordinate arrays
                coords = contour.points
                patch = MPLPolygon(coords, closed=True)
                all_patches.append(patch)
                all_colors.append(self.contour_colors.get(contour.type, '#888888'))
                all_alphas.append(self.contour_alphas.get(contour.type, 0.5))
            
            # Add child contours (holes and stress regions)
            for child in contour.children:
                if len(child.points) >= 3:
                    # Convert Point objects to coordinate arrays
                    child_coords = child.points
                    child_patch = MPLPolygon(child_coords, closed=True)
                    all_patches.append(child_patch)
                    
                    # Determine color based on stress class or contour type
                    color = self._get_contour_color(child)
                    alpha = self._get_contour_alpha(child)
                    
                    all_colors.append(color)
                    all_alphas.append(alpha)
        
        # Create patch collection for efficient rendering
        if all_patches:
            # Group patches by alpha for efficient rendering
            alpha_groups = {}
            for patch, color, alpha in zip(all_patches, all_colors, all_alphas):
                if alpha not in alpha_groups:
                    alpha_groups[alpha] = {'patches': [], 'colors': []}
                alpha_groups[alpha]['patches'].append(patch)
                alpha_groups[alpha]['colors'].append(color)
            
            # Add each alpha group as a separate collection
            for alpha, group in alpha_groups.items():
                collection = PatchCollection(group['patches'], alpha=alpha, 
                                           facecolors=group['colors'], 
                                           edgecolors='black', linewidths=0.5)
                self.ax_geometry.add_collection(collection)
                self._geometry_patches.append(collection)

        
        # Set axis limits based on data
        #self._set_axis_limits(self.ax_geometry, layer)
        
        # Re-add legend
        #self._add_legend()
    

    def _update_stress_panel(self, layer: StressLayerData):
        """Update the stress panel with heatmap visualization."""
        z_height = layer.layer_height
        # Check cache first
        if z_height in self._stress_cache:
            return self._stress_cache[z_height]
    
        
        
        if layer.nodes is None or len(layer.nodes) == 0:
            self.ax_stress.text(0.5, 0.5, f'No Stress Data\nat Z = {layer.z_height:.3f} mm',
                               ha='center', va='center', transform=self.ax_stress.transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            return
        
        # Prepare stress data dictionary
        stress_data = {
            'coordinates': layer.nodes[:, :2],
            'stress_values': layer.nodes[:, 3],
            'stress_thresholds': (
                layer.stress_statistics.get('low_threshold', 0),
                layer.stress_statistics.get('moderate_threshold', 0)
            )
            

        }

        self._create_stress_heatmap(stress_data)
        
        # Set axis limits to match geometry panel
        #self._set_axis_limits(self.ax_stress, layer)

    def _create_stress_heatmap(self, stress_data: Dict[str, Any]):
        """Create stress heatmap visualization."""
        coordinates = stress_data['coordinates']
        stress_values = stress_data['stress_values']
        
        # Create scatter plot with color mapping
        scatter = self.ax_stress.scatter(
            coordinates[:, 0], coordinates[:, 1], 
            c=stress_values, cmap=self.stress_colormap, norm=self.stress_norm,
            alpha=self.stress_alpha, s=20, edgecolors='none'
        )

        
        
        # Handle colorbar removal and creation more safely
        try:
            if hasattr(self, '_colorbar') and self._colorbar is not None:
                self._colorbar.remove()
        except (AttributeError, KeyError, ValueError):
            pass
        
        try:
            self._colorbar = self.fig.colorbar(scatter, ax=self.ax_stress, shrink=0.8)
            self._colorbar.set_label('Von Mises Stress (Pa)', rotation=270, labelpad=20)
            
            # Add stress threshold lines if available
            if 'stress_thresholds' in stress_data:
                low_max, mod_max = stress_data['stress_thresholds']
                
                # Add threshold indicators in the colorbar
                self._colorbar.ax.axhline(y=low_max, color='orange', linestyle='--', linewidth=2, alpha=0.8)
                self._colorbar.ax.axhline(y=mod_max, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add threshold labels
                self._colorbar.ax.text(1.1, low_max, 'Low/Mod', va='center', fontsize=8, color='orange')
                self._colorbar.ax.text(1.1, mod_max, 'Mod/High', va='center', fontsize=8, color='red')
        
        
        except Exception as e:
            print(f"Warning: Failed to create colorbar: {e}")
            self._colorbar = None

        

    def _get_contour_color(self, contour: Contour) -> str:
        """Get color for a contour based on its type and properties."""
        # Check if it's a stress zone with stress_class property
        if contour.properties and 'stress_class' in contour.properties:
            stress_class = contour.properties['stress_class']
            return self.contour_colors.get(stress_class, self.contour_colors.get('zone', '#888888'))
        
        # Fall back to contour type
        return self.contour_colors.get(contour.type, '#888888')
    
    def _get_contour_alpha(self, contour: Contour) -> float:
        """Get alpha value for a contour based on its type and properties."""
        # Check if it's a stress zone with stress_class property
        if contour.properties and 'stress_class' in contour.properties:
            stress_class = contour.properties['stress_class']
            return self.contour_alphas.get(stress_class, self.contour_alphas.get('zone', 0.6))
        
        # Fall back to contour type
        return self.contour_alphas.get(contour.type, 0.7)

    def show(self):
        """Display the visualization widget."""
        if not self.layer_geometry_data:
            print("No data loaded. Use load_production_data(), load_workflow_data(), or load_direct_data() first.")
            return
        
        self.setup_figure()
        self.update_visualization()
        plt.show()


def visualize_production_results(results: ProductionResults):
        widget = VisualizationWidget()
        widget.load_production_data(results)
        widget.show()


def main():

    from run_pipeline import run_standard_production
    

    # Run standard production
    results = run_standard_production()

    # Visu
    visualize_production_results(results)

if __name__ == "__main__":
    print("[🔍] Starting visualization...")
    main()
    print("[✅] Visualization completed successfully.")
    