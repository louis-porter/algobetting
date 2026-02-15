import matplotlib.pyplot as plt
import seaborn as sns

class StyleConfig:
    """Centralized style configuration for consistent visualization styling"""
    
    # Color settings
    COLORS = {
        'palette': 'deep',
        'edge_color': 'white',
        'grid_color': 'gray',
        'axis_color': 'gray',
        'text_color': 'black',
        'arrow_color': 'gray'
    }
    
    # Font settings
    FONTS = {
        'family': 'monospace',
        'title_size': 24,
        'label_size': 16,
        'text_size': 10,
        'legend_title_size': 14,
        'legend_size': 12,
        'quadrant_size': 14
    }
    
    # Plot settings
    PLOT = {
        'figure_size': (20, 16),
        'scatter_size': 120,
        'alpha': 0.9,
        'line_width': 0.8,
        'dpi': 300
    }
    
    # Text adjustment settings
    TEXT_ADJUST = {
        'expand_points': (1.5, 1.5),
        'force_points': (0.5, 0.8),
        'force_text': (0.8, 0.8),
        'arrowprops': dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
        'autoalign': 'xy',
        'only_move': {'points': 'xy', 'text': 'xy'},
        'lim': 300
    }
    
    # Text box styling
    TEXT_BOX = {
        'boxstyle': "round,pad=0.3",
        'facecolor': 'white',
        'alpha': 0.85,
        'linewidth': 1.5
    }
    
    # Quadrant label styling
    QUADRANT_BOX = {
        'facecolor': 'white',
        'alpha': 0.5,
        'boxstyle': 'round,pad=0.5'
    }
    
    @staticmethod
    def apply_style():
        """Apply all matplotlib style settings at once"""
        plt.rcParams['font.family'] = StyleConfig.FONTS['family']
        plt.rcParams['figure.dpi'] = StyleConfig.PLOT['dpi']
        plt.rcParams['savefig.dpi'] = StyleConfig.PLOT['dpi']
        plt.rcParams['savefig.bbox'] = 'tight'
    
    @staticmethod
    def get_color_palette(n_colors):
        """Get consistent color palette"""
        return sns.color_palette(StyleConfig.COLORS['palette'], n_colors)


# Usage in your existing code:
# from style_config import StyleConfig
# StyleConfig.apply_style()
# 
# Then use StyleConfig.FONTS['title_size'], StyleConfig.PLOT['figure_size'], etc.