# --- Cycling Art Dashboard Generator ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fitparse import FitFile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

# --- FIT Parser ---
def parse_fit_file(file_path: str) -> pd.DataFrame:
    """Parse FIT file and extract elevation, longitude, latitude, speed, and timestamp data."""
    fitfile = FitFile(file_path)
    data = {
        'altitude': [],
        'longitude': [],
        'latitude': [],
        'speed': [],
        'timestamp': []
    }
    
    # Get all records
    records = list(fitfile.get_messages('record'))
    
    for record in records:
        fields = {}
        
        # Extract all fields from the record
        try:
            # Try multiple methods to extract field data
            if hasattr(record, 'get_data'):
                record_data = record.get_data()
                for field_name, field_value in record_data.items():
                    fields[field_name] = field_value
            elif hasattr(record, 'fields'):
                for field in record.fields:
                    if hasattr(field, 'name') and hasattr(field, 'value'):
                        fields[field.name] = field.value
            else:
                # Direct iteration as fallback
                for field in record:
                    if hasattr(field, 'name') and hasattr(field, 'value'):
                        fields[field.name] = field.value
        except Exception as e:
            print(f"Warning: Could not parse record: {e}")
            continue
        
        # Handle different field name variations
        altitude = fields.get('altitude', fields.get('enhanced_altitude', np.nan))
        longitude = fields.get('longitude', fields.get('position_long', np.nan))
        latitude = fields.get('latitude', fields.get('position_lat', np.nan))
        speed = fields.get('speed', fields.get('enhanced_speed', np.nan))
        timestamp = fields.get('timestamp', np.nan)
        
        data['altitude'].append(altitude)
        data['longitude'].append(longitude)
        data['latitude'].append(latitude)
        data['speed'].append(speed)
        data['timestamp'].append(timestamp)
    
    df = pd.DataFrame(data)
    return df.dropna(subset=['altitude'])

# --- Animated Dashboard ---
def create_animated_dashboard(df: pd.DataFrame):
    """Create an animated dashboard that draws the route and elevation progressively."""
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    
    # Create grid layout
    gs = fig.add_gridspec(
        3, 7,
        height_ratios=[0.7, 0.9, 0.15],  # much more space for map and elevation
        width_ratios=[0.85, 0.05, 0.12, 0.22, 0.05, 0.01, 0.01],  # much wider plots
        hspace=0.22,  # much more vertical gap
        wspace=0.01
    )
    
    # Process data
    df['altitude_smooth'] = df['altitude'].rolling(window=60, min_periods=1, center=True).mean()
    
    # Calculate summary statistics
    total_distance = len(df) * 0.01
    total_elevation_gain = df['altitude'].diff().clip(lower=0).sum()
    max_elevation = df['altitude'].max()
    min_elevation = df['altitude'].min()
    avg_elevation = df['altitude'].mean()

    # Additional metrics
    if 'timestamp' in df.columns and len(df['timestamp'].dropna()) > 0:
        try:
            t0 = pd.to_datetime(df['timestamp'].dropna().iloc[0])
            t1 = pd.to_datetime(df['timestamp'].dropna().iloc[-1])
            total_seconds = int((t1 - t0).total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            total_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        except Exception:
            total_time_str = "N/A"
    else:
        total_time_str = "N/A"

    if 'speed' in df.columns and len(df['speed'].dropna()) > 0:
        avg_speed = df['speed'].mean() * 3.6
        max_speed = df['speed'].max() * 3.6
        avg_speed_str = f"{avg_speed:.1f} km/h"
        max_speed_str = f"{max_speed:.1f} km/h"
    else:
        avg_speed_str = "N/A"
        max_speed_str = "N/A"

    # Ride date title
    ride_date_str = None
    ride_date = None
    if 'timestamp' in df.columns and len(df['timestamp'].dropna()) > 0:
        ride_date = df['timestamp'].dropna().iloc[0]
    elif 'start_time' in df.columns and len(df['start_time'].dropna()) > 0:
        ride_date = pd.to_datetime(df['start_time'].dropna().iloc[0])
    if ride_date is not None:
        ride_date_str = ride_date.strftime('%B %d, %Y')
    else:
        ride_date_str = datetime.today().strftime('%B %d, %Y')

    # Title
    fig.text(
        0.5, 0.93, ride_date_str,
        ha='center', va='top',
        fontsize=28,
        color=(1, 1, 1, 0.82),
        fontweight='ultralight',
        fontfamily='DejaVu Sans',
        alpha=0.98,
        linespacing=1.1,
        zorder=10,
        transform=fig.transFigure
    )

    # Route map (top left)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor('none')
    ax_map.axis('off')
    
    # Elevation profile (bottom left)
    ax_elev = fig.add_subplot(gs[1, 0])
    ax_elev.set_facecolor('none')
    ax_elev.axis('off')
    
    # Summary statistics (right side)
    ax_stats = fig.add_subplot(gs[:, 3:5])
    ax_stats.set_facecolor('none')
    ax_stats.axis('off')
    
    # Process route data
    df_route = df.dropna(subset=['longitude', 'latitude'])
    if len(df_route) > 0:
        lon_min, lon_max = df_route['longitude'].min(), df_route['longitude'].max()
        lat_min, lat_max = df_route['latitude'].min(), df_route['latitude'].max()
        lon_padding = (lon_max - lon_min) * 0.1
        lat_padding = (lat_max - lat_min) * 0.1
        route_x = (df_route['longitude'] - lon_min + lon_padding) / (lon_max - lon_min + 2 * lon_padding)
        route_y = (df_route['latitude'] - lat_min + lat_padding) / (lat_max - lat_min + 2 * lat_padding)
    
    # Create coordinates for elevation
    x_coords = np.linspace(0, 100, len(df))
    y_coords = df['altitude_smooth']
    
    # Find the index and coordinates of the highest elevation point
    max_elev_idx = y_coords.values.argmax()
    max_elev_x = x_coords[max_elev_idx]
    max_elev_y = y_coords.iloc[max_elev_idx]

    # Define a KOM-style mountain shape as a custom marker (three peaks, center tallest)
    kom_vertices = [
        (-0.6, -1),    # left base
        (-0.35, 0.2),  # left peak
        (-0.15, -0.3), # left valley
        (0.0, 0.8),    # center (tallest) peak
        (0.15, -0.3),  # right valley
        (0.35, 0.4),   # right peak
        (0.6, -1),     # right base
        (-0.6, -1)     # close path
    ]
    kom_codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    kom_path = Path(kom_vertices, kom_codes)
    kom_marker = MarkerStyle(kom_path)
    
    # Set fixed axis limits for consistent scaling
    x_min, x_max = 0, 100
    y_min, y_max = df['altitude_smooth'].min(), df['altitude_smooth'].max()
    y_padding = (y_max - y_min) * 0.01
    y_min -= y_padding
    y_max += y_padding
    
    # Animation parameters
    n_frames = 150
    lines = []

    def update(frame):
        # Only clear/redraw axes that change
        ax_map.clear()
        ax_map.set_facecolor('none')
        ax_map.axis('off')
        ax_elev.clear()
        ax_elev.set_facecolor('none')
        ax_elev.axis('off')
        # Set fixed axis limits for consistent scaling
        ax_elev.set_xlim(x_min, x_max)
        ax_elev.set_ylim(y_min, y_max)
        progress = frame / n_frames
        
        # Update route map with thinner line
        if len(df_route) > 0:
            route_end_idx = int(progress * len(route_x))
            if route_end_idx > 0:
                ax_map.plot(route_x[:route_end_idx], route_y[:route_end_idx], 
                           color='#00e6d2', linewidth=2, alpha=0.9)
            ax_map.set_xlim(0, 1)
            ax_map.set_ylim(0, 1)
            ax_map.set_aspect('equal')
        
        # Update elevation profile
        end_idx = int(progress * len(x_coords))
        if end_idx > 0:
            x_plot = x_coords[:end_idx]
            y_plot = y_coords[:end_idx]
            
            for i in range(3):
                offset = i * 1.2  # slightly less offset for tighter lines
                alpha = 0.85 - (i * 0.15)  # slightly more transparent for modern look
                color_intensity = 1.0 - (i * 0.12)
                line, = ax_elev.plot(
                    x_plot + offset, y_plot,
                    color=(color_intensity, color_intensity, color_intensity),
                    linewidth=1.2,  # thinner line for tighter look
                    alpha=alpha
                )
                lines.append(line)
            # Place a small red dot directly on the max elevation point
            if isinstance(max_elev_idx, (int, np.integer)) and max_elev_idx < end_idx:
                ax_elev.plot(
                    max_elev_x, max_elev_y,
                    marker='o',
                    markersize=5,
                    color='#e10600',
                    markeredgewidth=0,
                    alpha=0.98,
                    zorder=10
                )
        
        # Update stats panel with animated text
        ax_stats.clear()
        ax_stats.set_facecolor('none')
        ax_stats.axis('off')
        
        # Calculate animated statistics based on progress
        animated_distance = total_distance * progress
        animated_elevation_gain = total_elevation_gain * progress
        animated_data_points = int(len(df) * progress)
        
        # Get current elevation stats based on progress
        if end_idx > 0:
            current_elevation_data = df['altitude'].iloc[:end_idx]
            current_max_elevation = current_elevation_data.max()
            current_min_elevation = current_elevation_data.min()
            current_avg_elevation = current_elevation_data.mean()
        else:
            current_max_elevation = 0
            current_min_elevation = 0
            current_avg_elevation = 0
        
        # Animated stats text
        animated_stats_text = (
            f"RIDE SUMMARY\n\n"
            f"Distance: {animated_distance:.1f} km\n"
            f"Elevation Gain: {animated_elevation_gain:.0f} m\n"
            f"Max Elevation: {current_max_elevation:.0f} m\n"
            f"Min Elevation: {current_min_elevation:.0f} m\n"
            f"Avg Elevation: {current_avg_elevation:.0f} m\n"
            f"Data Points: {animated_data_points:,}\n\n"
            f"---\n"
            f"Total Time: {total_time_str}\n"
            f"Avg Speed: {avg_speed_str}\n"
            f"Max Speed: {max_speed_str}"
        )
        ax_stats.text(0.08, 0.98, animated_stats_text,
            transform=ax_stats.transAxes, 
            fontsize=18, color='#eaf6fb', fontfamily='monospace',
            verticalalignment='top', linespacing=2.0, wrap=True, zorder=1)
        
        return lines
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=55, repeat=False)  # 10% slower
    
    plt.show()

# --- Main Function ---
def main():
    """Load data and create dashboard visualization."""
    # Load data
    data_files = [f for f in os.listdir('data') if f.endswith('.fit')]
    if not data_files:
        print("No .fit files found in data directory")
        return
    
    file_path = f"data/{data_files[0]}"
    print(f"Creating dashboard visualization with file: {file_path}")
    
    # Parse data
    df = parse_fit_file(file_path)
    print(f"Loaded {len(df)} data points")
    
    # Check if we have route data
    df_route = df.dropna(subset=['longitude', 'latitude'])
    if len(df_route) > 0:
        print(f"Found {len(df_route)} route points with GPS coordinates")
    else:
        print("Warning: No GPS coordinates found in the FIT file")
    
    # Create animated dashboard visualization
    create_animated_dashboard(df)

if __name__ == "__main__":
    main()