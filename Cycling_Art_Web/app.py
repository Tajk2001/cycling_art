from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import numpy as np
import pandas as pd
from fitparse import FitFile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Golden ratio for text scaling
GOLDEN_RATIO = 1.618033988749895

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates using Haversine formula."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

def calculate_total_distance(df):
    """Calculate total distance from GPS coordinates."""
    df_route = df.dropna(subset=['longitude', 'latitude'])
    if len(df_route) < 2:
        return 0.0
    
    # Filter out GPS outliers (coordinates that are clearly wrong)
    # Valid latitude range: -90 to 90, valid longitude range: -180 to 180
    df_route = df_route[
        (df_route['latitude'] >= -90) & (df_route['latitude'] <= 90) &
        (df_route['longitude'] >= -180) & (df_route['longitude'] <= 180)
    ]
    
    if len(df_route) < 2:
        return 0.0
    
    # If we have too many GPS points, sample them to avoid over-counting
    # GPS devices often record points every few seconds, which can lead to inflated distances
    if len(df_route) > 1000:
        # Sample every 10th point for distance calculation
        df_route = df_route.iloc[::10].reset_index(drop=True)
    
    total_distance = 0.0
    for i in range(1, len(df_route)):
        lat1 = df_route.iloc[i-1]['latitude']
        lon1 = df_route.iloc[i-1]['longitude']
        lat2 = df_route.iloc[i]['latitude']
        lon2 = df_route.iloc[i]['longitude']
        
        # Skip if coordinates are the same (no movement)
        if lat1 == lat2 and lon1 == lon2:
            continue
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        
        # Filter out unrealistic distances (more than 1km between consecutive points)
        if distance > 1.0:
            continue
            
        total_distance += distance
    
    return total_distance

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set modern matplotlib style with soothing colors
plt.style.use('default')
plt.rcParams['figure.facecolor'] = '#f8fafc'
plt.rcParams['axes.facecolor'] = '#f8fafc'
plt.rcParams['savefig.facecolor'] = '#f8fafc'
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#475569'
plt.rcParams['xtick.color'] = '#64748b'
plt.rcParams['ytick.color'] = '#64748b'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'

# --- FIT Parser (from test_features.py) ---
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
        
        # Convert GPS coordinates from semicircles to degrees if needed
        if longitude is not None and not np.isnan(longitude):
            # FIT files store coordinates in semicircles, convert to degrees
            longitude = longitude * (180.0 / 2**31)
        if latitude is not None and not np.isnan(latitude):
            # FIT files store coordinates in semicircles, convert to degrees
            latitude = latitude * (180.0 / 2**31)
        
        data['altitude'].append(altitude)
        data['longitude'].append(longitude)
        data['latitude'].append(latitude)
        data['speed'].append(speed)
        data['timestamp'].append(timestamp)
    
    df = pd.DataFrame(data)
    return df.dropna(subset=['altitude'])

def create_animated_dashboard(df: pd.DataFrame) -> str:
    """Create an animated dashboard that draws the route and elevation progressively."""
    fig = plt.figure(figsize=(24, 14), dpi=100)  # Increased from (20, 12) to (24, 14)
    fig.patch.set_facecolor('#f8fafc')  # Light, soothing background
    
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
    total_distance = calculate_total_distance(df)
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

    # Title - more minimalist with golden ratio scaling
    base_font_size = 24
    title_font_size = int(base_font_size * GOLDEN_RATIO)
    fig.text(
        0.5, 0.93, ride_date_str,
        ha='center', va='top',
        fontsize=title_font_size,  # Scaled with golden ratio
        color='#1e293b',
        fontweight='200',  # Lighter weight
        fontfamily='SF Pro Display',
        alpha=0.9,
        linespacing=1.1,
        zorder=10,
        transform=fig.transFigure
    )

    # Route map (top left)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor('#f8fafc')  # Light, soothing background
    ax_map.axis('off')
    
    # Elevation profile (bottom left)
    ax_elev = fig.add_subplot(gs[1, 0])
    ax_elev.set_facecolor('#f8fafc')  # Light, soothing background
    ax_elev.axis('off')
    
    # Summary statistics (right side) - more minimalist
    ax_stats = fig.add_subplot(gs[:, 3:5])
    ax_stats.set_facecolor('#f8fafc')  # Light, soothing background
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
        ax_map.set_facecolor('#f8fafc')  # Light, soothing background
        ax_map.axis('off')
        ax_elev.clear()
        ax_elev.set_facecolor('#f8fafc')  # Light, soothing background
        ax_elev.axis('off')
        # Set fixed axis limits for consistent scaling
        ax_elev.set_xlim(x_min, x_max)
        ax_elev.set_ylim(y_min, y_max)
        progress = frame / n_frames
        end_idx = int(progress * len(x_coords))
        
        # Draw route progressively
        if len(df_route) > 0 and end_idx > 0:
            route_end_idx = min(end_idx, len(df_route))
            route_x_plot = route_x.iloc[:route_end_idx]
            route_y_plot = route_y.iloc[:route_end_idx]
            
            # Single clean line with modern color
            line, = ax_map.plot(
                route_x_plot, route_y_plot,
                color='#3b82f6',
                linewidth=2.5,
                alpha=0.9
            )
            lines.append(line)
            
            # Add start and end markers
            if route_end_idx > 0:
                try:
                    # Start marker
                    start_dot = ax_map.scatter(route_x.iloc[0], route_y.iloc[0], s=80, color='#10b981', alpha=0.8, zorder=5, edgecolors='white', linewidth=1.5)
                    lines.append(start_dot)
                    
                    if route_end_idx > 1:
                        # End marker
                        end_dot = ax_map.scatter(route_x.iloc[route_end_idx-1], route_y.iloc[route_end_idx-1], s=80, color='#ef4444', alpha=0.8, zorder=5, edgecolors='white', linewidth=1.5)
                        lines.append(end_dot)
                except:
                    pass
        
        # Draw elevation progressively
        if end_idx > 0:
            elev_end_idx = min(end_idx, len(x_coords))
            x_plot = x_coords[:elev_end_idx]
            y_plot = y_coords[:elev_end_idx]
            
            # Main elevation line
            line, = ax_elev.plot(x_plot, y_plot, color='#3b82f6', linewidth=2.5, alpha=0.9)
            lines.append(line)
        
        # Update stats panel with minimalist design
        ax_stats.clear()
        ax_stats.set_facecolor('#f8fafc')  # Light, soothing background
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
        
        # Minimalist stats text - only key metrics with golden ratio scaling
        base_stats_font = 16
        stats_font_size = int(base_stats_font * GOLDEN_RATIO)
        animated_stats_text = (
            f"RIDE SUMMARY\n\n"
            f"Distance: {animated_distance:.1f} km\n"
            f"Elevation Gain: {animated_elevation_gain:.0f} m\n"
            f"Max Elevation: {current_max_elevation:.0f} m\n"
            f"Data Points: {animated_data_points:,}"
        )
        ax_stats.text(0.08, 0.98, animated_stats_text,
            transform=ax_stats.transAxes, 
            fontsize=stats_font_size, color='#1e293b', fontfamily='Inter',
            verticalalignment='top', linespacing=2.0, wrap=True, zorder=1)
        
        return lines
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=55, repeat=False)
    
    # Save animation as GIF using temporary file
    temp_gif_path = tempfile.mktemp(suffix='.gif')
    try:
        anim.save(temp_gif_path, writer='pillow', fps=15, dpi=100)
        
        # Read the saved GIF and convert to base64
        with open(temp_gif_path, 'rb') as f:
            gif_data = f.read()
        
        # Convert to base64
        img_str = base64.b64encode(gif_data).decode()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_gif_path):
            try:
                os.unlink(temp_gif_path)
            except:
                pass
    
    plt.close()
    return img_str

def create_elevation_plot(df: pd.DataFrame) -> str:
    """Create a modern elevation plot and return as base64 string."""
    df['altitude_smooth'] = df['altitude'].rolling(window=60, min_periods=1, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    
    # Create x-axis as percentage of distance
    x_coords = np.linspace(0, 100, len(df))
    y_coords = df['altitude_smooth']
    
    # Find the index and coordinates of the highest elevation point
    max_elev_idx = y_coords.values.argmax()
    max_elev_x = x_coords[max_elev_idx]
    max_elev_y = y_coords.iloc[max_elev_idx]
    
    # Create shadow effect
    shadow_offset = 2
    ax.plot(x_coords + shadow_offset, y_coords, color='#cbd5e1', linewidth=2.5, alpha=0.4, zorder=1)
    
    # Main elevation line
    ax.plot(x_coords, y_coords, color='#3b82f6', linewidth=2.5, alpha=0.9, zorder=2)
    
    # Add max elevation marker
    ax.scatter(max_elev_x, max_elev_y, s=100, color='#ef4444', alpha=0.8, zorder=4, edgecolors='white', linewidth=2)
    
    # Minimal styling - let the visualization speak for itself
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Remove all spines and grid
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8fafc')
    img_buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

def create_route_map(df: pd.DataFrame) -> str:
    """Create a modern route map and return as base64 string."""
    # Process route data
    df_route = df.dropna(subset=['longitude', 'latitude'])
    
    if len(df_route) == 0:
        return ""
    
    # Create the plot with bigger size
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#f8fafc')  # Bigger size
    ax.set_facecolor('#f8fafc')
    
    # Calculate route bounds
    lon_min, lon_max = df_route['longitude'].min(), df_route['longitude'].max()
    lat_min, lat_max = df_route['latitude'].min(), df_route['latitude'].max()
    lon_padding = (lon_max - lon_min) * 0.1
    lat_padding = (lat_max - lat_min) * 0.1
    
    # Normalize coordinates
    route_x = (df_route['longitude'] - lon_min + lon_padding) / (lon_max - lon_min + 2 * lon_padding)
    route_y = (df_route['latitude'] - lat_min + lat_padding) / (lat_max - lat_min + 2 * lat_padding)
    
    # Plot the route with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(route_x)))
    
    # Main route line
    ax.plot(route_x, route_y, color='#3b82f6', linewidth=3, alpha=0.8, zorder=2)
    
    # Add start and end markers
    if len(route_x) > 0:
        # Start marker (green)
        ax.scatter(route_x.iloc[0], route_y.iloc[0], s=150, color='#10b981', alpha=0.8, zorder=4, edgecolors='white', linewidth=2)
        
        # End marker (red)
        ax.scatter(route_x.iloc[-1], route_y.iloc[-1], s=150, color='#ef4444', alpha=0.8, zorder=4, edgecolors='white', linewidth=2)
    
    # Minimal styling - let the visualization speak for itself
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Remove all spines and grid
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8fafc')
    img_buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/test')
def test():
    """Simple test endpoint."""
    return jsonify({"message": "Flask app is working!", "status": "success"})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle FIT file upload and return CSV data with visualizations."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.fit'):
        return jsonify({"error": "Please upload a .fit file"}), 400
    
    temp_path = None
    try:
        # Save the uploaded file temporarily
        temp_fd, temp_path = tempfile.mkstemp(suffix='.fit')
        file.save(temp_path)
        
        # Parse the FIT file
        df = parse_fit_file(temp_path)
        
        if len(df) == 0:
            return jsonify({"error": "No valid data found in the FIT file"}), 400
        
        # Calculate some basic stats
        total_distance = calculate_total_distance(df)
        stats = {
            "total_points": len(df),
            "altitude_min": float(df['altitude'].min()),
            "altitude_max": float(df['altitude'].max()),
            "altitude_avg": float(df['altitude'].mean()),
            "gps_points": len(df.dropna(subset=['longitude', 'latitude'])),
            "total_distance": float(total_distance),
            "filename": file.filename
        }
        
        # Create visualizations
        elevation_plot = create_elevation_plot(df)
        route_map = create_route_map(df)
        animated_dashboard = create_animated_dashboard(df)
        
        # Save to CSV
        csv_filename = f"ride_data_{file.filename.replace('.fit', '')}.csv"
        csv_path = os.path.join(os.getcwd(), csv_filename)
        df.to_csv(csv_path, index=False)
        
        return jsonify({
            "message": "FIT file processed successfully!",
            "stats": stats,
            "csv_file": csv_filename,
            "elevation_plot": elevation_plot,
            "route_map": route_map,
            "animated_dashboard": animated_dashboard
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.route('/download/<filename>')
def download_csv(filename):
    """Download the generated CSV file."""
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 