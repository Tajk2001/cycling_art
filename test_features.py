# --- Cycling Art Dashboard Generator ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from fitparse import FitFile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Lightweight Theming & Config ---
THEME = {
    'bg': '#0f1115',
    'card': '#151a23',
    'text': '#E6EAF0',
    'muted': '#9AA5B1',
    'accent': '#00e6d2',
    'alert': '#e10600',
    'line': '#D7DCE1',
}

# Optional export toggles (set env vars to enable)
SAVE_ANIMATION = os.getenv('SAVE_ANIMATION', '0') == '1'  # mp4/gif export requires ffmpeg/pillow
SAVE_FRAME = os.getenv('SAVE_FRAME', '0') == '1'          # always safe PNG export of final frame
OUTPUTS_DIR = 'outputs'

def _ensure_montserrat() -> bool:
    """Ensure Montserrat font is available; try to download if missing.

    Returns True if Montserrat was found or successfully registered.
    Falls back silently if network is unavailable; matplotlib will then
    substitute the closest available sans-serif font.
    """
    try:
        import matplotlib.font_manager as fm
        # If it's already available, we're done
        if any('Montserrat' == f.name for f in fm.fontManager.ttflist):
            return True

        font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(font_dir, exist_ok=True)

        # Register any existing local Montserrat TTFs
        registered = False
        for fname in os.listdir(font_dir):
            if fname.lower().startswith('montserrat') and fname.lower().endswith('.ttf'):
                try:
                    fm.fontManager.addfont(os.path.join(font_dir, fname))
                    registered = True
                except Exception:
                    pass
        if registered:
            return True

        # Attempt to download regular and bold weights (reliable source repo)
        urls = {
            'Regular': 'https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf',
            'Bold': 'https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf',
        }
        local_paths = []
        for weight, url in urls.items():
            local = os.path.join(font_dir, f'Montserrat-{weight}.ttf')
            if not os.path.exists(local):
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, local)
                except Exception:
                    continue
            local_paths.append(local)
        for p in local_paths:
            try:
                fm.fontManager.addfont(p)
                registered = True
            except Exception:
                pass
        return registered
    except Exception:
        return False

have_montserrat = _ensure_montserrat()

# Global matplotlib polish
plt.rcParams.update({
    'figure.facecolor': THEME['bg'],
    'axes.facecolor': 'none',
    'savefig.facecolor': THEME['bg'],
    'font.family': ['Montserrat'] if have_montserrat else ['DejaVu Sans', 'sans-serif'],
    'text.color': THEME['text'],
    'axes.edgecolor': (1, 1, 1, 0.06),
    'axes.linewidth': 0.8,
    'figure.autolayout': False,
})

# --- Utilities ---
def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3

def format_km(km: float) -> str:
    return f"{km:.1f} km"

def format_m(meters: float) -> str:
    return f"{meters:.0f} m"

def format_speed(kmh: float) -> str:
    return f"{kmh:.1f} km/h"

def add_card(ax, pad=0.02, radius=10, facecolor=None, alpha=0.96, shadow=True):
    """Draw a rounded 'card' on the axes and return the visible patch.

    Returning the card patch allows us to clip subsequent artists to the
    rounded shape so content never bleeds outside the box.
    """
    if facecolor is None:
        facecolor = THEME['card']
    # Shadow (draw first so it's below)
    if shadow:
        shadow_patch = FancyBboxPatch(
            (0 + pad + 0.01, 0 + pad - 0.01), 1 - 2 * pad, 1 - 2 * pad,
            boxstyle=f"round,pad={pad},rounding_size={radius}",
            transform=ax.transAxes, linewidth=0,
            facecolor='black', alpha=0.25, zorder=0
        )
        ax.add_patch(shadow_patch)
    # Main card
    card_patch = FancyBboxPatch(
        (0 + pad, 0 + pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=1, facecolor=facecolor,
        edgecolor=(1, 1, 1, 0.06), alpha=alpha, zorder=0.5
    )
    ax.add_patch(card_patch)
    return card_patch

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

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

    # Convert lat/lon from FIT semicircles to degrees if needed
    # Some devices store GPS coordinates as semicircles (±2^31). Detect and convert.
    if 'latitude' in df.columns and df['latitude'].notna().any():
        if float(np.nanmax(np.abs(df['latitude'].to_numpy()))) > 90.0:
            df['latitude'] = df['latitude'].astype(float) * (180.0 / (2**31))
    if 'longitude' in df.columns and df['longitude'].notna().any():
        if float(np.nanmax(np.abs(df['longitude'].to_numpy()))) > 180.0:
            df['longitude'] = df['longitude'].astype(float) * (180.0 / (2**31))

    return df.dropna(subset=['altitude'])

def _prepare_metrics(df: pd.DataFrame):
    # Smooth elevation for visual stability
    df = df.copy()
    df['altitude_smooth'] = df['altitude'].rolling(window=60, min_periods=1, center=True).mean()

    # Time metrics
    if 'timestamp' in df.columns and len(df['timestamp'].dropna()) > 1:
        t0 = pd.to_datetime(df['timestamp'].dropna().iloc[0])
        t1 = pd.to_datetime(df['timestamp'].dropna().iloc[-1])
        total_seconds = max(1, int((t1 - t0).total_seconds()))
        hh, mm, ss = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        total_time_str = f"{hh:02}:{mm:02}:{ss:02}"
    else:
        total_seconds = 0
        total_time_str = 'N/A'

    # Distance via haversine using available lat/lon
    lat = df['latitude'].to_numpy() if 'latitude' in df.columns else np.full(len(df), np.nan)
    lon = df['longitude'].to_numpy() if 'longitude' in df.columns else np.full(len(df), np.nan)
    mask = ~np.isnan(lat) & ~np.isnan(lon)
    dist_increments = np.zeros(len(df))
    idx = np.arange(len(df))
    valid_idx = idx[mask]
    if len(valid_idx) > 1:
        lat_v, lon_v = lat[mask], lon[mask]
        d = haversine_m(lat_v[:-1], lon_v[:-1], lat_v[1:], lon_v[1:])
        # Map back into the full array (increment at the destination index)
        dist_increments[valid_idx[1:]] = d
    df['distance_m'] = np.cumsum(dist_increments)
    total_distance_km = df['distance_m'].iloc[-1] / 1000.0 if len(df) else 0.0

    # Elevation metrics
    elev_diff = np.diff(df['altitude_smooth'].to_numpy(), prepend=df['altitude_smooth'].iloc[0])
    # Ignore tiny noise (< 1 m)
    elev_gain = np.where(elev_diff > 1.0, elev_diff, 0).sum()
    max_elev = float(df['altitude_smooth'].max())
    min_elev = float(df['altitude_smooth'].min())
    avg_elev = float(df['altitude_smooth'].mean())

    # Speed metrics (prefer sensor speed, fallback to distance/time)
    if 'speed' in df.columns and len(df['speed'].dropna()) > 0:
        avg_speed_kmh = float(df['speed'].dropna().mean() * 3.6)
        max_speed_kmh = float(df['speed'].dropna().max() * 3.6)
    elif total_seconds > 0 and total_distance_km > 0:
        avg_speed_kmh = 3600.0 * total_distance_km / total_seconds
        # Approximate max speed from 1-second increments if timestamps exist
        if 'timestamp' in df.columns and len(df['timestamp'].dropna()) == len(df):
            inc_km = dist_increments / 1000.0
            max_speed_kmh = float(np.nanmax(inc_km) * 3600.0)
        else:
            max_speed_kmh = avg_speed_kmh
    else:
        avg_speed_kmh = float('nan')
        max_speed_kmh = float('nan')

    return {
        'df': df,
        'total_distance_km': total_distance_km,
        'elev_gain_m': float(elev_gain),
        'max_elev_m': max_elev,
        'min_elev_m': min_elev,
        'avg_elev_m': avg_elev,
        'total_time_str': total_time_str,
        'avg_speed_kmh': avg_speed_kmh,
        'max_speed_kmh': max_speed_kmh,
    }

# --- Animated Dashboard ---
def create_animated_dashboard(df: pd.DataFrame, *, save_animation: bool = SAVE_ANIMATION, save_frame: bool = SAVE_FRAME):
    """Create an animated dashboard that draws the route and elevation progressively with polished visuals."""
    m = _prepare_metrics(df)
    df = m['df']

    fig = plt.figure(figsize=(16, 9), dpi=110)

    # Grid layout: large left area, slim right stats
    gs = fig.add_gridspec(
        3, 7,
        height_ratios=[0.7, 0.95, 0.15],
        width_ratios=[0.9, 0.05, 0.08, 0.22, 0.05, 0.01, 0.01],
        hspace=0.22,
        wspace=0.02,
    )

    # Ride date title
    if 'timestamp' in df.columns and len(df['timestamp'].dropna()) > 0:
        ride_date = pd.to_datetime(df['timestamp'].dropna().iloc[0])
    elif 'start_time' in df.columns and len(df.get('start_time', pd.Series(dtype='datetime64[ns]')).dropna()) > 0:
        ride_date = pd.to_datetime(df['start_time'].dropna().iloc[0])
    else:
        ride_date = datetime.today()
    ride_date_str = ride_date.strftime('%B %d, %Y')

    fig.text(
        0.5, 0.955, ride_date_str,
        ha='center', va='top', fontsize=26, color=(1, 1, 1, 0.9),
        fontweight='bold', alpha=0.98, linespacing=1.1,
        zorder=10, transform=fig.transFigure,
    )

    # Route map (top left)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.axis('off')
    # Elevation profile (bottom left)
    ax_elev = fig.add_subplot(gs[1, 0])
    ax_elev.axis('off')
    # Summary statistics (right side)
    ax_stats = fig.add_subplot(gs[:, 3:5])
    ax_stats.axis('off')

    # Route normalization
    df_route = df.dropna(subset=['longitude', 'latitude'])
    if len(df_route) > 1:
        lon_min, lon_max = df_route['longitude'].min(), df_route['longitude'].max()
        lat_min, lat_max = df_route['latitude'].min(), df_route['latitude'].max()
        lon_pad = (lon_max - lon_min) * 0.08
        lat_pad = (lat_max - lat_min) * 0.08
        # Normalize to 0..1, then inset to avoid touching card edges
        route_x = (df_route['longitude'] - lon_min + lon_pad) / (lon_max - lon_min + 2 * lon_pad)
        route_y = (df_route['latitude'] - lat_min + lat_pad) / (lat_max - lat_min + 2 * lat_pad)
        inner_pad_axes = 0.06
        route_x = route_x * (1 - 2 * inner_pad_axes) + inner_pad_axes
        route_y = route_y * (1 - 2 * inner_pad_axes) + inner_pad_axes
        route_x = route_x.to_numpy()
        route_y = route_y.to_numpy()
    else:
        route_x = np.array([])
        route_y = np.array([])

    # Elevation series
    x_coords = np.linspace(0, 100, len(df))
    y_coords = df['altitude_smooth']
    max_elev_idx = int(y_coords.values.argmax()) if len(y_coords) else 0
    max_elev_x = float(x_coords[max_elev_idx]) if len(x_coords) else 0.0
    max_elev_y = float(y_coords.iloc[max_elev_idx]) if len(y_coords) else 0.0

    # Fixed axis limits for consistent scaling
    # Keep some left/right margin so the elevation line doesn't touch rounded corners
    x_min, x_max = 2, 98
    y_min, y_max = float(df['altitude_smooth'].min()), float(df['altitude_smooth'].max())
    y_pad = (y_max - y_min) * 0.04 if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad

    # Animation parameters
    n_frames = 160

    def update(frame):
        t = ease_out_cubic(frame / (n_frames - 1))

        # Map card backgrounds (and get patches for clipping)
        ax_map.clear(); ax_map.axis('off'); card_map = add_card(ax_map)
        ax_elev.clear(); ax_elev.axis('off'); card_elev = add_card(ax_elev)
        ax_stats.clear(); ax_stats.axis('off'); card_stats = add_card(ax_stats)

        # Route map with progress and subtle glow
        if len(route_x) > 1:
            end = max(1, int(t * len(route_x)))
            line_map, = ax_map.plot(route_x[:end], route_y[:end], color=THEME['accent'], linewidth=2.4, alpha=0.95, clip_on=True)
            line_map.set_path_effects([pe.Stroke(linewidth=5.5, foreground=(0, 0, 0, 0.35)), pe.Normal()])
            line_map.set_clip_path(card_map.get_path(), card_map.get_transform())
            # Progress marker
            marker, = ax_map.plot(route_x[end - 1], route_y[end - 1], marker='o', markersize=6.5, color=THEME['accent'], zorder=3, clip_on=True)
            marker.set_clip_path(card_map.get_path(), card_map.get_transform())
            ax_map.set_xlim(0, 1)
            ax_map.set_ylim(0, 1)
            ax_map.set_aspect('equal')

        # Elevation profile: main line + soft underfill + max point
        ax_elev.set_xlim(x_min, x_max)
        ax_elev.set_ylim(y_min, y_max)
        end_idx = max(1, int(t * len(x_coords))) if len(x_coords) else 0
        if end_idx > 0:
            x_plot = x_coords[:end_idx]
            y_plot = y_coords[:end_idx]
            # Trim to visible domain to prevent any bleed beyond card edges
            mask_dom = (x_plot >= x_min) & (x_plot <= x_max)
            x_plot = x_plot[mask_dom]
            y_plot = y_plot[mask_dom]
            fill = ax_elev.fill_between(x_plot, y_plot, y_min, color=THEME['accent'], alpha=0.08, zorder=0.6)
            fill.set_clip_path(card_elev.get_path(), card_elev.get_transform())
            line_elev, = ax_elev.plot(x_plot, y_plot, color=THEME['line'], linewidth=1.6, alpha=0.94, zorder=0.8, clip_on=True)
            line_elev.set_path_effects([pe.Stroke(linewidth=3.8, foreground=(0, 0, 0, 0.26)), pe.Normal()])
            line_elev.set_clip_path(card_elev.get_path(), card_elev.get_transform())
            if max_elev_idx < end_idx and (x_min <= max_elev_x <= x_max):
                pt, = ax_elev.plot(max_elev_x, max_elev_y, marker='o', markersize=5.5, color=THEME['alert'], zorder=2, clip_on=True)
                pt.set_clip_path(card_elev.get_path(), card_elev.get_transform())

        # Stats panel header (centered)
        ax_stats.text(0.5, 0.93, 'Ride Summary', transform=ax_stats.transAxes, fontsize=16, color=THEME['muted'],
                      fontweight='bold', alpha=0.95, zorder=2, clip_on=True, ha='center')

        # Animated values
        dist = m['total_distance_km'] * t
        gain = m['elev_gain_m'] * t
        max_elev_now = (m['max_elev_m'] if end_idx == 0 else float(df['altitude_smooth'].iloc[:end_idx].max()))
        min_elev_now = (m['min_elev_m'] if end_idx == 0 else float(df['altitude_smooth'].iloc[:end_idx].min()))
        avg_elev_now = (m['avg_elev_m'] if end_idx == 0 else float(df['altitude_smooth'].iloc[:end_idx].mean()))

        # Table layout with separate unit column for perfect alignment
        def split_units(txt: str):
            if isinstance(txt, str):
                for suf in (' km/h', ' km', ' m'):
                    if txt.endswith(suf):
                        return txt[:-len(suf)], suf.strip()
                return txt, ''
            return str(txt), ''

        rows_raw = [
            ('Distance', format_km(dist)),
            ('Elevation Gain', format_m(gain)),
            ('Max Elevation', format_m(max_elev_now)),
            ('Min Elevation', format_m(min_elev_now)),
            ('Avg Elevation', format_m(avg_elev_now)),
            ('Total Time', m['total_time_str']),
            ('Avg Speed', 'N/A' if np.isnan(m['avg_speed_kmh']) else format_speed(m['avg_speed_kmh'])),
            ('Max Speed', 'N/A' if np.isnan(m['max_speed_kmh']) else format_speed(m['max_speed_kmh'])),
        ]
        rows = [(label, *split_units(value)) for (label, value) in rows_raw]

        # Render as a table within the card bounds
        # Use six columns: label | gap | value | gap | unit | right-gap to keep everything inside
        rows = [(label, '', val, '', unit, '') for (label, val, unit) in rows]
        table_bbox = (0.08, 0.08, 0.84, 0.80)  # more generous right/left padding
        col_widths = [0.48, 0.03, 0.29, 0.03, 0.11, 0.06]  # sums to 1.0
        tbl = ax_stats.table(cellText=rows, colWidths=col_widths, loc='center', bbox=table_bbox)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.set_zorder(3)

        # Style cells and align columns
        for (r, c), cell in list(tbl.get_celld().items()):
            # transparent cells; subtle separators; make sure above the card
            cell.set_facecolor('none')
            cell.set_edgecolor((1, 1, 1, 0.06))
            cell.set_linewidth(0.5 if r > -1 else 0)
            cell.visible_edges = 'horizontal'
            # Clip to card so nothing can bleed out
            cell.get_text().set_clip_path(card_stats.get_path(), card_stats.get_transform())
            cell.set_zorder(3)
            cell.get_text().set_zorder(4)
            if c == 0:
                cell.get_text().set_color(THEME['muted'])
                cell.get_text().set_ha('left')
                cell.get_text().set_fontweight('regular')
            elif c == 1:
                # spacer column: keep empty
                cell.get_text().set_text('')
                cell.get_text().set_color(THEME['muted'])
                cell.get_text().set_ha('center')
                cell.get_text().set_fontweight('regular')
            elif c == 2:
                cell.get_text().set_color(THEME['text'])
                cell.get_text().set_ha('right')
                cell.get_text().set_fontweight('bold')
            elif c == 3:
                # second spacer between value and unit
                cell.get_text().set_text('')
                cell.get_text().set_color(THEME['muted'])
                cell.get_text().set_ha('center')
                cell.get_text().set_fontweight('regular')
            elif c == 4:  # unit
                cell.get_text().set_color(THEME['muted'])
                cell.get_text().set_ha('left')  # left align to avoid overlapping value
                cell.get_text().set_fontweight('regular')
            else:  # trailing right gap
                cell.get_text().set_text('')
                cell.get_text().set_color(THEME['muted'])
                cell.get_text().set_ha('center')
                cell.get_text().set_fontweight('regular')

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=55, repeat=False)

    # Optional save outputs
    if (save_animation or save_frame) and not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if save_frame:
        # Draw last frame, then save a PNG
        update(n_frames - 1)
        png_path = os.path.join(OUTPUTS_DIR, 'ride_dashboard.png')
        # Save with standard bounding box to prevent accidental cropping of titles
        fig.savefig(png_path, dpi=180)
        print(f"Saved final frame: {png_path}")

    if save_animation:
        # Try mp4 (ffmpeg) first, then fallback to GIF if pillow is available
        try:
            mp4_path = os.path.join(OUTPUTS_DIR, 'ride_dashboard.mp4')
            anim.save(mp4_path, fps=24)
            print(f"Saved animation: {mp4_path}")
        except Exception as e:
            try:
                from matplotlib.animation import PillowWriter  # optional dependency
                gif_path = os.path.join(OUTPUTS_DIR, 'ride_dashboard.gif')
                anim.save(gif_path, writer=PillowWriter(fps=20))
                print(f"Saved animation: {gif_path}")
            except Exception as e2:
                print("Could not save animation (need ffmpeg or pillow).", e, e2)

    # Show interactive animation
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
    create_animated_dashboard(df, save_animation=SAVE_ANIMATION, save_frame=SAVE_FRAME)

if __name__ == "__main__":
    main()
