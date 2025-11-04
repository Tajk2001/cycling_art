# Cycling Art Dashboard Generator

Transform your cycling data into beautiful animated dashboard visualizations that showcase your ride with route maps, elevation profiles, and live-updating statistics.

## Features

- **FIT File Support**: Reads cycling data from Garmin .fit files
- **Animated Dashboard**: Dynamic visualization with route map and elevation profile
- **Polished Visuals**: Themed dark UI, rounded “cards”, glow effects, easing
- **Live Statistics**: Real-time updating ride statistics during animation
- **Route Visualization**: Top-down view of your cycling route
- **Elevation Profile**: 3D-like depth effect showing elevation changes
- **Clean Design**: Minimalist, web-friendly interface
- **Optional Export**: Save a final PNG or animation (mp4/gif)

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Add your .fit files** to the `data/` directory
2. **Run the dashboard generator:**
   ```bash
   python test_features.py
   ```
3. **Watch the animation**: The script will display an interactive window showing your ride's animated dashboard

Optional: Export outputs without changing code

- Save final PNG of last frame:
  ```bash
  SAVE_FRAME=1 python test_features.py
  ```
- Save animation (requires ffmpeg for mp4 or pillow for gif):
  ```bash
  SAVE_ANIMATION=1 python test_features.py
  ```

## How It Works

The script:
1. Parses elevation, GPS, speed, and timestamp data from your .fit file
2. Computes accurate haversine distance and robust elevation gain
3. Creates a dashboard layout with route map, elevation profile, and statistics panel
4. Animates the route with easing, glow accents, and a progress marker
5. Displays live-updating statistics with a clean typographic hierarchy

## Dashboard Components

- **Route Map**: Top-down view showing your cycling route with animated drawing
- **Elevation Profile**: 3D-like visualization with depth effect showing elevation changes
- **Statistics Panel**: Live-updating metrics including:
  - Distance covered
  - Total elevation gain
  - Maximum, minimum, and average elevation
  - Total ride time
  - Average and maximum speed
  - Number of data points

## Visual Polish

- Dark theme with subtle contrast and rounded cards
- Easing on animation for smoother motion
- Route glow + progress marker
- Elevation underfill for depth and a highlighted max point
- Cleaner, aligned stats with bigger headline metrics

## Directory Structure

```
cycling_art/
├── data/           # Place your .fit files here
├── outputs/        # Output directory (currently unused)
├── venv/           # Virtual environment
├── requirements.txt
├── test_features.py # Main dashboard script
└── README.md
```

## Requirements

- Python 3.7+
- .fit files from Garmin devices or compatible cycling apps
- See `requirements.txt` for Python package dependencies

## Web Integration Ready

The cleaned-up code is designed for easy integration into web applications, with:
- Minimal dependencies
- Clean, modular functions
- No complex visual effects or file exports
- Simple matplotlib-based animations
