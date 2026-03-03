"""
AVERA-ATLAS Enhanced Visualization

Creates useful trajectory visualizations for conjunction assessment:
- 3D orbital view with proper scaling
- Conjunction timeline
- Risk-colored trajectory segments
- TCA markers with miss distance annotations

Output: MP4 video and PNG summary chart
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
INPUT_FILE = "prop_multi.npz"
OUTPUT_VIDEO = "planner_output.mp4"
OUTPUT_SUMMARY = "conjunction_summary.png"

# Earth radius for visualization
R_EARTH = 6371.0  # km

# Risk colors
COLORS = {
    'RED': '#ff4757',
    'AMBER': '#ffa502',
    'GREEN': '#2ed573',
    'NOMINAL': '#66fcf1',
    'asset': '#00d4ff',
    'earth': '#1a4a6e'
}


def create_earth_sphere(ax, radius=R_EARTH, resolution=30):
    """Add a wireframe Earth to 3D plot."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color=COLORS['earth'], linewidth=0)
    ax.plot_wireframe(x, y, z, alpha=0.1, color='white', linewidth=0.3)


def render_conjunction_summary(data: dict, output_path: str):
    """
    Create a static summary image of all conjunctions.
    """
    obj_ids = data['obj_ids']
    miss_distances = data['ca_table']
    pc_values = data['pc_values']
    risk_levels = data['risk_levels']
    tca_indices = data['tca_indices']
    dt_sec = 60.0  # Assume 60s steps
    
    n_objs = len(obj_ids)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='#0a0a12')
    
    # --- Panel 1: Miss Distance Bar Chart ---
    ax1 = axes[0]
    ax1.set_facecolor('#12131a')
    
    colors = [COLORS.get(str(r), COLORS['NOMINAL']) for r in risk_levels]
    y_pos = np.arange(n_objs)
    
    # Log scale for miss distance
    miss_m = miss_distances * 1000  # Convert to meters
    bars = ax1.barh(y_pos, miss_m, color=colors, edgecolor='white', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([str(oid) for oid in obj_ids], fontsize=9, color='white')
    ax1.set_xlabel('Miss Distance (m)', color='white', fontsize=10)
    ax1.set_xscale('log')
    ax1.set_title('MISS DISTANCE AT TCA', color='#66fcf1', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100m threshold')
    ax1.axvline(x=1000, color='yellow', linestyle='--', alpha=0.7, label='1km threshold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, miss_m)):
        if val < 1000:
            label = f'{val:.0f}m'
        else:
            label = f'{val/1000:.1f}km'
        ax1.text(val * 1.1, bar.get_y() + bar.get_height()/2, label,
                va='center', ha='left', color='white', fontsize=8)
    
    # --- Panel 2: Pc Values ---
    ax2 = axes[1]
    ax2.set_facecolor('#12131a')
    
    # Filter out zero Pc values for log scale
    pc_display = np.array([max(pc, 1e-12) for pc in pc_values])
    bars2 = ax2.barh(y_pos, pc_display, color=colors, edgecolor='white', linewidth=0.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(oid) for oid in obj_ids], fontsize=9, color='white')
    ax2.set_xlabel('Probability of Collision', color='white', fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xlim(1e-12, 1)
    ax2.set_title('COLLISION PROBABILITY (Pc)', color='#66fcf1', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    
    # Threshold lines
    ax2.axvline(x=1e-4, color='red', linestyle='--', alpha=0.7, label='RED (1e-4)')
    ax2.axvline(x=1e-5, color='yellow', linestyle='--', alpha=0.7, label='AMBER (1e-5)')
    
    # --- Panel 3: TCA Timeline ---
    ax3 = axes[2]
    ax3.set_facecolor('#12131a')
    
    tca_minutes = tca_indices * dt_sec / 60.0
    scatter = ax3.scatter(tca_minutes, y_pos, c=colors, s=200, edgecolors='white', linewidth=1, zorder=5)
    
    # Add horizontal lines to timeline
    for i in range(n_objs):
        ax3.hlines(i, 0, tca_minutes[i], colors=colors[i], alpha=0.3, linewidth=2)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([str(oid) for oid in obj_ids], fontsize=9, color='white')
    ax3.set_xlabel('Time to TCA (minutes)', color='white', fontsize=10)
    ax3.set_title('CONJUNCTION TIMELINE', color='#66fcf1', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.set_xlim(0, max(tca_minutes) * 1.1)
    
    # Add TCA labels
    for i, (t, risk) in enumerate(zip(tca_minutes, risk_levels)):
        ax3.text(t + 2, i, f'T+{t:.0f}min', va='center', ha='left', 
                color='white', fontsize=8)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['RED'], 
               markersize=10, label='RED (Pc≥1e-4)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['AMBER'],
               markersize=10, label='AMBER (Pc≥1e-5)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['GREEN'],
               markersize=10, label='GREEN (Pc≥1e-7)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['NOMINAL'],
               markersize=10, label='NOMINAL', linestyle='None'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               facecolor='#1a1b26', edgecolor='#66fcf1', labelcolor='white',
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    
    plt.suptitle('AVERA-ATLAS CONJUNCTION ASSESSMENT SUMMARY', 
                 color='#66fcf1', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Summary saved: {output_path}")


def render_3d_video(data: dict, output_path: str):
    """
    Create an animated 3D visualization of trajectories.
    """
    t_array = data['t_array']
    r_asset = data['r_asset']
    r_objects = data['r_objects']
    obj_ids = data['obj_ids']
    risk_levels = data['risk_levels']
    tca_indices = data['tca_indices']
    miss_distances = data['ca_table']
    pc_values = data['pc_values']
    
    n_steps = len(t_array)
    n_objs = len(obj_ids)
    
    # Downsample for smoother animation
    step = max(1, n_steps // 200)
    frame_indices = list(range(0, n_steps, step))
    
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0a12')
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Main 3D view
    ax3d = fig.add_subplot(gs[0, :2], projection='3d', facecolor='#0a0a12')
    
    # Info panel
    ax_info = fig.add_subplot(gs[0, 2], facecolor='#12131a')
    ax_info.axis('off')
    
    # Timeline panel
    ax_timeline = fig.add_subplot(gs[1, :], facecolor='#12131a')
    
    def update(frame_num):
        idx = frame_indices[frame_num]
        
        ax3d.clear()
        ax_info.clear()
        ax_info.axis('off')
        
        # 3D View setup
        ax3d.set_facecolor('#0a0a12')
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor('gray')
        ax3d.yaxis.pane.set_edgecolor('gray')
        ax3d.zaxis.pane.set_edgecolor('gray')
        ax3d.tick_params(colors='gray', labelsize=7)
        ax3d.set_xlabel('X (km)', color='gray', fontsize=8)
        ax3d.set_ylabel('Y (km)', color='gray', fontsize=8)
        ax3d.set_zlabel('Z (km)', color='gray', fontsize=8)
        
        # Draw Earth
        create_earth_sphere(ax3d)
        
        # Asset trajectory (trail)
        trail_start = max(0, idx - 100)
        ax3d.plot(r_asset[trail_start:idx+1, 0], 
                  r_asset[trail_start:idx+1, 1],
                  r_asset[trail_start:idx+1, 2],
                  color=COLORS['asset'], alpha=0.6, linewidth=1.5)
        
        # Asset current position
        ax3d.scatter(*r_asset[idx], color=COLORS['asset'], s=150, 
                     edgecolors='white', linewidth=2, zorder=10)
        
        # Debris trajectories and positions
        active_conjunctions = []
        for i in range(n_objs):
            risk = str(risk_levels[i])
            color = COLORS.get(risk, COLORS['NOMINAL'])
            
            # Trail
            ax3d.plot(r_objects[i, trail_start:idx+1, 0],
                      r_objects[i, trail_start:idx+1, 1],
                      r_objects[i, trail_start:idx+1, 2],
                      color=color, alpha=0.4, linewidth=1)
            
            # Current position
            marker = 'D' if risk in ['RED', 'AMBER'] else 'o'
            size = 120 if risk == 'RED' else (80 if risk == 'AMBER' else 50)
            ax3d.scatter(*r_objects[i, idx], color=color, s=size, marker=marker,
                        edgecolors='white', linewidth=1, zorder=9)
            
            # TCA marker
            tca_idx = tca_indices[i]
            if abs(idx - tca_idx) < 20:  # Near TCA
                ax3d.scatter(*r_objects[i, tca_idx], color=color, s=200, 
                            marker='*', edgecolors='white', linewidth=1, zorder=11)
                active_conjunctions.append(i)
            
            # Line to asset at TCA (if close to TCA)
            if abs(idx - tca_idx) < 10:
                ax3d.plot([r_asset[tca_idx, 0], r_objects[i, tca_idx, 0]],
                         [r_asset[tca_idx, 1], r_objects[i, tca_idx, 1]],
                         [r_asset[tca_idx, 2], r_objects[i, tca_idx, 2]],
                         color=color, linestyle='--', alpha=0.8, linewidth=2)
        
        # Auto-scale view to show relevant objects
        all_positions = np.vstack([r_asset[idx:idx+1]] + [r_objects[i, idx:idx+1] for i in range(n_objs)])
        center = np.mean(all_positions, axis=0)
        max_range = max(np.max(np.abs(all_positions - center)), 1000)
        
        ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
        ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
        ax3d.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # Info Panel
        time_min = idx * 60 / 60  # Assuming 60s steps
        ax_info.text(0.5, 0.95, 'CONJUNCTION STATUS', transform=ax_info.transAxes,
                    ha='center', va='top', color='#66fcf1', fontsize=12, fontweight='bold')
        
        ax_info.text(0.5, 0.85, f'T + {time_min:.0f} min', transform=ax_info.transAxes,
                    ha='center', va='top', color='white', fontsize=14, fontweight='bold')
        
        # List objects by risk
        y_pos = 0.72
        for i in range(n_objs):
            risk = str(risk_levels[i])
            color = COLORS.get(risk, COLORS['NOMINAL'])
            miss_m = miss_distances[i] * 1000
            tca_min = tca_indices[i]
            
            status = "◀ TCA" if abs(idx - tca_min) < 5 else ""
            
            if miss_m < 1000:
                miss_str = f"{miss_m:.0f}m"
            else:
                miss_str = f"{miss_m/1000:.1f}km"
            
            ax_info.text(0.05, y_pos, f"● {obj_ids[i]}", transform=ax_info.transAxes,
                        color=color, fontsize=9, fontweight='bold')
            ax_info.text(0.55, y_pos, f"{miss_str}", transform=ax_info.transAxes,
                        color='white', fontsize=9)
            ax_info.text(0.85, y_pos, status, transform=ax_info.transAxes,
                        color='yellow', fontsize=9)
            y_pos -= 0.1
        
        # Title
        n_red = np.sum([str(r) == 'RED' for r in risk_levels])
        n_amber = np.sum([str(r) == 'AMBER' for r in risk_levels])
        ax3d.set_title(f'AVERA-ATLAS Orbital View | 🔴 {n_red} RED | 🟠 {n_amber} AMBER',
                      color='#66fcf1', fontsize=11, fontweight='bold', pad=10)
    
    # Setup timeline (static)
    dt_sec = 60.0
    tca_minutes = tca_indices * dt_sec / 60.0
    total_minutes = n_steps * dt_sec / 60.0
    
    ax_timeline.set_xlim(0, total_minutes)
    ax_timeline.set_ylim(-0.5, n_objs - 0.5)
    ax_timeline.set_xlabel('Time (minutes)', color='white', fontsize=9)
    ax_timeline.set_title('CONJUNCTION TIMELINE', color='#66fcf1', fontsize=10, fontweight='bold')
    ax_timeline.tick_params(colors='white', labelsize=8)
    ax_timeline.set_yticks(range(n_objs))
    ax_timeline.set_yticklabels([str(oid) for oid in obj_ids], fontsize=8, color='white')
    
    # TCA markers on timeline
    for i in range(n_objs):
        risk = str(risk_levels[i])
        color = COLORS.get(risk, COLORS['NOMINAL'])
        ax_timeline.scatter(tca_minutes[i], i, c=color, s=100, marker='D', edgecolors='white', zorder=5)
        ax_timeline.hlines(i, 0, tca_minutes[i], colors=color, alpha=0.3, linewidth=3)
    
    # Current time indicator (will be updated)
    time_line = ax_timeline.axvline(x=0, color='white', linewidth=2, alpha=0.8)
    
    def update_with_timeline(frame_num):
        update(frame_num)
        idx = frame_indices[frame_num]
        current_time = idx * dt_sec / 60.0
        time_line.set_xdata([current_time, current_time])
        return []
    
    print(f"[VIZ] Rendering {len(frame_indices)} frames...")
    
    ani = animation.FuncAnimation(fig, update_with_timeline, frames=len(frame_indices),
                                  interval=50, blit=False)
    
    try:
        writer = animation.FFMpegWriter(fps=20, bitrate=3000,
                                        metadata={'artist': 'AVERA-ATLAS'})
        ani.save(output_path, writer=writer)
        print(f"[VIZ] ✅ Video saved: {output_path}")
    except Exception as e:
        print(f"[VIZ] FFmpeg error: {e}")
        # Fallback to GIF
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            ani.save(gif_path, writer='pillow', fps=10)
            print(f"[VIZ] ✅ GIF saved: {gif_path}")
        except Exception as e2:
            print(f"[VIZ] GIF error: {e2}")
    
    plt.close(fig)


def render_all():
    """Main rendering function."""
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"[VIZ] Waiting for {INPUT_FILE}...")
        return False
    
    # Check if already rendered
    video_path = os.path.join(DATA_DIR, OUTPUT_VIDEO)
    summary_path = os.path.join(DATA_DIR, OUTPUT_SUMMARY)
    
    if os.path.exists(video_path):
        input_mtime = os.path.getmtime(input_path)
        output_mtime = os.path.getmtime(video_path)
        if output_mtime >= input_mtime:
            return False  # Already up to date
    
    print(f"[VIZ] Loading {INPUT_FILE}...")
    
    try:
        data = dict(np.load(input_path, allow_pickle=True))
        # Convert arrays
        for key in data:
            if hasattr(data[key], 'item') and data[key].ndim == 0:
                data[key] = data[key].item()
        
        # Validate data - check for NaN values
        r_asset = data.get('r_asset')
        r_objects = data.get('r_objects')
        
        if r_asset is None or r_objects is None:
            print(f"[VIZ] Missing trajectory data")
            return False
        
        # Replace NaN values with interpolated or zero values
        if np.any(np.isnan(r_asset)):
            print(f"[VIZ] Warning: NaN in asset trajectory, attempting repair...")
            nan_mask = np.isnan(r_asset)
            # Replace NaN with nearest valid value
            for col in range(3):
                arr = r_asset[:, col]
                nans = np.isnan(arr)
                if np.all(nans):
                    arr[:] = 0
                else:
                    valid_idx = np.where(~nans)[0]
                    arr[nans] = np.interp(np.where(nans)[0], valid_idx, arr[valid_idx])
            data['r_asset'] = r_asset
        
        if np.any(np.isnan(r_objects)):
            print(f"[VIZ] Warning: NaN in debris trajectories, attempting repair...")
            for i in range(r_objects.shape[0]):
                for col in range(3):
                    arr = r_objects[i, :, col]
                    nans = np.isnan(arr)
                    if np.all(nans):
                        arr[:] = 0
                    elif np.any(nans):
                        valid_idx = np.where(~nans)[0]
                        arr[nans] = np.interp(np.where(nans)[0], valid_idx, arr[valid_idx])
            data['r_objects'] = r_objects
        
        # Fix NaN in other arrays
        for key in ['ca_table', 'pc_values', 'tca_indices']:
            if key in data and np.any(np.isnan(data[key])):
                data[key] = np.nan_to_num(data[key], nan=0.0)
        
    except Exception as e:
        print(f"[VIZ] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Render summary image
    try:
        render_conjunction_summary(data, summary_path)
    except Exception as e:
        print(f"[VIZ] Error rendering summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Render video
    try:
        render_3d_video(data, video_path)
    except Exception as e:
        print(f"[VIZ] Error rendering video: {e}")
        import traceback
        traceback.print_exc()
    
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    print(f"[VIZ] AVERA-ATLAS Enhanced Visualization")
    print(f"[VIZ] Watching {DATA_DIR}...")
    
    while True:
        try:
            render_all()
        except Exception as e:
            print(f"[VIZ] Error: {e}")
        time.sleep(5)
