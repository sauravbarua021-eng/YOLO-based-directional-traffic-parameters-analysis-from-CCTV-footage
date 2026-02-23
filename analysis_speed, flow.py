import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import io
import seaborn as sns
import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_excel('traffic_flow_speed_adjusted.xlsx')

# Display basic info about the dataset
print("Dataset Info:")
print(f"Total records: {len(df)}")
print(f"Time range: {df['time_s'].min():.2f} to {df['time_s'].max():.2f} seconds")
print(f"Directions: {df['direction'].unique()}")
print(f"Lines: {df['line'].unique()}")
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# FIXED: Create 5-second time bins correctly
# ============================================================================
print("\n" + "="*70)
print("CREATING 5-SECOND TIME BINS")
print("="*70)

# Create bin edges from 0 to max time + 5, with step of 5
bin_edges = np.arange(0, df['time_s'].max() + 5, 5)
bin_centers = bin_edges[:-1] + 2.5  # Center of each bin for labeling

print(f"Number of bins: {len(bin_edges) - 1}")
print(f"Bin edges: {bin_edges[:10]}...")
print(f"Bin centers: {bin_centers[:10]}...")

# Create time bins
df['time_bin_5s'] = pd.cut(df['time_s'], 
                           bins=bin_edges,
                           labels=bin_centers,  # Use bin centers as labels
                           right=False)

# Convert to numeric for easier handling
df['time_bin_5s'] = df['time_bin_5s'].astype(float)

# Verify the binning
print(f"\nSample of binned data:")
print(df[['time_s', 'time_bin_5s']].head(10))

# ============================================================================
# TRAFFIC FLOW THEORY CALCULATIONS
# ============================================================================

print("\n" + "="*70)
print("TRAFFIC FLOW THEORY CALCULATIONS")
print("="*70)

# 1. TIME MEAN SPEED (TMS) - Arithmetic mean of spot speeds
# Formula: TMS = (1/n) * Σ vi

# Overall Time Mean Speed
tms_overall = df['speed_kmh'].mean()
print(f"\n1. TIME MEAN SPEED (TMS):")
print(f"   Overall TMS: {tms_overall:.2f} km/h")

# TMS by direction
tms_by_direction = df.groupby('direction')['speed_kmh'].mean().round(2)
print(f"\n   TMS by Direction:")
for direction, tms in tms_by_direction.items():
    print(f"   {direction:8s}: {tms:.2f} km/h")

# TMS by 5-second intervals
tms_5s = df.groupby('time_bin_5s')['speed_kmh'].mean().round(2).reset_index()
tms_5s.columns = ['time_bin_5s', 'tms_kmh']

# 2. SPACE MEAN SPEED (SMS) - Harmonic mean of spot speeds
# Formula: SMS = n / Σ (1/vi)

def calculate_sms(speeds):
    """Calculate Space Mean Speed (harmonic mean)"""
    # Filter out zero or negative speeds if any
    speeds = speeds[speeds > 0]
    if len(speeds) == 0:
        return 0
    return 1 / np.mean(1.0 / speeds)

# Overall Space Mean Speed
sms_overall = calculate_sms(df['speed_kmh'])
print(f"\n2. SPACE MEAN SPEED (SMS):")
print(f"   Overall SMS: {sms_overall:.2f} km/h")
print(f"   Note: SMS is always ≤ TMS (TMS/SMS ratio: {tms_overall/sms_overall:.3f})")

# SMS by direction
sms_by_direction = df.groupby('direction')['speed_kmh'].apply(calculate_sms).round(2)
print(f"\n   SMS by Direction:")
for direction, sms in sms_by_direction.items():
    tms = tms_by_direction[direction]
    print(f"   {direction:8s}: {sms:.2f} km/h (TMS/SMS: {tms/sms:.3f})")

# SMS by 5-second intervals
sms_5s = df.groupby('time_bin_5s')['speed_kmh'].apply(calculate_sms).round(2).reset_index()
sms_5s.columns = ['time_bin_5s', 'sms_kmh']

# 3. TRAFFIC DENSITY (k) - vehicles per km per lane
# Formula: k = q / vs, where:
#   q = flow rate (vehicles per hour)
#   vs = space mean speed (km/h)

# Calculate time span in hours
time_span_hours = (df['time_s'].max() - df['time_s'].min()) / 3600

# Overall density
total_vehicles = len(df)
overall_flow_rate = total_vehicles / time_span_hours  # vehicles per hour
overall_density = overall_flow_rate / sms_overall if sms_overall > 0 else 0  # vehicles per km

print(f"\n3. TRAFFIC DENSITY:")
print(f"   Overall Density: {overall_density:.2f} vehicles/km")
print(f"   (Based on flow rate: {overall_flow_rate:.0f} veh/h, SMS: {sms_overall:.2f} km/h)")

# Density by direction
density_by_direction = {}
print(f"\n   Density by Direction:")
for direction in df['direction'].unique():
    dir_df = df[df['direction'] == direction]
    dir_vehicles = len(dir_df)
    dir_flow_rate = dir_vehicles / time_span_hours
    dir_sms = sms_by_direction[direction]
    dir_density = dir_flow_rate / dir_sms if dir_sms > 0 else 0
    density_by_direction[direction] = dir_density
    print(f"   {direction:8s}: {dir_density:.2f} veh/km (Flow: {dir_flow_rate:.0f} veh/h)")

# Density by 5-second intervals
density_5s = []
for time_bin in sorted(df['time_bin_5s'].unique()):
    bin_df = df[df['time_bin_5s'] == time_bin]
    bin_vehicles = len(bin_df)
    # Convert 5-second count to hourly flow rate
    bin_flow_rate = bin_vehicles * (3600 / 5)  # vehicles per hour
    bin_sms = calculate_sms(bin_df['speed_kmh'])
    bin_density = bin_flow_rate / bin_sms if bin_sms > 0 else 0
    density_5s.append({
        'time_bin_5s': time_bin,
        'density_veh_per_km': round(bin_density, 2),
        'flow_rate_veh_per_hour': round(bin_flow_rate, 2),
        'vehicle_count': bin_vehicles,
        'sms_kmh': round(bin_sms, 2)
    })

density_5s_df = pd.DataFrame(density_5s)

# 4. ADDITIONAL TRAFFIC PARAMETERS

# Headway (time between vehicles in seconds)
avg_headway = time_span_hours * 3600 / total_vehicles if total_vehicles > 0 else 0
print(f"\n4. ADDITIONAL PARAMETERS:")
print(f"   Average Time Headway: {avg_headway:.2f} seconds/vehicle")

# Spacing (distance between vehicles in meters)
avg_spacing = (1000 / overall_density) if overall_density > 0 else 0
print(f"   Average Space Headway: {avg_spacing:.2f} meters/vehicle")

# Occupancy percentage (assuming average vehicle length of 4.5 meters)
avg_vehicle_length = 4.5  # meters
occupancy = (overall_density * avg_vehicle_length / 1000) * 100
print(f"   Estimated Lane Occupancy: {occupancy:.2f}%")

# Create a comprehensive traffic flow dataframe
traffic_flow_metrics = pd.DataFrame({
    'Metric': [
        'Time Mean Speed (TMS) - Overall',
        'Space Mean Speed (SMS) - Overall',
        'TMS/SMS Ratio',
        'Traffic Density (Overall)',
        'Flow Rate (Overall)',
        'Average Time Headway',
        'Average Space Headway',
        'Estimated Lane Occupancy'
    ],
    'Value': [
        f"{tms_overall:.2f} km/h",
        f"{sms_overall:.2f} km/h",
        f"{tms_overall/sms_overall:.3f}",
        f"{overall_density:.2f} veh/km",
        f"{overall_flow_rate:.0f} veh/h",
        f"{avg_headway:.2f} s",
        f"{avg_spacing:.2f} m",
        f"{occupancy:.2f}%"
    ]
})

print("\n" + "="*70)
print("5-SECOND INTERVAL TRAFFIC METRICS")
print("="*70)
print(density_5s_df.head(10))

# ============================================================================
# CREATE VISUALIZATIONS FOR TRAFFIC FLOW METRICS
# ============================================================================

# Create a directory for plots
plots_dir = 'traffic_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print("\nCreating traffic flow visualizations...")

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== PLOT 1: Vehicle Count by Direction ====================
# Create pivot table for counts
count_pivot = df.pivot_table(
    index='time_bin_5s', 
    columns='direction', 
    values='id', 
    aggfunc='count',
    fill_value=0
).reset_index()

plt.figure(figsize=(14, 8))
for direction in count_pivot.columns[1:]:  # Skip time_bin_5s
    plt.plot(count_pivot['time_bin_5s'], count_pivot[direction], 
             marker='o', markersize=4, linewidth=2, label=direction)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Vehicle Count per 5-second Interval', fontsize=14)
plt.title('Vehicle Count by Direction (5-second Intervals)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/1_vehicle_count_by_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 2: Average Speed by Direction ====================
avg_speed_by_dir = df.groupby(['time_bin_5s', 'direction'])['speed_kmh'].mean().reset_index()
plt.figure(figsize=(14, 8))
for direction in avg_speed_by_dir['direction'].unique():
    dir_data = avg_speed_by_dir[avg_speed_by_dir['direction'] == direction]
    plt.plot(dir_data['time_bin_5s'], dir_data['speed_kmh'], 
             marker='s', markersize=4, linewidth=2, label=direction)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Average Speed (km/h)', fontsize=14)
plt.title('Average Speed by Direction (5-second Intervals)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/2_average_speed_by_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 3: Direction Distribution Pie Chart ====================
direction_totals = df.groupby('direction')['id'].count()
direction_percentages = (direction_totals / len(df) * 100).round(2)

plt.figure(figsize=(12, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(direction_percentages)))
wedges, texts, autotexts = plt.pie(direction_percentages.values, 
                                    labels=direction_percentages.index,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    startangle=90,
                                    textprops={'fontsize': 14})
plt.title('Directional Distribution of Vehicles', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{plots_dir}/3_direction_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 4: Speed Distribution by Direction (Box Plot) ====================
plt.figure(figsize=(14, 8))

# Create box plot using matplotlib
data_to_plot = [df[df['direction'] == d]['speed_kmh'].values for d in sorted(df['direction'].unique())]
bp = plt.boxplot(data_to_plot, labels=sorted(df['direction'].unique()), patch_artist=True, 
                 showmeans=True, meanline=True)

# Customize colors
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Customize other elements
plt.setp(bp['whiskers'], color='gray', linestyle='-', linewidth=1.5)
plt.setp(bp['caps'], color='gray', linewidth=1.5)
plt.setp(bp['medians'], color='red', linewidth=2)
plt.setp(bp['means'], color='blue', linestyle='--', linewidth=2)

plt.xlabel('Direction', fontsize=14)
plt.ylabel('Speed (km/h)', fontsize=14)
plt.title('Speed Distribution by Direction (Box Plot)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add statistical annotations
for i, direction in enumerate(sorted(df['direction'].unique())):
    dir_data = df[df['direction'] == direction]['speed_kmh']
    stats_text = f'n={len(dir_data)}\nmean={dir_data.mean():.1f}\nmedian={dir_data.median():.1f}'
    plt.text(i+1, dir_data.max() + 2, stats_text, ha='center', va='bottom', 
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{plots_dir}/4_speed_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 5: Cumulative Vehicle Count ====================
plt.figure(figsize=(14, 8))
for direction in df['direction'].unique():
    dir_data = df[df['direction'] == direction].sort_values('time_s')
    plt.plot(dir_data['time_s'], np.arange(1, len(dir_data) + 1), 
             label=direction, linewidth=2.5)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Cumulative Vehicle Count', fontsize=14)
plt.title('Cumulative Vehicle Count by Direction', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/5_cumulative_vehicle_count.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 6: Speed Histogram ====================
plt.figure(figsize=(14, 8))
for direction in df['direction'].unique():
    dir_data = df[df['direction'] == direction]['speed_kmh']
    plt.hist(dir_data, bins=30, alpha=0.6, label=direction, density=True, edgecolor='black')
plt.xlabel('Speed (km/h)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Speed Distribution Histogram by Direction', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/6_speed_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 7: Total Vehicle Count Over Time ====================
agg_total = df.groupby('time_bin_5s').size().reset_index(name='total_vehicle_count')
plt.figure(figsize=(14, 8))
plt.bar(agg_total['time_bin_5s'], agg_total['total_vehicle_count'], 
        width=4, alpha=0.7, color='steelblue', edgecolor='navy')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Total Vehicle Count per 5s', fontsize=14)
plt.title('Total Vehicle Count Over Time (5-second Intervals)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{plots_dir}/7_total_vehicle_count.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 8: Average Speed Over Time (All Directions) ====================
agg_speed = df.groupby('time_bin_5s')['speed_kmh'].agg(['mean', 'std']).reset_index()
plt.figure(figsize=(14, 8))
plt.plot(agg_speed['time_bin_5s'], agg_speed['mean'], 
         marker='o', markersize=4, linewidth=2, color='darkgreen', label='Average Speed')
plt.fill_between(agg_speed['time_bin_5s'], 
                 agg_speed['mean'] - agg_speed['std'],
                 agg_speed['mean'] + agg_speed['std'],
                 alpha=0.2, color='green', label='±1 Std Dev')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Speed (km/h)', fontsize=14)
plt.title('Average Speed Over Time with Variability', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/8_average_speed_with_variability.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 9: TMS vs SMS Comparison ====================
# Merge TMS and SMS data
tms_sms_comparison = pd.merge(tms_5s, sms_5s, on='time_bin_5s', how='outer').fillna(0)
plt.figure(figsize=(14, 8))
plt.plot(tms_sms_comparison['time_bin_5s'], tms_sms_comparison['tms_kmh'], 
         marker='o', markersize=4, linewidth=2, label='Time Mean Speed (TMS)', color='blue')
plt.plot(tms_sms_comparison['time_bin_5s'], tms_sms_comparison['sms_kmh'], 
         marker='s', markersize=4, linewidth=2, label='Space Mean Speed (SMS)', color='red')
plt.fill_between(tms_sms_comparison['time_bin_5s'], 
                 tms_sms_comparison['sms_kmh'], 
                 tms_sms_comparison['tms_kmh'],
                 alpha=0.2, color='purple', label='Difference (TMS-SMS)')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Speed (km/h)', fontsize=14)
plt.title('Time Mean Speed vs Space Mean Speed (5-second intervals)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/9_tms_vs_sms_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 10: Traffic Density Over Time ====================
plt.figure(figsize=(14, 8))
plt.plot(density_5s_df['time_bin_5s'], density_5s_df['density_veh_per_km'], 
         marker='o', markersize=4, linewidth=2, color='darkorange')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Density (vehicles/km)', fontsize=14)
plt.title('Traffic Density Over Time (5-second intervals)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/10_traffic_density_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 11: Flow-Density Diagram ====================
plt.figure(figsize=(14, 8))
scatter = plt.scatter(density_5s_df['density_veh_per_km'], 
                      density_5s_df['flow_rate_veh_per_hour'],
                      c=density_5s_df['sms_kmh'], cmap='viridis', 
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Space Mean Speed (km/h)')
plt.xlabel('Density (vehicles/km)', fontsize=14)
plt.ylabel('Flow Rate (vehicles/hour)', fontsize=14)
plt.title('Fundamental Diagram of Traffic Flow', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/11_fundamental_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 12: Speed-Density Relationship ====================
plt.figure(figsize=(14, 8))
scatter = plt.scatter(density_5s_df['density_veh_per_km'], 
                      density_5s_df['sms_kmh'],
                      c=density_5s_df['flow_rate_veh_per_hour'], cmap='plasma', 
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Flow Rate (vehicles/hour)')
plt.xlabel('Density (vehicles/km)', fontsize=14)
plt.ylabel('Space Mean Speed (km/h)', fontsize=14)
plt.title('Speed-Density Relationship', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/12_speed_density_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 13: TMS and SMS by Direction ====================
plt.figure(figsize=(12, 8))
directions = list(tms_by_direction.index)
x = np.arange(len(directions))
width = 0.35

tms_values = tms_by_direction.values
sms_values = sms_by_direction.values

bars1 = plt.bar(x - width/2, tms_values, width, label='Time Mean Speed', color='steelblue', alpha=0.8)
bars2 = plt.bar(x + width/2, sms_values, width, label='Space Mean Speed', color='coral', alpha=0.8)

plt.xlabel('Direction', fontsize=14)
plt.ylabel('Speed (km/h)', fontsize=14)
plt.title('Time Mean Speed vs Space Mean Speed by Direction', fontsize=16, fontweight='bold')
plt.xticks(x, directions, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{plots_dir}/13_tms_sms_by_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 14: Density by Direction ====================
plt.figure(figsize=(12, 10))
density_values = list(density_by_direction.values())
density_directions = list(density_by_direction.keys())
colors = plt.cm.Set2(np.linspace(0, 1, len(density_values)))

wedges, texts, autotexts = plt.pie(density_values, 
                                    labels=density_directions,
                                    autopct=lambda pct: f'{pct:.1f}%\n({(pct/100)*sum(density_values):.1f} veh/km)',
                                    colors=colors,
                                    startangle=90,
                                    textprops={'fontsize': 12})
plt.title('Traffic Density Distribution by Direction', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{plots_dir}/14_density_by_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 15: Flow Rate by Direction ====================
flow_rate_by_direction = {}
for direction in df['direction'].unique():
    dir_df = df[df['direction'] == direction]
    dir_flow_rate = len(dir_df) / time_span_hours
    flow_rate_by_direction[direction] = dir_flow_rate

plt.figure(figsize=(12, 8))
bars = plt.bar(flow_rate_by_direction.keys(), flow_rate_by_direction.values(), 
               color='lightseagreen', alpha=0.8, edgecolor='darkgreen')
plt.xlabel('Direction', fontsize=14)
plt.ylabel('Flow Rate (vehicles/hour)', fontsize=14)
plt.title('Traffic Flow Rate by Direction', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{plots_dir}/15_flow_rate_by_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== PLOT 16: Headway Distribution ====================
# Calculate instantaneous headways (time gaps between consecutive vehicles)
df_sorted = df.sort_values('time_s')
df_sorted['time_diff'] = df_sorted['time_s'].diff().fillna(avg_headway)
# Filter out unreasonable headways (e.g., > 60 seconds)
reasonable_headways = df_sorted[df_sorted['time_diff'] < 60]['time_diff']

plt.figure(figsize=(14, 8))
plt.hist(reasonable_headways, bins=50, alpha=0.7, color='mediumpurple', edgecolor='black', density=True)
plt.xlabel('Time Headway (seconds)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Distribution of Time Headways', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.axvline(avg_headway, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_headway:.2f}s')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{plots_dir}/16_headway_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll 16 traffic flow plots saved to '{plots_dir}' directory")

# ============================================================================
# CREATE EXCEL FILE WITH ALL METRICS
# ============================================================================

output_file = 'traffic_5s_directional_analysis.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Raw data
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    # 5-second directional aggregation
    agg_directional = df.groupby(['time_bin_5s', 'direction']).agg({
        'id': 'count',
        'speed_kmh': ['mean', 'std', 'min', 'max', 'median']
    }).round(2)
    agg_directional.columns = ['_'.join(col).strip() for col in agg_directional.columns.values]
    agg_directional = agg_directional.rename(columns={'id_count': 'vehicle_count'})
    agg_directional.reset_index().to_excel(writer, sheet_name='5s_Directional_Agg', index=False)
    
    # Traffic flow metrics sheet
    flow_metrics_df = pd.DataFrame({
        'time_bin_5s': density_5s_df['time_bin_5s'],
        'vehicle_count': density_5s_df['vehicle_count'],
        'tms_kmh': tms_5s['tms_kmh'],
        'sms_kmh': density_5s_df['sms_kmh'],
        'density_veh_per_km': density_5s_df['density_veh_per_km'],
        'flow_rate_veh_per_hour': density_5s_df['flow_rate_veh_per_hour']
    })
    flow_metrics_df.to_excel(writer, sheet_name='5s_Traffic_Flow_Metrics', index=False)
    
    # Directional traffic flow metrics
    directional_flow = pd.DataFrame({
        'Direction': list(tms_by_direction.index),
        'Time_Mean_Speed_kmh': tms_by_direction.values,
        'Space_Mean_Speed_kmh': sms_by_direction.values,
        'TMS_SMS_Ratio': (tms_by_direction / sms_by_direction).values,
        'Density_veh_per_km': [density_by_direction[d] for d in tms_by_direction.index],
        'Flow_Rate_veh_per_hour': [flow_rate_by_direction[d] for d in tms_by_direction.index],
        'Vehicle_Count': [len(df[df['direction'] == d]) for d in tms_by_direction.index],
        'Percentage': [(len(df[df['direction'] == d]) / len(df) * 100) for d in tms_by_direction.index]
    }).round(2)
    directional_flow.to_excel(writer, sheet_name='Directional_Flow_Metrics', index=False)
    
    # Overall traffic theory summary
    theory_summary = pd.DataFrame({
        'Metric': [
            'Total Vehicles',
            'Time Span (seconds)',
            'Time Span (hours)',
            'Overall Time Mean Speed (km/h)',
            'Overall Space Mean Speed (km/h)',
            'TMS/SMS Ratio',
            'Overall Density (veh/km)',
            'Overall Flow Rate (veh/h)',
            'Average Time Headway (s)',
            'Average Space Headway (m)',
            'Estimated Lane Occupancy (%)'
        ],
        'Value': [
            len(df),
            round(df['time_s'].max() - df['time_s'].min(), 2),
            round(time_span_hours, 4),
            round(tms_overall, 2),
            round(sms_overall, 2),
            round(tms_overall/sms_overall, 3),
            round(overall_density, 2),
            round(overall_flow_rate, 2),
            round(avg_headway, 2),
            round(avg_spacing, 2),
            round(occupancy, 2)
        ]
    })
    theory_summary.to_excel(writer, sheet_name='Traffic_Theory_Summary', index=False)
    
    # Count pivot
    count_pivot.to_excel(writer, sheet_name='Count_Pivot', index=False)
    
    # Speed statistics
    speed_stats = df.groupby('direction')['speed_kmh'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    speed_stats.to_excel(writer, sheet_name='Speed_Stats')
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Total Vehicles', 'Average Speed (km/h)', 'Speed Std Dev', 
                   'Min Speed', 'Max Speed', 'Median Speed'],
        'Value': [len(df), df['speed_kmh'].mean(), df['speed_kmh'].std(),
                  df['speed_kmh'].min(), df['speed_kmh'].max(), df['speed_kmh'].median()]
    }).round(2)
    summary_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)

print(f"\nAnalysis complete! Excel file saved as: {output_file}")

# ============================================================================
# ENHANCED INSIGHTS WITH TRAFFIC FLOW THEORY
# ============================================================================

print("\n" + "="*80)
print("TRAFFIC FLOW THEORY - COMPREHENSIVE ANALYSIS")
print("="*80)

print("\n🔹 FUNDAMENTAL RELATIONSHIP: q = k * vs")
print(f"   where q = flow rate (veh/h), k = density (veh/km), vs = space mean speed (km/h)")
print(f"   Overall: {overall_flow_rate:.0f} = {overall_density:.2f} × {sms_overall:.2f}")

print("\n🔹 TIME MEAN SPEED vs SPACE MEAN SPEED:")
print(f"   • TMS is always ≥ SMS")
print(f"   • The ratio TMS/SMS indicates speed variability")
print(f"   • Overall TMS/SMS ratio: {tms_overall/sms_overall:.3f}")
print(f"   • Higher ratio indicates greater speed variation")

print("\n🔹 DIRECTIONAL TRAFFIC CHARACTERISTICS:")
print("-" * 60)
print(f"{'Direction':<10} {'TMS':>8} {'SMS':>8} {'Ratio':>8} {'Density':>10} {'Flow':>10} {'%':>6}")
print("-" * 60)

for direction in sorted(tms_by_direction.index):
    tms = tms_by_direction[direction]
    sms = sms_by_direction[direction]
    ratio = tms/sms
    density = density_by_direction[direction]
    flow = flow_rate_by_direction[direction]
    pct = len(df[df['direction'] == direction]) / len(df) * 100
    print(f"{direction:<10} {tms:>8.2f} {sms:>8.2f} {ratio:>8.3f} {density:>10.2f} {flow:>10.0f} {pct:>6.1f}")

print("\n🔹 TRAFFIC LEVEL CLASSIFICATION:")
print("-" * 40)

def classify_traffic_level(density):
    if density < 10:
        return "Free Flow (A)"
    elif density < 20:
        return "Stable Flow (B)"
    elif density < 30:
        return "Stable Flow (C)"
    elif density < 40:
        return "Approaching Unstable (D)"
    elif density < 50:
        return "Unstable Flow (E)"
    else:
        return "Forced Flow (F)"

for direction in density_by_direction:
    level = classify_traffic_level(density_by_direction[direction])
    print(f"   {direction:8s}: {density_by_direction[direction]:.2f} veh/km - {level}")

print("\n🔹 PEAK DENSITY PERIODS:")
print("-" * 40)
peak_density = density_5s_df.nlargest(5, 'density_veh_per_km')[['time_bin_5s', 'density_veh_per_km', 'flow_rate_veh_per_hour', 'sms_kmh']]
for _, row in peak_density.iterrows():
    print(f"   Time {row['time_bin_5s']-2.5:.0f}-{row['time_bin_5s']+2.5:.0f}s: "
          f"Density={row['density_veh_per_km']:.1f} veh/km, "
          f"Flow={row['flow_rate_veh_per_hour']:.0f} veh/h, "
          f"Speed={row['sms_kmh']:.1f} km/h")

print("\n🔹 LOWEST SPEED PERIODS:")
print("-" * 40)
lowest_speed = sms_5s.nsmallest(5, 'sms_kmh')
for _, row in lowest_speed.iterrows():
    time_bin = row['time_bin_5s']
    matching_rows = density_5s_df[density_5s_df['time_bin_5s'] == time_bin]
    if not matching_rows.empty:
        density_row = matching_rows.iloc[0]
        print(f"   Time {time_bin-2.5:.0f}-{time_bin+2.5:.0f}s: "
              f"Speed={row['sms_kmh']:.1f} km/h, "
              f"Density={density_row['density_veh_per_km']:.1f} veh/km, "
              f"Flow={density_row['flow_rate_veh_per_hour']:.0f} veh/h")

print("\n🔹 CRITICAL TRAFFIC PARAMETERS:")
print("-" * 40)
max_density = density_5s_df['density_veh_per_km'].max()
max_flow = density_5s_df['flow_rate_veh_per_hour'].max()
max_speed = df['speed_kmh'].max()

print(f"   Maximum Density: {max_density:.2f} veh/km")
print(f"   Maximum Flow Rate: {max_flow:.0f} veh/h")
print(f"   Maximum Speed: {max_speed:.2f} km/h")
print(f"   Minimum Speed: {df['speed_kmh'].min():.2f} km/h")

# Check if critical density (capacity) is reached
capacity_flow_idx = density_5s_df['flow_rate_veh_per_hour'].idxmax()
critical_density = density_5s_df.loc[capacity_flow_idx, 'density_veh_per_km']
critical_speed = density_5s_df.loc[capacity_flow_idx, 'sms_kmh']

print(f"\n🔹 CAPACITY ANALYSIS:")
print("-" * 40)
print(f"   Capacity Flow: {max_flow:.0f} veh/h")
print(f"   Critical Density at capacity: {critical_density:.2f} veh/km")
print(f"   Speed at capacity: {critical_speed:.2f} km/h")

print("\n" + "="*80)
print("PROCESS COMPLETE")
print("="*80)
print(f"\n📊 Generated files:")
print(f"   • Excel Analysis: {output_file}")
print(f"   • Plots directory: {plots_dir}/ (16 individual plots)")
print(f"\n✅ Traffic flow theory analysis completed successfully!")