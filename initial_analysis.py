import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('t4t_synthetic_data_large.csv')
df['event_date'] = pd.to_datetime(df['event_date'])
df['year'] = df['event_date'].dt.year
df['month'] = df['event_date'].dt.to_period('M')
df['quarter'] = df['event_date'].dt.to_period('Q')
df['capacity_utilization'] = (df['products_distributed'] / df['carrying_capacity']) * 100
df['products_per_beneficiary'] = df['products_distributed'] / df['estimated_beneficiaries']
df['volunteer_efficiency'] = df['products_distributed'] / df['volunteers_present']
df['capacity_gap'] = df['carrying_capacity'] - df['products_distributed']

print("Teens4Teens Data Analysis - Resource Allocation Insights")
print(f"\nDataset Overview:")
print(f"Total Events: {len(df)}")
print(f"Date Range: {df['event_date'].min().strftime('%Y-%m-%d')} to {df['event_date'].max().strftime('%Y-%m-%d')}")
print(f"Analysis Period: {(df['event_date'].max() - df['event_date'].min()).days} days (~{(df['event_date'].max() - df['event_date'].min()).days/365:.1f} years)")
print(f"Chapters: {df['chapter_name'].nunique()}")
print(f"States: {df['state'].nunique()}")
print(f"Cities: {df['city'].nunique()}")
print(f"\nChapters: {', '.join(sorted(df['chapter_name'].unique()))}")

# Key Metrics Summary
print("Performance Metrics")
print(f"\nTotal Products Distributed: {df['products_distributed'].sum():,}")
print(f"Total Beneficiaries Reached: {df['estimated_beneficiaries'].sum():,}")
print(f"Total Volunteer Hours: {(df['volunteers_present'] * df['event_duration_hours']).sum():,.0f}")
print(f"Total Events Conducted: {len(df)}")
print(f"\nAverage Products per Event: {df['products_distributed'].mean():.0f}")
print(f"Median Products per Event: {df['products_distributed'].median():.0f}")
print(f"Average Capacity Utilization: {df['capacity_utilization'].mean():.1f}%")
print(f"Average Products per Beneficiary: {df['products_per_beneficiary'].mean():.2f}")
print(f"Average Volunteer Efficiency: {df['volunteer_efficiency'].mean():.1f} products/volunteer")
print(f"Average Volunteers per Event: {df['volunteers_present'].mean():.1f}")

# Year over Year Growth
print("Year Over Year Growth Analysis")

yearly_stats = df.groupby('year').agg({
    'products_distributed': 'sum',
    'estimated_beneficiaries': 'sum',
    'event_id': 'count',
    'volunteers_present': 'sum'
}).round(0)
yearly_stats.columns = ['Total Products', 'Total Beneficiaries', 'Events', 'Total Volunteers']

print("\n", yearly_stats)

if len(yearly_stats) > 1:
    yoy_growth = ((yearly_stats.loc[2024] - yearly_stats.loc[2023]) / yearly_stats.loc[2023] * 100)
    print("\nYear-over-Year Growth (2023 → 2024):")
    for metric in yoy_growth.index:
        print(f"  • {metric}: {yoy_growth[metric]:+.1f}%")

# Chapter Performance Analysis
print("Chapter Performance Comparison")

chapter_stats = df.groupby('chapter_name').agg({
    'products_distributed': ['sum', 'mean', 'count'],
    'capacity_utilization': 'mean',
    'estimated_beneficiaries': 'sum',
    'volunteers_present': 'mean',
    'volunteer_efficiency': 'mean',
    'capacity_gap': 'mean'
}).round(2)

chapter_stats.columns = ['Total Products', 'Avg Products/Event', 'Events', 
                         'Avg Capacity %', 'Total Beneficiaries', 
                         'Avg Volunteers', 'Volunteer Efficiency', 'Avg Capacity Gap']

print("\n", chapter_stats.sort_values('Total Products', ascending=False))

# Identify Resource Allocation Opportunities
print("Resource allocation insights")

# High demand chapters (>90% capacity utilization)
high_demand = df.groupby('chapter_name')['capacity_utilization'].mean()
high_demand_chapters = high_demand[high_demand > 90].sort_values(ascending=False)

print("\n High demand chapters (>90% capacity) - may need more resources:")
for chapter, util in high_demand_chapters.items():
    events = len(df[df['chapter_name'] == chapter])
    avg_gap = df[df['chapter_name'] == chapter]['capacity_gap'].mean()
    print(f"  • {chapter}: {util:.1f}% avg utilization ({events} events, {avg_gap:.0f} avg gap)")

# Underutilized chapters (<80% capacity)
low_demand_chapters = high_demand[high_demand < 80].sort_values(ascending=False)

if len(low_demand_chapters) > 0:
    print("\n Underutilized Chapters (<80% capacity) - Potential for reallocation:")
    for chapter, util in low_demand_chapters.items():
        events = len(df[df['chapter_name'] == chapter])
        avg_products = df[df['chapter_name'] == chapter]['products_distributed'].mean()
        print(f"  • {chapter}: {util:.1f}% avg utilization ({events} events, avg {avg_products:.0f} products)")

# Moderate chapters (80-90% capacity)
moderate_chapters = high_demand[(high_demand >= 80) & (high_demand <= 90)].sort_values(ascending=False)

if len(moderate_chapters) > 0:
    print("\n Balanced chapters (80-90% capacity):")
    for chapter, util in moderate_chapters.items():
        events = len(df[df['chapter_name'] == chapter])
        print(f"  • {chapter}: {util:.1f}% avg utilization ({events} events)")

# Volunteer efficiency analysis
print("Volunteer Allocation Analysis")

vol_efficiency = df.groupby('chapter_name')['volunteer_efficiency'].mean().sort_values(ascending=False)
print("\nVolunteer Efficiency by Chapter (products distributed per volunteer):")
for i, (chapter, eff) in enumerate(vol_efficiency.items(), 1):
    avg_vols = df[df['chapter_name'] == chapter]['volunteers_present'].mean()
    total_hours = (df[df['chapter_name'] == chapter]['volunteers_present'] * 
                   df[df['chapter_name'] == chapter]['event_duration_hours']).sum()
    print(f"  {i:2d}. {chapter}: {eff:.1f} products/volunteer (avg {avg_vols:.1f} volunteers, {total_hours:.0f} total hours)")

# Time-based trends
print("Temporal Trends")

quarterly_trends = df.groupby('quarter').agg({
    'products_distributed': 'sum',
    'estimated_beneficiaries': 'sum',
    'capacity_utilization': 'mean',
    'event_id': 'count'
}).round(1)
quarterly_trends.columns = ['Products', 'Beneficiaries', 'Avg Capacity %', 'Events']

print("\nQuarterly Trends:")
print(quarterly_trends)

# Seasonal patterns
df['season'] = df['event_date'].dt.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

seasonal = df.groupby('season').agg({
    'products_distributed': ['mean', 'sum'],
    'capacity_utilization': 'mean',
    'volunteers_present': 'mean',
    'event_id': 'count'
}).round(1)

print("\n Seasonal Patterns:")
print(seasonal)

# Geographic analysis
print("GEOGRAPHIC DISTRIBUTION")

state_stats = df.groupby('state').agg({
    'products_distributed': 'sum',
    'estimated_beneficiaries': 'sum',
    'event_id': 'count',
    'chapter_name': 'nunique'
}).round(0)
state_stats.columns = ['Total Products', 'Total Beneficiaries', 'Events', 'Chapters']

print("\n", state_stats.sort_values('Total Products', ascending=False))

# Product type analysis
print("Product Type Distribution")

product_stats = df.groupby('product_type').agg({
    'event_id': 'count',
    'products_distributed': 'sum'
}).round(0)
product_stats.columns = ['Events', 'Total Products']
product_stats['% of Events'] = (product_stats['Events'] / len(df) * 100).round(1)

print("\n", product_stats.sort_values('Total Products', ascending=False))

# Capacity gap analysis
print("Capacity Gap Analysis")

print("\nChapters Running Near/Over Capacity (Top 10 by utilization):")
capacity_issues = df.groupby('chapter_name').agg({
    'capacity_gap': ['mean', 'min', 'max'],
    'capacity_utilization': ['mean', 'max']
}).round(1)
capacity_issues.columns = ['Avg Gap', 'Min Gap', 'Max Gap', 'Avg Util %', 'Max Util %']
capacity_issues = capacity_issues.sort_values('Avg Util %', ascending=False).head(10)

print(capacity_issues)

# Statistical Analysis - Correlations
print("Correlation Analysis")

correlations = df[['products_distributed', 'carrying_capacity', 'volunteers_present', 
                   'estimated_beneficiaries', 'event_duration_hours']].corr()

print("\nKey Correlations:")
print(f"  • Products vs Volunteers: {correlations.loc['products_distributed', 'volunteers_present']:.3f}")
print(f"  • Products vs Capacity: {correlations.loc['products_distributed', 'carrying_capacity']:.3f}")
print(f"  • Products vs Beneficiaries: {correlations.loc['products_distributed', 'estimated_beneficiaries']:.3f}")

# Recommendations
print("Data-Driven Recommendations")

print("\n1. Increase capacity for high-demand chapters:")
for chapter in high_demand_chapters.head(5).index:
    current_capacity = df[df['chapter_name'] == chapter]['carrying_capacity'].mean()
    current_util = high_demand_chapters[chapter]
    if current_util > 95:
        increase = 1.30
    elif current_util > 92:
        increase = 1.20
    else:
        increase = 1.15
    suggested_capacity = current_capacity * increase
    print(f"   • {chapter}: Increase from {current_capacity:.0f} to {suggested_capacity:.0f} ({(increase-1)*100:.0f}%)")

print("\n2. Optimize volunteer allocation:")
low_efficiency = vol_efficiency.tail(3)
high_efficiency = vol_efficiency.head(3)
print(f"   • Benchmark efficiency: {vol_efficiency.median():.1f} products/volunteer")
print(f"   • Top performers ({', '.join(high_efficiency.index[:2])}): {high_efficiency.mean():.1f} avg")
print(f"   • Improvement opportunity ({', '.join(low_efficiency.index[:2])}): {low_efficiency.mean():.1f} avg")
print(f"   • Potential gain: {((high_efficiency.mean() - low_efficiency.mean()) / low_efficiency.mean() * 100):.0f}% improvement possible")

print("\n3. Geographic expansion opportunitiies:")
underserved_states = state_stats[state_stats['Events'] <= 15].sort_values('Events')
if len(underserved_states) > 0:
    print("   • States with limited presence:")
    for state, row in underserved_states.iterrows():
        print(f"     - {state}: {int(row['Events'])} events, {int(row['Chapters'])} chapter(s)")

print("\n4. Seasonal planning:")
seasonal_sorted = seasonal.sort_values(('products_distributed', 'mean'), ascending=False)
best_season = seasonal_sorted.index[0]
worst_season = seasonal_sorted.index[-1]
best_avg = seasonal_sorted.iloc[0][('products_distributed', 'mean')]
worst_avg = seasonal_sorted.iloc[-1][('products_distributed', 'mean')]
print(f"   • Peak season: {best_season} ({best_avg:.0f} avg products/event)")
print(f"   • Lowest season: {worst_season} ({worst_avg:.0f} avg products/event)")
print(f"   • Variance: {((best_avg - worst_avg) / worst_avg * 100):.1f}% difference")

print("\n5. Product type optimization:")
print("   • Current mix:")
for product_type, row in product_stats.iterrows():
    print(f"     - {product_type}: {row['% of Events']:.1f}% of events")

# Predictive estimates
print("Predictive inventory insights (Next Quarter)")

print("\nBased on historical trends and growth rates:")
for chapter in df['chapter_name'].unique():
    chapter_df = df[df['chapter_name'] == chapter]
    
    # Calculate trend
    recent_events = chapter_df.tail(5)['products_distributed'].mean()
    
    # Events per quarter (based on historical average)
    events_per_quarter = len(chapter_df) / 8  # 8 quarters in 2 years
    
    # Estimate for next quarter
    quarterly_estimate = recent_events * events_per_quarter
    
    print(f"  • {chapter}: ~{quarterly_estimate:.0f} products "
          f"({events_per_quarter:.1f} events × {recent_events:.0f} recent avg)")

# Advanced insights - Growth trajectory
print("Growth trajectory analysis")

chapter_growth = {}
for chapter in df['chapter_name'].unique():
    chapter_df = df[df['chapter_name'] == chapter].sort_values('event_date')
    if len(chapter_df) >= 4:
        first_half = chapter_df.head(len(chapter_df)//2)['products_distributed'].mean()
        second_half = chapter_df.tail(len(chapter_df)//2)['products_distributed'].mean()
        growth_rate = ((second_half - first_half) / first_half * 100)
        chapter_growth[chapter] = growth_rate

chapter_growth_sorted = dict(sorted(chapter_growth.items(), key=lambda x: x[1], reverse=True))

print("\nChapter Growth Rates (First half vs Second half of data):")
for chapter, growth in list(chapter_growth_sorted.items())[:10]:
    emoji = "UP " if growth > 5 else "STABLE " if growth > 0 else "DOWN "
    print(f"  {emoji} {chapter}: {growth:+.1f}%")

# Create comprehensive visualizations
fig = plt.figure(figsize=(24, 16))

# 1. Chapter Performance Comparison
ax1 = plt.subplot(4, 4, 1)
chapter_totals = df.groupby('chapter_name')['products_distributed'].sum().sort_values()
chapter_totals.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('Total Products Distributed by Chapter', fontsize=11, fontweight='bold')
ax1.set_xlabel('Products Distributed')
ax1.grid(True, alpha=0.3)

# 2. Capacity Utilization by Chapter
ax2 = plt.subplot(4, 4, 2)
chapter_capacity = df.groupby('chapter_name')['capacity_utilization'].mean().sort_values()
colors = ['red' if x > 90 else 'orange' if x > 80 else 'green' for x in chapter_capacity]
chapter_capacity.plot(kind='barh', ax=ax2, color=colors)
ax2.axvline(x=90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High (90%)')
ax2.axvline(x=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (80%)')
ax2.set_title('Avg Capacity Utilization by Chapter', fontsize=11, fontweight='bold')
ax2.set_xlabel('Capacity Utilization (%)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Time Series of Products Distributed
ax3 = plt.subplot(4, 4, 3)
quarterly_products = df.groupby('quarter')['products_distributed'].sum()
quarterly_products.plot(ax=ax3, marker='o', linewidth=2.5, markersize=8, color='darkblue')
ax3.set_title('Quarterly Products Distribution Trend', fontsize=11, fontweight='bold')
ax3.set_xlabel('Quarter')
ax3.set_ylabel('Total Products')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Year-over-Year Comparison
ax4 = plt.subplot(4, 4, 4)
yearly_products = df.groupby('year')['products_distributed'].sum()
yearly_products.plot(kind='bar', ax=ax4, color=['lightcoral', 'salmon'])
ax4.set_title('Year-over-Year Products Distributed', fontsize=11, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Total Products')
ax4.tick_params(axis='x', rotation=0)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Volunteer Efficiency
ax5 = plt.subplot(4, 4, 5)
vol_eff_sorted = df.groupby('chapter_name')['volunteer_efficiency'].mean().sort_values()
vol_eff_sorted.plot(kind='barh', ax=ax5, color='coral')
ax5.set_title('Volunteer Efficiency by Chapter', fontsize=11, fontweight='bold')
ax5.set_xlabel('Products per Volunteer')
ax5.grid(True, alpha=0.3)

# 6. Geographic Distribution
ax6 = plt.subplot(4, 4, 6)
state_products = df.groupby('state')['products_distributed'].sum().sort_values(ascending=False)
state_products.head(10).plot(kind='bar', ax=ax6, color='seagreen')
ax6.set_title('Top 10 States by Products Distributed', fontsize=11, fontweight='bold')
ax6.set_xlabel('State')
ax6.set_ylabel('Products')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(True, alpha=0.3)

# 7. Products per Beneficiary Distribution
ax7 = plt.subplot(4, 4, 7)
df['products_per_beneficiary'].hist(bins=30, ax=ax7, color='mediumpurple', edgecolor='black')
ax7.axvline(x=df['products_per_beneficiary'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f'Mean: {df["products_per_beneficiary"].mean():.2f}')
ax7.set_title('Distribution of Products per Beneficiary', fontsize=11, fontweight='bold')
ax7.set_xlabel('Products per Beneficiary')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Seasonal Patterns
ax8 = plt.subplot(4, 4, 8)
seasonal_avg = df.groupby('season')['products_distributed'].mean().reindex(
    ['Winter', 'Spring', 'Summer', 'Fall'])
seasonal_avg.plot(kind='bar', ax=ax8, color=['skyblue', 'lightgreen', 'gold', 'orange'])
ax8.set_title('Average Products by Season', fontsize=11, fontweight='bold')
ax8.set_xlabel('Season')
ax8.set_ylabel('Avg Products per Event')
ax8.tick_params(axis='x', rotation=45)
ax8.grid(True, alpha=0.3)

# 9. Monthly trend over time
ax9 = plt.subplot(4, 4, 9)
monthly_trend = df.groupby(df['event_date'].dt.to_period('M'))['products_distributed'].sum()
x_axis = range(len(monthly_trend))
ax9.plot(x_axis, monthly_trend.values, linewidth=1.5, color='darkgreen', alpha=0.7)
z = np.polyfit(x_axis, monthly_trend.values, 1)
p = np.poly1d(z)
ax9.plot(x_axis, p(x_axis), "r--", linewidth=2, label='Trend')
ax9.set_title('Monthly Products with Trend Line', fontsize=11, fontweight='bold')
ax9.set_xlabel('Month')
ax9.set_ylabel('Products')
ax9.legend()
ax9.grid(True, alpha=0.3)
# Set x-tick labels to show months at intervals
tick_positions = range(0, len(monthly_trend), 4)
tick_labels = [str(monthly_trend.index[i]) for i in tick_positions]
ax9.set_xticks(tick_positions)
ax9.set_xticklabels(tick_labels, rotation=45)

# 10. Events by Chapter
ax10 = plt.subplot(4, 4, 10)
event_counts = df['chapter_name'].value_counts().sort_values()
event_counts.plot(kind='barh', ax=ax10, color='teal')
ax10.set_title('Number of Events by Chapter', fontsize=11, fontweight='bold')
ax10.set_xlabel('Number of Events')
ax10.grid(True, alpha=0.3)

# 11. Capacity utilization over time
ax11 = plt.subplot(4, 4, 11)
monthly_capacity = df.groupby(df['event_date'].dt.to_period('M'))['capacity_utilization'].mean()
x_axis = range(len(monthly_capacity))
ax11.plot(x_axis, monthly_capacity.values, marker='s', linewidth=1.5, markersize=5, color='darkred', alpha=0.7)
ax11.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Demand (90%)')
ax11.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (80%)')
ax11.set_title('Monthly Avg Capacity Utilization', fontsize=11, fontweight='bold')
ax11.set_xlabel('Month')
ax11.set_ylabel('Capacity Utilization (%)')
ax11.legend(fontsize=8)
ax11.grid(True, alpha=0.3)
# Set x-tick labels
tick_positions = range(0, len(monthly_capacity), 4)
tick_labels = [str(monthly_capacity.index[i]) for i in tick_positions]
ax11.set_xticks(tick_positions)
ax11.set_xticklabels(tick_labels, rotation=45)

# 12. Beneficiaries by Chapter
ax12 = plt.subplot(4, 4, 12)
beneficiaries_total = df.groupby('chapter_name')['estimated_beneficiaries'].sum().sort_values()
beneficiaries_total.plot(kind='barh', ax=ax12, color='darkorange')
ax12.set_title('Total Beneficiaries by Chapter', fontsize=11, fontweight='bold')
ax12.set_xlabel('Beneficiaries')
ax12.grid(True, alpha=0.3)

# 13. Volunteer hours by chapter
ax13 = plt.subplot(4, 4, 13)
df['volunteer_hours'] = df['volunteers_present'] * df['event_duration_hours']
vol_hours = df.groupby('chapter_name')['volunteer_hours'].sum().sort_values()
vol_hours.plot(kind='barh', ax=ax13, color='purple')
ax13.set_title('Total Volunteer Hours by Chapter', fontsize=11, fontweight='bold')
ax13.set_xlabel('Volunteer Hours')
ax13.grid(True, alpha=0.3)

# 14. Product type breakdown
ax14 = plt.subplot(4, 4, 14)
product_type_counts = df['product_type'].value_counts()
product_type_counts.plot(kind='pie', ax=ax14, autopct='%1.1f%%', startangle=90)
ax14.set_title('Product Type Distribution', fontsize=11, fontweight='bold')
ax14.set_ylabel('')

# 15. Capacity gap distribution
ax15 = plt.subplot(4, 4, 15)
df['capacity_gap'].hist(bins=30, ax=ax15, color='indianred', edgecolor='black')
ax15.axvline(x=df['capacity_gap'].mean(), color='blue', linestyle='--', linewidth=2, 
             label=f'Mean: {df["capacity_gap"].mean():.0f}')
ax15.set_title('Distribution of Capacity Gap', fontsize=11, fontweight='bold')
ax15.set_xlabel('Unused Capacity (products)')
ax15.set_ylabel('Frequency')
ax15.legend()
ax15.grid(True, alpha=0.3)

# 16. Growth trajectory
ax16 = plt.subplot(4, 4, 16)
growth_df = pd.DataFrame(list(chapter_growth_sorted.items()), columns=['Chapter', 'Growth'])
growth_df = growth_df.set_index('Chapter').sort_values('Growth')
colors_growth = ['red' if x < 0 else 'lightgreen' if x > 5 else 'yellow' for x in growth_df['Growth']]
growth_df.plot(kind='barh', ax=ax16, color=colors_growth, legend=False)
ax16.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax16.set_title('Chapter Growth Rate (1st vs 2nd Half)', fontsize=11, fontweight='bold')
ax16.set_xlabel('Growth Rate (%)')
ax16.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('t4t_comprehensive_analysis_large.png', dpi=300, bbox_inches='tight')
print("\n Saved: t4t_comprehensive_analysis_large.png")

# Create correlation heatmap
fig2, ax = plt.subplots(figsize=(12, 10))
correlation_vars = df[['products_distributed', 'carrying_capacity', 'volunteers_present', 
                       'estimated_beneficiaries', 'event_duration_hours', 
                       'capacity_utilization', 'volunteer_efficiency', 'capacity_gap']]
correlation_matrix = correlation_vars.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=ax, fmt='.3f', annot_kws={'size': 10})
ax.set_title('Correlation Matrix: Key Operational Metrics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('t4t_correlation_heatmap_large.png', dpi=300, bbox_inches='tight')
print(" Saved: t4t_correlation_heatmap_large.png")

# Create geographic heatmap
fig3, axes = plt.subplots(2, 2, figsize=(18, 12))

# State-level analysis
ax1 = axes[0, 0]
state_stats_plot = df.groupby('state')['products_distributed'].sum().sort_values(ascending=False)
state_stats_plot.plot(kind='bar', ax=ax1, color='darkblue', alpha=0.7)
ax1.set_title('Total Products by State', fontsize=14, fontweight='bold')
ax1.set_xlabel('State')
ax1.set_ylabel('Products Distributed')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# Top cities
ax2 = axes[0, 1]
city_stats = df.groupby('city')['products_distributed'].sum().sort_values(ascending=False).head(15)
city_stats.plot(kind='barh', ax=ax2, color='darkgreen', alpha=0.7)
ax2.set_title('Top 15 Cities by Products Distributed', fontsize=14, fontweight='bold')
ax2.set_xlabel('Products Distributed')
ax2.grid(True, alpha=0.3)

# Events per state over time
ax3 = axes[1, 0]
for state in df['state'].value_counts().head(5).index:
    state_data = df[df['state'] == state].groupby('quarter')['event_id'].count()
    ax3.plot(state_data.index.astype(str), state_data.values, marker='o', label=state, linewidth=2)
ax3.set_title('Quarterly Events: Top 5 States', fontsize=14, fontweight='bold')
ax3.set_xlabel('Quarter')
ax3.set_ylabel('Number of Events')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)
ax3.set_xticks(ax3.get_xticks()[::2])

# Chapter performance scatter
ax4 = axes[1, 1]
chapter_summary = df.groupby('chapter_name').agg({
    'capacity_utilization': 'mean',
    'volunteer_efficiency': 'mean',
    'products_distributed': 'sum'
})
scatter = ax4.scatter(chapter_summary['capacity_utilization'], 
                     chapter_summary['volunteer_efficiency'],
                     s=chapter_summary['products_distributed']/50,
                     alpha=0.6, c=range(len(chapter_summary)), cmap='viridis')
ax4.axhline(y=vol_efficiency.median(), color='red', linestyle='--', alpha=0.5, label='Median Efficiency')
ax4.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='80% Capacity')
ax4.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90% Capacity')
ax4.set_xlabel('Average Capacity Utilization (%)', fontsize=12)
ax4.set_ylabel('Average Volunteer Efficiency', fontsize=12)
ax4.set_title('Chapter Performance Matrix\n(Bubble size = Total Products)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add chapter labels
for idx, chapter in enumerate(chapter_summary.index):
    ax4.annotate(chapter.split()[0], 
                (chapter_summary.iloc[idx]['capacity_utilization'], 
                 chapter_summary.iloc[idx]['volunteer_efficiency']),
                fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('t4t_geographic_analysis_large.png', dpi=300, bbox_inches='tight')
print("Saved: t4t_geographic_analysis_large.png")

print("Analysis")
print("\n Generated Files:")
print("  • t4t_synthetic_data_large.csv - Synthetic dataset (243 events, 15 chapters, 2 years)")
print("  • t4t_comprehensive_analysis_large.png - 16-panel comprehensive visualization")
print("  • t4t_correlation_heatmap_large.png - Correlation matrix of key metrics")
print("  • t4t_geographic_analysis_large.png - Geographic and performance analysis")
print("\n Key Findings Summary:")
print(f"  • {len(high_demand_chapters)} chapters operating >90% capacity - need resource expansion")
print(f"  • Top performer: {chapter_totals.idxmax()} with {chapter_totals.max():,.0f} products distributed")
print(f"  • Volunteer efficiency varies {vol_efficiency.max()/vol_efficiency.min():.1f}x between best and worst")
print(f"  • Year-over-year growth: {((yearly_stats.loc[2024, 'Total Products'] - yearly_stats.loc[2023, 'Total Products']) / yearly_stats.loc[2023, 'Total Products'] * 100):.1f}%")
print(f"  • Total impact: {df['estimated_beneficiaries'].sum():,} beneficiaries served")
print("\n Implementation Priority:")
print("  1. Deploy centralized data collection system (Google Forms → Sheets)")
print("  2. Automate monthly analytics pipeline (Python scripts)")
print("  3. Create real-time capacity monitoring dashboard")
print("  4. Implement predictive inventory allocation model")
print("  5. Establish quarterly review process with chapter leaders")
