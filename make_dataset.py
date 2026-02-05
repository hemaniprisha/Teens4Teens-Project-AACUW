import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define chapters with their characteristics
chapters = {
    'Bay Area Chapter': {'state': 'CA', 'cities': ['San Francisco', 'Oakland', 'Berkeley', 'San Jose', 'Palo Alto'], 
                         'base_capacity': 500, 'growth_rate': 1.02, 'efficiency': 0.95},
    'Austin Central': {'state': 'TX', 'cities': ['Austin', 'Round Rock', 'Cedar Park', 'Georgetown'], 
                       'base_capacity': 400, 'growth_rate': 1.03, 'efficiency': 0.94},
    'Boston Metro': {'state': 'MA', 'cities': ['Boston', 'Cambridge', 'Somerville', 'Brookline', 'Newton'], 
                     'base_capacity': 350, 'growth_rate': 1.01, 'efficiency': 0.93},
    'Seattle North': {'state': 'WA', 'cities': ['Seattle', 'Bellevue', 'Tacoma', 'Kirkland', 'Redmond'], 
                      'base_capacity': 450, 'growth_rate': 1.025, 'efficiency': 0.92},
    'Phoenix Valley': {'state': 'AZ', 'cities': ['Phoenix', 'Tempe', 'Scottsdale', 'Mesa', 'Chandler'], 
                       'base_capacity': 300, 'growth_rate': 1.04, 'efficiency': 0.88},
    'Denver Metro': {'state': 'CO', 'cities': ['Denver', 'Boulder', 'Aurora', 'Lakewood', 'Arvada'], 
                     'base_capacity': 350, 'growth_rate': 1.02, 'efficiency': 0.90},
    'Chicago South': {'state': 'IL', 'cities': ['Chicago', 'Oak Park', 'Evanston', 'Naperville', 'Schaumburg'], 
                      'base_capacity': 500, 'growth_rate': 1.03, 'efficiency': 0.91},
    'Portland West': {'state': 'OR', 'cities': ['Portland', 'Beaverton', 'Gresham', 'Hillsboro', 'Lake Oswego'], 
                      'base_capacity': 400, 'growth_rate': 1.015, 'efficiency': 0.89},
    'Miami South': {'state': 'FL', 'cities': ['Miami', 'Fort Lauderdale', 'West Palm Beach', 'Coral Gables', 'Boca Raton'], 
                    'base_capacity': 450, 'growth_rate': 1.035, 'efficiency': 0.92},
    'Atlanta Metro': {'state': 'GA', 'cities': ['Atlanta', 'Decatur', 'Marietta', 'Sandy Springs', 'Alpharetta'], 
                      'base_capacity': 425, 'growth_rate': 1.025, 'efficiency': 0.90},
    'Philadelphia East': {'state': 'PA', 'cities': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Erie'], 
                          'base_capacity': 380, 'growth_rate': 1.02, 'efficiency': 0.88},
    'Minneapolis North': {'state': 'MN', 'cities': ['Minneapolis', 'St. Paul', 'Bloomington', 'Rochester'], 
                          'base_capacity': 360, 'growth_rate': 1.015, 'efficiency': 0.87},
    'San Diego South': {'state': 'CA', 'cities': ['San Diego', 'La Jolla', 'Chula Vista', 'Carlsbad'], 
                        'base_capacity': 420, 'growth_rate': 1.02, 'efficiency': 0.91},
    'Dallas Fort Worth': {'state': 'TX', 'cities': ['Dallas', 'Fort Worth', 'Plano', 'Irving', 'Arlington'], 
                          'base_capacity': 480, 'growth_rate': 1.03, 'efficiency': 0.93},
    'New York Metro': {'state': 'NY', 'cities': ['Brooklyn', 'Queens', 'Bronx', 'Manhattan', 'Staten Island'], 
                       'base_capacity': 550, 'growth_rate': 1.025, 'efficiency': 0.94},
}

product_types = ['mixed', 'pads_tampons', 'pads_only', 'tampons_cups', 'mixed', 'mixed']

notes_templates = [
    'High turnout despite weather',
    'First event of semester',
    'Partnership with local school',
    'Record attendance',
    'Community center venue',
    'Exceeded capacity',
    'New volunteers joined',
    'College campus event',
    'Focus on reusable products',
    'Spring drive success',
    'Expanded coverage area',
    'Suburban outreach',
    'Growing chapter',
    'Strong volunteer turnout',
    'Consistent demand',
    'Nearly at capacity',
    'University partnership',
    'Earth Day event',
    'Back to school prep',
    'Summer program',
    'Peak performance',
    'Holiday giving surge',
    'Winter weather challenges',
    'New location tested',
    'Collaborative event with another org',
    'Strong community support',
    'First-time beneficiaries',
    'Repeat location',
    'Wealthy area outreach',
    'Underserved community focus',
]

# Generate events over 2 years
events = []
event_id = 1
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

for chapter_name, info in chapters.items():
    # Determine number of events (more established chapters have more events)
    if 'Bay Area' in chapter_name or 'Chicago' in chapter_name or 'New York' in chapter_name:
        num_events = random.randint(18, 24)  # Established chapters
    elif 'Phoenix' in chapter_name or 'Minneapolis' in chapter_name:
        num_events = random.randint(10, 14)  # Newer chapters
    else:
        num_events = random.randint(14, 18)  # Mid-tier chapters
    
    # Generate random dates throughout the period
    dates = []
    for _ in range(num_events):
        random_days = random.randint(0, (end_date - start_date).days)
        event_date = start_date + timedelta(days=random_days)
        dates.append(event_date)
    
    dates.sort()
    
    # Generate events for this chapter
    for i, event_date in enumerate(dates):
        # Calculate capacity with growth over time
        months_from_start = (event_date.year - start_date.year) * 12 + (event_date.month - start_date.month)
        capacity = int(info['base_capacity'] * (info['growth_rate'] ** (months_from_start / 12)))
        
        # Add seasonal variation (higher in fall/spring, lower in summer)
        season_multiplier = 1.0
        month = event_date.month
        if month in [3, 4, 5, 9, 10, 11]:  # Spring and Fall
            season_multiplier = 1.1
        elif month in [6, 7, 8]:  # Summer
            season_multiplier = 0.9
        elif month == 12:  # December holiday boost
            season_multiplier = 1.15
        
        capacity = int(capacity * season_multiplier)
        
        # Products distributed (based on efficiency and some randomness)
        efficiency = info['efficiency']
        # Add some random variation
        actual_efficiency = efficiency + np.random.normal(0, 0.05)
        actual_efficiency = max(0.7, min(1.0, actual_efficiency))  # Keep between 70-100%
        
        products_distributed = int(capacity * actual_efficiency)
        
        # Volunteers (scales with capacity)
        base_volunteers = max(6, int(capacity / 40))
        volunteers = base_volunteers + random.randint(-2, 4)
        volunteers = max(5, volunteers)
        
        # Event duration (3-5 hours typically)
        duration = random.choice([3, 3.5, 4, 4, 4.5])
        
        # Beneficiaries (roughly 2.5 products per beneficiary on average)
        beneficiaries = int(products_distributed / (2.3 + random.uniform(-0.3, 0.5)))
        
        # Product type
        product_type = random.choice(product_types)
        
        # City
        city = random.choice(info['cities'])
        
        # Notes
        note = random.choice(notes_templates)
        
        events.append({
            'event_id': f'E{event_id:04d}',
            'chapter_name': chapter_name,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'state': info['state'],
            'city': city,
            'products_distributed': products_distributed,
            'carrying_capacity': capacity,
            'volunteers_present': volunteers,
            'estimated_beneficiaries': beneficiaries,
            'product_type': product_type,
            'event_duration_hours': duration,
            'notes': note
        })
        
        event_id += 1

# Create DataFrame
df = pd.DataFrame(events)
df = df.sort_values('event_date').reset_index(drop=True)

# Save to CSV
df.to_csv('t4t_synthetic_data_large.csv', index=False)

print(f"Generated {len(df)} events across {len(chapters)} chapters")
print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
print(f"\nEvents per chapter:")
print(df['chapter_name'].value_counts().sort_values(ascending=False))
print(f"\nTotal products distributed: {df['products_distributed'].sum():,}")
print(f"Total beneficiaries: {df['estimated_beneficiaries'].sum():,}")