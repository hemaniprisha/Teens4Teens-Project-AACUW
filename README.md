# Teens4Teens Data Engineering Project - Comprehensive Analysis

## Overview

This analysis demonstrates the value of data-driven decision-making for Teens4Teens resource allocation and operational planning using a realistic synthetic dataset spanning 2 years of operations across 15 chapters.

## Deliverables

### 1. **t4t_synthetic_data_large.csv**
Comprehensive synthetic dataset containing **243 chapter giving events** across **15 chapters** spanning **January 2023 - December 2024** (2 full years).

**Dataset Characteristics:**
- **243 events** across **15 chapters**
- **13 states** and **69 cities** covered
- **102,033 total products** distributed
- **42,834 beneficiaries** reached
- **11,239 volunteer hours** contributed
- Realistic growth patterns (14.8% YoY)
- Seasonal variations modeled
- Chapter-specific characteristics (capacity, efficiency, growth rates)

**Columns:**
- `event_id`: Unique event identifier (E0001-E0243)
- `chapter_name`: T4T chapter name
- `event_date`: Date of giving event
- `state`, `city`: Geographic location
- `products_distributed`: Number of period products distributed
- `carrying_capacity`: Maximum products the chapter can distribute
- `volunteers_present`: Number of volunteers at the event
- `estimated_beneficiaries`: Estimated people who received products
- `product_type`: Type of products (mixed, pads_tampons, pads_only, tampons_cups)
- `event_duration_hours`: Length of event (3-5 hours)
- `notes`: Qualitative observations

### 2. **Analysis Scripts**

**make_dataset.py:**
- Synthetic data generation script with sophisticated modeling
- Chapter-specific growth rates and efficiency parameters
- Seasonal variation algorithms
- Realistic randomization with controlled variance

**initial_analysis.py:**
- Complete analysis pipeline (400+ lines)
- Statistical summaries and KPIs
- Year-over-year growth analysis
- Chapter performance comparison
- Resource allocation insights
- Temporal trend analysis (quarterly, seasonal, monthly)
- Geographic distribution analysis
- Correlation analysis
- Predictive inventory estimates
- Growth trajectory analysis
- Automated visualization generation (16 charts)

### 3. **Visualizations**

**t4t_comprehensive_analysis_large.png** (16-panel dashboard):
1. Total products distributed by chapter
2. Capacity utilization by chapter (color-coded: red >90%, orange 80-90%, green <80%)
3. Quarterly products distribution trend
4. Year-over-year comparison (2023 vs 2024)
5. Volunteer efficiency by chapter
6. Top 10 states by products distributed
7. Products per beneficiary distribution
8. Seasonal patterns (Winter, Spring, Summer, Fall)
9. Monthly products with trend line
10. Number of events by chapter
11. Monthly average capacity utilization
12. Total beneficiaries by chapter
13. Total volunteer hours by chapter
14. Product type distribution (pie chart)
15. Capacity gap distribution
16. Chapter growth rate (1st vs 2nd half comparison)

**t4t_correlation_heatmap_large.png:**
- Correlation matrix showing relationships between operational metrics
- Reveals key performance drivers
- Products vs Volunteers: 0.709
- Products vs Capacity: 0.968
- Products vs Beneficiaries: 0.910

**t4t_geographic_analysis_large.png** (4-panel geographic dashboard):
1. Total products by state (bar chart)
2. Top 15 cities by products distributed
3. Quarterly events trend for top 5 states
4. Chapter performance matrix (scatter plot showing capacity utilization vs volunteer efficiency, bubble size = total products)

## Findings

### Performance Overview
- **Total Impact**: 102,033 products distributed to 42,834 beneficiaries
- **Growth**: 14.8% year-over-year increase (2023→2024)
- **Average Capacity Utilization**: 91.1% (very high!)
- **Average Volunteer Efficiency**: 35.7 products/volunteer
- **Products per Beneficiary**: 2.41 average

### Top Performing Chapters
1. **New York Metro**: 12,696 products (23 events, 93.2% capacity)
2. **Chicago South**: 11,423 products (24 events, 90.2% capacity)
3. **Bay Area Chapter**: 11,112 products (23 events, 94.1% capacity)
4. **Dallas Fort Worth**: 7,101 products (14 events, 94.8% capacity)
5. **Miami South**: 6,814 products (15 events, 91.7% capacity)

### High-Demand Chapters (>90% Capacity) That Need Urgent Action
**10 chapters** running at >90% capacity - these need immediate resource expansion:

| Chapter | Avg Capacity | Events | Avg Gap | Recommendation |
|---------|--------------|--------|---------|----------------|
| Dallas Fort Worth | 94.8% | 14 | 28 | Increase capacity 20% |
| Bay Area Chapter | 94.1% | 23 | 30 | Increase capacity 20% |
| New York Metro | 93.2% | 23 | 40 | Increase capacity 20% |
| Austin Central | 92.4% | 14 | 31 | Increase capacity 20% |
| Miami South | 91.7% | 15 | 41 | Increase capacity 15% |
| Seattle North | 91.7% | 14 | 39 | Increase capacity 15% |
| Boston Metro | 91.0% | 14 | 33 | Increase capacity 15% |
| Atlanta Metro | 90.8% | 16 | 43 | Increase capacity 15% |
| San Diego South | 90.4% | 14 | 42 | Increase capacity 15% |
| Chicago South | 90.2% | 24 | 52 | Increase capacity 15% |

### Efficiency Insights for Volunteer Allocation

**Efficiency Variation**: 1.2x between highest and lowest

**Top Efficiency Chapters:**
1. Austin Central: 39.0 products/volunteer
2. New York Metro: 37.4 products/volunteer
3. Seattle North: 37.2 products/volunteer

**Chapters that Need Improvement:**
- Minneapolis North: 32.1 products/volunteer
- Atlanta Metro: 32.5 products/volunteer
- Denver Metro: 33.4 products/volunteer

**Potential Impact**: Bringing the bottom 3 chapters up to median efficiency could increase output by ~16%

### Seasonal Patterns

| Season | Avg Products/Event | Total Events |
|--------|-------------------|--------------|
| Winter | 445 | 54 |
| Spring | 443 | 62 |
| Fall | 436 | 65 |
| Summer | 357 | 62 |

**Discrepancy: t**: Summer shows 24.5% lower distribution. This is an opportunity for targeted campaigns

### Year-over-Year Growth (2023 → 2024)

| Metric | 2023 | 2024 | Growth |
|--------|------|------|--------|
| Products | 47,497 | 54,536 | +14.8% |
| Beneficiaries | 20,018 | 22,816 | +14.0% |
| Events | 114 | 129 | +13.2% |
| Volunteers | 1,360 | 1,583 | +16.4% |

### Geographic Distribution

**Top 5 States by Products Distributed:**
1. California: 16,596 products (2 chapters, 37 events)
2. New York: 12,696 products (1 chapter, 23 events)
3. Texas: 12,462 products (2 chapters, 28 events)
4. Illinois: 11,423 products (1 chapter, 24 events)
5. Florida: 6,814 products (1 chapter, 15 events)

**Expansion Opportunities:**
- States with <15 events: AZ (11), MN (12), MA (14), PA (14), WA (14), FL (15)
- Single-chapter states: All except CA and TX
- Major metros without chapters: Houston, Phoenix, Detroit, Charlotte, Columbus

### Product Type Distribution
- **Mixed products**: 51.4% of events (most versatile)
- **Tampons & cups**: 18.1% of events (focus on reusables)
- **Pads only**: 16.5% of events
- **Pads & tampons**: 14.0% of events

### Fastest Growing Chapters
1. Phoenix Valley: +18.2% (emerging market)
2. Austin Central: +10.5% (strong growth)
3. New York Metro: +8.5% (sustained expansion)
4. Miami South: +7.4% (building momentum)
5. Chicago South: +7.2% (steady growth)

## Recommendations

### 1. Capacity Expansion (Next 30 Days)

**Priority 1:**
- Dallas Fort Worth: 536 → 643 products (+20%)
- Bay Area Chapter: 513 → 616 products (+20%)
- New York Metro: 592 → 710 products (+20%)
- Austin Central: 414 → 497 products (+20%)

**Priority 2: :**
- Miami South: 495 → 569 products (+15%)
- Seattle North: 433 → 498 products (+15%)
- Boston Metro: 366 → 421 products (+15%)
- Atlanta Metro: 471 → 542 products (+15%)
- San Diego South: 438 → 504 products (+15%)
- Chicago South: 529 → 608 products (+15%)

**Expected Impact**: Accommodate ~15,000 additional products annually

### 2. Volunteer Optimization (short term)

**Benchmark**: 35.9 products/volunteer (median)

**Action Plan:**
1. Document best practices from Austin Central (39.0) and New York Metro (37.4)
2. Conduct efficiency audit in bottom 5 chapters
3. Implement cross-training program
4. Share workflow optimizations across network
5. Target 10% efficiency improvement in bottom quartile

**Expected Impact**: +3,000-5,000 additional products distributed with same volunteer base

### 3. Geographic Expansion (Next 90 Days)

**Part 1** (Existing states, add events):
- Florida: Add 1 chapter in Orlando or Tampa
- Texas: Establish chapter in Houston or San Antonio
- California: Add chapter in Sacramento or Fresno
- Target: +30 events/year, +12,000 products

**Part 2** (New states):
- Ohio (Columbus, Cleveland)
- Michigan (Detroit)
- North Carolina (Charlotte, Raleigh)
- Target: +20 events/year, +8,000 products

### 4. Seasonal Planning (Ongoing)

**Summer Boost Initiative:**
- Increase outreach by 25% in June-August
- Partner with summer programs and camps
- Target capacity: 445 products/event (match Winter levels)
- Expected lift: +5,000 products annually

**Holiday Surge Preparation:**
- November-December consistently show highest demand
- Pre-stock inventory +20% in Q4
- Recruit additional seasonal volunteers
- Plan for 15-20% above normal capacity

### 5. Product Mix Optimization (Next Quarter)

**Current State:**
- 51% mixed events (good baseline)
- Opportunity to increase reusable products (cups) to 25% of mix
- Partner with sustainable product manufacturers

**Recommended Mix:**
- Mixed: 45% (maintain flexibility)
- Tampons & cups: 25% (increase reusables)
- Pads only: 15% (maintain)
- Pads & tampons: 15% (maintain)

## Predictive Inventory Estimates

### Next Quarter Projections (Q1 2025)

| Chapter | Estimated Products | Estimated Events |
|---------|-------------------|------------------|
| New York Metro | 1,735 | 2.9 |
| Chicago South | 1,524 | 3.0 |
| Bay Area Chapter | 1,419 | 2.9 |
| Dallas Fort Worth | 909 | 1.8 |
| Miami South | 888 | 1.9 |
| Atlanta Metro | 874 | 2.0 |
| Denver Metro | 748 | 2.2 |
| Portland West | 741 | 2.1 |
| Seattle North | 727 | 1.8 |
| San Diego South | 711 | 1.8 |
| Philadelphia East | 706 | 1.8 |
| Austin Central | 694 | 1.8 |
| Boston Metro | 585 | 1.8 |
| Minneapolis North | 515 | 1.5 |
| Phoenix Valley | 425 | 1.4 |
| **TOTAL** | **~13,600** | **~29** |

**Planning Notes:**
- Add 15% buffer for high-demand chapters
- Account for Q1 seasonal patterns (higher than summer, lower than fall)
- Pre-order inventory by mid-December for January distribution

## Technical Implementation

### Data Pipeline Architecture
```
Raw Event Data (Google Form)
    ↓
Google Sheets (Master Dataset)
    ↓
Python ETL Pipeline (pandas)
    ├── Data Validation & Cleaning
    ├── Derived Metrics Calculation
    └── Historical Data Integration
    ↓
Analytics Engine
    ├── Statistical Analysis (scipy)
    ├── Predictive Modeling (sklearn)
    ├── Visualization Generation (matplotlib, seaborn)
    └── Report Generation
    ↓
Automated Outputs
    ├── Monthly Performance Reports (PDF)
    ├── Chapter Scorecards (Excel)
    ├── Executive Dashboard (Interactive)
    └── Inventory Alerts (Email)
```

### Analysis Techniques Applied

1. **Descriptive Statistics**
   - Central tendency (mean, median)
   - Dispersion (std dev, quartiles)
   - Distribution analysis

2. **Time Series Analysis**
   - Trend identification (linear regression)
   - Seasonal decomposition
   - Year-over-year comparisons
   - Quarterly aggregations

3. **Correlation Analysis**
   - Pearson correlation coefficients
   - Relationship mapping
   - Driver identification

4. **Performance Metrics**
   - Capacity utilization rates
   - Efficiency ratios
   - Growth trajectories
   - Comparative benchmarking

5. **Predictive Analytics**
   - Moving averages
   - Trend extrapolation
   - Seasonal adjustment
   - Inventory forecasting

6. **Geospatial Analysis**
   - State-level aggregation
   - City-level distribution
   - Coverage gap identification
   - Expansion opportunity mapping

## Implementation Roadmap

### Part 1: Data Centralization
**Objective**: Establish data collection infrastructure

- [ ] Design Google Form for post-event data entry
- [ ] Create automated Google Sheets master database
- [ ] Implement data validation rules
- [ ] Set up automated backup system
- [ ] Train chapter leaders on data entry (2 sessions)
- [ ] Create quick reference guide
- [ ] Launch pilot with 3 chapters

**Deliverables**: 
- Standardized form template
- Training materials
- Initial data from pilot chapters

### Part 2: ETL Pipeline
**Objective**: Automate data processing

- [ ] Develop Python scripts to pull from Google Sheets API
- [ ] Build data cleaning and validation logic
- [ ] Create derived metrics calculations
- [ ] Implement error handling and logging
- [ ] Set up automated monthly execution
- [ ] Create data quality dashboard
- [ ] Test with historical data

**Deliverables**:
- Python ETL codebase
- Data quality report
- Automated execution schedule

### Part 3: Analytics & Dashboards 
**Objective**: Deploy analytics and visualization

- [ ] Build automated report generation scripts
- [ ] Create interactive dashboards (Streamlit or Tableau)
- [ ] Implement alert system for capacity issues
- [ ] Develop predictive inventory models
- [ ] Create chapter-specific scorecards
- [ ] Build executive summary templates
- [ ] Set up email distribution lists

**Deliverables**:
- 3 dashboard views (Executive, Operational, Chapter)
- Monthly report templates
- Alert notification system

### Part 4: Rollout & Training (Weeks 7-8)
**Objective**: Operationalize across all chapters

- [ ] Conduct organization-wide training (webinar)
- [ ] Roll out to all 15 chapters
- [ ] Establish monthly review cadence
- [ ] Create feedback mechanism
- [ ] Document all processes
- [ ] Develop troubleshooting guides
- [ ] Launch continuous improvement program

**Deliverables**:
- User documentation
- Training recordings
- Feedback collection system
- Process documentation

### Part 5: Optimization
**Objective**: Continuous improvement

- [ ] Monthly data quality reviews
- [ ] Quarterly model refinement
- [ ] Semi-annual process audits
- [ ] Annual strategy sessions
- [ ] Feature enhancement backlog
- [ ] Best practice sharing sessions

## Expected Impact & ROI

- Admin time reduction
- Response time improvement
- Resource optimization
- Efficient inventory management
- Program growth
- Increased volunteer efficiency
- 
### Next Steps
1. Review findings and recommendations
2. Prioritize implementation phases
3. Allocate resources for Phase 1 (data collection infrastructure)
4. Schedule kickoff meeting with Teens4Teens 
5. Begin pilot program with 3 chapters

**Analysis Generated**: February 2026  
**Data Period**: January 2023 - December 2024 (synthetic)  
**Total Events**: 243  
**Chapters**: 15  
**States Covered**: 13  
**Total Products**: 102,033  
**Total Beneficiaries**: 42,834  
**Volunteer Hours**: 11,239  
**Year-over-Year Growth**: +14.8%
