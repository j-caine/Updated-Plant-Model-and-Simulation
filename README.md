# Updated-Plant-Model-and-Simulation
 This project models native plant growth and pollinator interactions using daily weather data, and provides outputs illustrating plant biomass over time, plant offspring, and pollinator movement.

# Overview
The simulation orechestrates:
 1. Data Loading: Uses 'DataProcessor' to read CSV files containing plant, bee, butterfly, and weather data.
 2. Plant Growth: Applies a logistic growth model based on Growing Degree Datas (GDD), temperature, precipitation, and competition among plants.
 3. Pollination: Matches pollinators (bees/butterflies) to blooming plants accodring to color affinity and active months, then tracks their movements over time.
 4. Visualization: Produces multi-line charts showing plant biomass over time, and a 2D scatter plot showing plant positions and pollinator trails.

# Features
 1. Flexible Data Loading: Adaptable to new plant or pollinator data by changing the CSV inputs.
 2. Growth Modeling: Extensible for more complex growth equations or varied environmental parameters.
 3. Pollinator Behavior: Supports color-based matching and monthly activity ranges.
 4. Visuals: Leverages Plotly for detailed charts and 2D scatter plots.

# Data Requirements
You'll need four CSV files
 1. plants.csv
 2. bees.csv
 3. butterflies.csv
 4. weather.csv
Place these CSV files in the same directory as the Python script or update the code paths accordingly.

# Output and Visualization
 1. Console Output: Prints a preview of the daily bioimass DataFrame (first 10 days) and the final plant states (with positions and biomass).
 2. Plotly Figures:
    - Daily Biomass: Multi-line chart showing each plant's biomass across simulation days.
    - 2D Scatter:
      - Large blue circles represent parent plants.
      - Smaller green circles represent newly sprouted "child plants".
      - Dashed lines represent pollinator movement.
        
# Extending the Project
- Additional Pollinators: Add new data columns or species in the CSV.
- Advanced Growth Equations: Replace or enhance the logic in PlantGrowthModel.update_biomass with new formulas to capture more nuanced growth factors.
- More Complex Weather Variables: Incorporate additional weather data/columns (wind speed, humidity, etc.).
- GUI: Expand the Visualizer to include interactive dashboards or embedded plots.
