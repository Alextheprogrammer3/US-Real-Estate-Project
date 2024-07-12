# US Real Estate Trends Analysis

## Project Overview

This project involves analyzing a dataset of real estate trends across the United States from the year 2000 to 2024. The goal was to clean, transform, and visualize the data to uncover insights into home value trends over time.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Methods and Approach](#methods-and-approach)
- [Data Cleaning and Transformation](#data-cleaning-and-transformation)
- [Data Analysis and Visualization](#data-analysis-and-visualization)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run the Code](#how-to-run-the-code)
- [License](#license)

## Data Description

The dataset `USRealEstateTrends.csv` contains historical data on home values across various regions in the United States from 2000 to 2024. The dataset includes the following columns:

- `RegionID`: Unique identifier for the region
- `SizeRank`: Rank of the region by size
- `RegionName`: Name of the region
- `StateName`: State abbreviation
- `2000-02-HomeValue` to `2024-05-HomeValue`: Monthly home values from February 2000 to May 2024
- `2024-03-DaysPending`: Number of days pending for the property in March 2024
- `2024-03-CutRaw`: Raw data cut value for March 2024

## Methods and Approach

1. **Data Loading and Initial Exploration:**
   - Loaded the dataset and examined the first few rows to understand its structure and identify issues.
   - Command:
     ```python
     import pandas as pd
     df = pd.read_csv('USRealEstateTrends.csv')
     print(df.head())
     ```

2. **Data Cleaning:**
   - Removed rows with missing values and duplicated entries.
   - Standardized column names for consistency.
   - Saved the cleaned data to a new CSV file.
   - Commands:
     ```python
     df_cleaned = df.dropna()
     df_cleaned.columns = [col.lower() for col in df_cleaned.columns]
     df_cleaned = df_cleaned.drop_duplicates()
     df_cleaned.to_csv('USRealEstateTrends_cleaned.csv', index=False)
     ```

3. **Further Data Cleaning:**
   - Filled remaining missing values with the mean of the columns.
   - Replaced spaces in column names with underscores.
   - Commands:
     ```python
     df_cleaned = df.dropna()
     df_cleaned = df.fillna(df.mean())
     df_cleaned.columns = [col.lower().replace(' ', '_') for col in df_cleaned.columns]
     df_cleaned = df_cleaned.drop_duplicates()
     df_cleaned.to_csv('USRealEstateTrends_cleaned.csv', index=False)
     ```

4. **Data Transformation:**
   - Reshaped the data from wide to long format for analysis.
   - Commands:
     ```python
     value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
     df_melted = df_cleaned.melt(id_vars=['regionid', 'sizerank', 'regionname', 'statename'], 
                                 value_vars=value_columns, 
                                 var_name='date', 
                                 value_name='homevalue')
     ```

5. **Data Analysis and Visualization:**
   - Analyzed the distribution of home values and created a histogram with gradient coloring.
   - Commands:
     ```python
     import matplotlib.pyplot as plt
     
     min_value = df_melted['homevalue'].min()
     max_value = df_melted['homevalue'].max()
     mean_value = df_melted['homevalue'].mean()
     
     plt.figure(figsize=(10, 6))
     n, bins, patches = plt.hist(df_melted['homevalue'], bins=50, edgecolor='black')
     
     bin_centers = 0.5 * (bins[:-1] + bins[1:])
     col = bin_centers - min(bin_centers)
     col /= max(col)
     
     cm = plt.cm.get_cmap('viridis')
     
     for c, p in zip(col, patches):
         plt.setp(p, 'facecolor', cm(c))
     
     plt.axvline(min_value, color='red', linestyle='dashed', linewidth=1)
     plt.axvline(max_value, color='green', linestyle='dashed', linewidth=1)
     plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=1)
     
     plt.text(min_value, max(n), f'Min: {min_value:.2f}', color='red', fontsize=12)
     plt.text(max_value, max(n), f'Max: {max_value:.2f}', color='green', fontsize=12)
     plt.text(mean_value, max(n), f'Mean: {mean_value:.2f}', color='blue', fontsize=12)
     
     plt.xlabel('Home Value')
     plt.ylabel('Frequency')
     plt.title('Distribution of Home Values with Gradient Coloring')
     
     plt.show()
     ```

   ![Histogram of Home Values](https://github.com/user-attachments/assets/ff3307c5-4dcd-448d-8776-5ebc0b38b3b3)

6. **Top 10 Regions by Average Home Value:**
   - Analyzed and visualized the top 10 regions with the highest average home values.
   - Commands:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt
     
     df = pd.read_csv('USRealEstateTrends.csv')
     
     df_cleaned = df.dropna()
     df_cleaned = df.fillna(df.mean(numeric_only=True))
     
     value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
     df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                                 value_vars=value_columns, 
                                 var_name='Date', 
                                 value_name='HomeValue')
     
     average_home_values = df_melted.groupby('RegionName')['HomeValue'].mean()
     
     top_regions = average_home_values.sort_values(ascending=False).head(10)
     
     plt.figure(figsize=(14, 8))
     bars = plt.bar(top_regions.index, top_regions.values, color='teal')
     
     plt.xlabel('Region')
     plt.ylabel('Average Home Value')
     plt.title('Top 10 Regions with the Highest Average Home Values')
     plt.xticks(rotation=45, ha='right')
     
     for bar in bars:
         height = bar.get_height()
         plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
                  ha='center', va='bottom', fontsize=10, color='black')
     
     plt.tight_layout()
     plt.show()
     ```

   ![Top 10 Regions by Average Home Value](https://github.com/user-attachments/assets/006f9532-9199-4007-aafa-ebc58266c9da)

7. **Average Home Values by State with Gradient Color Bar:**
   - Created a bar graph with a gradient color to represent average home values by state.
   - Commands:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt
     import matplotlib.colors as mcolors

     df = pd.read_csv('USRealEstateTrends.csv')

     df_cleaned = df.dropna()
     df_cleaned = df.fillna(df.mean(numeric_only=True))

     value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
     df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                                 value_vars=value_columns, 
                                 var_name='Date', 
                                 value_name='HomeValue')

     average_home_values_per_state = df_melted.groupby('StateName')['HomeValue'].mean()
     sorted_states = average_home_values_per_state.sort_values(ascending=False)

     plt.figure(figsize=(14, 10))

     cmap = plt.get_cmap('coolwarm')
     norm = mcolors.Normalize(vmin=sorted_states.min(), vmax=sorted_states.max())

     bars = plt.bar(sorted_states.index, sorted_states.values, color=cmap(norm(sorted_states.values)))

     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
     sm.set_array([])
     plt.colorbar(sm, label='Average Home Value')

     plt.xlabel('State')
     plt.ylabel('Average Home Value')
     plt.title('Average Home Values by State in the US (Gradient Color Bar)')

     plt.xticks(rotation=90)

     for bar in bars:
         height = bar.get_height()
         plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
                  ha='center', va='bottom', fontsize=10, color='black')

     plt.tight_layout()
     plt.show()
     ```

   ![Average Home Values by State with Gradient Color Bar](https://github.com/user-attachments/assets/965ce5c4-0ebc-49a6-9e58-98dfa0036bf0)

8. **Top 10 NYC Regions by Home Value:**
   - Analyzed and visualized the top 10 regions in New York City by average home value.
   - Commands:
     ```python
     df_nyc = df_cleaned[df_cleaned['regionname'].str.contains('New York')]
     
     df_melted_nyc = df_nyc.melt(id_vars=['regionname', 'statename'], 
                                 value_vars=value_columns, 
                                 var_name='Date', 
                                 value_name='HomeValue')
     
     average_home_values_nyc = df_melted_nyc.groupby('regionname')['HomeValue'].mean()
     
     top_nyc_regions = average_home_values_nyc.sort_values(ascending=False).head(10)
     
     plt.figure(figsize=(14, 8))
     bars = plt.bar(top_nyc_regions.index, top_nyc_regions.values, color='darkorange')
     
     plt.xlabel('NYC Region')
     plt.ylabel('Average Home Value')
     plt.title('Top 10 NYC Regions by Average Home Values')
     plt.xticks(rotation=45, ha='right')
     
     for bar in bars:
         height = bar.get_height()
         plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
                  ha='center', va='bottom', fontsize=10, color='black')
     
     plt.tight_layout()
     plt.show()
     ```

   ![Top 10 NYC Regions by Average Home Value](https://github.com/user-attachments/assets/31e6f430-b058-42e7-8b0c-df37ecf1461b)

## Results

The analysis provided several key insights:

- The distribution of home values exhibits a wide range, with significant variation across different states and regions.
- The top 10 regions with the highest average home values are predominantly located in major metropolitan areas.
- The average home values by state show that coastal states, especially those on the East and West Coasts, tend to have higher home values.
- Specific regions in New York City exhibit some of the highest home values in the dataset.

## Technologies Used

- **Python**: For data cleaning, transformation, and analysis.
- **Pandas**: For data manipulation and cleaning.
- **Matplotlib**: For data visualization.

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/USRealEstateTrendsAnalysis.git
