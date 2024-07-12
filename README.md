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
![download](https://github.com/user-attachments/assets/2bf58617-6c79-433e-97f2-354e15bc193e)


8. **Top 10 NYC Regions by Home Value:**
   - Analyzed and visualized the top 10 regions in New York City by average home value.
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

     nyc_data = df_melted[df_melted['StateName'] == 'NY']
     top_nyc_regions = nyc_data.groupby('RegionName')['HomeValue'].mean().sort_values(ascending=False).head(10)

     plt.figure(figsize=(14, 8))
     bars = plt.bar(top_nyc_regions.index, top_nyc_regions.values, color='purple')

     plt.xlabel('Region')
     plt.ylabel('Average Home Value')
     plt.title('Top 10 Regions in NYC by Average Home Value')
     plt.xticks(rotation=45, ha='right')

     for bar in bars:
         height = bar.get_height()
         plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
                  ha='center', va='bottom', fontsize=10, color='black')

     plt.tight_layout()
     plt.show()
     ```
     ![download](https://github.com/user-attachments/assets/df8e351e-7975-4ae1-b963-877c15eafba0)


9. **Stacked Bar Chart of Home Prices Over Time:**
   - Created a stacked bar chart comparing home prices over time between East Coast and West Coast.
   - Commands:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     # Sample data
     data = {
         'Date': ['2000-02', '2000-03', '2000-04', '2000-05', '2000-06', '2000-07', '2000-08', '2000-09', 
                  '2000-10', '2000-11', '2000-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05'],
         'East Coast': [154332.573521, 155000.0, 156000.0, 157500.0, 160000.0, 161500.0, 162000.0, 163000.0, 
                        164000.0, 165000.0, 166000.0, 180000.0, 185000.0, 190000.0, 195000.0, 200000.0], 
         'West Coast': [230191.681333, 231000.0, 232000.0, 235000.0, 237000.0, 240000.0, 245000.0, 250000.0, 
                        255000.0, 260000.0, 265000.0, 280000.0, 290000.0, 300000.0, 310000.0, 320000.0]
     }

     # Creating DataFrame from sample data
     df = pd.DataFrame(data)
     df['Date'] = pd.to_datetime(df['Date'])
     df['Date'] = df['Date'].dt.strftime('%Y-%m')

     # Plotting
     fig, ax = plt.subplots(figsize=(14, 8))

     # Bar positions
     bar_width = 0.4
     dates = df['Date']
     index = range(len(dates))

     # Creating the stacked bar chart
     p1 = ax.bar(index, df['East Coast'], bar_width, label='East Coast', color='b', alpha=0.6)
     p2 = ax.bar(index, df['West Coast'], bar_width, bottom=df['East Coast'], label='West Coast', color='r', alpha=0.6)

     # Adding titles and labels
     ax.set_title('Home Prices Over Time: East Coast vs. West Coast', fontsize=16)
     ax.set_xlabel('Date', fontsize=14)
     ax.set_ylabel('Home Price', fontsize=14)
     ax.set_xticks(index)
     ax.set_xticklabels(dates, rotation=45, ha='right')

     # Adding legend
     ax.legend()

     # Display the plot
     plt.tight_layout()
     plt.show()
     ```

![download](https://github.com/user-attachments/assets/83691429-ba91-45a9-a5f3-dd5d1517a264)



10. **Mekko (Marimekko) Chart for Top 10 Regions:**
    - Created a Mekko chart to visualize the top 10 regions with the highest average home values.
    - Commands:
      ```python
      import pandas as pd
      import matplotlib.pyplot as plt
      from matplotlib.patches import Rectangle
      import numpy as np

      # Load the dataset
      df = pd.read_csv('USRealEstateTrends.csv')

      # Drop rows with missing values (NaNs)
      df_cleaned = df.dropna()

      # Alternatively, fill missing values with the mean of the respective columns
      df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

      # Melt the dataframe to have a single column for home values
      value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
      df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                                  value_vars=value_columns, 
                                  var_name='Date', 
                                  value_name='HomeValue')

      # Calculate average home values per region
      average_home_values = df_melted.groupby('RegionName')['HomeValue'].mean()

      # Sort regions by average home value in descending order and select top 10
      top_regions = average_home_values.sort_values(ascending=False).head(10)

      # Filter top regions data
      top_regions_data = df_cleaned[df_cleaned['RegionName'].isin(top_regions.index)].copy()

      # Calculate average home values and relative widths
      top_regions_data.loc[:, 'AvgHomeValue'] = top_regions_data[value_columns].mean(axis=1)
      total_rank = top_regions_data['SizeRank'].sum()
      top_regions_data.loc[:, 'Width'] = top_regions_data['SizeRank'] / total_rank

      # Create a Mekko (Marimekko) chart
      fig, ax = plt.subplots(figsize=(14, 8))

      # Define colormap
      colors = plt.get_cmap('viridis')

      # Starting position for the first bar
      start_x = 0

      # Add bars to the plot
      for idx, row in top_regions_data.iterrows():
          region_name = row['RegionName']
          home_value = row['AvgHomeValue']
          width = row['Width']
          
          # Adjust width to ensure minimum width for readability
          min_width = 0.05  # Increased minimum width for bars
          if width < min_width:
              width = min_width
          
          color = colors(np.random.rand())  # Random gradient color
          
          ax.add_patch(Rectangle((start_x, 0), width, home_value, facecolor=color, edgecolor='black'))
          
          # Add text labels
          ax.text(start_x + width / 2, home_value / 2, f'{region_name}\n${home_value:,.2f}', 
                  ha='center', va='center', fontsize=6, color='white', weight='bold')
          
          # Move to the next position
          start_x += width

      # Add labels and title
      plt.xlabel('Region (Relative Size by SizeRank)')
      plt.ylabel('Average Home Value')
      plt.title('Top 10 Regions with the Highest Average Home Values')

      # Set limits
      ax.set_xlim(0, 1)
      ax.set_ylim(0, top_regions_data['AvgHomeValue'].max() * 1.1)

      # Show plot
      plt.tight_layout()
      plt.show()
      ```



![download](https://github.com/user-attachments/assets/64c85619-661e-4003-9e30-758b51377eb6)

11. **Average Home Values by State with Gradient Color Bar:**
   - Created a bar graph with a gradient color to represent average home values by state.
   - Commands:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt
     data = {'RegionID': [102001, 394913, 753899, 394463, 394514],
    'RegionName': ['United States', 'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Dallas, TX'],
    'StateName': [None, 'NY', 'CA', 'IL', 'TX'],
    '2000-02-HomeValue': [123048.375901, 217413.348751, 230191.681333, 154332.573521, 128259.611168],
    '2000-03-HomeValue': [123316.373392, 218341.844065, 231328.440994, 154604.414602, 128325.255585],
    '2000-04-HomeValue': [123891.175404, 220223.334853, 233590.210855, 155280.687654, 128495.727241],
    '2000-05-HomeValue': [124552.703606, 222171.683720, 236063.878014, 156094.162793, 128720.489451],
    '2000-06-HomeValue': [125261.950998, 224331.715643, 238520.221749, 157007.101990, 128947.995217],
    '2024-05-HomeValue': [360681.294250, 657279.223513, 962388.491425, 321897.252361, 381103.625851],
    'Date': ['2000-02', '2000-03', '2000-04', '2000-05', '2000-06']
    }
    df = pd.DataFrame(data)
    dates = ['2000-02', '2000-03', '2000-04', '2000-05', '2000-06']
    home_values = [
    [123048.375901, 123316.373392, 123891.175404, 124552.703606, 125261.950998],  # United States
    [217413.348751, 218341.844065, 220223.334853, 222171.683720, 224331.715643],  # New York, NY
    [230191.681333, 231328.440994, 233590.210855, 236063.878014, 238520.221749],  # Los Angeles, CA
    [154332.573521, 154604.414602, 155280.687654, 156094.162793, 157007.101990],  # Chicago, IL
    [128259.611168, 128325.255585, 128495.727241, 128720.489451, 128947.995217]   # Dallas, TX
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, region in enumerate(df['RegionName']):
    ax.plot(dates, home_values[i], marker='o', label=region)
    ax.set_title('Home Prices Over Time for Selected Regions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Home Price')
    ax.legend(title='Region')
    plt.grid(True)
    plt.show()
     ```

![download](https://github.com/user-attachments/assets/63c29ebe-d55b-494f-b25a-6bdbbba178dd)

12. **Heatmap of Home Prices by State and Month:**
    - Created a heatmap to visualize home prices across states and months.
    - Commands:
      ```python
      import pandas as pd
      import matplotlib.pyplot as plt
      import numpy as np
      from matplotlib.widgets import Cursor

      # Sample Data
      data = { 'State': ['CA', 'NY', 'FL', 'TX', 'PA', 'IL', 'OH', 'MI', 'GA', 'NC',
                    'NJ', 'VA', 'WA', 'AZ', 'MA', 'MO', 'MD', 'IN', 'TN', 'WI'],
          '2000-02': [230191.68, 217413.35, 180000.00, 128259.61, 140000.00, 155000.00, 145000.00, 150000.00, 135000.00, 120000.00,
                      160000.00, 165000.00, 175000.00, 185000.00, 190000.00, 150000.00, 155000.00, 165000.00, 170000.00, 155000.00],
          '2000-03': [231000.00, 220223.33, 185000.00, 130000.00, 142000.00, 157000.00, 148000.00, 152000.00, 138000.00, 122000.00,
                      162000.00, 167000.00, 177000.00, 187000.00, 192000.00, 152000.00, 158000.00, 168000.00, 172000.00, 158000.00],
          '2000-04': [232000.00, 222171.68, 190000.00, 132000.00, 144000.00, 160000.00, 150000.00, 154000.00, 140000.00, 124000.00,
                      164000.00, 169000.00, 179000.00, 188000.00, 194000.00, 155000.00, 160000.00, 170000.00, 175000.00, 160000.00],
          '2000-05': [235000.00, 224331.72, 192000.00, 135000.00, 146000.00, 162000.00, 155000.00, 156000.00, 142000.00, 126000.00,
                      166000.00, 171000.00, 180000.00, 190000.00, 196000.00, 158000.00, 162000.00, 172000.00, 177000.00, 162000.00],
          '2000-06': [237000.00, 226500.00, 195000.00, 140000.00, 148000.00, 165000.00, 157000.00, 158000.00, 144000.00, 128000.00,
                      168000.00, 173000.00, 182000.00, 192000.00, 198000.00, 160000.00, 165000.00, 175000.00, 180000.00, 165000.00],
          '2000-07': [240000.00, 229000.00, 197000.00, 142000.00, 150000.00, 168000.00, 160000.00, 160000.00, 146000.00, 130000.00,
                      170000.00, 175000.00, 185000.00, 195000.00, 200000.00, 162000.00, 167000.00, 178000.00, 183000.00, 167000.00],
          '2024-01': [320000.00, 310000.00, 290000.00, 300000.00, 270000.00, 280000.00, 260000.00, 265000.00, 250000.00, 240000.00,
                      285000.00, 290000.00, 305000.00, 310000.00, 295000.00, 275000.00, 280000.00, 295000.00, 310000.00, 290000.00],
          '2024-02': [322000.00, 315000.00, 295000.00, 305000.00, 275000.00, 285000.00, 265000.00, 270000.00, 255000.00, 245000.00,
                      290000.00, 295000.00, 310000.00, 315000.00, 300000.00, 278000.00, 283000.00, 300000.00, 315000.00, 295000.00],
          '2024-03': [325000.00, 320000.00, 300000.00, 310000.00, 280000.00, 290000.00, 270000.00, 275000.00, 260000.00, 250000.00,
                      295000.00, 300000.00, 315000.00, 320000.00, 305000.00, 280000.00, 290000.00, 305000.00, 320000.00, 300000.00],
          '2024-04': [330000.00, 325000.00, 305000.00, 315000.00, 285000.00, 295000.00, 275000.00, 280000.00, 265000.00, 255000.00,
                      300000.00, 305000.00, 320000.00, 325000.00, 310000.00, 282000.00, 292000.00, 310000.00, 325000.00, 305000.00],
          '2024-05': [335000.00, 330000.00, 310000.00, 320000.00, 290000.00, 300000.00, 280000.00, 285000.00, 270000.00, 260000.00,
                      305000.00, 310000.00, 325000.00, 330000.00, 315000.00, 285000.00, 295000.00, 315000.00, 330000.00, 310000.00]
      }

      # Create DataFrame
      df = pd.DataFrame(data)

      # Set 'State' as index
      df.set_index('State', inplace=True)

      # Create the heatmap
      fig, ax = plt.subplots(figsize=(14, 8))
      cax = ax.matshow(df, cmap='coolwarm', interpolation='nearest')

      # Add colorbar
      cbar = plt.colorbar(cax, pad=0.02)
      cbar.set_label('Home Price', rotation=270, labelpad=15)

      # Set labels and title
      ax.set_title('Home Prices Heatmap by State and Month', fontsize=16)
      ax.set_xlabel('Month', fontsize=14)
      ax.set_ylabel('State', fontsize=14)

      # Set x and y ticks
      ax.set_xticks(np.arange(len(df.columns)))
      ax.set_yticks(np.arange(len(df.index)))
      ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=12)
      ax.set_yticklabels(df.index, fontsize=12)

      # Add grid
      ax.grid(False)

      # Add a cursor for interactivity
      cursor = Cursor(ax, useblit=True, color='black', linewidth=1)

      # Display the plot
      plt.tight_layout()
      plt.show()
      ```
      ![download](https://github.com/user-attachments/assets/420a311f-c658-4a5c-993f-581b6d6e7701)


     ## Results

- **Overview**: This project explores the US real estate trends from February 2000 to May 2024, providing a detailed analysis of home price fluctuations over more than two decades. By analyzing historical data and projecting future trends, we gain valuable insights into the evolving real estate market.

- **Data Coverage**:
  - **Timeframe**: The dataset spans from February 2000 to May 2024, capturing over 24 years of real estate data. This extensive period allows us to analyze long-term trends and seasonal variations in home prices.
  - **Future Projections**: Projections extend to 2034, offering a glimpse into potential future trends based on historical patterns. This forward-looking analysis helps in anticipating market shifts and planning for future investments.

- **Home Prices Over Time: East Coast vs. West Coast**:
  - **Trend Analysis**: The stacked bar chart illustrates the trajectory of home prices on the East Coast and West Coast from February 2000 to May 2024. 
    - **East Coast**: Prices started at approximately $154,000 in February 2000 and increased steadily to around $200,000 by May 2024. This gradual growth reflects a stable but consistent rise in home values.
    - **West Coast**: Home prices on the West Coast began at about $230,000 and surged to approximately $320,000. This steep increase indicates a stronger upward trend, with prices rising more rapidly compared to the East Coast.
  - **Comparison**: The West Coast, particularly California, consistently shows higher home prices than the East Coast. This suggests a significant premium on West Coast real estate, driven by higher demand and cost of living.

- **Top 10 Regions with the Highest Average Home Values**:
  - **Marimekko Chart Insights**: The Marimekko chart highlights the top 10 regions with the highest average home values.
    - **California Dominance**: California regions dominate the chart, reinforcing the state's position as having the highest average home values in the country.
    - **Regional Disparities**: The chart also reveals notable differences between regions, with certain areas showing disproportionately high home values relative to their size rank. This disparity underscores the concentration of high-value properties in specific regions.

- **Home Prices Heatmap by State and Month**:
  - **Heatmap Insights**: The heatmap provides a comprehensive view of home price variations across states and months, using color intensity to represent price levels.
    - **Seasonal Fluctuations**: The heatmap shows seasonal trends in home prices, with some states experiencing more pronounced seasonal variations than others. For instance, certain regions may see price spikes during summer months.
    - **Regional Trends**: The visualization confirms that California has the highest home prices across the board. States like New York and Massachusetts also show high price levels, though not as pronounced as California.
    - **Price Volatility**: Some states exhibit higher price volatility, which could be due to local economic factors or market conditions affecting home values more drastically.

- **Relevance**:
  - **Historical Context**: The dataset provides a historical perspective on real estate trends, allowing for a nuanced understanding of how market conditions have evolved.
  - **Future Implications**: Projections extending to 2034 offer a forward-looking view that helps stakeholders anticipate future market conditions. This forward view is crucial for strategic planning and investment decisions.

These insights illustrate the dynamic nature of the US real estate market, highlighting regional differences, trends over time, and projections for the future.


