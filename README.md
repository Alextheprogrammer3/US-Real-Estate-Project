# US-Real-Estate-Project
Real
import pandas as pd

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Display the first few rows of the dataset
print(df.head())

# Drop rows with missing values
df_cleaned = df.dropna()

# Alternatively, fill missing values with a specific value (e.g., 0)
# df_cleaned = df.fillna(0)

# Fill missing values with the mean of the column
# df_cleaned = df.fillna(df.mean())

# Convert column names to lowercase
df_cleaned.columns = [col.lower() for col in df_cleaned.columns]

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Display summary statistics
print(df_cleaned.describe())

# Save the 
df_cleaned.to_csv('USRealEstateTrends_cleaned.csv', index=False)

# Display the first few rows of the dataset
print(df.head())

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with a specific value (e.g., 0)
# df_cleaned = df.fillna(0)

# Fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean())

# Convert column names to lowercase and replace spaces with underscores
df_cleaned.columns = [col.lower().replace(' ', '_') for col in df_cleaned.columns]

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Display summary statistics to ensure data is cleaned
print(df_cleaned.describe())

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('USRealEstateTrends_cleaned.csv', index=False)
    RegionID  SizeRank       RegionName StateName  2000-02-HomeValue  \
0    102001         0    United States       NaN      123048.375901   
1    394913         1     New York, NY        NY      217413.348751   
2    753899         2  Los Angeles, CA        CA      230191.681333   
3    394463         3      Chicago, IL        IL      154332.573521   
4    394514         4       Dallas, TX        TX      128259.611168   

   2000-03-HomeValue  2000-04-HomeValue  2000-05-HomeValue  2000-06-HomeValue  \
0      123316.373392      123891.175404      124552.703606      125261.950998   
1      218341.844065      220223.334853      222171.683720      224331.715643   
2      231328.440994      233590.210855      236063.878014      238520.221749   
3      154604.414602      155280.687654      156094.162793      157007.101990   
4      128325.255585      128495.727241      128720.489451      128947.995217   

   2000-07-HomeValue  ...  2024-02-CutRaw  2024-03-HomeValue  \
0      126013.182906  ...         10000.0      357374.362080   
1      226691.604401  ...         25000.0      646508.747102   
2      241038.221287  ...         30000.5      952665.578913   
3      157978.803091  ...         10000.0      316875.294320   
4      129186.531899  ...         10000.0      379972.349667   

   2024-03-DaysPending  2024-03-CutRaw  2024-04-HomeValue  \
0                 42.0         10000.0      359240.114070   
1                 55.0         25000.0      652619.099940   
2                 32.0         34000.0      956266.687926   
3                 29.0         10000.0      319764.144323   
4                 42.0         10000.0      380957.392395   

   2024-04-DaysPending  2024-04-CutRaw  2024-05-HomeValue  \
0                 38.0         10000.0      360681.294250   
1                 51.0         26000.0      657279.223513   
2                 29.0         40012.0      962388.491425   
3                 25.0         10000.0      321897.252361   
4                 38.0         10000.0      381103.625851   

   2024-05-DaysPending  2024-05-CutRaw  
0                 37.0         10000.0  
1                 47.0         30000.0  
2                 28.0         40000.0  
3                 23.0         10000.0  
4                 38.0         10000.0  

[5 rows x 450 columns]
<ipython-input-2-339e1d13d82a>:16: FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
  df_cleaned = df.fillna(df.mean())
            regionid    sizerank  2000-02-homevalue  2000-03-homevalue  \
count     895.000000  895.000000         895.000000         895.000000   
mean   412099.672626  461.751955      109823.132178      109978.654801   
std     78377.355083  268.710532       33623.325119       33799.783441   
min    102001.000000    0.000000       24508.884229       24496.582232   
25%    394546.000000  230.500000      100241.757887      100015.977962   
50%    394795.000000  460.000000      109823.132178      109978.654801   
75%    395044.500000  689.500000      109823.132178      109978.654801   
max    753929.000000  939.000000      388736.507581      391739.288868   

       2000-04-homevalue  2000-05-homevalue  2000-06-homevalue  \
count         895.000000         895.000000         895.000000   
mean       110669.025722      111418.259858      111992.900574   
std         34343.743085       34858.616321       35347.024815   
min         24470.277297       24457.747178       24455.818269   
25%        100452.120723      101040.619762      101126.135014   
50%        110669.025722      111418.259858      111992.900574   
75%        110669.025722      111418.259858      111992.900574   
max        399871.036264      408396.675198      418165.004118   

       2000-07-homevalue  2000-08-homevalue  2000-09-homevalue  ...  \
count         895.000000         895.000000         895.000000  ...   
mean       112763.427730      113389.379398      114077.797909  ...   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean(numeric_only=True))

# Melt the dataframe to have a single column for home values
value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                            value_vars=value_columns, 
                            var_name='Date', 
                            value_name='HomeValue')

# Calculate statistics
min_value = df_melted['HomeValue'].min()
max_value = df_melted['HomeValue'].max()
mean_value = df_melted['HomeValue'].mean()

# Plot histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df_melted['HomeValue'], bins=50, edgecolor='black')

# Normalize the data to 0-1 for the gradient coloring
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)

# Use a color map
cm = plt.cm.get_cmap('viridis')

# Color the patches
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

# Plot statistics lines
plt.axvline(min_value, color='red', linestyle='dashed', linewidth=1)
plt.axvline(max_value, color='green', linestyle='dashed', linewidth=1)
plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=1)

# Add text annotations for the statistics
plt.text(min_value, max(n), f'Min: {min_value:.2f}', color='red', fontsize=12)
plt.text(max_value, max(n), f'Max: {max_value:.2f}', color='green', fontsize=12)
plt.text(mean_value, max(n), f'Mean: {mean_value:.2f}', color='blue', fontsize=12)

# Labels and title
plt.xlabel('Home Value')
plt.ylabel('Frequency')
plt.title('Distribution of Home Values with Gradient Coloring')

# Show plot
plt.show()


![download](https://github.com/user-attachments/assets/ff3307c5-4dcd-448d-8776-5ebc0b38b3b3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean(numeric_only=True))

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

# Plot the bar graph
plt.figure(figsize=(14, 8))
bars = plt.bar(top_regions.index, top_regions.values, color='teal')

# Add labels and title
plt.xlabel('Region')
plt.ylabel('Average Home Value')
plt.title('Top 10 Regions with the Highest Average Home Values')
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')

# Show plot
plt.tight_layout()
plt.show()
![download](https://github.com/user-attachments/assets/006f9532-9199-4007-aafa-ebc58266c9da)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean(numeric_only=True))

# Melt the dataframe to have a single column for home values
value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                            value_vars=value_columns, 
                            var_name='Date', 
                            value_name='HomeValue')

# Calculate average home values per state
average_home_values_per_state = df_melted.groupby('StateName')['HomeValue'].mean()

# Sort states by average home value in descending order
sorted_states = average_home_values_per_state.sort_values(ascending=False)

# Plot the bar graph with a gradient color
plt.figure(figsize=(14, 10))

# Define a colormap
cmap = plt.get_cmap('coolwarm')

# Normalize the average home values for the color gradient
norm = mcolors.Normalize(vmin=sorted_states.min(), vmax=sorted_states.max())

# Plot bars
bars = plt.bar(sorted_states.index, sorted_states.values, color=cmap(norm(sorted_states.values)))

# Add color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Average Home Value')

# Add labels and title
plt.xlabel('State')
plt.ylabel('Average Home Value')
plt.title('Average Home Values by State in the US (Gradient Color Bar)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')

# Show plot
plt.tight_layout()import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean(numeric_only=True))

# Melt the dataframe to have a single column for home values
value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                            value_vars=value_columns, 
                            var_name='Date', 
                            value_name='HomeValue')

# Filter the data for New York City (NYC) regions
df_nyc = df_melted[df_melted['StateName'] == 'NY']

# Find the top 10 regions in NYC by the most recent home value
df_nyc_recent = df_nyc[df_nyc['Date'].str.contains('2024-05')]
top_10_nyc = df_nyc_recent.sort_values(by='HomeValue').head(10)

# Plot the histogram
plt.figure(figsize=(12, 8))

# Define a colormap
cmap = plt.get_cmap('coolwarm')

# Normalize the home values for the color gradient
norm = mcolors.Normalize(vmin=top_10_nyc['HomeValue'].min(), 
                         vmax=top_10_nyc['HomeValue'].max())

# Create a bar plot with gradient color
bars = plt.barh(top_10_nyc['RegionName'], top_10_nyc['HomeValue'], color=[cmap(norm(val)) for val in top_10_nyc['HomeValue']])

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Home Value ($)')

# Add labels and title
plt.xlabel('Home Value ($)')
plt.ylabel('Region')
plt.title('Top 10 NYC Regions by Home Value (Lowest to Highest)')

# Add value labels on the bars
for bar in bars:
    height = bar.get_width()
    plt.text(height, bar.get_y() + bar.get_height()/2.0, f'{height:,.2f}', va='center', fontsize=10, color='black')

# Show plot
plt.tight_layout()
plt.show()

![download](https://github.com/user-attachments/assets/965ce5c4-0ebc-49a6-9e58-98dfa0036bf0)

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Drop rows with missing values (NaNs)
df_cleaned = df.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_cleaned = df.fillna(df.mean(numeric_only=True))

# Melt the dataframe to have a single column for home values
value_columns = [col for col in df_cleaned.columns if 'HomeValue' in col]
df_melted = df_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                            value_vars=value_columns, 
                            var_name='Date', 
                            value_name='HomeValue')

# Define a list of East Coast states
east_coast_states = [
    'CT', 'DE', 'FL', 'GA', 'MA', 'MD', 'ME', 'NH', 'NJ', 'NY', 'NC', 'PA', 'RI', 'SC', 'VA', 'WV'
]

# Filter the data for East Coast states
df_east_coast = df_melted[df_melted['StateName'].isin(east_coast_states)]

# Extract year-month from the Date column
df_east_coast['YearMonth'] = df_east_coast['Date'].str[:7]

# Group by state and YearMonth to calculate the average home value
df_avg_home_value = df_east_coast.groupby(['StateName', 'YearMonth'])['HomeValue'].mean().reset_index()

# Create the box plot
plt.figure(figsize=(14, 8))

# Create a box plot with East Coast states
df_avg_home_value.boxplot(column='HomeValue', by='StateName', grid=False, patch_artist=True, medianprops=dict(color='black'))

# Add labels and title
plt.title('Box Plot of Average Home Values on the East Coast (2012-2022)')
plt.suptitle('')  # Suppress the default title to keep the custom title
plt.xlabel('State')
plt.ylabel('Average Home Value ($)')
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()
![download](https://github.com/user-attachments/assets/9cc0e118-5bdf-4304-8323-192a298b7fca)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor

# Sample Data
data = {
    'State': ['CA', 'NY', 'FL', 'TX', 'PA', 'IL', 'OH', 'MI', 'GA', 'NC',
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

![download](https://github.com/user-attachments/assets/9541aba1-540c-48ab-a9bb-314c064fe70e)

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

![download](https://github.com/user-attachments/assets/c1033638-e3c4-4b87-a0ff-4c37d975c2e4)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Starting position for the first bar
start_x = 0

# Add bars to the plot
for idx, row in top_regions_data.iterrows():
    region_name = row['RegionName']
    home_value = row['AvgHomeValue']
    width = row['Width']
    
    ax.add_patch(Rectangle((start_x, 0), width, home_value, facecolor='teal', edgecolor='black'))
    
    # Add text labels
    ax.text(start_x + width / 2, home_value / 2, f'{region_name}\n${home_value:,.2f}', 
            ha='center', va='center', fontsize=10, color='white', weight='bold')
    
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

![download](https://github.com/user-attachments/assets/426bdd8d-cd36-4a6f-8d52-3329fc42c08e)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the dataset
df = pd.read_csv('USRealEstateTrends.csv')

# Filter for California regions
df_ca = df[df['StateName'] == 'CA']

# Drop rows with missing values (NaNs)
df_ca_cleaned = df_ca.dropna()

# Alternatively, fill missing values with the mean of the respective columns
df_ca_cleaned = df_ca_cleaned.fillna(df_ca_cleaned.mean(numeric_only=True))

# Melt the dataframe to have a single column for home values
value_columns = [col for col in df_ca_cleaned.columns if 'HomeValue' in col]
df_melted = df_ca_cleaned.melt(id_vars=['RegionID', 'SizeRank', 'RegionName', 'StateName'], 
                               value_vars=value_columns, 
                               var_name='Date', 
                               value_name='HomeValue')

# Extracting '2000-02' and '2024-05' home values
df_pivot = df_melted.pivot_table(index=['RegionID', 'RegionName'], 
                                 columns='Date', 
                                 values='HomeValue', 
                                 aggfunc='mean').reset_index()

# Correct the column names
df_pivot.columns = [col if col in ['RegionID', 'RegionName'] else col.replace('-HomeValue', '') for col in df_pivot.columns]

# Filter for required dates
required_columns = ['RegionID', 'RegionName', '2000-02', '2024-05']
if all(col in df_pivot.columns for col in required_columns):
    df_pivot = df_pivot[required_columns]
else:
    raise KeyError(f"Columns {required_columns} not found in the DataFrame")

# Calculate the change in home values
df_pivot['Change'] = df_pivot['2024-05'] - df_pivot['2000-02']

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Starting point
start_value = df_pivot['2000-02'].iloc[0]
x = 0
tick_positions = []
tick_labels = []

# Plot the initial value
ax.bar(x, start_value, color='blue')
tick_positions.append(x)
tick_labels.append('2000-02')

# Plot the changes
for idx, row in df_pivot.iterrows():
    x += 1
    ax.bar(x, row['Change'], bottom=start_value, color='green' if row['Change'] > 0 else 'red')
    start_value += row['Change']
    tick_positions.append(x)
    tick_labels.append(row['RegionName'])

# Plot the final value
x += 1
ax.bar(x, start_value, color='blue')
tick_positions.append(x)
tick_labels.append('2024-05')

# Customize the plot
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.set_ylabel('Home Value')
ax.set_title('Waterfall Chart of Home Value Changes in California from 2000-02 to 2024-05')

plt.tight_layout()
plt.show()

![download](https://github.com/user-attachments/assets/1ecb2139-27d1-43b7-a571-b8417f25fb2a)
