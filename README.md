# Customer Segmentation for E-commerce Personalization

## Project Overview

This project aims to segment customers based on their browsing behavior, purchase history, and demographic data to personalize product recommendations and marketing campaigns. By leveraging multiple free-to-use APIs, we will gather data on customers, locations, and weather conditions. The analysis will be performed using Python libraries such as `pandas`, `numpy`, and `scikit-learn`.

## Objectives

- Fetch customer data using the Odoo API.
- Obtain location data and geocode addresses using the OpenStreetMap API.
- Fetch weather data for the locations of the customers using a Weather API.
- Perform data cleaning and preparation to handle missing values and merge datasets.
- Conduct exploratory data analysis (EDA) to identify trends and correlations.
- Segment customers using clustering techniques.
- Summarize key findings and provide actionable insights.

## APIs Used

1. **Odoo API**: For fetching customer data, including browsing behavior, purchase history, and demographic information.
2. **OpenStreetMap API**: For obtaining location data and geocoding addresses.
3. **Weather API**: For fetching weather data that might influence customer behavior.

## Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `requests`

## Steps to Implement

1. **Set Up Environment**:
   - Install necessary libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`.
   - Obtain API keys for Odoo, OpenStreetMap, and a Weather API.

2. **Fetch Data**:
   - Use the Odoo API to fetch customer data, including browsing behavior, purchase history, and demographic information.
   - Use the OpenStreetMap API to get location data and geocode addresses.
   - Use the Weather API to fetch weather data for the locations of the customers.

3. **Data Cleaning and Preparation**:
   - Clean the fetched data to handle missing values, incorrect data types, and duplicates.
   - Perform feature engineering to create new features or transform existing ones.

4. **Exploratory Data Analysis (EDA)**:
   - **Descriptive Statistics**: Calculate mean, median, standard deviation, and other statistics for numerical features.
   - **Correlation Analysis**: Analyze the correlation between features.
   - **Visualization**:
     - **Histograms**: Plot histograms for numerical features.
     - **Box Plots**: Create box plots to visualize the distribution of numerical features.
     - **Heatmaps**: Create heatmaps to visualize the correlation matrix.

5. **Clustering**:
   - Use K-means clustering to segment customers based on their behavior and demographic data.
   - Determine the optimal number of clusters using the Elbow Method or Silhouette Analysis.

6. **Cluster Analysis**:
   - Analyze the characteristics of each cluster to understand customer segments.
   - Visualize the clusters using scatter plots, pair plots, and other visualization techniques.

7. **Personalization**:
   - Use the customer segments to personalize product recommendations and marketing campaigns.
   - Implement a recommendation system based on the clusters.

8. **Insights and Conclusions**:
   - Summarize the key findings from the clustering analysis.
   - Identify the most influential features for customer segmentation.
   - Provide actionable insights or recommendations based on the analysis.

9. **Documentation and Presentation**:
   - Document the entire process, including code, visualizations, and findings.
   - Create a presentation or report to showcase the results.

## My Pitch
We shall provide a comprehensive analysis of customer segmentation using clustering techniques and real-time data from APIs. By leveraging K-means clustering and performing EDA, we can segment customers based on their behavior and demographic data. The insights and findings from this project can be used to personalize product recommendations and marketing campaigns, ultimately enhancing customer experience and driving sales

## Example Code Snippet

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import requests

# Fetch customer data from Odoo API
odoo_api_key = 'your_odoo_api_key'
odoo_url = f'https://your-odoo-instance.com/api/customers?apikey={odoo_api_key}'
odoo_response = requests.get(odoo_url)
odoo_data = odoo_response.json()

# Fetch location data from OpenStreetMap API
osm_url = 'https://nominatim.openstreetmap.org/search'
params = {'q': 'New York, USA', 'format': 'json'}
osm_response = requests.get(osm_url, params=params)
osm_data = osm_response.json()

# Fetch weather data from Weather API
weather_api_key = 'your_weather_api_key'
weather_url = f'http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q=New York'
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Example data processing and visualization
odoo_df = pd.DataFrame(odoo_data['customers'])
osm_df = pd.DataFrame(osm_data)
weather_df = pd.DataFrame([weather_data['current']])

# Merge dataframes
merged_df = pd.concat([odoo_df, osm_df, weather_df], axis=1)

# Feature engineering
merged_df['TotalSpent'] = merged_df['Quantity'] * merged_df['UnitPrice']

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(merged_df[['Quantity', 'UnitPrice', 'TotalSpent']])

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(data_scaled)

# Cluster analysis
sns.pairplot(merged_df, hue='Cluster')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(data_scaled, merged_df['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
