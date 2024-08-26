# Portfolio
# 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("Top_1000_Companies_Dataset.csv")
df.head()
df.info()
df.describe()
duplicated_data = df.duplicated().any()
duplicated_data
df.isnull().sum()
#drop accelerator, btype
df = df.drop(['Accelerator', 'btype'], axis=1)
#Convert growth_percentage to float
df['growth_percentage'] = df['growth_percentage'].str.rstrip('%').astype(float)
import re

def convert_funding_to_numeric(funding_str):
    if pd.isnull(funding_str):
        return None
    
    # Regular expression pattern to match the desired formats
    pattern = r'^([€£$CA])([\d.]+)[MB]$'
    
    match = re.match(pattern, funding_str)
    if match:
        currency = match.group(1)
        value = float(match.group(2))
        if 'B' in funding_str:
            multiplier = 1e9
        elif 'M' in funding_str:
            multiplier = 1e6
        else:
            return None
        
        if currency in ['$', 'CA']:
            return value * multiplier
        elif currency == '€':
            # Convert Euros to dollars (approximate conversion rate)
            return value * multiplier * 1.2  # Example conversion rate, adjust as needed
        elif currency == '£':
            # Convert British Pounds to dollars (approximate conversion rate)
            return value * multiplier * 1.4  # Example conversion rate, adjust as needed
        
    return None

df['total_funding'] = df['total_funding'].apply(convert_funding_to_numeric)

# Display the DataFrame
df.head()

# Display the DataFrame
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

features =['valuation', 'estimated_revenues', 'total_funding', 'growth_percentage']
# Visualize distributions
plt.figure(figsize=(12,6))
for i, feature in enumerate(features, 1):
    plt.subplot(1, len(features), i)
    plt.hist(df[feature].dropna(), bins=20, color='blue', alpha=0.7)
    plt.title(f'Distribution of {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
plt.tight_layout()
plt.show()
#Scatter plot
sns.pairplot(df[features].dropna(), diag_kind='kde')
plt.title('Scatter Plot Matrix', y=1.02)
plt.show()
import plotly.express as px

industry_count = df['Industry'].value_counts()
industry_df = pd.DataFrame({'Industry': industry_count.index, 'Company Counts': industry_count.values})
industry_df = industry_df.sort_values(by='Company Counts', ascending=False)

fig = px.treemap(industry_df, path=['Industry'], values='Company Counts', title='Distribution of Companies Across Industries')
fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))  
fig.show()
#Top 10 industries
top_industries = df.dropna().groupby('Industry')['valuation'].mean().nlargest(10).reset_index()
print(top_industries.head(10))

plt.figure(figsize=(10,6))
sns.barplot(data=top_industries, x='Industry', y='valuation', palette='rocket')
plt.title('Top 10 Industries with Higest Average Valuation')
plt.xticks(rotation=45)
plt.show()
filtered_df = df[df['Industry'].isin(top_industries['Industry'])]
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='Industry', y='valuation')
plt.xticks(rotation=45)
plt.title('Valuation Distribution by Industry')
plt.show()
#Top 10 companies with highest valuation 
key_features = ['total_funding', 'estimated_revenues', 'growth_percentage']
top_10_company = df.dropna().nlargest(10, 'valuation')
for i in key_features:
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_10_company, x='company_name', y=i, palette='viridis')
    plt.title(f'{i} of Top 10 Companies with Highest Valuation')
plt.xticks(rotation=45)
plt.show()
df['country'] = df['country'].replace('USA', 'United States')
geo_df = df.groupby(['country', 'state', 'city']).size().reset_index(name='count')

fig = px.treemap(geo_df, path=['country', 'state', 'city'], values='count',
                 labels={'city': 'City', 'state':'State','country':'Country', 'count':'Number of Companies'},
                 title='Hierarchical Treemap: Companies by Country, State, and City')
                
fig.show()
import plotly.express as px

# Create a DataFrame with top 20 countries and their total valuation
top_countries = df.groupby('country')['valuation'].sum().nlargest(20).reset_index()

# Calculate company counts by country
company_count_by_country = df['country'].value_counts().reset_index()
company_count_by_country.columns = ['country', 'company_count']

# Merge company counts into the top_countries DataFrame
top_countries = top_countries.merge(company_count_by_country, on='country')

# Create the choropleth map
fig = px.choropleth(top_countries, 
                    locations='country', 
                    locationmode='country names',
                    color='valuation',
                    hover_name='country',
                    hover_data=['company_count'],  # Add company count to hover information
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title='Choropleth Map of Top 20 Countries by Total Valuation')

# Custom hover template
hover_template = '<b>%{hovertext}</b><br>' + \
                 'Valuation: $%{z:.2f}<br>' + \
                 'Company Count: %{customdata}'
fig.update_traces(hovertemplate=hover_template)

fig.show()
# Correlation
corr_matrix = df[features].dropna().corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
# Define criteria
port_criteria = ['company_name', 'valuation', 'growth_percentage']
# Filter and sort companies
port_company = df.sort_values(by=port_criteria, ascending=False).head(10)
# Group companies by industry and select top company from each industry
port_company = port_company.groupby('Industry', group_keys=False).apply(lambda x: x.nlargest(1, 'valuation'))
print("Selected Portforlio Companies:")
print(port_company[['company_name', 'valuation', 'growth_percentage']])
# Calculate allocation for each company
total_invest = 1000000
port_company['allocation'] = (port_company['valuation']/port_company['valuation'].sum()) * total_invest

print("\nAllocation for Each Company:")
print(port_company[['company_name', 'allocation']])
import numpy as np
from scipy.optimize import minimize

# Calculate log returns for each company
log_returns = np.log(1 + port_company['growth_percentage'] / 100)
# Calculate portfolio expected return and volatility
weights = port_company['allocation'] / total_invest
portfolio_return = np.sum(weights * log_returns.mean())

# Calculate covariance matrix
cov_matrix = np.cov(log_returns, rowvar=False)

# Calculate portfolio volatility
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print("\nPortfolio Expected Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
# Define the objective function for portfolio optimization
def objective(weights):
    port_return = np.sum(weights * log_returns.mean())
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(log_returns, rowvar=False), weights)))
    return -port_return / port_volatility
# Define constraints and initial weights
constraints = tuple((0, 1) for _ in range(len(port_company)))
initial_weights = np.ones(len(port_company)) / len(port_company)
# Portfolio optimization
result = minimize(objective, initial_weights, method='SLSQP', bounds=constraints)
optimal_weights = result.x
print("\nOptimal Portfolio Allocation:")
for i in range(len(port_company)):
    print(f"{port_company['company_name'].iloc[i]}: {optimal_weights[i]:.2%}")
