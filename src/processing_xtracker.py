import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file just copy the information from the website in a googe sheet and download it as a CSV file
file_path = '.csv'
df = pd.read_csv(file_path)

# Function to parse 'Fine [€]' column
def parse_fine(value):
    try:
        return float(value.replace(',', '').replace(' ', ''))
    except:
        return np.nan

# Apply transformation
df['Fine [€]'] = df['Fine [€]'].apply(parse_fine)

# Parse dates and extract year
df['Year'] = pd.to_datetime(df['Date of Decision'], errors='coerce').dt.year

# Calculate statistics
total_fines = len(df)
unknown_fines = df['Fine [€]'].isna().sum()

# Grouping fines by year
fines_by_year = df.groupby('Year').agg(
    number_of_fines=('Fine [€]', 'count'),
    sum_of_fines=('Fine [€]', 'sum')
).reset_index()

# Grouping by country
fines_by_country = df.groupby('Country').agg(
    number_of_fines=('Fine [€]', 'count'),
    sum_of_fines=('Fine [€]', 'sum')
).reset_index()

# Handling multiple quoted articles
df['Quoted Art.'] = df['Quoted Art.'].fillna('')
df['Quoted Art. List'] = df['Quoted Art.'].apply(lambda x: x.split(', '))
exploded_articles = df.explode('Quoted Art. List')

# Grouping by article
violated_articles = exploded_articles.groupby('Quoted Art. List').agg(
    number_of_violations=('Fine [€]', 'count'),
    total_fines=('Fine [€]', 'sum')
).reset_index().sort_values(by='number_of_violations', ascending=False)

# Grouping by Type
fines_by_type = df.groupby('Type').agg(
    number_of_fines=('Fine [€]', 'count'),
    sum_of_fines=('Fine [€]', 'sum')
).reset_index().sort_values(by='number_of_fines', ascending=False)

# Plotting distribution of fines by year
plt.figure(figsize=(10, 6))
plt.bar(fines_by_year['Year'], fines_by_year['sum_of_fines'], color='lightblue')
plt.title('Sum of Fines by Year')
plt.xlabel('Year')
plt.ylabel('Sum of Fines (€)')
plt.show()

# Plotting distribution of fines by country
plt.figure(figsize=(10, 6))
plt.bar(fines_by_country['Country'], fines_by_country['sum_of_fines'], color='lightcoral')
plt.title('Sum of Fines by Country')
plt.xlabel('Country')
plt.ylabel('Sum of Fines (€)')
plt.xticks(rotation=90)
plt.show()

# Basic statistics
average_fine = df['Fine [€]'].mean()
median_fine = df['Fine [€]'].median()

top_10_percent_avg = df['Fine [€]'].quantile(0.9)
top_10_percent = df[df['Fine [€]'] >= top_10_percent_avg]['Fine [€]'].mean()

bottom_10_percent_avg = df['Fine [€]'].quantile(0.1)
bottom_10_percent = df[df['Fine [€]'] <= bottom_10_percent_avg]['Fine [€]'].mean()

# Top 5 largest fines
top_5_fines = df.nlargest(5, 'Fine [€]')

# Top 3 fines by article
top_fines_by_article = exploded_articles.groupby('Quoted Art. List').apply(
    lambda x: x.nlargest(3, 'Fine [€]')
).reset_index(drop=True)

# Top 3 sum of fines aggregated by article
top_3_sum_fines_by_article = exploded_articles.groupby('Quoted Art. List').agg(
    total_fines=('Fine [€]', 'sum')
).reset_index().sort_values(by='total_fines', ascending=False).head(3)

# Total sum of fines, earliest and latest date
total_sum_of_fines = df['Fine [€]'].sum()
min_date = pd.to_datetime(df['Date of Decision'], errors='coerce').min()
max_date = pd.to_datetime(df['Date of Decision'], errors='coerce').max()

# Output the statistics
print(f"Total fines: {total_fines}")
print(f"Unknown fines: {unknown_fines}")
print(f"Average Fine: €{average_fine}")
print(f"Median Fine: €{median_fine}")
print(f"Top 10% Average Fine: €{top_10_percent}")
print(f"Bottom 10% Average Fine: €{bottom_10_percent}")
print(f"Total sum of fines: €{total_sum_of_fines}")
print(f"Fines recorded from {min_date} to {max_date}")

# Output relevant tables
print("\nTop 5 Largest Fines:\n", top_5_fines[['Country', 'Fine [€]', 'Controller/Processor']])
print("\nMost Violated Articles:\n", violated_articles)
print("\nFines by Year:\n", fines_by_year)
print("\nFines by Country:\n", fines_by_country)
print("\nTop 3 Fines by Article:\n", top_fines_by_article)
print("\nTop 3 Sum of Fines by Article:\n", top_3_sum_fines_by_article)
