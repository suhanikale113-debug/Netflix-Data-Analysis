import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.show()

# Set graph style
sns.set_style("whitegrid")

os.makedirs("charts", exist_ok=True)
# ==========================================
# TASK 1 - DATA CLEANING & PREPARATION
# ==========================================

df = pd.read_csv("Dataset.csv")


print("\n===== FIRST 5 ROWS =====")
print(df.head())


print("\n===== DATASET SHAPE =====")
print(df.shape)


print("\n===== COLUMN NAMES =====")
print(df.columns)


print("\n===== DATA TYPES =====")
print(df.dtypes)


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())


# ==========================================
# CLEANING THE DATA
# ==========================================

df = df.drop_duplicates()

text_columns = [
    'type',
    'title',
    'director',
    'country',
    'rating',
    'listed_in'
]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()


df['date_added'] = pd.to_datetime(df['date_added'])


print("\n===== AFTER CLEANING =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

# ==========================================
# TASK 2 - CONTENT TYPE ANALYSIS
# ==========================================

# Count Movies and TV Shows
type_counts = df['type'].value_counts()

print("\n===== MOVIES VS TV SHOWS =====")
print(type_counts)


type_percentage = df['type'].value_counts(normalize=True) * 100

print("\n===== CONTENT PERCENTAGE =====")
print(type_percentage.round(2))


# ------------------------------------------
# Bar Chart
# ------------------------------------------



plt.figure(figsize=(8, 5))

plt.bar(
    ["Movie", "TV Show"],
    [6126, 2664]
)

plt.title("Netflix Content Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.savefig(
    "charts/Task1.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Task1.png created successfully")
# ------------------------------------------
# Pie Chart
# ------------------------------------------

plt.figure(figsize=(7, 7))

plt.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Netflix Movies vs TV Shows")

plt.show()


# ==========================================
# TASK 3 - COUNTRY-WISE ANALYSIS
# ==========================================


country_counts = df['country'].value_counts()

print("\n===== TOP 10 COUNTRIES =====")
print(country_counts.head(10))


# ------------------------------------------
# Top 10 Countries Chart
# ------------------------------------------

top_countries = country_counts.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()
plt.show()


# ------------------------------------------
# Country Ranking
# ------------------------------------------

country_ranking = country_counts.reset_index()

country_ranking.columns = [
    'Country',
    'Content_Count'
]

country_ranking['Rank'] = (
    country_ranking['Content_Count']
    .rank(method='dense', ascending=False)
    .astype(int)
)

country_ranking = country_ranking[
    ['Rank', 'Country', 'Content_Count']
]

print("\n===== COUNTRY RANKING =====")
print(country_ranking.head(15))



# ==========================================
# TASK 4 - TREND ANALYSIS BY RELEASE YEAR
# ==========================================

# Count content by release year
yearly_content = (
    df.groupby('release_year')
      .size()
      .reset_index(name='content_count')
)

print("\n===== YEARLY CONTENT =====")
print(yearly_content)


# ------------------------------------------
# Year-over-Year Growth
# ------------------------------------------

yearly_content['YoY_Growth_%'] = (
    yearly_content['content_count']
    .pct_change() * 100
)

print("\n===== YEARLY GROWTH =====")
print(
    yearly_content[
        yearly_content['release_year'] >= 2010
    ].round(2)
)


# ------------------------------------------
# Release Year Trend Chart
# ------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    yearly_content['release_year'],
    yearly_content['content_count'],
    marker='o'
)

plt.title("Netflix Content Releases by Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------
# Recent Trend: 2010-2021
# ------------------------------------------

recent_years = yearly_content[
    yearly_content['release_year'] >= 2010
]

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=recent_years,
    x='release_year',
    y='content_count',
    marker='o'
)

plt.title("Netflix Content Trend: 2010-2021")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ------------------------------------------
# Find Peak Year
# ------------------------------------------

peak_year = yearly_content.loc[
    yearly_content['content_count'].idxmax()
]

print("\n===== PEAK RELEASE YEAR =====")
print("Year:", peak_year['release_year'])
print("Number of Titles:", peak_year['content_count'])


# ==========================================
# MOVIES VS TV SHOWS OVER TIME
# ==========================================

year_type = pd.crosstab(
    df['release_year'],
    df['type']
)

recent_type = year_type.loc[2010:2021]

recent_type.plot(
    figsize=(14, 6),
    marker='o'
)

plt.title("Movies vs TV Shows Released by Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()