import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

sns.set_style("whitegrid")

# Create charts folder automatically
os.makedirs("charts", exist_ok=True)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("Dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# TASK 1 - DATA CLEANING & PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("TASK 1 - DATA CLEANING & PREPARATION")
print("=" * 60)

# Check missing values
missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

# Check duplicates
duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

# Remove duplicate rows
df = df.drop_duplicates()

# Clean text columns if they exist
text_columns = [
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "rating",
    "duration",
    "listed_in"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

# Convert date_added if available
if "date_added" in df.columns:
    df["date_added"] = pd.to_datetime(
        df["date_added"],
        errors="coerce"
    )

# Save cleaned dataset
df.to_csv("Netflix_Cleaned.csv", index=False)

print("\nCleaned dataset saved as:")
print("Netflix_Cleaned.csv")


# ------------------------------------------------------------
# TASK 1 CHART
# Missing Values
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    x=missing_values.index,
    y=missing_values.values
)

plt.title("Task 1 - Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "charts/Task1_Missing_Values.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task1_Missing_Values.png")


# ============================================================
# TASK 2 - CONTENT TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TASK 2 - CONTENT TYPE ANALYSIS")
print("=" * 60)

# Count Movie and TV Show
type_counts = df["type"].value_counts()

print("\nMovies vs TV Shows:")
print(type_counts)

# Percentage
type_percentage = (
    df["type"]
    .value_counts(normalize=True)
    * 100
)

print("\nContent Percentage:")
print(type_percentage.round(2))


# ------------------------------------------------------------
# TASK 2 CHART 1
# Movie vs TV Show Bar Chart
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.barplot(
    x=type_counts.index,
    y=type_counts.values
)

plt.title("Task 2 - Netflix Content Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    "charts/Task2_Content_Distribution.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task2_Content_Distribution.png")


# ------------------------------------------------------------
# TASK 2 CHART 2
# Movie vs TV Show Pie Chart
# ------------------------------------------------------------

plt.figure(figsize=(8, 8))

plt.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Task 2 - Movies vs TV Shows")

plt.tight_layout()

plt.savefig(
    "charts/Task2_Content_Percentage.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task2_Content_Percentage.png")


# ------------------------------------------------------------
# TASK 2 CHART 3
# Top 10 Ratings
# ------------------------------------------------------------

if "rating" in df.columns:

    rating_counts = df["rating"].value_counts().head(10)

    print("\nTop Ratings:")
    print(rating_counts)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=rating_counts.values,
        y=rating_counts.index
    )

    plt.title("Task 2 - Top 10 Netflix Content Ratings")
    plt.xlabel("Number of Titles")
    plt.ylabel("Rating")

    plt.tight_layout()

    plt.savefig(
        "charts/Task2_Top_Ratings.png",
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: charts/Task2_Top_Ratings.png")


# ============================================================
# TASK 3 - COUNTRY-WISE NETFLIX CONTENT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 - COUNTRY-WISE CONTENT ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Clean and split country information
# ------------------------------------------------------------

country_data = df[
    ["country", "type"]
].dropna(subset=["country"]).copy()

# Split multiple countries
country_data["country"] = country_data["country"].str.split(",")

# One country per row
country_data = country_data.explode("country")

# Remove spaces
country_data["country"] = country_data["country"].str.strip()

# Remove empty countries
country_data = country_data[
    country_data["country"] != ""
]

# Count content by country
country_counts = country_data["country"].value_counts()

print("\nTop 10 Countries:")
print(country_counts.head(10))


# ------------------------------------------------------------
# TASK 3 CHART 1
# Top 10 Countries
# ------------------------------------------------------------

top_countries = country_counts.head(10)

plt.figure(figsize=(11, 7))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Task 3 - Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    "charts/Task3_Top_Countries.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task3_Top_Countries.png")


# ------------------------------------------------------------
# TASK 3 CHART 2
# Movies vs TV Shows by Top Countries
# ------------------------------------------------------------

country_type = pd.crosstab(
    country_data["country"],
    country_data["type"]
)

top_country_names = country_counts.head(10).index

country_type_top = country_type.loc[
    top_country_names
]

country_type_top.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Task 3 - Movies vs TV Shows by Top 10 Countries"
)

plt.xlabel("Country")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "charts/Task3_Country_Content_Type.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task3_Country_Content_Type.png")


# ============================================================
# TASK 4 - TREND ANALYSIS BY RELEASE YEAR
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 - TREND ANALYSIS BY RELEASE YEAR")
print("=" * 60)

# Count content by release year
yearly_content = (
    df.groupby("release_year")
      .size()
      .reset_index(name="content_count")
)

print("\nYearly Content:")
print(yearly_content.tail(10))


# ------------------------------------------------------------
# TASK 4 CHART 1
# Netflix Content Releases by Year
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    yearly_content["release_year"],
    yearly_content["content_count"],
    marker="o"
)

plt.title(
    "Task 4 - Netflix Content Releases by Year"
)

plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "charts/Task4_Content_Trend.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task4_Content_Trend.png")


# ------------------------------------------------------------
# TASK 4 CHART 2
# Movies vs TV Shows Over Time
# ------------------------------------------------------------

year_type = pd.crosstab(
    df["release_year"],
    df["type"]
)

# Use recent years where enough data exists
recent_type = year_type.loc[
    year_type.index >= 2010
]

recent_type.plot(
    figsize=(14, 7),
    marker="o"
)

plt.title(
    "Task 4 - Movies vs TV Shows Released by Year"
)

plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    "charts/Task4_Movies_vs_TV_Shows.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/Task4_Movies_vs_TV_Shows.png")


# ============================================================
# TASK 4 - PEAK YEAR
# ============================================================

peak_year = yearly_content.loc[
    yearly_content["content_count"].idxmax()
]

print("\nPeak Release Year:")
print("Year:", peak_year["release_year"])
print("Number of Titles:", peak_year["content_count"])


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("ALL 4 TASKS COMPLETED!")
print("=" * 60)

print("\nCharts have been saved inside the 'charts' folder.")