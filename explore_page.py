import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
def shorten_categories(categories,cutoff):
    categorical_map={}
    for i in range(len(categories)):
        if categories.values[i]>=cutoff:
            categorical_map[categories.index[i]]=categories.index[i]
        else:
            categorical_map[categories.index[i]]='Other'
    return categorical_map
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x== 'Less than 1 year':
        return 0.5
    return float(x)
def clean_education(x):
    x = str(x) # Handle NaN or non-string types
    
    if "Bachelor" in x:
        return "Bachelor's degree"
    if "Master" in x:
        return "Master's degree"
    if "Professional degree" in x or "doctoral" in x:
        return "Post grad"
    
    return "Less than a Bachelor"

import os

@st.cache_data
def load_data():
    try:
        # Try to load from local file first
        df = pd.read_csv("survey_results_public.csv")
    except FileNotFoundError:
        try:
            # Try working GitHub URL as backup
            url = "https://raw.githubusercontent.com/stackoverflow/survey-data/main/2020/survey_results_public.csv"
            df = pd.read_csv(url)
        except Exception as e:
            # If all else fails, create sample data
            st.warning("Could not load survey data. Using sample data instead.")
            df = pd.DataFrame({
                'Country': ['United States', 'India', 'Germany', 'United Kingdom', 'Canada'] * 100,
                'EdLevel': ['Bachelor\'s degree', 'Master\'s degree', 'Post grad', 'Less than a Bachelor'] * 125,
                'YearsCodePro': [1, 3, 5, 10, 15, 20, 25, 30] * 62 + [1],
                'ConvertedCompYearly': [60000, 80000, 100000, 120000, 140000] * 100,
                'Employment': 'Employed full-time'
            })
    
    salary_column = None
    for candidate in ["ConvertedCompYearly", "ConvertedComp", "ConvertedCompTotal"]:
        if candidate in df.columns:
            salary_column = candidate
            break
    if salary_column is None:
        raise KeyError("No salary column found in survey_results_public.csv")

    df = df[["Country", "EdLevel", "YearsCodePro", salary_column, "Employment"]]
    df = df.rename({salary_column: "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]
    df = df[df["Country"] != "Other"]
    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    return df
df=load_data()
def show_explore_page():
    st.title("Explore Software Developer Salaries")
    st.write("""### Stack Overflow Developer Survey 2020""")
    data=df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.write("""#### Number of Data from different countries""")
    st.pyplot(fig1)# Create a pie chart for the distribution of education levels
    st.write("""#### Mean Salary based on Country""")
    data=df.groupby("Country")["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)
    st.write("""#### Mean Salary based on Experience""")
    data=df.groupby("YearsCodePro")["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)