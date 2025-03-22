#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = "hotel_bookings.csv"
data = pd.read_csv(file_path)

# Display the first few rows
data.head()


# In[5]:

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry as pc

pd.options.display.max_columns = None


# In[3]:

df = data.copy()


# In[4]:

df.isnull().sum().sort_values(ascending=False)


# In[5]:

df.columns, len(df.index)


# In[6]:

df = df.drop(df[(df.adults+df.babies+df.children)==0].index)


## If no id of agent or company is null, just replace it with 0
df[['agent','company']] = df[['agent','company']].fillna(0.0)


## For the missing values in the country column, replace it with mode (value that appears most often)
df['country'].fillna(data['country'].mode()[0], inplace=True)



## for missing children value, replace it with rounded mean value
df['children'].fillna(round(data.children.mean()), inplace=True)


# In[7]:

# Ensure that the columns exist and contain valid integer values before converting
df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')

# In[8]:

import seaborn as sns
import matplotlib.pyplot as plt

def plot(x, y, x_label='', y_label='', title='', figsize=(10, 6), type='bar'):
    fig, ax = plt.subplots(figsize=figsize)

    if type == 'bar':
        sns.barplot(x=x, y=y, ax=ax)  # Corrected function call
    elif type == 'line':
        sns.lineplot(x=x, y=y, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()


# In[9]:

def get_count(series, limit=None):

    '''
    INPUT:
        series: Pandas Series (Single Column from DataFrame)
        limit:  If value given, limit the output value to first limit samples.
    OUTPUT:
        x = Unique values
        y = Count of unique values
    '''

    if limit != None:
        series = series.value_counts()[:limit]
    else:
        series = series.value_counts()

    x = series.index
    y = series/series.sum()*100

    return x.values,y.values


# In[10]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Create total nights column
df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']

# Calculate total revenue
df['revenue'] = df['adr'] * df['total_nights']

# Dropdown menu for selecting the time unit
time_unit = widgets.Dropdown(
    options=['Year', 'Month', 'Week'],
    value='Month',
    description='Time Unit:',
    style={'description_width': 'initial'}
)

# Function to plot revenue trends based on selected time unit
def plot_revenue_trends(time_unit):
    plt.figure(figsize=(12, 6))

    if time_unit == 'Year':
        revenue_trends = df.groupby(['arrival_date_year'])['revenue'].sum().reset_index()
        sns.lineplot(data=revenue_trends, x='arrival_date_year', y='revenue', marker='o')
        plt.title("Revenue Trends Over Time (Yearly)")
        plt.xlabel("Year")

    elif time_unit == 'Month':
        revenue_trends = df.groupby(['arrival_date_year', 'arrival_date_month'])['revenue'].sum().reset_index()
        sns.lineplot(data=revenue_trends, x='arrival_date_month', y='revenue', hue='arrival_date_year', marker='o')
        plt.title("Revenue Trends Over Time (Monthly)")
        plt.xlabel("Month")

    elif time_unit == 'Week':
        revenue_trends = df.groupby(['arrival_date_year', 'arrival_date_week_number'])['revenue'].sum().reset_index()
        sns.lineplot(data=revenue_trends, x='arrival_date_week_number', y='revenue', hue='arrival_date_year', marker='o')
        plt.title("Revenue Trends Over Time (Weekly)")
        plt.xlabel("Week Number")

    plt.xticks(rotation=45)
    plt.ylabel("Total Revenue")
    plt.show()

# Display the dropdown and update plot based on selection
widgets.interactive(plot_revenue_trends, time_unit=time_unit)


# In[11]:

import matplotlib.pyplot as plt
import seaborn as sns

# Set modern style
sns.set_style("whitegrid")

# Calculate cancellation rate
cancellation_rate = df['is_canceled'].mean() * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Get counts for canceled vs. not canceled
cancel_counts = df['is_canceled'].value_counts(normalize=True) * 100
labels = ['Not Canceled', 'Canceled']
colors = ['#2ecc71', '#e74c3c']  # Green for not canceled, red for canceled

# Plot
plt.figure(figsize=(6, 6))
ax = sns.barplot(x=labels, y=cancel_counts, palette=colors)

# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Titles and labels
plt.ylabel("Percentage", fontsize=12)
plt.title("Cancellation Rate", fontsize=14, fontweight="bold")
plt.ylim(0, 100)

plt.show()


# In[12]:

import matplotlib.pyplot as plt
import seaborn as sns

# Set modern style
sns.set_style("whitegrid")

# Count bookings per country
country_counts = df['country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Number of Bookings']

# Select top 10 countries
top_countries = country_counts.head(10)

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=top_countries, x='Country', y='Number of Bookings', palette="Spectral")

# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",  
                (p.get_x() + p.get_width() / 2., p.get_height()),  
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Titles and labels
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Number of Bookings", fontsize=12)
plt.title("Top 10 Countries by Number of Bookings", fontsize=14, fontweight="bold")

plt.show()


# In[13]:

import matplotlib.pyplot as plt
import seaborn as sns

# Set modern style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.histplot(df['lead_time'], bins=50, kde=True, color='#3498db', edgecolor='black', linewidth=1.2)

# Titles and labels
plt.title("Distribution of Booking Lead Time", fontsize=14, fontweight="bold")
plt.xlabel("Days Before Arrival", fontsize=12)
plt.ylabel("Number of Bookings", fontsize=12)


plt.show()


# In[14]:

import matplotlib.pyplot as plt
import seaborn as sns

# Set modern style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(x=df['customer_type'], palette='pastel', edgecolor='black')

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()}",  
                (p.get_x() + p.get_width() / 2., p.get_height()),  
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Titles and labels
plt.title("Distribution of Customer Types", fontsize=14, fontweight="bold")
plt.xlabel("Customer Type", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Rotate x-labels if needed
plt.xticks(rotation=20)

plt.show()


# In[15]:

plt.figure(figsize=(12, 6))
sns.histplot(df['adr'], bins=50, kde=True, color='green')
plt.title("Distribution of Average Daily Rate (ADR)")
plt.xlabel("ADR Value")
plt.ylabel("Frequency")
plt.show()


# In[16]:

# In[17]:

import faiss
from sklearn.preprocessing import StandardScaler

# Select numerical columns for vector embeddings
numeric_features = ['lead_time', 'adr', 'stays_in_week_nights', 'stays_in_weekend_nights']

# Convert dataframe to numpy array
scaler = StandardScaler()
vectors = scaler.fit_transform(df[numeric_features].values)

# Get dimensions of the vector space
d = vectors.shape[1]  # Number of features


# In[18]:

# Initialize FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(vectors)  # Add our booking data as vectors

print(f"FAISS Index trained with {index.ntotal} booking records")


# In[19]:

# Example new booking (randomly selected from dataset)
new_booking = df[numeric_features].iloc[0].values.reshape(1, -1)

# Scale it using the same scaler
new_booking_scaled = scaler.transform(new_booking)

# Search for 5 most similar bookings
k = 5  # Number of similar results
distances, indices = index.search(new_booking_scaled, k)

# Show results
print("Top similar bookings:")
print(df.iloc[indices[0]])


# In[20]:

from sklearn.preprocessing import OneHotEncoder

# Select categorical features
categorical_features = ['hotel', 'customer_type']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_vectors = encoder.fit_transform(df[categorical_features])

# Combine categorical and numerical features
full_vectors = np.hstack((vectors, cat_vectors))  # Merge both embeddings

# Update dimension size
d = full_vectors.shape[1]


# In[21]:

# Define number of clusters
nlist = 100  # Increase for better performance on large datasets

# Initialize IVF Index
quantizer = faiss.IndexFlatL2(d)  # Base index
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index (important for IVF)
index_ivf.train(full_vectors)
index_ivf.add(full_vectors)

print(f"Optimized FAISS index built with {index_ivf.ntotal} vectors")


# In[22]:

# Process new booking with numerical + categorical data
new_booking = df.iloc[10]  # Sample input

# Transform input (scale + one-hot encode)
num_part = scaler.transform(new_booking[numeric_features].values.reshape(1, -1))
cat_part = encoder.transform(new_booking[categorical_features].values.reshape(1, -1))
query_vector = np.hstack((num_part, cat_part))

# Search in FAISS index
k = 5  # Number of results
index_ivf.nprobe = 10  # Number of clusters to search in (tune this for accuracy/speed tradeoff)
distances, indices = index_ivf.search(query_vector, k)

# Display results
print("Top similar bookings:")
print(df.iloc[indices[0]])


# In[23]:

faiss.write_index(index_ivf, "faiss_bookings.index")
index_loaded = faiss.read_index("faiss_bookings.index")


# In[24]:

# Check if the loaded index has the same number of vectors
print(f"Total vectors in original index: {index_ivf.ntotal}")
print(f"Total vectors in loaded index: {index_loaded.ntotal}")

# Perform a simple query on the loaded index
k = 5  # Number of nearest neighbors
random_vector = full_vectors[0].reshape(1, -1)  # Take the first vector for testing

distances, indices = index_loaded.search(random_vector, k)

print("Indices of closest bookings:", indices)
print("Distances:", distances)


# In[25]:

from huggingface_hub import login

login("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")


# hugging face access token: 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

# In[28]:

df.columns


# In[29]:

df[(df['arrival_date_year'] == 2017) & (df['arrival_date_month'] == 'July')]['adr'].sum()


# In[30]:

df.groupby('country')['is_canceled'].sum().idxmax()


# In[31]:

df['adr'].mean()


# In[46]:

# In[43]:

import pandas as pd

# Load dataset
df = pd.read_csv("hotel_bookings.csv")

# Convert column names to lowercase (to avoid case mismatches)
df.columns = df.columns.str.lower()

# Convert 'arrival_date_month' to proper format (capitalize first letter)
df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()

# Ensure 'adr' is numeric and fill missing values with 0
df['adr'] = pd.to_numeric(df['adr'], errors='coerce').fillna(0)

# Check for any duplicate rows and remove them
df.drop_duplicates(inplace=True)

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)


# In[48]:

total_revenue_july_2018 = df[
    (df['arrival_date_year'] == 2018) & (df['arrival_date_month'] == 'July')
]['adr'].sum()

print("Total Revenue for July 2016:", total_revenue_july_2018)


# In[49]:

import pandas as pd
from transformers import pipeline

# Load dataset
df = pd.read_csv("/kaggle/input/hotel-bookings/hotel_bookings.csv")

# Load text generation model (Flan-T5 Small for efficiency)
text_generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Cache to store previously generated queries
query_cache = {}

# Predefined query templates
query_templates = {
    "total revenue for": "df[(df['arrival_date_year'] == {year}) & (df['arrival_date_month'] == '{month}')]['adr'].sum()",
    "highest booking cancellations": "df.groupby('country')['is_canceled'].sum().idxmax()",
    "average price of a hotel booking": "df['adr'].mean()"
}

def generate_pandas_query(query):
    """Convert a natural language question into a Pandas query."""
    if query in query_cache:
        return query_cache[query]

    # Handle "total revenue for" separately
    if "total revenue for" in query.lower():
        words = query.split()
        month, year = words[-2], words[-1].replace(".", "")
        pandas_query = query_templates["total revenue for"].format(year=year, month=month)
        query_cache[query] = pandas_query
        return pandas_query

    # Check for other predefined queries
    for key, value in query_templates.items():
        if key in query.lower():
            query_cache[query] = value
            return value

    # If no match, use LLM to generate a query
    prompt = f"""
    Convert the following question into a valid Pandas query:
    The DataFrame has these columns: {', '.join(df.columns)}.
    Question: "{query}"
    Pandas Query:
    """

    response = text_generator(prompt, max_new_tokens=50, do_sample=False)
    pandas_query = response[0]['generated_text']

    query_cache[query] = pandas_query
    return pandas_query

def execute_query(query):
    """Execute a Pandas query and return the result."""
    try:
        pandas_query = generate_pandas_query(query)
        print(f"Generated Query: {pandas_query}")  # Debugging output
        result = eval(pandas_query, {'df': df})

        # If result is a scalar, return it directly
        if isinstance(result, (int, float, str)):
            return result

        return result if not result.empty else "No results found."

    except Exception as e:
        return f"Error: {str(e)}"

# Example Queries
queries = [
    "Show me total revenue for July 2017.",
    "Which locations had the highest booking cancellations?",
    "What is the average price of a hotel booking?"
]

# Process queries
for user_query in queries:
    print(f"\nQ: {user_query}")
    print(f"A: {execute_query(user_query)}\n")


# In[ ]:
