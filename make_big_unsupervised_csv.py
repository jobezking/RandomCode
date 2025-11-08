# to use in Jupyter
#df = pd.read_csv('unsupervised_data.csv')
#print(df.head())
#print(df.info()) # Get a summary of the data, including missing values

# --- Configuration ---
NUM_ROWS = 1_000_000  # Number of rows in the CSV
NUM_CATEGORIES = 100 # Number of unique categories for categorical columns
CSV_FILE_NAME = 'unsupervised_data.csv'

# --- Generate Synthetic Data ---

print("Generating synthetic data...")

# 1. Numerical Data (for clustering and dimensionality reduction)
# We'll create several clusters to make the data interesting for algorithms like K-Means.
# We'll also add some noise.
num_features = 5
cluster_means = np.random.rand(4, num_features) * 20
cluster_std = np.random.rand(4) * 3 + 1
cluster_data = []
for i in range(4):
    num_points = int(NUM_ROWS / 4)
    cluster_data.append(np.random.normal(loc=cluster_means[i], scale=cluster_std[i], size=(num_points, num_features)))

# Combine all cluster data and shuffle
numerical_data = np.vstack(cluster_data)
np.random.shuffle(numerical_data)

# Add some outliers
num_outliers = int(NUM_ROWS * 0.005)
outliers = np.random.uniform(low=-50, high=50, size=(num_outliers, num_features))
numerical_data = np.vstack([numerical_data, outliers])

# Ensure the final size is exactly NUM_ROWS by trimming if necessary
if numerical_data.shape[0] > NUM_ROWS:
    numerical_data = numerical_data[:NUM_ROWS, :]
elif numerical_data.shape[0] < NUM_ROWS:
    # This case is unlikely but good practice
    missing_rows = NUM_ROWS - numerical_data.shape[0]
    numerical_data = np.vstack([numerical_data, np.random.normal(loc=10, scale=5, size=(missing_rows, num_features))])

# Create a DataFrame for numerical data
numerical_df = pd.DataFrame(numerical_data, columns=[f'feature_{i+1}' for i in range(num_features)])

# 2. Categorical Data (for tasks like mixed-data clustering)
categories = [f'cat_{i+1}' for i in range(NUM_CATEGORIES)]
categorical_data = np.random.choice(categories, size=NUM_ROWS, replace=True)
categorical_df = pd.DataFrame({'category': categorical_data})

# 3. Ordinal Data (e.g., ratings)
ratings = ['low', 'medium', 'high', 'very_high']
ordinal_data = np.random.choice(ratings, size=NUM_ROWS, replace=True, p=[0.4, 0.3, 0.2, 0.1])
ordinal_df = pd.DataFrame({'rating': ordinal_data})

# 4. Binary Data
binary_data = np.random.randint(0, 2, size=NUM_ROWS)
binary_df = pd.DataFrame({'is_active': binary_data})

# 5. Text-like Data (for potential NLP-based unsupervised learning)
# We will generate a simple text feature with some recurring keywords.
keywords = ['analysis', 'model', 'data', 'algorithm', 'machine learning', 'predictive', 'unsupervised']
text_data = ['The ' + np.random.choice(keywords) + ' ' + np.random.choice(keywords) + ' project.' for _ in range(NUM_ROWS)]
text_df = pd.DataFrame({'project_description': text_data})

# 6. A Column with Missing Values
missing_data = numerical_df['feature_1'].copy()
# Introduce NaN values (e.g., 5% of the data)
missing_data[np.random.choice(range(NUM_ROWS), size=int(NUM_ROWS * 0.05), replace=False)] = np.nan
missing_df = pd.DataFrame({'feature_with_nans': missing_data})

# 7. A Timestamp Column
timestamps = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(NUM_ROWS), unit='h')
timestamp_df = pd.DataFrame({'timestamp': timestamps})

# --- Combine all DataFrames ---
print("Combining data into a single DataFrame...")
all_data = pd.concat([
    numerical_df,
    categorical_df,
    ordinal_df,
    binary_df,
    text_df,
    missing_df,
    timestamp_df
], axis=1)

# --- Save to CSV ---
print(f"Saving DataFrame to {CSV_FILE_NAME}...")
# Setting index=False to avoid writing the pandas index as a column
all_data.to_csv(CSV_FILE_NAME, index=False)

print(f"Successfully generated {CSV_FILE_NAME} with {NUM_ROWS} rows.")
print("File size will be significant. The generation process may take a few moments.")
