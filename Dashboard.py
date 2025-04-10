# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import os

# # Page 1: Sign In Page
# def sign_in_page():
#     st.title("Sign In")
#     st.write("Please enter your credentials to sign in.")
    
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
    
#     if st.button("Sign In"):
#         if username == "admin" and password == "password":  # Dummy credentials
#             st.success("Sign-in successful!")
#             return True
#         else:
#             st.error("Invalid credentials. Please try again.")
#             return False
#     return False

# # Page 2: Ad and Sales Metrics
# def ad_sales_metrics_page(df):
    
#     df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#     df['YearMonth'] = df['Date'].dt.to_period('M')
#     print()
#     SystemExit()

#     # Group by Year-Month and calculate the total for each month
#     monthly_data = df.groupby('YearMonth').agg({
#         'Budget': 'sum',
#         'Spend': 'sum',
#         'Revenue': 'sum',
#         'ROAS': lambda x: (x.sum() / df.loc[x.index, 'Spend'].sum() if df.loc[x.index, 'Spend'].sum() != 0 else 0),  # Calculate ROAS for each month
#     }).reset_index()

#     # Calculating the percentage change between the most recent month and the previous month
#     monthly_data['Change_Budget'] = monthly_data['Budget'].pct_change() * 100
#     monthly_data['Change_Spend'] = monthly_data['Spend'].pct_change() * 100
#     monthly_data['Change_Revenue'] = monthly_data['Revenue'].pct_change() * 100
#     monthly_data['Change_ROAS'] = monthly_data['ROAS'].pct_change() * 100

#     # Get the most recent month
#     latest_month = monthly_data['YearMonth'].max()
#     latest_data = monthly_data[monthly_data['YearMonth'] == latest_month]

#     # Get the previous month data
#     previous_month = (latest_month.to_timestamp() - pd.DateOffset(months=1)).to_period('M')  # Convert to timestamp, subtract, and convert back to Period
#     previous_data = monthly_data[monthly_data['YearMonth'] == previous_month]

#     # Option map for selecting between Total and Average
#     option_map = {
#         0: "Total",
#         1: "Average",
#     }

#     # Streamlit segmented control for selecting between Total and Average
#     selection = st.segmented_control(
#         "Select/toggle between the two options below",
#         options=option_map.keys(),
#         format_func=lambda option: option_map[option],
#         selection_mode="single",
#     )

#     # Display the results based on the selected option (Total or Average)
#     if selection == 0:
#         a, b = st.columns(2)
#         c, d = st.columns(2)

#         # Displaying total values and change over time
#         a.metric("Total Budget", f"${latest_data['Budget'].iloc[0]:,.2f}", f"{latest_data['Change_Budget'].iloc[0]:.2f}% change over time", border=True)
#         b.metric("Total Spend", f"${latest_data['Spend'].iloc[0]:,.2f}", f"{latest_data['Change_Spend'].iloc[0]:.2f}% change over time", border=True)

#         c.metric("Total Revenue", f"${latest_data['Revenue'].iloc[0]:,.2f}", f"{latest_data['Change_Revenue'].iloc[0]:.2f}% change over time", border=True)
#         d.metric("Total ROAS", f"{latest_data['ROAS'].iloc[0]:.2f}", f"{latest_data['Change_ROAS'].iloc[0]:.2f}% change over time", border=True)

#     elif selection == 1:
#         a, b = st.columns(2)
#         c, d = st.columns(2)

#         # Displaying average values and change over time
#         a.metric("Average Budget", f"${monthly_data['Budget'].mean():,.2f}", f"{monthly_data['Change_Budget'].mean():.2f}% change over time", border=True)
#         b.metric("Average Spend", f"${monthly_data['Spend'].mean():,.2f}", f"{monthly_data['Change_Spend'].mean():.2f}% change over time", border=True)

#         c.metric("Average Revenue", f"${monthly_data['Revenue'].mean():,.2f}", f"{monthly_data['Change_Revenue'].mean():.2f}% change over time", border=True)
#         d.metric("Average ROAS", f"{monthly_data['ROAS'].mean():.2f}", f"{monthly_data['Change_ROAS'].mean():.2f}% change over time", border=True)



# # Page 3: Trend Identification with Linear Regression
# def trend_identification_page(df):
#     st.title("Trend Identification & Linear Regression")
#     st.write("This page is for identifying trends and performing linear regression on your ad data.")
    
#     st.write("Dataset preview:")
#     st.write(df.head())
    
#     if all(col in df.columns for col in ["Budget", "Impressions", "CTR", "Conversions"]):
#         X = df[["Budget", "Impressions", "CTR"]]
#         y = df["Conversions"]
        
#         model = LinearRegression()
#         model.fit(X, y)
        
#         y_pred = model.predict(X)
        
#         st.write(f"Coefficients: {model.coef_}")
#         st.write(f"Intercept: {model.intercept_}")
        
#         r2 = r2_score(y, y_pred)
#         mse = mean_squared_error(y, y_pred)
        
#         st.write(f"R-squared: {r2}")
#         st.write(f"Mean Squared Error: {mse}")
        
#         fig, ax = plt.subplots()
#         ax.scatter(y, y_pred, color='blue', alpha=0.5)
#         ax.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
#         ax.set_xlabel("Actual Conversions")
#         ax.set_ylabel("Predicted Conversions")
#         ax.set_title("Actual vs Predicted Conversions")
#         st.pyplot(fig)
        
#         fig, ax = plt.subplots()
#         ax.scatter(y_pred, y_pred - y, color='blue', alpha=0.5)
#         ax.axhline(y=0, color='red', linestyle='--')
#         ax.set_xlabel("Predicted Conversions")
#         ax.set_ylabel("Residuals")
#         ax.set_title("Residual Plot")
#         st.pyplot(fig)
#     else:
#         st.warning("Dataset does not contain the required columns ('Budget', 'Impressions', 'CTR', 'Conversions').")

# # Page 4: Exploratory Data Analysis (EDA)
# def eda_page(df):
#     st.title("Exploratory Data Analysis (EDA)")
#     st.write("Explore the data through visualizations and summary statistics.")

#     st.write("Dataset preview:")
#     st.write(df.head())
    
#     st.write("**Summary Statistics**:")
#     st.write(df.describe())
    
#     numeric_df = df.select_dtypes(include=[np.number])
    
#     if numeric_df.shape[1] > 1:
#         st.write("**Correlation Heatmap**:")
#         corr = numeric_df.corr()
#         fig, ax = plt.subplots()
#         sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
#     else:
#         st.warning("Not enough numeric columns to create a correlation heatmap.")
    
#     st.write("**Histograms of Numerical Features**:")
#     for col in numeric_df.columns:
#         st.subheader(f"Histogram of {col}")
#         fig, ax = plt.subplots()
#         numeric_df[col].hist(bins=20, ax=ax)
#         ax.set_title(f"Histogram of {col}")
#         st.pyplot(fig)
    
#     st.write("**Box Plots of Numerical Features**:")
#     for col in numeric_df.columns:
#         st.subheader(f"Box Plot of {col}")
#         fig, ax = plt.subplots()
#         sns.boxplot(data=numeric_df, x=col, ax=ax)
#         st.pyplot(fig)

# # Page 5: Automated Insights & Recommendations
# def insights_recommendations_page(df):
#     st.title("Automated Insights & Recommendations")
#     st.write("This page provides insights based on the data analysis and model results.")

#     st.write("Dataset preview:")
#     st.write(df.head())
    
#     if all(col in df.columns for col in ["Budget", "Impressions", "CTR", "Conversions"]):
#         X = df[["Budget", "Impressions", "CTR"]]
#         y = df["Conversions"]
#         model = LinearRegression()
#         model.fit(X, y)
        
#         coefficients = model.coef_
#         intercept = model.intercept_
#         r2 = r2_score(y, model.predict(X))

#         st.write(f"Linear Regression Coefficients: {coefficients}")
#         st.write(f"Intercept: {intercept}")
#         st.write(f"R-squared: {r2}")
        
#         st.write("**Automated Recommendations:**")
#         st.write("1. Increase ad budget to improve conversions.")
#         st.write("2. Focus on high CTR ads to optimize conversions.")
#         st.write("3. Lower budget on low-performing ads based on the regression analysis.")
#     else:
#         st.warning("Dataset does not contain the required columns ('Budget', 'Impressions', 'CTR', 'Conversions').")


# # Main function that handles navigation
# def main():
#     file_path = "ad-data.xlsx"
#     df = pd.read_excel('ad-data.xlsx', engine='openpyxl')
#             # Sidebar for navigation
#     page = st.sidebar.radio("Select a page", [
#         "Sign in", "Ad and Sales Metrics", "Trend Identification", 
#         "Exploratory Data Analysis", "Automated Insights & Recommendations"
#     ])
        
#     # Call the corresponding page function
#     if page == "Sign in":
#         sign_in_page()
#     elif page == "Ad and Sales Metrics":
#         ad_sales_metrics_page(df)
#     elif page == "Trend Identification":
#         trend_identification_page(df)
#     elif page == "Exploratory Data Analysis":
#         eda_page(df)
#     elif page == "Automated Insights & Recommendations":
#         insights_recommendations_page(df)


# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go

def histogram(df, x):
    df[x].hist(bins=5, edgecolor='black')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.histplot(df[x], ax=ax)
    ax.set_title(f'Histogram for {x}')
    ax.set_xlabel(x)
    ax.set_ylabel('Count')
    st.pyplot(fig)

def ad_page(df):
    st.title("Ad and sales metrics")
    st.write("## Dataset used:")
    st.write("Link to dataset: https://www.kaggle.com/datasets/aashwinkumar/ppc-campaign-performance-data/data")
    st.dataframe(df)
    st.write("## Data statistics")
    st.write(df.describe())
    st.write("## Performance over time")
    value = st.selectbox("Select variable", ['Revenue', 'Budget', 'Spend', 'ROAS'])
    if value:
        df['Date'] = pd.to_datetime(df['Date'])
        df_sum = df.groupby('Date').agg({value: 'sum'}).reset_index()
        fig = go.Figure(go.Scatter(x=df_sum['Date'], y=df_sum[value], mode='lines', name=value))
        st.plotly_chart(fig)
    st.write("## Campaign Demographics")
    age_chart = df['Target_Age'].value_counts()
    a, b = st.columns(2)
    c, d = st.columns(2)
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax.pie(df['Target_Age'].value_counts(), labels=df['Target_Age'].value_counts().index)
    ax2.pie(df['Target_Gender'].value_counts(), labels=df['Target_Gender'].value_counts().index)
    ax3.pie(df['Region'].value_counts(), labels=df['Region'].value_counts().index)
    ax4.pie(df['Platform'].value_counts(), labels=df['Platform'].value_counts().index)
    ax.set_title("Target age group")
    ax2.set_title("Target gender")
    ax3.set_title("Region")
    ax4.set_title("Platform")
    a.pyplot(fig)
    b.pyplot(fig2)
    c.pyplot(fig3)
    d.pyplot(fig4)

def linear_regression(df, xlabel, ylabel):
    X = df[[xlabel]]
    y = df[ylabel]
    model = LR()
    model.fit(X, y)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, label="Data Points", color='green')
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    x_pos = np.min(X) + (np.max(X) - np.min(X)) * 0.65
    y_pos = np.max(y) - (np.max(y) - np.min(y)) * 0.09
    ax.plot(X, y_pred, color='red', label="Regression Line")
    stats = f"Intercept (constant): {model.intercept_:.2f}\nCoefficient (slope) for x: {model.coef_[0]:.2f}\nR-squared: {r_squared:.2f}"
    ax.text(x_pos, y_pos, stats, fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)

def decision(df, ylabel, depth):
    X = df.drop(columns=ylabel)
    y = df[ylabel]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeRegressor(random_state=42, max_depth=depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=X.columns, ax=ax)
    st.pyplot(fig)

def random_forest(df):
    df.drop(columns=['Campaign_ID', 'Date'], inplace=True)
    feature_names = ['Target_Gender', 'Platform', 'Region', 'Content_Type', 'Target_Age']
    for feature in feature_names:
        df[feature] = LabelEncoder().fit_transform(df[feature])
    X = df.drop(columns='ROAS')
    y = df['ROAS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RFR(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    st.write(importance_df)
    st.write("We can see that CPA and Revenue are most important to predict ROI which is intuitive but "
             "scrolling further down we can see that age and platform are the most important demographics to predict ROI.")

def k_cluster(df):
    df_temp = df.copy()
    df = StandardScaler().fit_transform(df)
    model = PCA(n_components=4)
    model.fit(df)
    pca_df = model.transform(df)
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    kmeans.fit(pca_df)
    combined_df = pd.concat([df_temp, pd.DataFrame(pca_df)], axis=1)
    combined_df.columns.values[-4:] = ['PC_1', 'PC_2', 'PC_3', 'PC_4']
    combined_df['kmeans'] = kmeans.labels_
    combined_df['Clusters'] = combined_df['kmeans'].map({0:'Cluster 1', 1:'Cluster 2', 2:'Cluster 3', 3:'Cluster 4'})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x = combined_df['PC_1'], y = combined_df['PC_2'], hue=combined_df['Clusters'], ax=ax)
    ax.set_title("Plot of principle component 1 vs 2 grouped by K-means")
    ax.set_xlabel('Principle component 1')
    ax.set_ylabel('Principle component 2')
    st.pyplot(fig)
    st.write('Principle component 1 is most influenced by conversions, revenue, and clicks')
    st.write('Principle component 2 is most influenced by budget and spend')
    st.write("From this we can identify for example that cluster 1 (orange) and cluster 4 (red) failed to generate" 
             " high conversions, revenue, or clicks despite high budget and spend. Thus further analyzing the"
             "campaigns belonging to this cluster will help identify issues.")

def trends_page(df):
    st.title("Trends created using ML models")
    st.write("## Dataset used:")
    st.write("Link to dataset: https://www.kaggle.com/datasets/aashwinkumar/ppc-campaign-performance-data/data")
    st.dataframe(df)
    st.write("## Linear Regression")
    # x_col = st.selectbox("Select X variable", data.columns)
    df_copy = df.copy()
    df_copy.drop(columns=['Campaign_ID','Target_Gender', 'Platform', 'Date', 'Region', 'Content_Type', 'Target_Age'], inplace=True)
    X = st.selectbox("Choose independent variable", df_copy.columns)
    y = st.selectbox("Choose dependent variable", df_copy.columns)

    if y:
        Q1 = df_copy[X].quantile(0.25)
        Q3 = df_copy[X].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_copy[(df_copy[X] >= (Q1 - 1.5 * IQR)) & (df_copy[X] <= (Q3 + 1.5 * IQR))]
        linear_regression(df, X, y)
    
    st.write("## Decision Tree")
    y = st.selectbox("Choose value to predict", df_copy.columns)
    depth = st.selectbox("Choose depth of tree (max set to 5)", [1,2,3,4,5])
    if y:
        decision(df_copy, y, depth)
    st.write("## Trend 1: What features (variables) are most important to predict ROI?")
    st.write("Random forest regression used for analysis")
    random_forest(df)
    st.write("## Trend 2: Data clustered and graphed by princple components")
    k_cluster(df_copy)
    

def exp_page(df):
    st.title("Some exploratory data analysis performed on customer data")
    st.write("Link to dataset: https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis")
    st.write(df[:5000])
    st.write("Since real data isn't perfect we start by cleaning but we skip that step since I am using a fake dataset (probably generated but still has errors)."
             " Notice that the returns column has null values and there are duplicate columns customer age and age, these are some"
             " things that would be fixed in case real data was involved.")
    
    df = df[:5000]
    df_temp = df.drop(columns=['Customer Age'])
    st.write("## Descriptive Statistics")
    st.write(df_temp.describe())
    st.write("## Correlation heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    df_copy = df.copy()
    df_copy.drop(columns=['Customer ID', 'Gender', 'Payment Method', 'Customer Age', 'Purchase Date', 'Customer Name', 'Product Category'], inplace=True)
    sns.heatmap(df_copy.corr(), annot=True, ax=ax)
    st.pyplot(fig)
    st.write("Can see for example age is positively correlated to returns or that churn is most positively correlated to product price.")
    st.write("## Histogram")
    x = st.selectbox("Choose variable to make histogram", df.columns.difference(['Customer ID', 'Customer Age', 'Purchase Date', 'Customer Name']))
    if x:
        histogram(df, x)
    st.write("The data was most likely generated as you can see all the values in the histograms are mostly even. These graphs would look different with real data.")
    
def insights_page():
    st.title("Page to show automated insights and recommendations")
    st.write("These recommendations below are just based on the exploratory data analysis and trend identification"
             "I hard coded them as I didn't have time to implement this properly. Ideally however the analysis from the trend identification page "
             "would automatically generate the recommendations and insights similar to what's shown below.")
    st.write("1) Focus on targeting specific age groups as they are the most important demographic to predict ROI.\n"
             "2) Check why the campaigns in clusters 1 and 4 (Trend 2 from trend identification page) generated less revenue despite high budgets.\n"
             "3) Further analysis can be performed on cluster 3 that generated most revenue. Notice that the cluster equally "
             "includes campaigns of high and low budget, so it can be used to identify similarities and found out factors which made the low budget campaigns successful. "
             "Further A/B testing can also be recommended based on further analysis of this cluster/data")

st.sidebar.title("Go through the different pages below")

page = st.sidebar.radio("Pages", ["README", "Ad and Sales Metrics", "Trend Identification", "Exploratory Data Analysis", "Automated Insights & Recommendations"])


ad_data = pd.read_excel('ad-data.xlsx', engine='openpyxl')
customer_data = pd.read_csv('customer-data.csv')


if page == "Ad and Sales Metrics":
    ad_page(ad_data)
elif page == "Trend Identification":
    trends_page(ad_data)
elif page == "Exploratory Data Analysis":
    exp_page(customer_data)
elif page == "Automated Insights & Recommendations":
    insights_page()
