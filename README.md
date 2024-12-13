# Store Sales Prediction

## Project Overview
The **Store Sales Prediction** project aims to forecast the sales of items across various outlets based on historical data. By leveraging machine learning techniques, this project predicts the `Item_Outlet_Sales` for individual items and outlets using several features like item type, outlet type, and more.

## Dataset Details
The dataset used for this project consists of 8523 entries and includes the following features:

- **Item_Identifier**: Unique product ID
- **Item_Weight**: Weight of product
- **Item_Fat_Content**: Whether the product is low fat or not
- **Item_Visibility**: The % of total display area of all products in a store allocated to the particular product
- **Item_Type**: The category to which the product belongs
- **Item_MRP**: Maximum Retail Price (list price) of the product
- **Outlet_Identifier**: Unique store ID
- **Outlet_Establishment_Year**: The year in which store was established
- **Outlet_Size**: The size of the store in terms of ground area covered
- **Outlet_Location_Type**: The type of city in which the store is located
- **Outlet_Type**: Whether the outlet is just a grocery store or some sort of supermarket
- **Item_Outlet_Sales**: Sales of the product in the particular store. This is the outcome variable to be predicted.

## Key Objectives
1. **Predictive Modeling**: Develop a machine learning model to predict the `Item_Outlet_Sales`.
2. **Feature Analysis**: Explore and preprocess features like missing values, encoding categorical variables, and scaling numerical variables.
3. **Data Visualization**: Create visualizations to understand the relationship between features and sales.
4. **Model Evaluation**: Evaluate the model's performance using metrics like RMSE (Root Mean Square Error).

## Methodology
1. **Data Preprocessing**:
   - Handle missing values (e.g., `Item_Weight`, `Outlet_Size`).
   - Encode categorical variables using techniques like One-Hot Encoding and Label Encoding.
   - Normalize/Scale numerical variables.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze relationships between features (e.g., `Item_MRP` vs `Item_Outlet_Sales`).
   - Visualize patterns using libraries like Matplotlib, Seaborn, and Altair.

3. **Model Development**:
   - Train and test a regression model using algorithms like Linear Regression, Random Forest, or Gradient Boosting.
   - Use log transformation to handle skewness in features or target variables.

4. **Model Deployment**:
   - Build an interactive web application using Streamlit for predicting sales based on user input.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Altair
- **Deployment Platform**: Streamlit

## File Details
1. **`Sales_Pred_Original-2.pkl`**: Pre-trained model file used for predictions.
2. **`app.py`**: Streamlit application script.
3. **Dataset**: CSV file containing the sales data.

## How to Run the Application
1. Install the required Python packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn altair
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Use the web interface to input features and view sales predictions along with data visualizations.

## Visualizations
The app includes:
- **Scatter Plot**: `Item_MRP` vs `Item_Outlet_Sales`.
- **Bar Chart**: `Outlet_Size` vs Average Item Sales.

## Future Enhancements
- **Advanced Models**: Incorporate deep learning models for improved accuracy.
- **Dynamic Visualizations**: Enable users to customize visualization parameters.
- **Extended Features**: Include additional data points like seasonal trends, promotional effects, etc.

---

For questions or suggestions, feel free to contact me at [abineshbalasubramaniyam@gmail.com].

