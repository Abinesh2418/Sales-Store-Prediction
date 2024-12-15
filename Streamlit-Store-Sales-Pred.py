import pickle
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒")

# Load the trained model
MODEL_PATH = "Store-Sales-Pred.pkl"
with open(MODEL_PATH, "rb") as model_file:
    regressor = pickle.load(model_file)

# Initialize a DataFrame to store user input data
user_data = pd.DataFrame(columns=[
    "Item_Weight", "Item_MRP", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Outlet_Sales"
])

def predict_values(item_weight, item_mrp, outlet_size, outlet_location_type, outlet_type):
    """Predict sales using the input features."""
    # Encode categorical features
    outlet_size_encoded = {"Small": 0, "Medium": 1, "High": 2}[outlet_size]
    outlet_location_type_encoded = {
        "Urban Area": 0,
        "Suburban Area": 1,
        "Rural Area": 2,
    }[outlet_location_type]

    outlet_type_encoded = {
        "Convenience Store": 0,
        "Mini Mart": 1,
        "Neighborhood Market": 2,
        "Hypermarket": 3,
    }[outlet_type]

    # Create input feature array
    features = [
        item_weight,
        item_mrp,
        outlet_size_encoded,
        outlet_location_type_encoded,
        outlet_type_encoded,
    ]

    # Make prediction
    prediction = regressor.predict([features])
    return prediction

def plot_data(df):
    """Generate and display scatter and bar charts for the given data."""
    if df.empty:
        st.warning("No data to display. Please provide inputs and click Predict.")
        return

    st.markdown(
        "<h2 style='text-align: center; color: #00796B;'>Visualizations</h2>",
        unsafe_allow_html=True
    )

    # Scatter plot for Item MRP vs Item Outlet Sales
    scatter_chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('Item_MRP', title='Item MRP'),
        y=alt.Y('Item_Outlet_Sales', title='Item Outlet Sales'),
        color=alt.Color('Outlet_Size', title='Outlet Size'),
        tooltip=['Item_MRP', 'Item_Outlet_Sales', 'Outlet_Size']
    ).properties(
        width=400,
        height=400,
        title='Item MRP vs Item Outlet Sales'
    )
    st.altair_chart(scatter_chart, use_container_width=True)

    # Bar chart for Outlet Size vs Average Item Sales
    sales_by_size = df.groupby("Outlet_Size")["Item_Outlet_Sales"].mean().reset_index()
    bar_chart = alt.Chart(sales_by_size).mark_bar().encode(
        x=alt.X('Outlet_Size', title='Outlet Size'),
        y=alt.Y('Item_Outlet_Sales', title='Average Item Sales'),
        color=alt.Color('Outlet_Size', title='Outlet Size')
    ).properties(
        width=400,
        height=400,
        title='Outlet Size vs Average Item Sales'
    )
    st.altair_chart(bar_chart, use_container_width=True)

def main():
    """Main function to render the Streamlit app."""
    global user_data

    # Set background color and font styling using CSS
    st.markdown(
        """
        <style>
        body {
            background-color: #F0F4C3;
        }
        .stButton>button {
            background-color: #64B5F6;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #42A5F5;
            color: white;
        }
        h1 {
            color: #2E7D32;
            text-align: center;
        }
        .prediction-box {
            background-color: #00796B;
            color: white;
            font-size: 22px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1>Item Sale Predictor</h1>", unsafe_allow_html=True)

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        item_weight = st.number_input("Item Weight", min_value=0.0, format="%.2f", value=0.50)
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_type = st.radio(
            "Outlet Type",
            options=["Convenience Store", "Mini Mart", "Neighborhood Market", "Hypermarket"],
            horizontal=True
        )

    with col2:
        item_mrp = st.number_input("Item MRP", min_value=0.0, value=249.8092)
        outlet_location_type = st.selectbox("Outlet Location Type", ["Urban Area", "Suburban Area", "Rural Area"])

    # Predict button
    if st.button("Predict"):
        prediction = predict_values(item_weight, item_mrp, outlet_size, outlet_location_type, outlet_type)

        # Add input data to the DataFrame
        user_data = pd.concat([
            user_data, 
            pd.DataFrame({
                "Item_Weight": [item_weight],
                "Item_MRP": [item_mrp],
                "Outlet_Size": [outlet_size],
                "Outlet_Location_Type": [outlet_location_type],
                "Outlet_Type": [outlet_type],
                "Item_Outlet_Sales": [prediction]
            })
        ], ignore_index=True)

        st.markdown(
            f"<div class='prediction-box'>Predicted Value: {np.expm1(prediction[0]):.2f}</div>",
            unsafe_allow_html=True
        )

    # Display data visualizations
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    plot_data(user_data)

# Run the app
if __name__ == "__main__":
    main()
