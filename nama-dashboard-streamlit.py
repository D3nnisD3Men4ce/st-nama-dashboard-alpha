import supabase
from supabase import create_client
import json
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import time
import datetime
import streamlit as st
import plost
import gspread
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from gspread_pandas import Spread, Client
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PIL import Image
import json
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv();

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

# Create Supabase client
supabase: Client = create_client(API_URL, API_KEY)

st.set_page_config(
    page_title="Nama Microgreens Dashboard",
    layout='wide'
)

st.title("Nama Microgreens Dashboard")

# Fetch data from Supabase
sensors_data = supabase.table('device_readings').select('*').execute()
settings_data = supabase.table('device_settings').select('*').execute()

sensors_data_df = pd.DataFrame(sensors_data.data)
settings_data_df = pd.DataFrame(settings_data.data)


# Assuming you have already imported Streamlit
datePicker, startDate, endDate, refreshData = st.columns(4)

with datePicker:
    # Get the current date
    today = datetime.date.today()

    # Set the min and max dates
    min_date = today
    max_date = today + datetime.timedelta(days=1)

    a_date = st.date_input("Pick a date", (min_date, max_date))

# Parse the selected dates from the datePicker into datetime objects
date_start_str = a_date[0].strftime("%B %d, %Y")
date_end_str = a_date[-1].strftime("%B %d, %Y")

with startDate:
    st.write(f"Start: {date_start_str}")

with endDate:
    st.write(f"End: {date_end_str}")
    
# Convert the string dates to pandas Timestamp objects
date_start = pd.Timestamp(a_date[0])
date_end = pd.Timestamp(a_date[-1])


with refreshData:
    if st.button('Refresh Data'):
        # Code not intended for proper use, only used for its side effect which refreshes the page
        print('lol')



# Define a function to convert timestamp format (b) to format (a) and add 8 hours
def convert_timestamp(timestamp_b):
    # Parse the timestamp using pd.to_datetime() to convert it to a datetime object
    datetime_b = pd.to_datetime(timestamp_b)
    # Convert the datetime object to the desired format
    formatted_timestamp_a = (datetime_b + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    return formatted_timestamp_a

# Apply the conversion function to the 'created_at' column
sensors_data_df['created_at'] = sensors_data_df['created_at'].apply(convert_timestamp)


sensors_data_df['created_at'] = sensors_data_df['created_at'].apply(pd.to_datetime).dt.to_pydatetime()


sensors_data_df = sensors_data_df[(sensors_data_df['created_at'] >= date_start) & (
    sensors_data_df['created_at'] <= date_end)]


# # Print types for troubleshooting
# print("Type of date_start:", type(date_start))
# print("Type of date_end:", type(date_end))
# print("Type of 'created_at' column:", type(sensors_data_df['created_at'].iloc[0]))


# Create a new DataFrame with 'created_at' and 'temp' columns
temp_df = sensors_data_df[['created_at', 'temp']].copy()
# Divide all 'temp' values by 100
temp_df['temp'] = temp_df['temp'] / 100
humi_df = sensors_data_df[['created_at', 'humi']]
moisture_a_df = sensors_data_df[['created_at', 'moisture_a']]
moisture_b_df = sensors_data_df[['created_at', 'moisture_b']]
moisture_c_df = sensors_data_df[['created_at', 'moisture_c']]
moisture_d_df = sensors_data_df[['created_at', 'moisture_d']]
moisture_e_df = sensors_data_df[['created_at', 'moisture_e']]
moisture_f_df = sensors_data_df[['created_at', 'moisture_f']]
moisture_g_df = sensors_data_df[['created_at', 'moisture_g']]
moisture_h_df = sensors_data_df[['created_at', 'moisture_h']]



temp, humi = st.columns(2)

with temp:
    st.markdown("## Ambient Temperature")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_df['created_at'], y=temp_df['temp'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with humi:
    st.markdown("## Humidity")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=humi_df['created_at'], y=humi_df['humi'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)



moisture_a, moisture_b, moisture_c, moisture_d = st.columns(4)

with moisture_a:
    st.markdown("## Moisture Sensor A")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_a_df['created_at'], y=moisture_a_df['moisture_a'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with moisture_b:
    st.markdown("## Moisture Sensor B")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_b_df['created_at'], y=moisture_b_df['moisture_b'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
with moisture_c:
    st.markdown("## Moisture Sensor C")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_c_df['created_at'], y=moisture_c_df['moisture_c'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with moisture_d:
    st.markdown("## Moisture Sensor D")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_d_df['created_at'], y=moisture_d_df['moisture_d'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    
moisture_e, moisture_f, moisture_g, moisture_h = st.columns(4)

with moisture_e:
    st.markdown("## Moisture Sensor E")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_e_df['created_at'], y=moisture_e_df['moisture_e'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with moisture_f:
    st.markdown("## Moisture Sensor F")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_f_df['created_at'], y=moisture_f_df['moisture_f'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
with moisture_g:
    st.markdown("## Moisture Sensor G")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_g_df['created_at'], y=moisture_g_df['moisture_g'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with moisture_h:
    st.markdown("## Moisture Sensor H")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moisture_h_df['created_at'], y=moisture_h_df['moisture_h'], mode='lines'))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)







# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

# Display DataFrames
st.subheader('Sensors Data')
st.dataframe(sensors_data_df, use_container_width=True)

st.subheader('Settings Data')
st.dataframe(settings_data_df, use_container_width=True)





# # Combine the 'date' and 'time' columns into a single column
# plant_data_df['datetime'] = pd.to_datetime(
#     plant_data_df['date'] + ' ' + plant_data_df['time'])

# # Drop the original 'date' and 'time' columns
# plant_data_df = plant_data_df.drop(['date', 'time'], axis=1)
# column_order = ['datetime', 'ec', 'ph', 'windSpeed',
#                 'ambientTemp', 'waterTemp', 'hum', 'atmosPressure']
# plant_data_df = plant_data_df[column_order]




# # Convert the datetime64[ns] Series to a Series of datetime.datetime objects
# plant_data_df['datetime'] = plant_data_df['datetime'].apply(
#     lambda x: x.to_pydatetime() if pd.notnull(x) else x)

# plant_data_df = plant_data_df[(plant_data_df['datetime'] >= date_start) & (
#     plant_data_df['datetime'] <= date_end)]

# latest_data = plant_data_df[['datetime', 'ec', 'ph', 'windSpeed',
#                              'ambientTemp', 'waterTemp', 'hum', 'atmosPressure']].iloc[-1]


# def convert_to_black_fill(image_path):
#     # Open the image file
#     img = Image.open(image_path).convert("RGBA")
#     # Get the image data
#     data = img.getdata()

#     # Create new image data
#     new_data = []
#     for item in data:
#         # change all white (also shades of whites)
#         # pixels to yellow
#         if item[0] in list(range(200, 256)):
#             # change all white (also shades of whites) pixels to black
#             new_data.append((0, 0, 0, 255))
#         else:
#             new_data.append(item)  # append original data
#     # Update image data
#     img.putdata(new_data)
#     return img


# # Create a row with four columns
# col1, col2, col3, col4 = st.columns(4)

# # Column 1
# with col1:
#     img = convert_to_black_fill("assets/NutrientsEC.png")
#     st.image(img, width=80)
#     st.markdown("Nutrient Level")
#     st.write(f"{latest_data['ec']}")

# # Column 2
# with col2:
#     img = convert_to_black_fill("assets/pH.png")
#     st.image(img, width=50)
#     st.markdown("pH Level")
#     st.write(f"{latest_data['ph']}")

# # Column 3
# with col3:
#     img = convert_to_black_fill("assets/WindSpeed.png")
#     st.image(img, width=80)
#     st.markdown("Windspeed")
#     st.write(f"{latest_data['windSpeed']}")

# # Column 4
# with col4:
#     img = convert_to_black_fill("assets/AmbientTemperature.png")
#     st.image(img, width=40)
#     st.markdown("Ambient Temperature")
#     st.write(f"{latest_data['ambientTemp']}")

# # Create another row with three columns for the remaining data
# col5, col6, col7, col8 = st.columns(4)

# # Column 5
# with col5:
#     img = convert_to_black_fill("assets/WaterTemperature.png")
#     st.image(img, width=60)
#     st.markdown("Water Temperature")
#     st.write(f"{latest_data['waterTemp']}")

# # Column 6
# with col6:
#     img = convert_to_black_fill("assets/Humidity.png")
#     st.image(img, width=80)
#     st.markdown("Humidity")
#     st.write(f"{latest_data['hum']}")

# # Column 7
# with col7:
#     img = convert_to_black_fill("assets/AtmosphericPressure.png")
#     st.image(img, width=80)
#     st.markdown("Atmospheric Pressure")
#     st.write(f"{latest_data['atmosPressure']}")

