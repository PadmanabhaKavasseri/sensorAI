import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

# Directory containing CSV file
data_dir = 'raw_data'
filename = 'circle_small_fast.csv'
file_path = os.path.join(data_dir, filename)

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    exit(1)

# Read the CSV file
df = pd.read_csv(file_path)

# Convert timestamp strings to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

# Drop any rows where timestamp couldn't be parsed
df = df.dropna(subset=['timestamp'])

# Convert to seconds since start for plotting
df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

# Create Dash app
app = dash.Dash(__name__)

# Create subplots for accelerometer and gyroscope
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=("Accelerometer", "Gyroscope"),
                    vertical_spacing=0.1)

# Accelerometer traces
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['accel_x'], name='accel_x', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['accel_y'], name='accel_y', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['accel_z'], name='accel_z', line=dict(color='red')), row=1, col=1)

# Gyroscope traces
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['gyro_x'], name='gyro_x', line=dict(color='purple')), row=2, col=1)
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['gyro_y'], name='gyro_y', line=dict(color='orange')), row=2, col=1)
fig.add_trace(go.Scatter(x=df['time_sec'], y=df['gyro_z'], name='gyro_z', line=dict(color='brown')), row=2, col=1)

# Update layout with range slider and title
fig.update_layout(
    title='circle_1',
    xaxis2=dict(
        rangeslider=dict(visible=True, thickness=0.05),  # Draggable range slider
        type="linear",
        title="Time (s)"
    ),
    yaxis=dict(title="Acceleration"),
    yaxis2=dict(title="Angular Velocity"),
    showlegend=True,
    height=800,
    width=1000
)

# Dash layout
app.layout = html.Div([
    dcc.Graph(id='sensor-plot', figure=fig),
    html.Div(id='click-output', style={'margin-top': '20px'}),
    html.Label('Start Time (seconds):'),
    dcc.Input(id='start-time', type='number', value=0),
    html.Label('End Time (seconds):'),
    dcc.Input(id='end-time', type='number', value=df['time_sec'].max()),
    html.Button('Save Trimmed CSV', id='save-button', n_clicks=0),
    html.Div(id='save-output', style={'margin-top': '20px'})
])

# Callback to handle click events
@app.callback(
    Output('click-output', 'children'),
    Input('sensor-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click a point on the graph to see its details."
    
    point = clickData['points'][0]
    time_sec = point['x']
    curve_number = point['curveNumber']  # Corrected reference
    # Map curveNumber to column name
    curve_map = {0: 'accel_x', 1: 'accel_y', 2: 'accel_z', 3: 'gyro_x', 4: 'gyro_y', 5: 'gyro_z'}
    column = curve_map.get(curve_number, 'unknown')
    
    # Find the closest row in the DataFrame
    closest_row = df.iloc[(df['time_sec'] - time_sec).abs().idxmin()]
    row_index = closest_row.name
    timestamp = closest_row['timestamp']
    value = closest_row[column]
    
    return f"Clicked point: time_sec={time_sec:.2f}, {column}={value:.2f}, Row Index={row_index}, Timestamp={timestamp}"

# Callback to save trimmed CSV
@app.callback(
    Output('save-output', 'children'),
    Input('save-button', 'n_clicks'),
    [dash.dependencies.State('start-time', 'value'),
     dash.dependencies.State('end-time', 'value')]
)
def save_trimmed_csv(n_clicks, start_time, end_time):
    if n_clicks > 0:
        if start_time is None or end_time is None:
            return "Please enter valid start and end times."
        
        trimmed_df = df[(df['time_sec'] >= start_time) & (df['time_sec'] <= end_time)]
        if trimmed_df.empty:
            return "No data in the selected time range."
        
        output_file = os.path.join(data_dir, f"trimmed_{filename}")
        trimmed_df.to_csv(output_file, index=False)
        return f"Trimmed data saved to {output_file}"
    return ""

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)