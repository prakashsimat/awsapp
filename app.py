# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# from sklearn.preprocessing import StandardScaler
# import random
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output
# import plotly.express as px

# app = Flask(__name__)

# # Define the custom loss function
# custom_objects = {"mse": MeanSquaredError()}

# # Load the trained QGAN model with custom objects
# model = load_model('qgan_model.h5', custom_objects=custom_objects)

# # Sample data
# data = {
#     'UserID': [f'user{i}' for i in range(1, 51)],
#     'CPU_Usage': [random.uniform(0, 100) for _ in range(50)],
#     'Memory_Usage': [random.uniform(0, 100) for _ in range(50)],
#     'Network_Usage': [random.uniform(0, 100) for _ in range(50)],
#     'Login_Time': [random.uniform(0, 24) for _ in range(50)],
#     'File_Access_Count': [random.randint(0, 20) for _ in range(50)]
# }
# df = pd.DataFrame(data)

# # Create Dash app
# dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Layout for the Dash app
# dash_app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H1("User Properties Dashboard"), className="mb-2")
#     ]),
#     dbc.Row([
#         dbc.Col(dcc.Graph(id='cpu-usage-graph'), width=6),
#         dbc.Col(dcc.Graph(id='memory-usage-graph'), width=6)
#     ], className="mb-2"),
#     dbc.Row([
#         dbc.Col(dcc.Graph(id='network-usage-graph'), width=6),
#         dbc.Col(dcc.Graph(id='login-time-graph'), width=6)
#     ], className="mb-2"),
#     dbc.Row([
#         dbc.Col(dcc.Graph(id='file-access-graph'), width=6)
#     ])
# ], fluid=True)

# # Callbacks to update graphs
# @dash_app.callback(
#     [Output('cpu-usage-graph', 'figure'),
#      Output('memory-usage-graph', 'figure'),
#      Output('network-usage-graph', 'figure'),
#      Output('login-time-graph', 'figure'),
#      Output('file-access-graph', 'figure')],
#     [Input('cpu-usage-graph', 'id')]
# )
# def update_graphs(_):
#     fig_cpu = px.bar(df, x='UserID', y='CPU_Usage', title='CPU Usage')
#     fig_memory = px.bar(df, x='UserID', y='Memory_Usage', title='Memory Usage')
#     fig_network = px.bar(df, x='UserID', y='Network_Usage', title='Network Usage')
#     fig_login = px.bar(df, x='UserID', y='Login_Time', title='Login Time')
#     fig_file_access = px.bar(df, x='UserID', y='File_Access_Count', title='File Access Count')
#     return fig_cpu, fig_memory, fig_network, fig_login, fig_file_access

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html', data=df.to_dict(orient='records'))

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Define the suspicious threshold values
#         CPU_Usage_threshold = 49.889191
#         Memory_Usage_threshold = 50.233692
#         Network_Usage_threshold = 50.296352
#         Login_Time_threshold = 12.012202
#         File_Access_Count_threshold = 10.000000

#         # Check and print column names for debugging
#         print("Columns in DataFrame:", df.columns)

#         # Ensure 'UserID' column is dropped before feeding data to the model
#         if 'UserID' in df.columns:
#             test_data = df.drop(columns=['UserID']).values
#         else:
#             test_data = df.values

#         print("Test Data Shape:", test_data.shape)  # Print shape of test data

#         # Normalize the data using the same scaler used during training
#         scaler = StandardScaler()
#         test_data_scaled = scaler.fit_transform(test_data)

#         print("Scaled Test Data Shape:", test_data_scaled.shape)  # Print shape after scaling

#         # Make predictions
#         predictions = model.predict(test_data_scaled)

#         print("Predictions Shape:", predictions.shape)  # Print shape of predictions

#         # Ensure predictions are in the correct format
#         if predictions.shape[0] != test_data.shape[0]:
#             raise ValueError(f"Predictions shape {predictions.shape} does not match the number of input samples {test_data.shape[0]}")

#         predictions = predictions.flatten()  # Flatten the array if needed

#         # Add a column for Suspicious flag based on the conditions
#         df['Suspicious'] = (
#             (df['CPU_Usage'] > CPU_Usage_threshold) &
#             (df['Memory_Usage'] > Memory_Usage_threshold) &
#             (df['Network_Usage'] > Network_Usage_threshold) &
#             (df['Login_Time'] > Login_Time_threshold) &
#             (df['File_Access_Count'] > File_Access_Count_threshold)
#         ).astype(int)

#         # Filter suspicious users
#         suspicious_users = df[df['Suspicious'] == 1].to_dict(orient='records')

#         return render_template('dashboard.html', data=df.to_dict(orient='records'), suspicious_users=suspicious_users)
#     except Exception as e:
#         # Print the error for debugging
#         print(f"Error during prediction: {e}")
#         return render_template('dashboard.html', data=df.to_dict(orient='records'), error=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

# Define the custom loss function
custom_objects = {"mse": MeanSquaredError()}

# Load the trained QGAN model with custom objects
model = load_model('qgan_model.h5', custom_objects=custom_objects)

# Sample data
data = {
    'UserID': [f'user{i}' for i in range(1, 51)],
    'CPU_Usage': [random.uniform(0, 100) for _ in range(50)],
    'Memory_Usage': [random.uniform(0, 100) for _ in range(50)],
    'Network_Usage': [random.uniform(0, 100) for _ in range(50)],
    'Login_Time': [random.uniform(0, 24) for _ in range(50)],
    'File_Access_Count': [random.randint(0, 20) for _ in range(50)]
}
df = pd.DataFrame(data)

# Create Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the Dash app
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("User Properties Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cpu-usage-graph'), width=6),
        dbc.Col(dcc.Graph(id='memory-usage-graph'), width=6)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='network-usage-graph'), width=6),
        dbc.Col(dcc.Graph(id='login-time-graph'), width=6)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='file-access-graph'), width=6)
    ])
], fluid=True)

# Callbacks to update graphs
@dash_app.callback(
    [Output('cpu-usage-graph', 'figure'),
     Output('memory-usage-graph', 'figure'),
     Output('network-usage-graph', 'figure'),
     Output('login-time-graph', 'figure'),
     Output('file-access-graph', 'figure')],
    [Input('cpu-usage-graph', 'id')]
)
def update_graphs(_):
    fig_cpu = px.bar(df, x='UserID', y='CPU_Usage', title='CPU Usage')
    fig_memory = px.bar(df, x='UserID', y='Memory_Usage', title='Memory Usage')
    fig_network = px.bar(df, x='UserID', y='Network_Usage', title='Network Usage')
    fig_login = px.bar(df, x='UserID', y='Login_Time', title='Login Time')
    fig_file_access = px.bar(df, x='UserID', y='File_Access_Count', title='File Access Count')
    return fig_cpu, fig_memory, fig_network, fig_login, fig_file_access

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', data=df.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the suspicious threshold values
        CPU_Usage_threshold = 49.889191
        Memory_Usage_threshold = 50.233692
        Network_Usage_threshold = 50.296352
        Login_Time_threshold = 12.012202
        File_Access_Count_threshold = 10.000000

        # Ensure 'UserID' column is dropped before feeding data to the model
        if 'UserID' in df.columns:
            test_data = df.drop(columns=['UserID']).values
        else:
            test_data = df.values

        # Normalize the data using the same scaler used during training
        scaler = StandardScaler()
        test_data_scaled = scaler.fit_transform(test_data)

        # Make predictions
        predictions = model.predict(test_data_scaled)

        # Add a column for Suspicious flag based on the conditions
        df['Suspicious'] = (
            (df['CPU_Usage'] > CPU_Usage_threshold) &
            (df['Memory_Usage'] > Memory_Usage_threshold) &
            (df['Network_Usage'] > Network_Usage_threshold) &
            (df['Login_Time'] > Login_Time_threshold) &
            (df['File_Access_Count'] > File_Access_Count_threshold)
        ).astype(int)

        # Filter suspicious users
        suspicious_users = df[df['Suspicious'] == 1].to_dict(orient='records')

        return render_template('dashboard.html', data=df.to_dict(orient='records'), suspicious_users=suspicious_users)
    except Exception as e:
        # Print the error for debugging
        print(f"Error during prediction: {e}")
        return render_template('dashboard.html', data=df.to_dict(orient='records'), error=str(e))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Dummy user credentials check (replace with real authentication logic)
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the index page or dashboard
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
