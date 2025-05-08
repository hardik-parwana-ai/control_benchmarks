import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Dummy data: 3x3 grid of plots
rows, cols = 3, 3
x = np.linspace(0, 10, 100)
data = [np.sin(x + i + j) for i in range(rows) for j in range(cols)]

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H3("Click on any subplot to expand it"),
    html.Div(id='plots-container', style={'display': 'grid', 'gridTemplateColumns': f'repeat({cols}, 300px)'}),
    html.Div(id='expanded-plot', style={'marginTop': '40px'}),
    dcc.Store(id='clicked-index')
])

# Create the subplot grid
@app.callback(
    Output('plots-container', 'children'),
    Input('plots-container', 'id')  # Dummy input to trigger on app load
)
def create_subplots(_):
    plots = []
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=data[index], mode='lines'))
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=20),
                height=250,
                width=300,
                title=f"Plot ({i}, {j})",
                clickmode='event+select'
            )
            plots.append(
                dcc.Graph(
                    id={'type': 'subplot', 'index': index},
                    figure=fig,
                    style={'cursor': 'pointer'}
                )
            )
    return plots

# Callback to show expanded plot
@app.callback(
    Output('expanded-plot', 'children'),
    Input({'type': 'subplot', 'index': dash.ALL}, 'clickData'),
    State({'type': 'subplot', 'index': dash.ALL}, 'id')
)
def display_big_plot(click_data_list, ids):
    for click_data, id_ in zip(click_data_list, ids):
        if click_data is not None:
            row, col = divmod(id_['index'], cols)
            y = data[id_['index']]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
            fig.update_layout(title=f"Expanded Plot ({row}, {col})", height=500)
            return html.Div([
                html.H4(f"Clicked Plot Coordinates: Row={row}, Col={col}"),
                dcc.Graph(figure=fig)
            ])
    return None

if __name__ == '__main__':
    app.run(debug=True)

