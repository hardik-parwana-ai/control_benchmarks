import plotly.graph_objects as go
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
metadata = ['A', 'B', 'C']

fig = go.Figure(data=go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(size=10),
    customdata=np.array(metadata).reshape(-1, 1),
    hovertemplate='X: %{x}<br>Y: %{y}<br>Meta: %{customdata[0]}<extra></extra>'
))

fig.show()

