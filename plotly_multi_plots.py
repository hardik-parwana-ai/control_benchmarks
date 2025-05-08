import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a 3x3 grid of subplots
fig = make_subplots(rows=3, cols=3, subplot_titles=[f"Plot {i+1}" for i in range(9)])

# Sample data
x = np.linspace(0, 10, 100)

# Add a plot in each subplot
for i in range(3):
    for j in range(3):
        y = np.sin(x + (i + j))  # just to vary each subplot a little
        row = i + 1
        col = j + 1
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name=f"sin(x + {i + j})"),
            row=row, col=col
        )

# Update layout with titles, size, spacing etc.
fig.update_layout(
    height=900, width=900,
    title_text="3x3 Grid of Line Plots",
    showlegend=False,
    template='plotly_white'
)

fig.show()
