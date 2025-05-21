# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def adjust_bar_width(
    fig: go.Figure,
    max_full_width_bars: int = 10,
    base_width: float = 0.8,
    bargap: float = 0.1,
) -> go.Figure:
    """
    Adjust bar width dynamically based on number of bars in the figure.

    Parameters
    ----------
    fig : go.Figure
        A Plotly figure containing one or more bar traces.
    max_full_width_bars : int, optional
        Number of bars to display at full base_width before scaling kicks in.
    base_width : float, optional
        Width of each bar (as a fraction of its category slot) when few bars.
    bargap : float, optional
        Fractional gap between bars (0 to 1).

    """
    try:
        first_bar = next(trace for trace in fig.data if trace.type == "bar")
        n_bars = len(first_bar.x)
    except StopIteration:
        return fig

    scale = min(n_bars, max_full_width_bars) / max_full_width_bars
    width = base_width * scale
    fig.update_traces(selector={"type": "bar"}, width=width, offset=0)  # type: ignore
    fig.update_layout(xaxis_type="category", bargap=bargap)  # type: ignore
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    color_column: Optional[str] = None,
    custom_data: Optional[List[str]] = None,
    hover_template: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = None,
    x_tickvals: Optional[List[Any]] = None,
    x_ticktext: Optional[List[str]] = None,
    count_column: Optional[str] = None,
) -> go.Figure:
    """
    Create a standardized bar chart with consistent styling.
    """
    # Set default labels if not provided
    if x_label is None:
        x_label = x_column
    if y_label is None:
        y_label = y_column

    # Create labels dictionary
    labels = {x_column: x_label, y_column: y_label}

    # if count column provided created hover templete with it
    hover_data = None
    if count_column and count_column in df.columns:
        if not custom_data:
            custom_data = []
        if count_column not in custom_data:
            custom_data.append(count_column)

        # Create a default hover template if none is provided
        if hover_template is None:
            hover_template = f"<b>{x_label}:</b> %{{x}}<br><b>{y_label}:</b> %{{y:.2f}}<br><b>Count:</b> %{{customdata[{len(custom_data) - 1}]}}"

    # Create the chart
    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        title=title,
        labels=labels,
        color=color_column,
        barmode="group" if color_column else "relative",
        custom_data=custom_data,
        hover_data=hover_data,
    )

    # Apply common styling
    fig.update_layout(xaxis_tickangle=-45)  # type: ignore

    # Apply optional customizations
    if hover_template:
        fig.update_traces(hovertemplate=hover_template)  # type: ignore

    if y_range:
        fig.update_yaxes(range=y_range)  # type: ignore

    if x_tickvals and x_ticktext:
        fig.update_xaxes(tickvals=x_tickvals, ticktext=x_ticktext)  # type: ignore

    return adjust_bar_width(fig=fig)


def wrap_text(text: str, max_width: int = 50):
    """Wrap text to multiple lines if it's too long"""
    words = text.split()
    lines: List[str] = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= max_width:
            current_line.append(word)
            current_length += len(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)
