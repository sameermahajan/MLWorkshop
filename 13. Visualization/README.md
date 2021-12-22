You can read more about these Data Plots [here](https://www.analyticsvidhya.com/blog/2021/12/12-data-plot-types-for-visualization/)

Here we cover various **Data Plots** namely

- [Bar Graph](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Bar%20Graph.ipynb)
- [Line Graph](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Line%20Graph.ipynb)
- [Pie Chart](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Pie%20Chart.ipynb)
- [Histogram](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Histogram.ipynb)
- [Area Chart](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Area%20Chart.ipynb)
- [Dot Graph](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Dot%20Graph.ipynb)
- [Scatter Plot](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Scatter%20Plot.ipynb)
- [Bubble Chart](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Bubble%20Chart.ipynb)
- [Radar Chart](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Radar%20Chart.ipynb)
- [Pictogram](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Pictogram.ipynb)
- [Spline Chart](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Spline%20Chart.ipynb)
- [Box Plot](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Box%20Plot.ipynb)
- [Violin Plot](https://github.com/sameermahajan/MLWorkshop/blob/master/13.%20Visualization/Violin%20Plot.ipynb)

The code to generate these plots in plotly and seaborn is given along with their generated visualizations.

Here is a reference cheat sheet for methods and attributes to be used:

| Plot type | plotly | seaborn |
|---|---|---|
| Simple bar graph | express bar | barplot |
| Grouped bar graph | color attribute and barmode=’group’ | hue attribute |
| Stacked bar graph | color attribute | label and color attributes with multiple plots |
| Simple line graph	| express line	| lineplot |
| Multiple line graph	| color and symbol attributes	| hue attribute |
| Simple pie chart	| express pie	| matplotlib.pyplot.pie |
| Exploded pie chart	| graph_objects Pie with pull attribute	| explode attribute |
| Donut chart	| graph_objects Pie with hole attribute	| Add matplotlib.pyplot.Circle |
| 3D pie chart	| Use pygooglechart package	| shadow attribute |
| Normal histogram | express histogram |	histplot |
| Bimodal histogram	| color attribute	| kdeplot |
| Area chart	| express area	| matplotlib.pyplot.stackplot |
| Dot graph	| express scatter	| stripplot |
| Scatter plot	| express scatter	| scatterplot |
| Bubble chart	| express scatter with color and size attributes	| scatterplot with size attribute |
| Radar chart	| express line_polar	| matplotlib.pyplot figure |
| Pictogram 	| graph_objects Figure having Scatter with marker attribute	| matplotlib.pyplot figure with pywaffle package |
| Spline chart	| express line with line_shape=’spline’	| Scipy.interpolate.make_interp_spline |
| Box plot	| express box	| boxplot |
| Violin Plot	| express violin	| violinplot |
