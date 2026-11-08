import pandas as pd
import plotly.express as px
import itertools

data = pd.read_csv("NipahVisual/data/nipah_deaths.csv")

countries = data["Country"].unique()
years = range(1997, data["Year"].max() + 1)
full_grid = pd.DataFrame(list(itertools.product(countries, years)), columns=["Country", "Year"])

data_full = pd.merge(
    full_grid, 
    data[["Country", "Year", "Fatal cases"]], 
    on=["Country", "Year"], 
    how="left"
)

data_full["Fatal cases"] = data_full["Fatal cases"].fillna(0)

data_full = data_full.sort_values(by=["Country", "Year"])
data_full["Cumulative_Cases"] = data_full.groupby("Country")["Fatal cases"].cumsum()

data_full = data_full[data_full["Year"] == 2024]

fig = px.choropleth(
    data_full,
    locations="Country",
    locationmode="country names",
    color="Cumulative_Cases",
    hover_name="Country",
    color_continuous_scale="Reds",
    range_color=[0, data_full["Cumulative_Cases"].max()],
    title="Cumulative Annual Nipah Virus Fatalities in South and Southeast Asia Since 1997"
)

fig.update_geos(
    scope="asia",
    lonaxis_range=[60, 130],
    lataxis_range=[0, 35],
    showcountries=True,
    countrycolor="white"
)

fig.update_layout(coloraxis_colorbar=dict(title="Cumulative Fatalities"))

fig.show()