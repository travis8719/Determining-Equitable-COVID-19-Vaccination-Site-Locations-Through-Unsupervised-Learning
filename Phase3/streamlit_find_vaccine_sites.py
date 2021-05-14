import folium
from folium import plugins
import pandas as pd
import json
import geopandas as gpd
from jenkspy import jenks_breaks
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import pickle
import geopy.distance
import scipy.spatial
from haversine import haversine, Unit
from streamlit_folium import folium_static
import streamlit as st
import plotly.graph_objs as go
import pyproj
from mapbox import Geocoder
import seaborn as sns
from matplotlib.figure import Figure
import copy
import time
import base64


start_time = time.time()
st.set_page_config(layout='wide')


# MAIN PAGE
st.title("Determining Equitable COVID-19 Vaccination Site Locations Through Weighted K-Means")
st.text(" ")
st.text(" ")
st.write("This web app allows users to identify equitable vaccination sites based on American Community Survey data. \
Users select a county and a data attribute to weight and the app will provide a list of the closest possible sites and stats regarding access.")
st.markdown(" ")

# SIDEBAR - GET PARAMS - DISPLAY BACKGROUND
st.sidebar.header('Parameters') 
county = st.sidebar.selectbox('Select a County', ['Baltimore City, MD','Fulton, GA','Dekalb, GA','East Baton Rouge, LA','Travis, TX', 'Mobile, AL'])
attr = st.sidebar.selectbox('Select an attribute to weight', ['Total Population','Income','BIPOC Population'])
if attr in ['Income','BIPOC Population']:
   weight = st.sidebar.number_input("Enter weight (1-9)", min_value=1, max_value = 9, value=7)
else:
    weight = 1

st.sidebar.markdown('<br><strong>Health Equity</strong><em> &nbsp; <small>(noun)</small></em><br>*/helTH ekwədē/*<br>The absence of unfair and avoidable or remediable differences in health among population groups defined socially, \
economically, demographically or geographically (World Health Organization, 2011).',unsafe_allow_html=True)

st.sidebar.markdown("<br><strong>Background</strong> <br>In the United States and worldwide, the COVID-19 pandemic has magnified the intersectional and pervasive impacts of social and structural \
inequities in society. COVID-19 is having a disproportionate impact on people who are already disadvantaged by virtue of their race and ethnicity, \
age, health status, residence, occupation, socioeconomic conditions, and/or other contributing factors (Williams and Cooper, 2020). <br><br>Given the legacies of inequality, \
injustice, and discrimination that have undermined the health and well-being of certain populations in the United States for centuries, considerations of equity should \
factor into plans for allocating and distributing COVID-19 treatments and vaccines to the population at large (Essien et al., 2020).",unsafe_allow_html=True)

# FILE PATHS
# GET PATH FOR SHP
script_path = os.path.abspath(__file__) 
script_dir = os.path.split(script_path)[0]
script_dir = os.path.split(script_dir)[0]

shp_rel_path = "data/shapefiles/"
shp_abs_file_path = os.path.join(script_dir, shp_rel_path)

# GET PATH FOR CSV
csv_rel_path = "data/blockgroups/"
csv_abs_file_path = os.path.join(script_dir, csv_rel_path)

shp_paths = os.listdir(shp_abs_file_path)
shp_paths = [shp_abs_file_path + p for p in shp_paths if p.endswith('.shp')]
county_names = ['Dekalb, GA','Mobile, AL','Fulton, GA','East Baton Rouge, LA','Baltimore City, MD','Travis, TX']
shp_path_dict = {}
for i in range(len(shp_paths)):
    shp_path_dict[county_names[i]] = shp_paths[i]

csv_paths = os.listdir(csv_abs_file_path)
csv_paths = [csv_abs_file_path + c for c in csv_paths if c.endswith('.csv')]
county_names2 = ['Fulton, GA','Baltimore City, MD','Mobile, AL','Dekalb, GA','Travis, TX','East Baton Rouge, LA']
csv_path_dict = {}
for i in range(len(csv_paths)):
    csv_path_dict[county_names2[i]] = csv_paths[i]


# LOOKUP DICTS
attr_lookup = {'Total Population':'CSV_TOTAL_','Income':'CSV_INCOME','BIPOC Population':'BIPOC_PERCENT'}
params = {'county':county,'attribute':attr_lookup[attr],'weight':weight}
attr_alias_lookup = {'CSV_TOTAL_':'Population:','CSV_INCOME':'Median Household Income ($):','BIPOC_PERCENT':'BIPOC Population (%):'}
color_lookup = {'Total Population':'Purples','Income':'PuBu','BIPOC Population':'BuGn'}
seaborn_color_lookup = {'Total Population':'#9e9ac8','Income':'#74a9cf','BIPOC Population':'#66c2a4'}
icon_color_lookup = {'Total Population':'#9e9ac8','Income':'#74a9cf','BIPOC Population':'#66c2a4'}

# SET UP FIRST ROW OF COLS
row1_col1, row1_col2 = st.beta_columns((1,1))

# HISTORGRAM PLOT
def eda_plot(geo_df,version=1):
    version = 0 #use plt instead of sns
    if version:
        label_dict = {"CSV_INCOME":'Median Household Income','CSV_TOTAL_':'Total Population Per Blockgroup','BIPOC_PERCENT':'BIPOC Percentage of Total Population Per Blockgroup'}
        title = label_dict[attr_lookup[attr]] + ' for ' + county
        fig = Figure()
        ax = fig.subplots()
        ax.set_facecolor('#EAEAF2')
        fig.set_edgecolor('#EAEAF2')
        fig.patch.set_alpha(.7)
        ax.patch.set_alpha(1)    
        sns.histplot(ax=ax,data=geo_df, x=attr_lookup[attr],color=seaborn_color_lookup[attr]);
        ax.set(xlabel=label_dict[attr_lookup[attr]], ylabel='Count');
        with row1_col2:
            st.pyplot(fig)
    else:
        label_dict = {"CSV_INCOME":'Median Household Income','CSV_TOTAL_':'Total Population Per Blockgroup','BIPOC_PERCENT':'BIPOC Percentage of Total Population Per Blockgroup'}
        title = label_dict[attr_lookup[attr]] + ' for ' + county
        fig = Figure()
        ax = fig.subplots()
        ax.set_facecolor('#EAEAF2')
        fig.set_edgecolor('#EAEAF2')
        fig.patch.set_alpha(.7)
        ax.patch.set_alpha(1)
        n, bins, patches = ax.hist(geo_df[attr_lookup[attr]], bins=25, facecolor='#EAEAF2', edgecolor='#EAEAF2', linewidth=0.5, alpha=.9)
        n = n.astype('int')
        
        for i in range(len(patches)):
            patches[i].set_facecolor(seaborn_color_lookup[attr])

        ax.set(xlabel=label_dict[attr_lookup[attr]], ylabel='Count');
        with row1_col2:
            st.pyplot(fig)

# DISTRIBUTION MAP  
def map_choro():
    geo_df = gpd.read_file(shp_path_dict[params['county']])
    # fix issue in mobile column names
    mobile_new_cols = {'CSV_POPULA': 'CSV_TOTAL_',
        'CSV_WHIT_1': 'CSV_PERCEN'}
    geo_df.rename(columns=mobile_new_cols,
          inplace=True)
    
    geo_df['BIPOC_PERCENT'] = 1 - geo_df['CSV_PERCEN']
    style_function = lambda x: {'weight': '.5','color':'#969696'}
    n = folium.Map(tiles="cartodbpositron",control=True)
    layer = folium.GeoJson(data=geo_df["geometry"],style_function=style_function,control=True,name='County').add_to(n)

    n.fit_bounds(layer.get_bounds())
    
    folium.TileLayer("Stamen Watercolor").add_to(n)
    folium.TileLayer("Stamen Terrain").add_to(n)
    folium.TileLayer("Stamen Toner").add_to(n)
    folium.TileLayer("OpenStreetMap").add_to(n)

    from jenkspy import jenks_breaks
    breaks = jenks_breaks(geo_df[params['attribute']], nb_class=7)
    
    choropleth = folium.Choropleth(
       geo_data=geo_df.to_json(), 
       data=geo_df,
       columns=('GISJOIN', params['attribute']),
       key_on='feature.properties.GISJOIN',
       fill_color=color_lookup[attr],
       fill_opacity=0.7,
       nan_fill_color='white',
       nan_fill_opacity=0.4,
       #line_weight=2,
       #line_color = 'black',
       line_opacity=0.2,
       legend_name='Legend Title',
       highlight=True,
       reset=True,
       control=True,
       name='Choropleth',
       bins = breaks)
    
    # remove legend
    for key in choropleth._children:
        if key.startswith('color_map'):
           del(choropleth._children[key])

    folium.GeoJsonTooltip(
            fields=[params['attribute']], 
            aliases=[attr_alias_lookup[params['attribute']]],
            localize=True,control=False).add_to(choropleth.geojson)

    choropleth.add_to(n)
    folium.LayerControl().add_to(n)

    t1 = attr + ' Map'
    t2 = attr + ' Distribution Map'
    title = {'Total Population':t1,'Income':t2,'BIPOC Population':t1}

    tp = 'Population Distribution'
    inc = 'Median Income Distribution'
    p1 = attr + ' Distribution'
    title2 = {'Total Population':tp,'Income':inc,'BIPOC Population':p1}
    with row1_col1:
        st.markdown("<center><h3>"+title[attr]+"</h3>", unsafe_allow_html=True)
        folium_static(n,width=420,height=347)
    with row1_col2:
        st.markdown("<center><h3>"+title2[attr]+"</h3>", unsafe_allow_html=True)
        eda_plot(geo_df,1)

# CACHE CSV DATA
@st.cache(allow_output_mutation=True)
def load_csv_data(weight):

    '''
    READS IN CENSUS DATA FOR EACH BLOCKGROUP
    CALCULATES WEIGHT FOR EACH BLOCKGROUP BASED ON USER INPUT
    OUTPUTS 2D LIST OF [WEIGHT, LAT, LON] FOR EACH BLOCKGROUP
    '''
    county_df = pd.read_csv(csv_path_dict[params['county']])
    df_pop = pd.DataFrame({'gisjoin':county_df['GISJOIN'],'weight':county_df['TOTAL_POP'],'lat':county_df['LAT'],'lon':county_df['LON']}, columns=['weight','lat','lon','gisjoin'])
    df_pop_income = pd.DataFrame({'low_income':county_df['MAJ_LOW_INCOME'],'gisjoin':county_df['GISJOIN'],'pop':county_df['TOTAL_POP'],'income':county_df['INCOME'],'lat':county_df['LAT'],'lon':county_df['LON']}, columns=['pop','income','lat','lon','gisjoin','low_income'])
    df_pop_demo = pd.DataFrame({'bipoc':county_df['MAJ_BIPOC'],'gisjoin':county_df['GISJOIN'],'pop':county_df['TOTAL_POP'],'demo':county_df['BIPOC_POP'],'lat':county_df['LAT'],'lon':county_df['LON']}, columns=['pop','demo','lat','lon','gisjoin','bipoc'])

    # weighting parameters
    a = weight/10
    
    #st.write(a)
    b = 1 - a

    # create weight for population and income
    df_pop_income['weight'] = ((df_pop_income['income'].max()- df_pop_income['income'])**a) * (df_pop_income['pop']**b)

    # create weight for population and demographic
    df_pop_demo['weight'] = (df_pop_demo['demo']**a) * (df_pop_demo['pop']**b)

    # get data into numpy arrays for algorithm
    df_pop_values = df_pop[['weight','lat','lon']].values
    df_pop_demo_values = df_pop_demo[['weight','lat','lon']].values
    df_pop_income_values = df_pop_income[['weight','lat','lon']].values

    return df_pop_income, df_pop_values, df_pop_demo_values, df_pop_income_values


# WK-MEANS FUNCTIONS
def distance(coord1,coord2):
  return haversine(coord1[1:],coord2[1:])

def cluster_centroids(values, clusters, k):
  weighted_clusters=[]
  for c in range(k):
    weights=np.squeeze(np.asarray(values[clusters == c][:,0:1]))

    if weights.shape == ():
        weights=np.asarray(values[clusters == c][:,0:1])[0] #Save the day
    avg = np.average(values[clusters == c],weights=weights, axis=0) #throwing an error sometimes
    weighted_clusters.append(avg)
  return weighted_clusters

@st.cache(suppress_st_warning=True)
def wkmeans(values, k=None, centroids=None, steps=200):
  # initialize k points randomly
  centroids = values[np.random.choice(np.arange(len(values),), k, False)]

  for step in range(max(steps, 1)):
    # compute distance between each pair of the two collections of inputs
    dists = scipy.spatial.distance.cdist(centroids, values, lambda u, v: distance(u,v)**2)
    # closest centroid to each point
    clusters = np.argmin(dists, axis=0)

    new_centroids = cluster_centroids(values, clusters, k)

    if np.array_equal(new_centroids, centroids):
      break
    centroids = new_centroids
  
  return clusters, centroids

# GET DATA FOR WK-MEANS
df_pop_income, df_pop_values, df_pop_demo_values, df_pop_income_values = load_csv_data(weight)


# RUN WK-MEANS
k_values={'Dekalb, GA':25,'Mobile, AL':30,'Fulton, GA':25,'East Baton Rouge, LA':20,'Baltimore City, MD':25,'Travis, TX':40}
k = k_values[county]
np.random.seed(0)
clusters,centroids=wkmeans(df_pop_values,k)
clusters2,centroids2=wkmeans(df_pop_demo_values,k)
clusters3,centroids3=wkmeans(df_pop_income_values,k)


# SEARCH NEARBY
centroid_lon = [centroids[i][2] for i in range(len(centroids))]
centroid_lat = [centroids[i][1] for i in range(len(centroids))]
centroid_lon2 = [centroids2[i][2] for i in range(len(centroids2))]
centroid_lat2 = [centroids2[i][1] for i in range(len(centroids2))]
centroid_lon3 = [centroids3[i][2] for i in range(len(centroids3))]
centroid_lat3 = [centroids3[i][1] for i in range(len(centroids3))]

pop_centroids = pd.DataFrame({'lon':centroid_lon,'lat':centroid_lat})
demo_centroids = pd.DataFrame({'lon':centroid_lon2,'lat':centroid_lat2})
income_centroids = pd.DataFrame({'lon':centroid_lon3,'lat':centroid_lat3})


# MAPBOX FORWARD GEOCODING
mapbox_token = 'pk.eyJ1Ijoiamw4NzgxNyIsImEiOiJja244emZ6YjMwZWR0Mm9waWh0d3FpY24zIn0.d4S4s9v5PE93-J3othM21g'
geocoder = Geocoder(access_token=mapbox_token)

# DISTANCE FUNCTION
def geodesic_distance(coord1,coord2):
  return geopy.distance.distance(coord1, coord2).km

# CONVERT FORWARD GEOCODING RESULTS INTO DF
@st.cache()
def geojson_to_df(geojson, point):
  df = gpd.GeoDataFrame(geojson['features'])
  df = df[['place_name','center','properties']]
  df.columns = ['name','coordinates','properties']
  df['new_coord'] = [tuple(x[::-1]) for x in tuple(df['coordinates'])]
  types = [i['category'].split(',')[0] if 'category' in i else 'unknown' for i in df['properties']]
  df['type'] = types
  df['distance'] = df.apply(lambda x: geodesic_distance(point,x.new_coord),axis=1)
  df = df.sort_values('distance')
  df = df.drop('properties',axis=1)
  df['old_lat'] = point[0]
  df['old_lon'] = point[1]
  # The fast food restaurant Church's Chicken confuses the geolocate api and is returned when looking for churches...
  df = df[~df.name.str.contains("Chicken")]
  return df

# FORWARD GEOCODING
@st.cache(allow_output_mutation=True)
def get_nearby_locations(df):

    locations = []

    for idx,i in enumerate(df['lon']):
        point = (df['lat'][idx], df['lon'][idx])

        med_center = geocoder.forward(address="medical center", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        mdf = geojson_to_df(med_center,point)  
        
        pharmacy = geocoder.forward(address="pharmacy", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        pdf = geojson_to_df(pharmacy,point)

        grocery = geocoder.forward(address="grocery", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        gdf = geojson_to_df(grocery,point)

        com_center = geocoder.forward(address="community center", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        cdf = geojson_to_df(com_center,point)

        church = geocoder.forward(address="church", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        churchdf = geojson_to_df(church,point)
        
        school = geocoder.forward(address="school", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        sdf = geojson_to_df(school,point)

        uni = geocoder.forward(address="university", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        udf = geojson_to_df(uni,point)

        college = geocoder.forward(address="college", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        collegedf = geojson_to_df(college,point) 

        clinic = geocoder.forward(address="clinic", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        clinicdf = geojson_to_df(clinic,point)

        hospital = geocoder.forward(address="hospital", lon=df['lon'][idx], lat=df['lat'][idx]).geojson()
        hdf = geojson_to_df(hospital,point) 

        all_df = [pdf,gdf,cdf,churchdf,sdf,clinicdf,hdf,mdf,udf,collegedf]
        final_df = pd.concat(all_df).reset_index().drop('index',axis=1)
        final_df = final_df.sort_values('distance')


        best = final_df.iloc[0].values
        locations.append(best)
    return locations


centroid_list_lookup = {'Total Population':pop_centroids,'Income':income_centroids,'BIPOC Population':demo_centroids}

location_list = get_nearby_locations(centroid_list_lookup[attr])

# CONVERT SEARCH RESULTS TO CLEANED DF
@st.cache()
def searchresult_to_df(location_list):
    df = pd.DataFrame(location_list, columns=['place','coordinates','new_coord','type','distance_new','old_lat','old_lon'])
    new_lat = [i[0] for i in df.new_coord ]
    new_lon = [i[1] for i in df.new_coord ]
    df['type'] = df['type'].str.capitalize()
    df['new_lat'] = new_lat
    df['new_lon'] = new_lon
    df['name'] = df.place.str.split(',',expand=True)[0]
    df['address'] = [x.split(',')[1:-2] for x in df['place']]
    df['address'] = df['address'].apply(lambda x: ' '.join(x))
    df = df.drop(['coordinates','new_coord','place'],axis=1)
    df = df[['name', 'address','type', 'distance_new', 'old_lat', 'old_lon', 'new_lat', 'new_lon']]
    return df

vaccinnation_locations = searchresult_to_df(location_list)
vaccinnation_locations2 = vaccinnation_locations[['name','address']]
vaccinnation_locations2.columns = ['Name','Address']


# CALCULATE DISTANCE FOR END METRIC
@st.cache()
def add_distances(centroids,clusters,df):
  centroid_coords = [centroids[i][1:] for i in clusters]
  df['centroid_lon'] = [i[1] for i in centroid_coords]
  df['centroid_lat'] = [i[0] for i in centroid_coords]
  #df['centroid_id']=clusters
  df['distance'] = df.apply(lambda x: haversine([x.LAT,x.LON],[x.centroid_lat,x.centroid_lon]), axis = 1)
  return df
    
blockgroup_centers_lookup = {'Total Population':df_pop_values,'Income':df_pop_income_values,'BIPOC Population':df_pop_demo_values}
cluster_output_lookup = {'Total Population':clusters,'Income':clusters2,'BIPOC Population':clusters3}
centroid_output_lookup = {'Total Population':centroids,'Income':centroids2,'BIPOC Population':centroids3}
values_df = pd.DataFrame(blockgroup_centers_lookup[attr],columns=['w','lat','lon'])

# READ IN CENSUS DATA FOR CURRENT COUNTY
county_df = pd.read_csv(csv_path_dict[params['county']])

# CALCULATE DISTANCE OF EACH BLOCKGROUP TO NEAREST VACCINNATION LOCATION
test_df = add_distances(centroid_output_lookup[attr],cluster_output_lookup[attr],county_df)

distance_label_lookup={'Total Population':'Mean Distance to Vaccination Site (Miles)','Income':'Mean Distance to Vaccination Site <br>for Lower-Income Communities (Miles)','BIPOC Population':'Mean Distance to Vaccination Site <br>for BIPOC Communities (Miles)'}
conv_fac_km_miles = 0.621371

# CREATE DF FOR FINAL METRICS
if attr == 'Total Population':
    mean_dist_TP_km = test_df.distance.mean()
    mean_dist_TP_mi = round(mean_dist_TP_km * conv_fac_km_miles,2)
    dist_df = pd.DataFrame({distance_label_lookup['Total Population']:[str(mean_dist_TP_mi)]})

elif attr == 'Income':
    mean_dist_km = test_df[test_df['MAJ_LOW_INCOME']== 1].distance.mean()
    mean_dist_mi = round(mean_dist_km * conv_fac_km_miles,2)
    dist_df = pd.DataFrame({distance_label_lookup[attr]:[str(mean_dist_mi)]})

    mean_dist_km2 = test_df[test_df['MAJ_LOW_INCOME']== 0].distance.mean()
    mean_dist_mi2 = round(mean_dist_km2 * conv_fac_km_miles,2)
    dist_df2 = pd.DataFrame({'Mean Distance to Vaccination Site <br>for Higher-Income Communities (Miles)':[str(mean_dist_mi2)]})
    
    mean_dist_TP_km = test_df.distance.mean()
    mean_dist_TP_mi = round(mean_dist_TP_km * conv_fac_km_miles,2)
    dist_TP_df = pd.DataFrame({distance_label_lookup['Total Population']:[str(mean_dist_TP_mi)]})    

elif attr == 'BIPOC Population':
    mean_dist_km = test_df[test_df['MAJ_BIPOC']== 1].distance.mean()
    mean_dist_mi = round(mean_dist_km * conv_fac_km_miles,2)
    dist_df = pd.DataFrame({distance_label_lookup[attr]:[str(mean_dist_mi)]})

    mean_dist_km3 = test_df[test_df['MAJ_BIPOC']== 0].distance.mean()
    mean_dist_mi3 = round(mean_dist_km3 * conv_fac_km_miles,2)
    dist_df3 = pd.DataFrame({'Mean Distance to Vaccination Site <br>for White Communities (Miles)':[str(mean_dist_mi3)]})
    
    mean_dist_TP_km = test_df.distance.mean()
    mean_dist_TP_mi = round(mean_dist_TP_km * conv_fac_km_miles,2)
    dist_TP_df = pd.DataFrame({distance_label_lookup['Total Population']:[str(mean_dist_TP_mi)]})


map_choro()

# DISIPLAY VACCINE SITES
def map_choro_markers():
    geo_df = gpd.read_file(shp_path_dict[params['county']])
    # fix issue in mobile column names
    mobile_new_cols = {'CSV_POPULA': 'CSV_TOTAL_',
        'CSV_WHIT_1': 'CSV_PERCEN'}
    geo_df.rename(columns=mobile_new_cols,
          inplace=True)
    
    geo_df['BIPOC_PERCENT'] = 1 - geo_df['CSV_PERCEN']
    style_function = lambda x: {'weight': '.5','color':'#969696'}
    m = folium.Map(tiles="cartodbpositron",control=True)
    layer = folium.GeoJson(data=geo_df["geometry"],style_function=style_function,control=True,name='County').add_to(m)

    m.fit_bounds(layer.get_bounds())
    
    folium.TileLayer("Stamen Watercolor").add_to(m)
    folium.TileLayer("Stamen Terrain").add_to(m)
    folium.TileLayer("Stamen Toner").add_to(m)
    folium.TileLayer("OpenStreetMap").add_to(m)

    from jenkspy import jenks_breaks
    breaks = jenks_breaks(geo_df[params['attribute']], nb_class=7)
    
    choropleth = folium.Choropleth(
       geo_data=geo_df.to_json(), 
       data=geo_df,
       columns=('GISJOIN', params['attribute']),
       key_on='feature.properties.GISJOIN',
       fill_color=color_lookup[attr],
       #fill_opacity=0.4,
       nan_fill_color='white',
       nan_fill_opacity=0.4,
       #line_weight=2,
       #line_color = 'black',
       line_opacity=0.2,
       legend_name='Legend Title',
       highlight=True,
       reset=True,
       control=True,
       name='Choropleth',
       bins = breaks)
    
    # remove legend
    for key in choropleth._children:
        if key.startswith('color_map'):
           del(choropleth._children[key])

    folium.GeoJsonTooltip(
            fields=[params['attribute']], 
            aliases=[attr_alias_lookup[params['attribute']]],
            localize=True,control=False).add_to(choropleth.geojson)

    choropleth.add_to(m)
    folium.LayerControl().add_to(m)

   
    for i in range(0,len(vaccinnation_locations)):
       folium.Marker(
          location=[vaccinnation_locations.iloc[i]['new_lat'], vaccinnation_locations.iloc[i]['new_lon']],
          popup=folium.Popup('<b>'+vaccinnation_locations.iloc[i]['name'] +'</  b>'+'<br>' + vaccinnation_locations.iloc[i]['address'] +'<br>' + '(' +
                             str(vaccinnation_locations.iloc[i]['new_lon']) + ',' + str(vaccinnation_locations.iloc[i]['new_lat']) +')' ,min_width=300,max_width=300,min_height=500,max_height=500),
          icon=folium.plugins.BeautifyIcon(icon="plus-square", icon_shape='marker', border_width=0, background_color=icon_color_lookup[attr],innerIconAnchor=[-1,8],textColor='white')
).add_to(m)

    return m


m = map_choro_markers()

# DISPLAY TABLE OF VACCINE SITES
@st.cache(allow_output_mutation=True)
def plotly_table(results,cols,width):
    layout = go.Layout(autosize=False,width=250, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
    header_values = list(results.columns)
    cell_values = []
    for index in range(0, len(results.columns)):
        cell_values.append(results.iloc[:, index : index + 1])
    fig = go.Figure(layout=layout,data=[go.Table(columnwidth = width,header=dict(values=cols,align=['left']), cells=dict(values=cell_values,align=['left']))])
    return fig

   
# DOWNLOAD TABLE 
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    file_name = 'vaccination_locations_'+county
    file_name = file_name.replace(" ", "_").replace(",", "")
    href = f'<a href="data:file/csv;base64,{b64}" download={file_name}>Download Vaccination Site <br>Locations as CSV File</a>'
    return href


# METRIC: DETERMINE HOW MANY SITES ARE INSIDE NEIGHBORHOODS OF INTEREST
intersection_attr_lookup = {'Income':'CSV_INCOME','BIPOC Population':'CSV_PERCEN'}
@st.cache()
def intersection_metric(points,poly):
   geo_df = gpd.read_file(poly)

   
   # fix issue in mobile column names
   mobile_new_cols = {'CSV_POPULA': 'CSV_TOTAL_','CSV_WHIT_1': 'CSV_PERCEN'}
   geo_df.rename(columns=mobile_new_cols,inplace=True)
    
   geo_df['BIPOC_PERCENT'] = 1 - geo_df['CSV_PERCEN']

   intersection = gpd.sjoin(points, geo_df, how="inner", op='intersects')
   if attr == 'Income':
      cutoff = geo_df.CSV_INCOME.mean()
   elif attr == 'BIPOC Population':
      cutoff = .5
   return cutoff, intersection
intersection_label_lookup={'Income':'Percent of Vaccination Sites <br>inside Lower-Income Communities','BIPOC Population':'Percent of Vaccination Sites <br>inside BIPOC Communities'}

if attr in ['Income','BIPOC Population']:
   p = vaccinnation_locations[['new_lat','new_lon']]
   points = gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p.new_lon,p.new_lat))
   points.set_crs(epsg=4326, inplace=True)

   cutoff, df_int = intersection_metric(points,shp_path_dict[params['county']])

   df_int['inside_poly'] = np.where(df_int[intersection_attr_lookup[attr]]<=cutoff, 1, 0)
   total = df_int['inside_poly'].sum()
   percentage_inside = total*100/k
   percent_df = pd.DataFrame({intersection_label_lookup[attr]:[str(percentage_inside)+'%']})

# DISPLAY MAPS, PLOTS, AND TABLES
# SECOND ROW OF COLS    
col1, col2, col3 = st.beta_columns([1,6,1])

with col1:
    st.write("")
with col2:
    st.markdown("<center><h3>Vaccination Sites<br></h3>", unsafe_allow_html=True)
    folium_static(m)
with col3:
    st.write("")

col1b, col2b = st.beta_columns([2,1])


with col1b:
    st.markdown("")
    st.markdown("<center><h3>Vaccination Site Locations</h3>", unsafe_allow_html=True)
    st.plotly_chart(plotly_table(vaccinnation_locations2,['Name','Address'],[60,60]),use_container_width=True)
with col2b:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown(get_table_download_link(vaccinnation_locations2), unsafe_allow_html=True)

    fig_dist = plotly_table(dist_df,[distance_label_lookup[attr]],[1])
    fig_dist.update_layout(height=90)
    st.write(fig_dist,use_container_width=True)
    if attr in ['Income','BIPOC Population']:
       fig_per = plotly_table(percent_df,[intersection_label_lookup[attr]],[1])
       fig_per.update_layout(height=90)
       st.write(fig_per,use_container_width=True)

st.write('Runtime: {:.2f} seconds'.format(time.time() - start_time))



    

