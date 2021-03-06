# Determining Equitable COVID-19 Vaccination Site Locations Through Unsupervised Learning

This repository contains all data files and code utilized in my capstone project for the Data Science graduate program at University of Maryland, Baltimore County.

## Project Goal

This project aims to use an unsupervised weighted centroid-based clustering algorithm and neighborhood-level demographic Census data to determine equitable vaccination distribution sites in six counties across five states that were previously identified by NPR as containing the largest vaccination disparity between majority-White communities and majority-BIPOC communities. Results will be displayed on an interactive web app that will allow users to input parameters and compare the vaccination distribution maps.

## Phase 1
Data Collection: 
*   Tabular Census data and spatial boundary data from [NHGIS](https://www.nhgis.org/).
*   Current vaccination sites from health department sites of: Alabama, Louisiana, Texas, Georgia, Maryland.

Data cleaning & wrangling:
*   Filter he Census data to the counties of interest and join to the spatial boundary shapefiles.
*   Calculate demographic statistics and created categorical variables.
*   Compute centroids of neighborhoods to serve as clustering points. 

## Phase 2
EDA & Model Construction:
*   EDA on the spatial and demographic data.
*   Construction of weighted k-means algorithm.
*   Creation of custom metrics to evaluate effectiveness.

## Phase 3
Execution and Interpretation:
*   Add functionality to search for infrastructure near established sites.
*   Create web app framework for user interaction.


## Streamlit Web App
<img src="https://github.com/travis8719/Determining-Equitable-COVID-19-Vaccination-Site-Locations-Through-Unsupervised-Learning/blob/main/Phase3/webapp.png"/>

## To Use:
*   Create new virtual environment: `conda create -n streamlit_env python=3.7.6 anaconda`
*   Activate your new virtual environment: `conda activate streamlit_env`
*   Navigate to where you want to clone repo.
*   Clone repo: git clone `https://github.com/travis8719/Determining-Equitable-COVID-19-Vaccination-Site-Locations-Through-Unsupervised-Learning.git`
*   Install packages needed for the project: `pip install -r requirements.txt`
*   Run python file: `python3 streamlit_find_vaccine_sites.py`
*   To view locally in the browser: `streamlit run streamlit_find_vaccine_sites.py`


### Check out my [Capstone Project Site](https://sites.google.com/umbc.edu/data606/spring-21-section-2/travis-twigg) for additonal details.
