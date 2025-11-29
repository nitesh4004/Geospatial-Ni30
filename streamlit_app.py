import streamlit as st
import ee
import geemap.foliumap as geemap

# --- 1. CONFIGURATION & AUTHENTICATION ---
st.set_page_config(layout="wide", page_title="Landslide Susceptibility Mapping")

st.title("🏔️ Landslide Susceptibility Mapping Tool")
st.markdown("""
This application uses **Google Earth Engine** to perform a weighted overlay analysis for landslide susceptibility. 
Adjust the weights in the sidebar to visualize how different factors (Slope, Rainfall, Vegetation) contribute to risk.
""")

# Initialize Earth Engine
# Tries to initialize; if it fails, it prompts the user to authenticate via CLI.
try:
    ee.Initialize(project='my-project') # Replace 'my-project' if you have a specific GEE project, or remove argument if using default.
except Exception as e:
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as e:
        st.error(f"GEE Initialization failed. Please run `earthengine authenticate` in your terminal. Error: {e}")
        st.stop()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("1. Analysis Parameters")

# Date Selection for Rainfall
start_date = st.sidebar.date_input("Start Date (Rainfall)", value=pd.to_datetime("2023-06-01"))
end_date = st.sidebar.date_input("End Date (Rainfall)", value=pd.to_datetime("2023-09-30"))

st.sidebar.header("2. Factor Weights")
st.sidebar.info("Weights must sum to 1.0 ideally, but the map will visualize relative scores regardless.")

w_slope = st.sidebar.slider("Slope Weight (Topography)", 0.0, 1.0, 0.5, 0.1)
w_rain = st.sidebar.slider("Rainfall Weight (Trigger)", 0.0, 1.0, 0.3, 0.1)
w_ndvi = st.sidebar.slider("Vegetation Weight (Inverse)", 0.0, 1.0, 0.2, 0.1)

st.sidebar.header("3. Region of Interest")
location_options = {
    "Kerala, India (Waynad)": {'lat': 11.605, 'lon': 76.083, 'zoom': 11},
    "Uttarakhand, India (Joshimath)": {'lat': 30.55, 'lon': 79.56, 'zoom': 12},
    "Rio de Janeiro, Brazil": {'lat': -22.9068, 'lon': -43.1729, 'zoom': 11},
    "Custom (Use Map Center)": None
}
selected_loc = st.sidebar.selectbox("Jump to Location", options=location_options.keys())

# --- 3. GEOSPATIAL ANALYSIS FUNCTIONS ---

def get_factors(roi, start, end):
    """
    Fetches and processes GEE layers for Slope, Rainfall, and NDVI.
    Returns normalized images (0 to 1).
    """
    
    # --- A. SLOPE (Topography) ---
    # Data: NASA NASADEM (Global 30m)
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
    slope = ee.Terrain.slope(dem)
    
    # Normalize Slope: 
    # We assume 0 degrees is low risk, 45+ degrees is max risk.
    # .unitScale(min, max) clamps values to 0-1.
    slope_norm = slope.unitScale(0, 45)

    # --- B. RAINFALL (Trigger) ---
    # Data: CHIRPS Pentad (Global Rainfall)
    rain_col = ee.ImageCollection("UCSB-CHIRPS/PENTAD") \
        .filterDate(str(start), str(end)) \
        .filterBounds(roi)
    
    # Calculate mean daily rainfall over the period
    rain_mean = rain_col.mean()
    
    # Normalize Rainfall:
    # We assume 0 mm is low risk, 20 mm/day (avg) is very high risk for the period.
    rain_norm = rain_mean.unitScale(0, 20)

    # --- C. VEGETATION (Land Cover) ---
    # Data: Sentinel-2 (High Res) or Landsat 8
    # We use Landsat 8 for speed and lower cloud issues in composites for this demo
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterDate(str(start), str(end)) \
        .filterBounds(roi) \
        .median()
    
    # Calculate NDVI
    ndvi = l8.normalizedDifference(['SR_B5', 'SR_B4']) # NIR, Red
    
    # Normalize NDVI:
    # NDVI usually -1 to 1. We look at 0 to 0.8 for vegetation.
    # INVERSE Relationship: High Vegetation (Forest) = Low Landslide Risk.
    # Formula: 1 - Normalized_NDVI
    ndvi_val = ndvi.unitScale(-0.1, 0.8)
    ndvi_inv = ee.Image(1).subtract(ndvi_val)

    return slope_norm, rain_norm, ndvi_inv, dem

def compute_susceptibility(slope, rain, veg, w_s, w_r, w_v):
    """
    Weighted Linear Combination:
    LSM = (Slope * W_s) + (Rain * W_r) + (Veg * W_v)
    """
    lsm = slope.multiply(w_s) \
        .add(rain.multiply(w_r)) \
        .add(veg.multiply(w_v))
    return lsm

# --- 4. MAP RENDERING ---

# Set up the map
m = geemap.Map()
m.add_basemap("HYBRID")

# Determine ROI based on user selection or current map bounds
if selected_loc != "Custom (Use Map Center)":
    loc = location_options[selected_loc]
    m.setCenter(loc['lon'], loc['lat'], loc['zoom'])
    # Create a geometry for computation (small buffer around center)
    roi = ee.Geometry.Point([loc['lon'], loc['lat']]).buffer(20000).bounds()
else:
    # Default to a generic view if custom
    roi = ee.Geometry.Point([76.083, 11.605]).buffer(20000).bounds()

# Perform Calculation
with st.spinner("Crunching satellite data on Google Earth Engine..."):
    slope_layer, rain_layer, veg_layer, dem_layer = get_factors(roi, start_date, end_date)
    susceptibility_map = compute_susceptibility(slope_layer, rain_layer, veg_layer, w_slope, w_rain, w_ndvi)

# Visualization Parameters
vis_lsm = {
    'min': 0,
    'max': 0.8, # Threshold max at 0.8 to make colors pop for "High" risk
    'palette': ['green', 'yellow', 'orange', 'red', 'darkred']
}

vis_slope = {'min': 0, 'max': 45, 'palette': ['white', 'black']}
vis_rain = {'min': 0, 'max': 20, 'palette': ['blue', 'purple', 'black']}

# Add Layers to Map
# We add them as separate layers so the user can toggle them in the layer control
m.addLayer(dem_layer, {'min': 0, 'max': 3000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}, 'Elevation (DEM)', False)
m.addLayer(slope_layer, {'min': 0, 'max': 1}, 'Factor: Slope (Normalized)', False)
m.addLayer(rain_layer, {'min': 0, 'max': 1, 'palette': ['white', 'blue']}, 'Factor: Rainfall (Normalized)', False)
m.addLayer(veg_layer, {'min': 0, 'max': 1, 'palette': ['green', 'white']}, 'Factor: Vegetation Inverse', False)

# The Main Result
m.addLayer(susceptibility_map, vis_lsm, 'Landslide Susceptibility Index')

# Add a colorbar/legend
m.add_colorbar(vis_lsm, label="Susceptibility Index (Low -> High)")

# Render Map in Streamlit
m.to_streamlit(height=700)

# --- 5. EXPLANATION ---
with st.expander("See Calculation Logic"):
    st.write(f"""
    The map is calculated using the following formula:
    
    $$ \\text{{LSI}} = ({w_slope} \\times \\text{{Slope}}_{{norm}}) + ({w_rain} \\times \\text{{Rain}}_{{norm}}) + ({w_ndvi} \\times (1 - \\text{{NDVI}}_{{norm}})) $$
    
    * **Slope:** Derived from NASA NASADEM. Steeper slopes (up to 45°) increase risk.
    * **Rainfall:** Mean precipitation from CHIRPS data for the selected date range.
    * **Vegetation:** Inverse NDVI from Landsat 8. Bare soil (low NDVI) contributes to higher risk.
    """)
