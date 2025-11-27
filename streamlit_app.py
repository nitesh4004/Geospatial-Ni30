import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# --- ML/DL IMPORTS ---
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="NI30 Orbital Analytics", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --bg-color: #050509;
        --card-bg: rgba(20, 24, 35, 0.7);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        --accent-primary: #00f2ff;
        --accent-secondary: #7000ff;
        --text-primary: #e2e8f0;
    }

    .stApp { 
        background-image: radial-gradient(circle at 50% 0%, #1a1f35 0%, #050509 100%);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .title-font { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; letter-spacing: 1px; }
    p, label, .stMarkdown, div { color: var(--text-primary) !important; }

    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 16, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px;
        color: #fff !important;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, var(--accent-secondary) 0%, #4c1d95 100%);
        border: none;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(112, 0, 255, 0.4);
    }

    .glass-card {
        background: var(--card-bg);
        border: var(--glass-border);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 5px;
    }
    
    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
try:
    # Try using streamlit secrets if available
    service_account = st.secrets["gcp_service_account"]["client_email"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
    ee.Initialize(credentials)
except Exception:
    try:
        # Fallback to local auth
        ee.Initialize()
    except Exception as e:
        st.error(f"⚠️ Authentication Error: {e}")
        st.stop()

if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'dates' not in st.session_state: st.session_state['dates'] = []
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = 'Spectral Monitor'

# --- 4. HELPER FUNCTIONS ---
def parse_kml(content):
    try:
        if isinstance(content, bytes): content = content.decode('utf-8')
        match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL | re.IGNORECASE)
        if match: return process_coords(match.group(1))
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.lower().endswith('coordinates') and elem.text:
                return process_coords(elem.text)
    except: pass
    return None

def process_coords(text):
    raw = text.strip().split()
    coords = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in raw if len(x.split(',')) >= 2]
    return ee.Geometry.Polygon([coords]) if len(coords) > 2 else None

def preprocess_landsat(img):
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

def rename_landsat_bands(img):
    return img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )

def compute_index(img, platform, index, formula=None):
    if platform == "Sentinel-2 (Optical)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {
                'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 
                'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7'),
                'B8':img.select('B8'), 'B8A':img.select('B8A'), 
                'B11':img.select('B11'), 'B12':img.select('B12')
            }
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B8','B4'], 'GNDVI': ['B8','B3'], 'NDWI (Water)': ['B3','B8'], 'NDMI': ['B8','B11']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif "Landsat" in platform:
        if index == '🛠️ Custom (Band Math)':
            map_b = {'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7')}
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B5','B4'], 'GNDVI': ['B5','B3'], 'NDWI (Water)': ['B3','B5'], 'NDMI': ['B5','B6']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif platform == "Sentinel-1 (Radar)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {'VV': img.select('VV'), 'VH': img.select('VH')}
            return img.expression(formula, map_b).rename('Custom')
        if index == 'VV': return img.select('VV')
        if index == 'VH': return img.select('VH')
        if index == 'VH/VV Ratio': return img.select('VH').subtract(img.select('VV')).rename('Ratio')
    return img.select(0)

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_lulc_indices(image):
    nir = image.select("B8")
    red = image.select("B4")
    green = image.select("B3")
    blue = image.select("B2")
    swir1 = image.select("B11")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {"NIR": nir, "RED": red, "BLUE": blue}).rename("EVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    return image.addBands([ndvi, evi, gndvi, ndwi, ndmi])

def get_tiled_samples(image, roi, scale=20, num_points=1000, class_band='label', tile_scale=4):
    """Robust tile-based sampling to prevent memory errors."""
    try:
        samples = image.stratifiedSample(
            numPoints=num_points,
            classBand=class_band,
            region=roi,
            scale=scale,
            geometries=True,
            tileScale=tile_scale
        )
        return samples
    except Exception as e:
        st.warning(f"Sampling error (retrying with higher scale): {e}")
        return None

def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    thumb_url = image.getThumbURL({
        'min': vis_params['min'], 'max': vis_params['max'],
        'palette': vis_params['palette'], 'region': roi,
        'dimensions': 600, 'format': 'png'
    })
    response = requests.get(thumb_url)
    img_pil = Image.open(BytesIO(response.content))
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, facecolor='#050509')
    ax.set_facecolor('#050509')
    ax.imshow(img_pil)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#00f2ff')
    
    if is_categorical and class_names and 'palette' in vis_params:
        patches = []
        for name, color in zip(class_names, vis_params['palette']):
            patches.append(mpatches.Patch(color=color, label=name))
        legend = ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, title="Classes")
        plt.setp(legend.get_title(), color='white', fontweight='bold')
        for text in legend.get_texts(): text.set_color("white")
    elif cmap_colors:
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
        norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#050509')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 5. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-family: 'Rajdhani'; color: #fff; margin:0;">NI30</h2>
            <p style="font-size: 0.8rem; color: #00f2ff; letter-spacing: 2px; margin:0;">GEOSPATIAL CORE</p>
        </div>
    """, unsafe_allow_html=True)
    
    # MAIN MODE SELECTOR
    mode = st.radio("System Mode", 
        ["Spectral Monitor", "LULC (Supervised)", "Clustering (Unsupervised)", "Change Detection"], 
        index=0
    )
    st.session_state['mode'] = mode

    st.markdown("---")
    
    # ROI Selection
    with st.container():
        st.markdown("### 1. Target Acquisition (ROI)")
        roi_method = st.radio("Selection Mode", ["Upload KML", "Point & Buffer", "Manual Coordinates"], label_visibility="collapsed")
        
        new_roi = None
        if roi_method == "Upload KML":
            kml = st.file_uploader("Drop KML File", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            c1, c2 = st.columns([1, 1])
            lat = c1.number_input("Lat", 20.59)
            lon = c2.number_input("Lon", 78.96)
            rad = st.number_input("Radius (km)", 5)
            if lat and lon: new_roi = ee.Geometry.Point([lon, lat]).buffer(rad*1000).bounds()
        elif roi_method == "Manual Coordinates":
            c1, c2 = st.columns(2)
            min_lon = c1.number_input("Min Lon", 78.0)
            min_lat = c2.number_input("Min Lat", 20.0)
            max_lon = c1.number_input("Max Lon", 79.0)
            max_lat = c2.number_input("Max Lat", 21.0)
            if min_lon < max_lon: new_roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        if new_roi:
            if st.session_state['roi'] is None or new_roi.getInfo() != st.session_state['roi'].getInfo():
                st.session_state['roi'] = new_roi
                st.session_state['calculated'] = False
                st.toast("Target Locked: ROI Updated", icon="🎯")

    st.markdown("---")
    
    # Variables Init
    rf_trees, svm_kernel, svm_gamma, gtb_trees = 100, 'RBF', 0.5, 100
    ann_layers, ann_iter, ann_alpha = (100, 100), 500, 0.0001
    model_choice = "Random Forest"
    target_year = 2023
    cluster_count = 5
    change_year_1, change_year_2 = 2020, 2023

    # --- DYNAMIC CONFIG BASED ON MODE ---
    if mode == "Spectral Monitor":
        st.markdown("### 2. Sensor Config")
        platform = st.selectbox("Satellite Network", ["Sentinel-2 (Optical)", "Landsat 9 (Optical)", "Landsat 8 (Optical)", "Sentinel-1 (Radar)"])
        
        is_optical = "Optical" in platform
        formula, vmin, vmax, orbit = "", 0, 1, "BOTH"
        
        if is_optical:
            idx = st.selectbox("Spectral Product", ['NDVI', 'GNDVI', 'NDWI (Water)', 'NDMI', '🛠️ Custom (Band Math)'])
            if 'Custom' in idx:
                formula = st.text_input("Math Expression", "(B8-B4)/(B8+B4)")
                pal_name = "Viridis"
            elif 'Water' in idx:
                vmin, vmax = -0.5, 0.5
                pal_name = "Blue-White-Green"
            else:
                vmin, vmax = 0.0, 0.8
                pal_name = "Red-Yellow-Green"
            
            c1, c2 = st.columns(2)
            vmin = c1.number_input("Min", value=vmin)
            vmax = c2.number_input("Max", value=vmax)
            cloud = st.slider("Cloud Tolerance %", 0, 30, 10)
        else:
            idx = st.selectbox("Polarization", ['VV', 'VH', 'VH/VV Ratio'])
            vmin = -25.0
            vmax = -5.0
            pal_name = "Greyscale"
            orbit = st.radio("Pass Direction", ["DESCENDING", "ASCENDING", "BOTH"])
            cloud = 0

        pal_name = st.selectbox("Color Ramp", ["Red-Yellow-Green", "Blue-White-Green", "Magma", "Viridis", "Greyscale"], index=0)
        pal_map = {
            "Red-Yellow-Green": ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
            "Blue-White-Green": ['blue', 'white', 'green'],
            "Magma": ['black', 'purple', 'orange', 'white'],
            "Viridis": ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
            "Greyscale": ['black', 'white']
        }
        cur_palette = pal_map.get(pal_name, pal_map["Red-Yellow-Green"])

    elif mode == "LULC (Supervised)":
        st.markdown("### 2. Classifier Model")
        model_choice = st.selectbox(
            "Select Architecture", 
            [
                "AlphaEarth Embeddings (Random Forest)",
                "Google Dynamic World (Pre-trained)",
                "Standard Random Forest (Pixel-based)"
            ]
        )

        if model_choice == "AlphaEarth Embeddings (Random Forest)":
            st.info("🧬 Uses Google's 64-dim embeddings trained on ESA WorldCover. Best for generalisation.")
            rf_trees = st.slider("Trees", 50, 300, 100)
            target_year = st.selectbox("Target Year", [2020, 2021, 2022, 2023], index=3)
            cloud = 10

        elif model_choice == "Standard Random Forest (Pixel-based)":
            st.info("🌲 Standard Sentinel-2 spectral bands + RF.")
            rf_trees = st.slider("Trees", 50, 300, 100)
            cloud = st.slider("Cloud %", 0, 30, 10)

    elif mode == "Clustering (Unsupervised)":
        st.markdown("### 2. Clustering Config")
        st.info("🧬 Unsupervised K-Means on AlphaEarth Embeddings. Useful for finding patterns without labels.")
        target_year = st.selectbox("Year", [2020, 2021, 2022, 2023], index=3)
        cluster_count = st.slider("Number of Clusters (K)", 3, 10, 5)
        cloud = 10

    elif mode == "Change Detection":
        st.markdown("### 2. Change Config")
        st.info("🧬 Semantic Change Detection using Euclidean distance of AlphaEarth Embeddings.")
        c1, c2 = st.columns(2)
        change_year_1 = c1.selectbox("Year 1", [2019, 2020, 2021, 2022], index=0)
        change_year_2 = c2.selectbox("Year 2", [2020, 2021, 2022, 2023], index=3)
        cloud = 10

    st.markdown("---")
    
    # Date Inputs (Only for Spectral/Standard LULC)
    if mode in ["Spectral Monitor", "LULC (Supervised)"] and "AlphaEarth" not in model_choice:
        st.markdown("### 3. Temporal Window")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", datetime.now()-timedelta(60))
        end = c2.date_input("End", datetime.now())
    else:
        # Defaults for Embedding modes (Annual composites)
        start, end = datetime(target_year, 1, 1), datetime(target_year, 12, 31)

    st.markdown("###")
    if st.button("INITIALIZE SCAN 🚀", use_container_width=True):
        if st.session_state['roi']:
            st.session_state.update({
                'calculated': True, 
                'start': start.strftime("%Y-%m-%d") if hasattr(start, 'strftime') else start,
                'end': end.strftime("%Y-%m-%d") if hasattr(end, 'strftime') else end,
                'cloud': cloud,
                'model_choice': model_choice,
                'rf_trees': rf_trees,
                'target_year': target_year,
                'cluster_count': cluster_count,
                'change_year_1': change_year_1,
                'change_year_2': change_year_2,
                'platform': platform if mode == "Spectral Monitor" else "S2",
                'idx': idx if mode == "Spectral Monitor" else "NDVI",
                'formula': formula if mode == "Spectral Monitor" else "",
                'orbit': orbit if mode == "Spectral Monitor" else "BOTH",
                'vmin': vmin if mode == "Spectral Monitor" else 0,
                'vmax': vmax if mode == "Spectral Monitor" else 1,
                'palette': cur_palette if mode == "Spectral Monitor" else []
            })
            st.session_state['dates'] = []
        else:
            st.error("❌ Error: ROI not defined.")

# --- 6. MAIN CONTENT ---
st.markdown(f"""
<div class="glass-card" style="display:flex; justify-content:between; align-items:center; padding:15px;">
    <div>
        <h3 style="margin:0; color:#fff;">NI30 ANALYTICS</h3>
        <div style="color:#00f2ff; font-size:0.8rem;">MODE: {st.session_state['mode'].upper()}</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="color:#94a3b8; font-size:0.8rem;">SYSTEM ONLINE</div>
        <div style="color:#fff; font-family:'Rajdhani'; font-weight:bold;">{datetime.now().strftime("%H:%M UTC")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.info("👈 Please define ROI and parameters in the sidebar to start.")
    m = geemap.Map(height=500, basemap="HYBRID")
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': '#00f2ff'}, 'ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    p = st.session_state
    
    # -----------------------------------
    # MODE 1: SPECTRAL MONITOR (Original)
    # -----------------------------------
    if p['mode'] == "Spectral Monitor":
        with st.spinner("Processing Spectral Data..."):
            if "Sentinel-2" in p['platform']:
                col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(roi).filterDate(p['start'], p['end'])
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', p['cloud'])))
                processed = col.map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
            elif "Landsat" in p['platform']:
                col_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") if "Landsat 9" in p['platform'] else ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                col = col_raw.filterBounds(roi).filterDate(p['start'], p['end']).filter(ee.Filter.lt('CLOUD_COVER', p['cloud']))
                processed = col.map(lambda img: rename_landsat_bands(preprocess_landsat(img))).map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
            else: # Radar
                col = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(p['start'], p['end']).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                if p['orbit'] != "BOTH": col = col.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))
                processed = col.map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
            
            if processed.size().getInfo() > 0:
                # Basic reduction for visualization
                final_img = processed.median().clip(roi)
                band = 'Custom' if 'Custom' in p['idx'] else p['idx'].split()[0]
                if 'Ratio' in p['idx']: band = 'Ratio'
                
                vis = {'min': p['vmin'], 'max': p['vmax'], 'palette': p['palette']}
                
                m = geemap.Map(height=600, basemap="HYBRID")
                m.centerObject(roi, 13)
                m.addLayer(final_img.select(band), vis, f"{band} Median")
                m.add_colorbar(vis, label=band)
                
                col1, col2 = st.columns([3, 1])
                with col1: m.to_streamlit()
                with col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### Export")
                    if st.button("Download GeoTIFF"):
                        url = final_img.select(band).getDownloadURL({'scale': 30, 'region': roi, 'name': 'Spectral'})
                        st.markdown(f"[Download Link]({url})")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("No imagery found matching criteria.")

    # -----------------------------------
    # MODE 2: SUPERVISED LULC
    # -----------------------------------
    elif p['mode'] == "LULC (Supervised)":
        col1, col2 = st.columns([3, 1])
        m = geemap.Map(height=600, basemap="HYBRID")
        m.centerObject(roi, 13)

        if "AlphaEarth" in p['model_choice']:
            with st.spinner(f"🧬 Computing AlphaEarth Embeddings ({p['target_year']})..."):
                # 1. Training Data (Fixed to 2021/2022 for Ground Truth)
                train_year = 2021
                train_embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                    .filterDate(f'{train_year}-01-01', f'{train_year+1}-01-01') \
                    .filterBounds(roi).mosaic().clip(roi)
                
                label_img = ee.Image('ESA/WorldCover/v200/2021').clip(roi)
                training_image = train_embeddings.addBands(label_img.rename('label'))
                
                # Robust Sampling
                points = get_tiled_samples(training_image, roi, scale=20, num_points=2000, class_band='label')
                
                if points:
                    classifier = ee.Classifier.smileRandomForest(p['rf_trees']).train(points, 'label', train_embeddings.bandNames())
                    
                    # Inference
                    target_emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                        .filterDate(f"{p['target_year']}-01-01", f"{p['target_year']+1}-01-01") \
                        .filterBounds(roi).mosaic().clip(roi)
                    
                    classified = target_emb.classify(classifier)
                    
                    # Visualization (ESA Palette)
                    esa_vis = {'min': 10, 'max': 100, 'palette': ['006400', 'ffbb22', 'ffff4c', 'f096ff', 'fa0000', 'b4b4b4', 'f0f0f0', '0064c8', '0096a0', '00cf75', 'fae6a0']}
                    m.addLayer(classified, esa_vis, f"LULC {p['target_year']}")
                    m.add_legend(title="ESA Classes", builtin_legend='ESA_WorldCover')
                    
                    with col2:
                        st.success("Classification Complete")
                        st.metric("Training Points", points.size().getInfo())
                else:
                    st.error("Failed to sample training points.")

        elif "Dynamic World" in p['model_choice']:
            dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(roi).filterDate(p['start'], p['end']).select('label').mode().clip(roi)
            dw_vis = {"min": 0, "max": 8, "palette": ['#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635', '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1']}
            m.addLayer(dw, dw_vis, "Dynamic World")
            
        with col1: m.to_streamlit()

    # -----------------------------------
    # MODE 3: UNSUPERVISED CLUSTERING
    # -----------------------------------
    elif p['mode'] == "Clustering (Unsupervised)":
        with st.spinner(f"🧬 Running K-Means (k={p['cluster_count']}) on Embeddings..."):
            emb_img = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                .filterDate(f"{p['target_year']}-01-01", f"{p['target_year']+1}-01-01") \
                .filterBounds(roi).mosaic().clip(roi)
            
            # Sampling for clustering
            training = emb_img.sample(region=roi, scale=50, numPixels=3000)
            
            # Weka K-Means
            clusterer = ee.Clusterer.wekaKMeans(p['cluster_count']).train(training)
            result = emb_img.cluster(clusterer)
            
            # Random colors for clusters
            import random
            colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(p['cluster_count'])]
            
            m = geemap.Map(height=600, basemap="HYBRID")
            m.centerObject(roi, 13)
            m.addLayer(result.randomVisualizer(), {}, "Clusters (Random Colors)")
            
            st.markdown(f"### 🧬 Unsupervised Landscape Segmentation ({p['target_year']})")
            m.to_streamlit()

    # -----------------------------------
    # MODE 4: CHANGE DETECTION
    # -----------------------------------
    elif p['mode'] == "Change Detection":
        with st.spinner(f"🧬 Calculating Semantic Distance: {p['change_year_1']} vs {p['change_year_2']}..."):
            emb1 = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                .filterDate(f"{p['change_year_1']}-01-01", f"{p['change_year_1']+1}-01-01") \
                .filterBounds(roi).mosaic().clip(roi)
            
            emb2 = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                .filterDate(f"{p['change_year_2']}-01-01", f"{p['change_year_2']+1}-01-01") \
                .filterBounds(roi).mosaic().clip(roi)
            
            # Euclidean Distance between embedding vectors
            # distance = sqrt(sum((a-b)^2))
            diff = emb1.subtract(emb2).pow(2).reduce(ee.Reducer.sum()).sqrt().rename("Semantic Change")
            
            # Visualization
            # High distance = High change
            # Normalize visual roughly 0-0.5 depending on embedding range
            vis_change = {'min': 0, 'max': 1.0, 'palette': ['black', 'blue', 'purple', 'red', 'yellow']}
            
            m = geemap.Map(height=600, basemap="HYBRID")
            m.centerObject(roi, 13)
            m.addLayer(diff, vis_change, f"Change {p['change_year_1']}-{p['change_year_2']}")
            
            c1, c2 = st.columns([3, 1])
            with c1:
                m.to_streamlit()
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Change Intensity")
                st.markdown("Lighter colors (Red/Yellow) indicate **structural** landscape changes (e.g., construction, deforestation).")
                st.markdown("Darker colors indicate stability.")
                st.markdown('</div>', unsafe_allow_html=True)
