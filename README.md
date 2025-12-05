# 🌍 SpecTralNi30 – Real-time Satellite Analytics

SpecTralNi30 is a geospatial web application built with Streamlit and Google Earth Engine (GEE) for real-time analysis of Sentinel-1 (SAR) and Sentinel-2 (optical) imagery without writing code. It is designed for rapid vegetation, water, and land-change monitoring over any user-defined region of interest.

**Live app:** https://spectralni30.streamlit.app/

---

## ✨ Key Features

- **Multi-sensor support**  
  Switch seamlessly between Sentinel-2 (optical) and Sentinel-1 (SAR) workflows for robust monitoring under clear and cloudy conditions.

- **Spectral & radar indices**  
  Compute NDVI, GNDVI, NDWI (water), NDMI, and other band-math expressions such as $(B8 - B4) / (B8 + B4)$ directly from the UI.

- **Flexible ROI definition**  
  - Upload KML/KMZ  
  - Point + buffer (radius in km)  
  - Manual bounding box coordinates  

- **Time-series & compositing**  
  Analyze single dates or median composites over a user-defined date range to reduce noise and clouds.

- **Export & visualization**  
  - Generate GeoTIFF download URLs  
  - Export to Google Drive  
  - Create high-quality JPG maps with legends and scale bars for reports and presentations  

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/nitesh4004/SpecTralNi30.git
cd SpecTralNi30
```

### 2. Create environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure you have a valid Google Earth Engine account and have initialized the GEE Python API locally.

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the provided local URL in your browser to start using SpecTralNi30.

---

## 🧠 Application Workflow

1. **Select data source**  
   Choose Sentinel-2 or Sentinel-1 and configure band combinations or radar polarizations.

2. **Define ROI**  
   Upload KML/KMZ, click a point and set buffer distance, or provide a coordinate bounding box.

3. **Set temporal filters**  
   Provide a single date or start–end range to generate median composites and time-series products.

4. **Choose analysis**  
   - Spectral index computation  
   - Custom band math  
   - Time-series charts (where applicable)  

5. **Export & download**  
   - Get GeoTIFF URL  
   - Export to Drive  
   - Download rendered map as high-quality JPG with legend and scale bar  

---

## 📂 Project Structure

- `streamlit_app.py` – Main Streamlit app logic (UI + Earth Engine workflows).  
- `requirements.txt` – Python dependencies for local or cloud deployment.  
- `LULC_Sentinel2_Indian_Region_15000rows.csv` – Sample LULC dataset for experimentation.  
- `lulc_spectral_indices_1000.csv`, `lulc_spectral_indices_3000.csv` – Example spectral index datasets.  
- `sentinel2_lulc_synthetic.csv` – Synthetic Sentinel-2 LULC samples for testing and demos.  

---

## 🧪 Sample Use Cases

- Rapid vegetation health assessment using NDVI/GNDVI over agricultural fields.  
- Water body and flood-related analysis with NDWI and Sentinel-1 backscatter.  
- Land-use / land-cover exploration using pre-computed spectral features and sample CSVs.  

---

## 📦 Deployment Notes

- The app is currently deployed on Streamlit Community Cloud at `spectralni30.streamlit.app`.  
- For custom deployment (Docker, VM, or on-prem), ensure:  
  - Python environment matches `requirements.txt`  
  - GEE credentials are configured in the target environment  
  - Required environment variables (if any) are set correctly  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.  
To contribute:

1. Fork the repository  
2. Create a new branch: `git checkout -b feature/my-feature`  
3. Commit your changes and push the branch  
4. Open a pull request with a clear description  

---

## 📜 License

MIT License – See LICENSE file for details.

---

## 📬 Contact

**Author:** Nitesh Kumar  
**Role:** Geospatial Data Scientist  
**Email:** nitesh.kumar@swan.co.in  
**GitHub:** [@nitesh4004](https://github.com/nitesh4004)  

---


## 🎯 Comprehensive Features

### 🌍 **System Modes**

SpecTralNi30 provides four advanced analysis workflows:

#### 1. 🔍 **Spectral Monitor** (Vegetation Indices Analysis)

**Satellite Options:**
- Sentinel-2 (Optical) - 10m/20m resolution
- Sentinel-1 (SAR) - VV/VH polarization  
- Landsat-8/9 (Thermal & Optical)

**Spectral Indices:**
- NDVI, GNDVI, NDWI, NDMI, EVI, OSAVI
- Custom band-math expressions: (B8-B4)/(B8+B4)

**Capabilities:**
- Real-time satellite scene discovery
- Automatic cloud filtering (0-100% threshold)
- Median composite generation
- Dynamic value stretching (P2-P98)
- Statistical analysis (Mean, Min, Max, Std Dev)
- GeoTIFF export for GIS analysis

---

#### 2. 🏘️ **LULC Classifier** (Land Use/Land Cover)

**Pre-trained Models:**
- Google Dynamic World - FCN deep learning (10m global)

**Custom ML Models:**
- Artificial Neural Network (MLP)
- Random Forest (10-500 trees)
- Support Vector Machine (RBF, LINEAR, POLY)
- Gradient Tree Boost
- CART Decision Tree
- Naive Bayes

**Features:**
- 9-class LULC (Water, Trees, Grass, Crops, Built, Bare, Shrub, Flooded Veg, Snow/Ice)
- Automatic spectral index computation
- Area statistics per class (hectares)
- Train/validation split (50-90%)
- Overall accuracy & Kappa coefficient

---

#### 3. 🌐 **Geospatial Embeddings** (AI Foundation Model)

**Data Source:**
- Google Satellite Embeddings V1 (64 bands A00-A63)

**Use Cases:**
- LULC with ESA ground truth
- Water/Change detection (unsupervised)
- KMeans clustering on embeddings

---

#### 4. 🏔️ **Landslide Detection (SAR)**

**Sensor:** Sentinel-1 Radar (VV polarization)

**Method:**
- Backscatter change analysis
- DEM-based slope filtering
- Pre/Post event comparison

**Configuration:**
- Backscatter threshold: 1.0-5.0 dB
- Slope filter: 0-30 degrees
- Adjustable temporal windows

---

### 📍 **ROI Selection Methods**

1. **Upload KML/KMZ** - 200MB limit
2. **Point & Buffer** - Configurable radius (meters)
3. **Manual Coordinates** - Lat/Lon, bounding box

---

### 💾 **Export Formats**

- **GeoTIFF** - Direct download URLs
- **Google Drive** - Batch processing
- **JPG Maps** - Publication-ready cartography with legends & scale

---

### 🎨 **Visualization**

**Color Palettes:** Red-Yellow-Green, Blue-Green-Red, Viridis, Plasma, Terrain

**Interactive Mapping:**
- Leaflet.js interface
- Drawing tools (polygon, polyline, rectangle, circle)
- Layer management
- Full-screen mode
- Hybrid basemap

---

### 📊 **Live Testing Results** (Central India: 20.59°N, 78.96°E)

**NDVI Analysis:**
- Mean: 0.597 | Max: 0.875 | Min: -0.796 | Std: 0.162
- Scenes available: 10

**LULC Results:**
- Trees: 5,102.5 ha (53%) | Grass: 2,445 ha (25%)
- Crops: 2,006 ha (21%) | Built: 303 ha (3%)
- Water: 29 ha | Other: ~90 ha

---

### ✨ **Unique Capabilities**

✅ Multi-modal analysis (Optical + Radar)
✅ Pre-trained AI (no manual training)
✅ Hybrid ML execution
✅ Publication-ready maps
✅ Real-time processing
✅ Serverless architecture
✅ Global GEE coverage
✅ Code-free interface

---

### 🎯 **Use Cases**

🌾 Agricultural monitoring | 💧 Flood detection | 🏗️ Urban mapping
🌳 Forest change | 🏔️ Landslide hazard | 🛰️ Infrastructure planning
📡 Environmental assessment | 🔬 Climate research
### 🎯 Repository Topics
`geospatial` | `remote-sensing` | `sentinel-1` | `sentinel-2` | `earth-engine` | `streamlit` | `lulc` | `satellite-imagery`
