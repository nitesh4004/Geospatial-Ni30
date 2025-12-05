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

### 🎯 Repository Topics
`geospatial` | `remote-sensing` | `sentinel-1` | `sentinel-2` | `earth-engine` | `streamlit` | `lulc` | `satellite-imagery`
