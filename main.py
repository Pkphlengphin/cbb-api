from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- 1. เพิ่ม import นี้
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import os
from typing import List

app = FastAPI(title="CBB Prediction API (Hybrid Model)")

# =========================================================================
# 🔴 CORS SETUP (ส่วนที่เพิ่มเข้ามาแก้ปัญหา Failed to fetch)
# =========================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้ทุกเว็บ (รวมถึง localhost และ vercel) เข้าถึงได้
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Method (GET, POST, etc.)
    allow_headers=["*"],  # อนุญาตทุก Header
)
# =========================================================================


# =========================================================================
# 1. ⚠️ FEATURE LISTS (CRUCIAL: MUST MATCH TRAINED MODELS)
# =========================================================================

MODEL_FEATURES_ADULT = [
    'Site',
    'Ripeness',
    'Average of Temperature (°C)',
    'Average of Humidity %',
    'Sum of Rain (mm)',
    'Total_Adult_Stock',
    'Flight_Activity',
    'Altitude_Risk',
    'Target_Mean_7d',
    'Target_Mean_14d',
]

MODEL_FEATURES_PUPAE_ALIVE = [
    'Site',
    'Ripeness',
    'Average of Temperature (°C)',
    'Average of Humidity %',
    'Sum of Rain (mm)',
    'Larvae_To_Pupae',
    'Survival_Rate',
    'Pupae_Potential',
    'Altitude_Risk',
    'Target_Mean_7d',
    'Target_Mean_14d',
]

MODEL_FEATURES_LARVAE_ALIVE = [
    'Site',
    'Ripeness',
    'Average of Temperature (°C)',
    'Average of Humidity %',
    'Sum of Rain (mm)',
    'Larvae_Potential',
    'Altitude_Risk',
    'Target_Mean_7d',
    'Target_Mean_14d',
]

MODEL_FEATURES_EGGS = [
    'Target_Mean_14d',
    'Target_Mean_7d',
    'Ripeness',
    'Average of Humidity %',
    'Rain_Roll14',
    'Average of Temperature (°C)',
    'Parent_x_Risk',
    'Sum of Rain (mm)',
    'Parent_Mean_7d',
    'Alt_x_Temp',
    'Humid_Optimal_Days',
    'Site',
]
# -------------------------------------------------------------------------

# 2. โหลดโมเดล
models = {}
TARGETS = ["Adult", "Pupae Alive", "Larvae Alive", "Eggs"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("⏳ Loading models...")
for target in TARGETS:
    safe_name = target.replace(" ", "_")
    try:
        # พยายามโหลดไฟล์ .cbm (CatBoost)
        path = os.path.join(BASE_DIR, f"cbb_model_{safe_name}.cbm")
        
        # เช็คเผื่อกรณีไฟล์ชื่อ .pkl (ถ้ามี)
        if not os.path.exists(path):
             path_pkl = os.path.join(BASE_DIR, f"cbb_model_{safe_name}.pkl")
             if os.path.exists(path_pkl):
                 path = path_pkl

        if os.path.exists(path):
            m = CatBoostRegressor()
            m.load_model(path)
            models[target] = m
            print(f"   ✅ Loaded: {target}")
        else:
            print(f"   ⚠️ Not found: {path}")
    except Exception as e:
        print(f"   ❌ Error {target}: {e}")

# 3. กำหนด Input
class InsectInput(BaseModel):
    # Raw Weather & Host (Current Day Data)
    Site: float = Field(..., description="ความสูง (Altitude) ของ Site ปัจจุบัน")
    Ripeness: str = Field(..., description="ความสุกของผลกาแฟ (เช่น R, G, Unknown) ณ วันนี้")
    Temperature: float = Field(..., description="Average Temperature (°C) ณ วันนี้")
    Humidity: float = Field(..., description="Average Humidity (%) ณ วันนี้")
    Rain: float = Field(..., description="Sum of Rain (mm) ณ วันนี้")
    
    # Required Lag Features
    Parent_Mean_7d: float = Field(..., description="ค่าเฉลี่ย Log(Parent + 1) ของ Target จาก 7 วันก่อน")
    Target_Mean_7d: float = Field(..., description="ค่าเฉลี่ย Log(Target + 1) ของ Target จาก 7 วันก่อน")
    Target_Mean_14d: float = Field(..., description="ค่าเฉลี่ย Log(Target + 1) ของ Target จาก 14 วันก่อน")
    
    # History Data for Rolling Window Calculation
    History_Rain: List[float] = Field(..., description="Rain (mm) ย้อนหลัง")
    History_Humidity: List[float] = Field(..., description="Humidity (%) ย้อนหลัง")
    History_Larvae_Log: List[float] = Field(..., description="Larvae Alive Log(N+1) ย้อนหลัง")
    History_Pupae_Log: List[float] = Field(..., description="Pupae Alive Log(N+1) ย้อนหลัง")
    History_Adult_Log: List[float] = Field(..., description="Adult Log(N+1) ย้อนหลัง")


# -------------------------------------------------------------------------
# --- HELPER 1: คำนวณ FEATURES ที่ต้องใช้ ROLLING WINDOW
# -------------------------------------------------------------------------

def calculate_rolling_features(current_R, current_H, hist_R, hist_H):
    """Calculates all necessary Rolling Window Features"""
    
    hist_R_full = hist_R[::-1] + [current_R] 
    hist_H_full = hist_H[::-1] + [current_H]

    series_rain = pd.Series(hist_R_full)
    series_humid = pd.Series(hist_H_full)
    
    # 1. Humidity Features
    if len(series_humid) < 8:
        H_window_7 = pd.Series([0.0] * 7)
    else:
        H_window_7 = series_humid.iloc[-8:-1] 

    Humid_Optimal_Days = ((H_window_7 >= 90) & (H_window_7 <= 95)).sum()
    Humid_Stress_Days = (H_window_7[-3:] < 50).sum()
    Too_Wet_Days = (H_window_7[-3:] >= 98).sum()

    # 2. Rain Features
    if len(series_rain) < 15:
        R_roll_window = pd.Series([0.0] * 14)
    else:
        R_roll_window = series_rain.iloc[-15:-1]
    Rain_Roll14 = R_roll_window.sum()
    
    if len(series_rain) >= 5:
        rain_yesterday = series_rain.iloc[-2]
        rain_prev_3d = series_rain.iloc[-5:-2].sum()
        Rain_Trigger = int((rain_yesterday > 2) and (rain_prev_3d < 1))
    else:
        Rain_Trigger = 0 
    
    return {
        'Humid_Optimal_Days': int(Humid_Optimal_Days),
        'Humid_Stress_Days': int(Humid_Stress_Days),
        'Too_Wet_Days': int(Too_Wet_Days),
        'Rain_Roll14': float(Rain_Roll14),
        'Rain_Trigger': Rain_Trigger
    }


# -------------------------------------------------------------------------
# --- HELPER 2: คำนวณ Features ทั้งหมด
# -------------------------------------------------------------------------

def calculate_all_features(data: InsectInput, target_name: str):
    
    # Raw Inputs
    T, H, R, Site = data.Temperature, data.Humidity, data.Rain, data.Site
    Parent_Mean_7d = data.Parent_Mean_7d
    
    # 1. Calculate Environmental Rolling Features
    rolling_features = calculate_rolling_features(R, H, data.History_Rain, data.History_Humidity)
    
    # 2. Calculate Derived Features
    Altitude_Risk = np.clip((1600 - Site) / (1600 - 800), 0, 1)
    Alt_x_Temp = Altitude_Risk * T
    
    Fungal_Pressure = 0 

    # 3. Stage Transition Features
    
    # Larvae To Pupae
    Larvae_To_Pupae = 0
    if target_name in ['Pupae Alive', 'Adult']:
        Larvae_hist = pd.Series(data.History_Larvae_Log[::-1])
        if len(Larvae_hist) >= 24:
            Larvae_To_Pupae = Larvae_hist.iloc[-24:-14].mean() 
        else: Larvae_To_Pupae = 0.0

    # Total Adult Stock & Flight
    Total_Adult_Stock = 0
    Flight_Activity = 0
    if target_name == 'Adult':
        Adult_hist = pd.Series(data.History_Adult_Log[::-1])
        Pupae_hist = pd.Series(data.History_Pupae_Log[::-1])
        
        if len(Pupae_hist) >= 14:
            New_Born = Pupae_hist.iloc[-14:-7].mean()
        else: New_Born = 0.0

        if len(Adult_hist) >= 61:
            Old_Population = Adult_hist.iloc[-61:-1].mean()
        else: Old_Population = 0.0

        Total_Adult_Stock = New_Born + (Old_Population * 0.8)
        Flight_Activity = Total_Adult_Stock * rolling_features['Rain_Trigger']

    # Pupae Potential & Survival
    Pupae_Potential = 0
    Survival_Rate = 1.0
    if target_name == 'Pupae Alive':
        Rot_Risk = (rolling_features['Humid_Optimal_Days'] >= 3) or (rolling_features['Rain_Roll14'] > 150)
        Survival_Rate = 0.2 if Rot_Risk else 1.0
        Pupae_Potential = Larvae_To_Pupae * Survival_Rate
    
    # Larvae Potential
    Larvae_Potential = 0
    if target_name == 'Larvae Alive':
        Larvae_Potential = Parent_Mean_7d * 5

    # Egg Survival Potential
    Egg_Survival_Potential = 0
    if target_name == 'Eggs':
        Egg_Survival_Potential = Parent_Mean_7d / (1 + Fungal_Pressure)

    # --- 4. Construct Row Data ---
    row_data = {
        'Site': Site,
        'Ripeness': str(data.Ripeness),
        'Average of Temperature (°C)': T,
        'Average of Humidity %': H,
        'Sum of Rain (mm)': R,
        
        'Altitude_Risk': Altitude_Risk,
        'Alt_x_Temp': Alt_x_Temp,
        'Humid_Optimal_Days': rolling_features['Humid_Optimal_Days'],
        'Humid_Stress_Days': rolling_features['Humid_Stress_Days'],
        'Too_Wet_Days': rolling_features['Too_Wet_Days'],
        'Rain_Trigger': rolling_features['Rain_Trigger'],
        'Rain_Roll14': rolling_features['Rain_Roll14'],
        
        'Parent_Mean_7d': Parent_Mean_7d,
        'Target_Mean_7d': data.Target_Mean_7d,
        'Target_Mean_14d': data.Target_Mean_14d,
        
        'Larvae_To_Pupae': Larvae_To_Pupae,
        'Survival_Rate': Survival_Rate,
        'Pupae_Potential': Pupae_Potential,
        'Larvae_Potential': Larvae_Potential,
        'Total_Adult_Stock': Total_Adult_Stock,
        'Flight_Activity': Flight_Activity,
        'Egg_Survival_Potential': Egg_Survival_Potential,
        
        'Parent_x_Trigger': Parent_Mean_7d * rolling_features['Rain_Trigger'],
        'Parent_x_Risk': Parent_Mean_7d * Altitude_Risk,
    }
    
    return row_data


def get_feature_list(target_name):
    """Returns the correct feature list based on the target name."""
    if target_name == 'Adult':
        return MODEL_FEATURES_ADULT
    elif target_name == 'Pupae Alive':
        return MODEL_FEATURES_PUPAE_ALIVE
    elif target_name == 'Larvae Alive':
        return MODEL_FEATURES_LARVAE_ALIVE
    elif target_name == 'Eggs':
        return MODEL_FEATURES_EGGS
    return [] 

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CBB Prediction API is running!"}

@app.post("/predict/{target_name}")
def predict(target_name: str, data: InsectInput):
    if target_name not in models:
        raise HTTPException(404, detail="Model not found. Available: Adult, Pupae Alive, Larvae Alive, Eggs.")
    
    try:
        # 1. คำนวณ Features ทั้งหมด
        row_data = calculate_all_features(data, target_name)

        # 2. ดึง List Feature ที่ถูกต้อง
        current_model_features = models[target_name].feature_names_
        if not current_model_features or len(current_model_features) == 0:
             current_model_features = get_feature_list(target_name)

        # 3. สร้าง DataFrame
        df = pd.DataFrame([row_data])
        
        # 4. Filter and reorder columns
        for col in current_model_features:
             if col not in df.columns:
                 df[col] = 0.0 
        
        df = df[current_model_features] 
        
        # 5. ทำนาย
        pred_log = models[target_name].predict(df)
        pred_count = max(0, float(np.expm1(pred_log[0])))
        
        return {
            "target": target_name,
            "prediction": round(pred_count, 2),
            "note": "Prediction successful."
        }

    except Exception as e:
        print(f"Error processing {target_name}: {e}") # Log error ใน console ของ Render
        raise HTTPException(500, detail=f"Prediction Error: {str(e)}")

# (Optional) For Local Testing without command line
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)