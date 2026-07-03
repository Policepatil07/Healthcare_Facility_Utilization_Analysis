import pandas as pd

def load_data():
    df = pd.read_csv("healthcare_facility_utilization.csv")

    df["Admission_Date"] = pd.to_datetime(df["Admission_Date"])
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"])

    df["Bed_Utilization_%"] = (df["Occupied_Beds"] / df["Total_Beds"]) * 100

    return df