import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

data = pd.read_pickle(r"C:\Users\mathi\OneDrive\Desktop\P5_OCR\data\data.pkl")

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=data.iloc[:60], reference_data=data.iloc[60:], column_mapping=None)
data_drift_report

data_drift_report.save_html("data_drif_report.html")