import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


#initial report
data = pd.read_pickle(r"C:\Users\mathi\OneDrive\Desktop\P5_OCR\data\data.pkl")

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=data.iloc[:60], reference_data=data.iloc[60:], column_mapping=None)
data_drift_report

data_drift_report.save_html("data_drif_report.html")

#first section report on 2022 datas
data_1 = pd.read_pickle(r"C:\Users\mathi\OneDrive\Desktop\P5_OCR\data\data_report_drift.pkl")

data_drift_report_1 = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report_1.run(current_data=data_1.iloc[:60], reference_data=data.iloc[:60], column_mapping=None)
data_drift_report_1

data_drift_report_1.save_html("data_drif_report_1.html")

#second report on 2022 datas, second section
data_drift_report_2 = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report_2.run(current_data=data_1.iloc[1500:1560], reference_data=data.iloc[:60], column_mapping=None)
data_drift_report_2

data_drift_report_2.save_html("data_drif_report_2.html")

#third report on 2022 datas, third section
data_drift_report_3 = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report_3.run(current_data=data_1.iloc[:60], reference_data=data.iloc[:60], column_mapping=None)
data_drift_report_3

data_drift_report_3.save_html("data_drif_report_2.html")