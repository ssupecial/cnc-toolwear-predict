import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

data_path = "dataverse_files/Dataset"
T_LIST = [i for i in range(1, 9)]
SENSOR_LIST = ["Accelerometer_Data", "Acoustic_Emission_Data", "Force_Data"]
EXP_LIST = [i for i in range(1, 13)]

# Create Feature DataFrame
columns = ["T", "Exp"]
for sensor in ["Acc_X", "Acc_Y", "Acc_Z", "AE", "Force_X", "Force_Y", "Force_Z"]:
    columns.append(sensor + "_Mean")
    columns.append(sensor + "_RMS")
    columns.append(sensor + "_Std")
    columns.append(sensor + "_SF")
    columns.append(sensor + "_Skewness")
    columns.append(sensor + "_Kurtosis")
    columns.append(sensor + "_Peak")
    columns.append(sensor + "_CF")
    columns.append(sensor + "_IF")
    columns.append(sensor + "_MSF")
    columns.append(sensor + "_MPS")
    columns.append(sensor + "_FC")
df = pd.DataFrame(columns=columns)


def feature_extraction(data):
    # 1) Mean Value
    mean_value = np.mean(np.abs(data))

    # 2) Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.square(data)))

    # 3) Standard Deviation
    std_dev = np.sqrt(np.mean(np.square(np.abs(data) - np.mean(data))))

    # 4) Shape Factor
    shape_factor = rms / mean_value

    # 5) Skewness
    skewness = np.mean(((np.abs(data) - mean_value) / std_dev) ** 3)

    # 6) Kurtosis
    kurt = np.mean(((np.abs(data) - mean_value) / std_dev) ** 4)

    # 7) Peak value
    peak_value = np.max(np.abs(data))

    # 8) Crest Factor
    crest_factor = peak_value / rms

    # 9) Impact Factor
    impact_factor = peak_value / mean_value

    # 10) Mean Square Frequency
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    mean_square_frequency = np.sum((freqs**2) * np.abs(fft_data) ** 2) / np.sum(
        np.abs(fft_data) ** 2
    )

    # 11) Mean of Power Spectrum
    power_spectrum = np.abs(fft_data) ** 2
    mean_power_spectrum = np.mean(power_spectrum)

    # 12) Frequency Centroid
    frequency_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

    return (
        mean_value,
        rms,
        std_dev,
        shape_factor,
        skewness,
        kurt,
        peak_value,
        crest_factor,
        impact_factor,
        mean_square_frequency,
        mean_power_spectrum,
        frequency_centroid,
    )


# Data check
for t in T_LIST:
    for exp in EXP_LIST:
        df_acc = pd.read_csv(data_path + f"/T{t}/Accelerometer_Data/Expt_{exp}.csv")
        df_ae = pd.read_csv(data_path + f"/T{t}/Acoustic_Emission_Data/Expt_{exp}.csv")
        df_force = pd.read_csv(data_path + f"/T{t}/Force_Data/Expt_{exp}.csv")

        if (
            (len(df_acc.columns) != 3)
            or (len(df_ae.columns) != 1)
            or (len(df_force.columns) != 3)
        ):
            print(f"Column의 개수가 맞지 않습니다. -> T{t}/Expt_{exp}")
            continue

        df_acc.columns = ["X", "Y", "Z"]
        df_ae.columns = ["AE"]
        df_force.columns = ["X", "Y", "Z"]
        columns = (
            df_acc.columns.tolist() + df_ae.columns.tolist() + df_force.columns.tolist()
        )

        expected_columns = set(["X", "Y", "Z", "AE"])

        if set(columns) != expected_columns:
            print(f"Column명이 잘못된 것 같습니다. -> T{t}/Expt_{exp}")
            print(columns)
            continue

# Feature Extraction
for t in T_LIST:
    for exp in EXP_LIST:
        df_acc = pd.read_csv(data_path + f"/T{t}/Accelerometer_Data/Expt_{exp}.csv")
        df_ae = pd.read_csv(data_path + f"/T{t}/Acoustic_Emission_Data/Expt_{exp}.csv")
        df_force = pd.read_csv(data_path + f"/T{t}/Force_Data/Expt_{exp}.csv")

        df_acc.columns = ["X", "Y", "Z"]
        df_ae.columns = ["AE"]
        df_force.columns = ["X", "Y", "Z"]

        acc_x = df_acc["X"]
        acc_y = df_acc["Y"]
        acc_z = df_acc["Z"]
        ae = df_ae["AE"]
        force_x = df_force["X"]
        force_y = df_force["Y"]
        force_z = df_force["Z"]

        acc_x_features = feature_extraction(acc_x)
        acc_y_features = feature_extraction(acc_y)
        acc_z_features = feature_extraction(acc_z)
        ae_features = feature_extraction(ae)
        force_x_features = feature_extraction(force_x)
        force_y_features = feature_extraction(force_y)
        force_z_features = feature_extraction(force_z)

        df.loc[len(df)] = (
            [t, exp]
            + list(acc_x_features)
            + list(acc_y_features)
            + list(acc_z_features)
            + list(ae_features)
            + list(force_x_features)
            + list(force_y_features)
            + list(force_z_features)
        )

# Add Tool Wear Data to DataFrame column(This is the target variable)
tool_values = []
for i in T_LIST:
    df_tool = pd.read_csv(
        data_path
        + f"/T{t}/Tool_Wear_Values/T{t}_Tool_wear_values_for_all_Experiments.csv"
    )
    # df_tool.info()
    tool_values.extend(df_tool.iloc[:, 1].values.tolist())

print(len(tool_values) == 96)
if len(tool_values) == 96:
    df["Tool_Wear"] = tool_values

print(df.head())
print(df.info())
df.to_csv("feature_extraction.csv", index=False)
