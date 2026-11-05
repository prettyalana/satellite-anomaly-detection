import pandas as pd
import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt

start_time = time.perf_counter()

# Captures all csv files in the Support Files directory
csv_files = glob.glob("Support Files/*.csv")

# Combines all csv files except beta into one pandas table
li = []

for file in csv_files:
    if Path(file).stem == "beta_sun_deg":
        continue
    df = pd.read_csv(file, index_col=None, header=0)
    df["satellite"] = Path(file).stem
    li.append(df)

new_table = pd.concat(li, axis=0, ignore_index=True)

# Converts timestamp to use datetime
new_table["timestamp"] = pd.to_datetime(new_table["timestamp"], unit="ms")

# Creates a separate beta table 
beta_csv = pd.read_csv("Support Files/beta_sun_deg.csv", index_col=0)

# Converts index to dates 
beta_csv.index = pd.to_datetime(beta_csv.index, format="%m/%d/%Y")


# Plots 

Path("plots").mkdir(exist_ok=True)

# beta plot 
beta_csv.plot(figsize=(12,6), style=["-","--",":","-.","-","--",":"])
plt.title("Beta Sun Angle by Satellite")
plt.xlabel("Dates")
plt.ylabel("Beta Sun Angle Degrees")
plt.savefig("plots/beta_sun_angles.png")
plt.close() 


# Plot per satellite including all telemetry channels

# Collapse all satellite names into 7 distinct satellite names
satellite_names = new_table["satellite"].unique()

for name in satellite_names:
    # Keep only this satellite's rows
    belongs_to_satellite = new_table["satellite"] == name
    satellite_data = new_table[belongs_to_satellite]

    # 4 stacked panels sharing the same time axis
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))

    # Panel 1: gyros
    gyros = satellite_data[["timestamp", "gyro_x", "gyro_y", "gyro_z"]].dropna()
    gyros.plot(x="timestamp", ax=axes[0])
    axes[0].set_ylabel("Gyro rate (deg/s)")

    # Panel 2: wheels
    wheels = satellite_data[["timestamp", "wheel_s", "wheel_x", "wheel_y", "wheel_z"]].dropna()
    wheels.plot(x="timestamp", ax=axes[1])
    axes[1].set_ylabel("Wheel speed (rpm)")

    # Panel 3: bus voltage
    voltage = satellite_data[["timestamp", "bus_voltage"]].dropna()
    voltage.plot(x="timestamp", ax=axes[2])
    axes[2].set_ylabel("Bus voltage (V)")

    # Panel 4: low voltage flag
    flag = satellite_data[["timestamp", "low_voltage"]].dropna()
    flag.plot(x="timestamp", ax=axes[3])
    axes[3].set_ylabel("Low voltage flag")

    axes[3].set_xlabel("Time (UTC)")
    fig.suptitle(f"{name} telemetry")
    plt.savefig(f"plots/{name}_telemetry.png")
    plt.close()

# Plot per telemetry channel including all satellites
channels = {
    "gyro_x": "Gyro rate (deg/s)",
    "gyro_y": "Gyro rate (deg/s)",
    "gyro_z": "Gyro rate (deg/s)",
    "wheel_s": "Wheel speed (rpm)",
    "wheel_x": "Wheel speed (rpm)",
    "wheel_y": "Wheel speed (rpm)",
    "wheel_z": "Wheel speed (rpm)",
    "bus_voltage": "Bus voltage (V)",
    "low_voltage": "Low voltage flag",
}

for channel, y_label in channels.items():
    plt.figure(figsize=(12, 6))

    for name, satellite_data in new_table.groupby("satellite"):
        readings = satellite_data[["timestamp", channel]].dropna()
        plt.plot(readings["timestamp"], readings[channel], label=name, linewidth=0.8, alpha=0.7)

    # Reference lines for the limits given in the instructions
    if channel.startswith("wheel"):
        plt.axhline(8000, color="black", linestyle="--", label="Saturation (8000 rpm)")
    if channel.startswith("gyro"):
        plt.axhline(2, color="black", linestyle="--", label="+/- 2 deg/s")
        plt.axhline(-2, color="black", linestyle="--")

    plt.title(f"{channel} by satellite")
    plt.xlabel("Time (UTC)")
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    plt.savefig(f"plots/{channel}_all_satellites.png")
    plt.close()

# Percentage of available telemetry points with each condition
wheel_cols = ["wheel_s", "wheel_x", "wheel_y", "wheel_z"]
gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

results = []

for name in satellite_names:
    belongs_to_satellite = new_table["satellite"] == name
    satellite_data = new_table[belongs_to_satellite]

    # One or more wheels saturated (|speed| >= 8000 rpm)
    wheels = satellite_data[wheel_cols]
    has_wheel_reading = wheels.notna().any(axis=1)
    any_wheel_saturated = (wheels.abs() >= 8000).any(axis=1)
    pct_wheels = any_wheel_saturated[has_wheel_reading].mean() * 100

    # Low voltage flag active
    flag = satellite_data["low_voltage"].dropna()
    pct_low_voltage = (flag == 1).mean() * 100

    # One or more gyro rates above 2 deg/s (in either direction)
    gyros = satellite_data[gyro_cols]
    has_gyro_reading = gyros.notna().any(axis=1)
    any_gyro_fast = (gyros.abs() > 2).any(axis=1)
    pct_gyros = any_gyro_fast[has_gyro_reading].mean() * 100

    results.append({
        "satellite": name,
        "wheel_saturated_%": pct_wheels,
        "low_voltage_%": pct_low_voltage,
        "gyro_above_2_%": pct_gyros,
    })

percentages = pd.DataFrame(results).set_index("satellite").sort_index().round(2)
print(percentages)
percentages.to_csv("percentages.csv")

# Anomaly detection functions
def is_stuck_at_zero(channel):
    """True if a channel reads 0 for every measurement (dead wheel or dead sensor)."""
    readings = channel.dropna()
    if readings.empty:
        return False
    return bool((readings.abs() < 1).all())


def has_low_voltage_events(channel):
    """True if the low voltage flag was ever active. In nominal operations it should
    never trigger, since it shuts off hardware like the camera."""
    readings = channel.dropna()
    return bool((readings == 1).any())


def is_spinning_out_of_control(channel, limit=2, minutes=30):
    """True if a gyro rate stays above the limit for a sustained period.
    Normal turns cause brief spikes; a rate above the limit for many minutes
    in a row means the satellite is rotating and not stopping."""
    readings = channel.dropna().sort_index()
    if readings.empty:
        return False
    per_minute = readings.abs().resample("1min").median()
    above_limit = (per_minute > limit).astype(int)
    sustained = above_limit.rolling(minutes).min()
    return bool((sustained == 1).any())


# Test every function on every satellite to confirm there are no false positives
checks = []

for name in satellite_names:
    belongs_to_satellite = new_table["satellite"] == name
    satellite_data = new_table[belongs_to_satellite].set_index("timestamp").sort_index()

    # Find which wheels are stuck at zero
    dead_wheels = []
    for wheel in wheel_cols:
        if is_stuck_at_zero(satellite_data[wheel]):
            dead_wheels.append(wheel)

    # Find which gyros are spinning out of control
    spinning_gyros = []
    for gyro in gyro_cols:
        if is_spinning_out_of_control(satellite_data[gyro]):
            spinning_gyros.append(gyro)

    # Turn each list into readable text for the report
    if dead_wheels:
        dead_wheels_text = ", ".join(dead_wheels)
    else:
        dead_wheels_text = "none"

    if spinning_gyros:
        spinning_gyros_text = ", ".join(spinning_gyros)
    else:
        spinning_gyros_text = "none"

    checks.append({
        "satellite": name,
        "dead_wheels": dead_wheels_text,
        "spinning_gyros": spinning_gyros_text,
        "low_voltage_events": has_low_voltage_events(satellite_data["low_voltage"]),
        "voltage_sensor_reads_zero": is_stuck_at_zero(satellite_data["bus_voltage"]),
    })

anomaly_report = pd.DataFrame(checks).set_index("satellite").sort_index()
print(anomaly_report.to_string())

# Run Time Timer 
end_time = time.perf_counter()

final_run_time = end_time - start_time

print(f"Total run time: {final_run_time:.2f} seconds")