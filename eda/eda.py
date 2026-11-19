#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) Pipeline for Metro Manila Air Quality Datasets
Auto-installs required packages on first run if missing.

Supports:
  1. OpenAQ/               (Long-format station CSVs)
  2. QC Data Converted/    (Long-format station CSVs + QC Sensor Locations.csv)
  3. QC Data/              (Wide-format yearly CSVs 2021-2025 + QCDATA_SensorLoc.csv)

Usage:
  python eda_air_quality.py
  python eda_air_quality.py --data-dir "path/to/downloaded/folder" --out-dir "eda_outputs"
"""

import argparse
import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path


def ensure_dependencies() -> None:
    """Checks for required third-party packages and installs missing ones via pip."""
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tabulate": "tabulate",
    }
    missing = [
        pip_name
        for import_name, pip_name in required_packages.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        print(f"[SETUP] Installing missing dependencies: {', '.join(missing)}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
            print("[SETUP] All required packages installed successfully.\n")
        except subprocess.CalledProcessError as exc:
            print(
                f"[ERROR] Automatic installation failed ({exc}).\n"
                f"Please run manually: pip install {' '.join(missing)}"
            )
            sys.exit(1)


# Run dependency check before importing third-party data science libraries
ensure_dependencies()

import matplotlib
matplotlib.use("Agg")  # Headless-safe rendering for all OS/terminals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set consistent plot styling
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# WHO 2021 & Philippine DENR NAAQS Reference Thresholds (PM2.5 in ug/m3)
WHO_ANNUAL_GUIDELINE = 5.0
WHO_24HR_GUIDELINE = 15.0
PH_NAAQS_24HR_GUIDELINE = 35.0
EXTREME_SPIKE_THRESHOLD = 150.0

# Built-in coordinate lookup for OpenAQ sensors (since OpenAQ/ has no local metadata CSV)
OPENAQ_COORDS = {
    4797640: (14.56880, 121.01795),  # Kalayaan Avenue - South Avenue Intersection
    3370151: (14.58455, 121.01877),  # Mandaluyong City
    4797656: (14.37853, 121.04615),  # Barangay Tunasan - Maharlika Highway
    4797661: (14.58734, 121.04665),  # Old Wack Wack Road corner Shaw Boulevard
    4797648: (14.51769, 120.99638),  # Barangay Tambo - Quirino Avenue
    4963739: (14.59083, 121.07037),  # ASMPH
    6303937: (14.59871, 120.98947),  # San Sebastian Residences
    6303935: (14.61825, 120.97142),  # Near Puregold Tayuman
    4795498: (14.69158, 120.97246),  # Valenzuela CENRO (ARCA North)
    4797651: (14.60346, 121.04448),  # Ortigas Avenue-Roosevelt Intersection
    4526988: (14.63842, 121.07670),  # Near Katipunan Ave., Quezon City
    4795496: (14.66963, 120.94676),  # Gen. Luna St. corner Gov. Pascual Ave.
    3990426: (14.45627, 121.04569),  # Near Sucat Interchange/Exit
    4526987: (14.65744, 120.98362),  # Monumento, Caloocan
    4797652: (14.71405, 121.00088),  # 3S Center Bagbaguin
    4797637: (14.56746, 121.03929),  # J.P. Rizal Avenue corner Camia
    4795497: (14.69397, 121.00919),  # 3S Center Ugong
    6303934: (14.58385, 121.01645),  # Sta. Ana Hospital
    4797641: (14.56383, 121.05797),  # Unnamed (DNVBA1576 -> AL739RG7)
    6303936: (14.59064, 120.97112),  # Anda Circle
    4797658: (14.66115, 121.12476),  # Barangay Fortune - Santan Street
    3027756: (14.63525, 121.07784),  # NASA GSFC Rutgers Calib. N12
    4797653: (14.44481, 121.04586),  # Barangay Buli - Concepcion Street
    4797643: (14.54939, 121.07264),  # Sto. Rosario Bridge
    6340868: (14.67666, 121.09581),  # Tivoli
    4797642: (14.54848, 121.00053),  # Santa Clara - Pasay City General Hospital
    6338564: (14.56900, 120.98366),  # Rajah Sulayman Park
    4797644: (14.55157, 120.98649),  # World Trade Center
    4797638: (14.58655, 121.02549),  # Mandaluyong Elementary School
    5975180: (14.68856, 120.97071),  # Valenzuela
    3015285: (14.58853, 121.05822),  # Ortigas
    3990425: (14.57863, 121.05675),  # Along Shaw Blvd.
    4526989: (14.52917, 120.98988),  # Bradco Ave. cor Macapagal
    4795495: (14.63666, 120.96414),  # Barangay San Rafael Hall
    4795500: (14.67508, 120.93989),  # Barangay Tanza 2 Hall
    4795499: (14.66916, 120.97576),  # Malabon Action Center
    4797645: (14.56838, 121.02845),  # J.P. Rizal Avenue corner Pertierra
    4797646: (14.60399, 121.02781),  # San Juan Mini Park
    4797647: (14.63410, 121.09803),  # Marikina Sports Complex
    4797649: (14.52904, 121.00472),  # Aurora Boulevard (Tramo) - Andrews Avenue
    4797650: (14.54570, 121.06514),  # Pateros Bridge
    4797654: (14.45671, 121.03519),  # Barangay San Antonio - Paranaque-Sucat Road
    4797655: (14.55070, 121.02221),  # Makati Avenue - Antonio Arnaiz Avenue
    4797659: (14.64443, 120.98409),  # C3 Road (5th Avenue) - Rizal Avenue Ext.
    4797660: (14.71071, 120.99545),  # 3S Center Paso De Blas
    5976206: (14.43755, 121.00009),  # Las Pinas
}


def locate_data_root(user_path: str = None) -> Path:
    """Finds the root folder containing OpenAQ, QC Data Converted, or QC Data."""
    candidates = []
    if user_path:
        candidates.append(Path(user_path).expanduser().resolve())
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, script_dir])

    # Also check immediate subdirectories (e.g. if unzipped as 'Air Data (Old)/')
    for base in [cwd, script_dir]:
        if base.exists():
            for sub in base.iterdir():
                if sub.is_dir():
                    candidates.append(sub)

    target_subfolders = {"OpenAQ", "QC Data Converted", "QC Data"}
    for cand in candidates:
        if not cand.exists():
            continue
        existing = {p.name for p in cand.iterdir() if p.is_dir()}
        if existing.intersection(target_subfolders):
            return cand

    # Fallback if user placed script inside one of the subfolders directly
    if cwd.name in target_subfolders:
        return cwd.parent

    raise FileNotFoundError(
        "Could not locate 'OpenAQ', 'QC Data Converted', or 'QC Data' folders.\n"
        "Please place this script inside the downloaded folder or run with:\n"
        "  python eda_air_quality.py --data-dir \"/path/to/downloaded/folder\""
    )


def load_sensor_metadata(root: Path) -> pd.DataFrame:
    """Loads and merges QC Sensor Locations.csv, QCDATA_SensorLoc.csv, and OpenAQ coords."""
    meta_records = {}

    # 1. Load QC Sensor Locations.csv from 'QC Data Converted'
    qc_loc_csv = root / "QC Data Converted" / "QC Sensor Locations.csv"
    if qc_loc_csv.exists():
        df_loc = pd.read_csv(qc_loc_csv)
        df_loc.columns = [c.strip() for c in df_loc.columns]
        for _, row in df_loc.iterrows():
            loc_id = int(row["location_id"])
            meta_records[loc_id] = {
                "location_id": loc_id,
                "meta_name": str(row["name"]).strip(),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "district": "Unknown",
                "classification": "Unknown",
                "installation_date": None,
            }

    # 2. Enrich with District & Roadside Classification from 'QC Data/QCDATA_SensorLoc.csv'
    qc_raw_meta = root / "QC Data" / "QCDATA_SensorLoc.csv"
    if qc_raw_meta.exists():
        df_raw_meta = pd.read_csv(qc_raw_meta)
        df_raw_meta.columns = [c.strip() for c in df_raw_meta.columns]
        for _, row in df_raw_meta.iterrows():
            try:
                site_id = int(row["ID"])
                loc_id = 9000000 + site_id
            except (ValueError, KeyError):
                continue

            coords_str = str(row.get("Coordinates", "")).replace('"', "").strip()
            lat, lon = np.nan, np.nan
            if "," in coords_str:
                parts = coords_str.split(",")
                try:
                    lat, lon = float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass

            if loc_id not in meta_records:
                meta_records[loc_id] = {
                    "location_id": loc_id,
                    "meta_name": str(row.get("Site name", "")).strip(),
                    "latitude": lat,
                    "longitude": lon,
                }
            meta_records[loc_id]["district"] = f"District {str(row.get('District', '')).strip()}"
            meta_records[loc_id]["classification"] = str(row.get("Classification", "Unknown")).strip()
            meta_records[loc_id]["installation_date"] = str(row.get("Installation Date", "")).strip()

    # 3. Add OpenAQ built-in coordinates
    for loc_id, (lat, lon) in OPENAQ_COORDS.items():
        if loc_id not in meta_records:
            meta_records[loc_id] = {
                "location_id": loc_id,
                "meta_name": f"OpenAQ_{loc_id}",
                "latitude": lat,
                "longitude": lon,
                "district": "Metro Manila (OpenAQ)",
                "classification": "OpenAQ Node",
                "installation_date": None,
            }

    return pd.DataFrame(list(meta_records.values()))


def load_long_format_folder(folder: Path, source_label: str, file_prefix: str) -> pd.DataFrame:
    """Loads all station CSVs in long format from OpenAQ/ or QC Data Converted/."""
    if not folder.exists():
        print(f"[WARN] Folder not found, skipping: {folder}")
        return pd.DataFrame()

    csv_files = sorted(folder.glob(f"{file_prefix}*.csv"))
    if not csv_files:
        csv_files = [
            f for f in sorted(folder.glob("*.csv"))
            if "location" not in f.name.lower() and "sensorloc" not in f.name.lower()
        ]

    frames = []
    required_cols = {"location_name", "location_id", "local_timestamp", "pm25_value"}

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            df.columns = [c.strip() for c in df.columns]
            if not required_cols.issubset(set(df.columns)):
                continue
            df["source"] = source_label
            df["file_name"] = fpath.name
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed reading {fpath.name}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["location_name"] = combined["location_name"].astype(str).str.strip()
    combined["location_id"] = pd.to_numeric(combined["location_id"], errors="coerce").astype("Int64")
    combined["pm25_value"] = pd.to_numeric(combined["pm25_value"], errors="coerce")

    # Parse local_timestamp to naive Asia/Manila local time (UTC+8)
    ts_utc = pd.to_datetime(combined["local_timestamp"], errors="coerce", utc=True)
    combined["datetime_local"] = ts_utc.dt.tz_convert("Asia/Manila").dt.tz_localize(None)
    combined = combined.dropna(subset=["datetime_local"]).copy()

    # Deduplicate same station + timestamp if any exist
    combined = combined.sort_values("datetime_local").drop_duplicates(
        subset=["source", "location_id", "datetime_local"], keep="last"
    )
    return combined


def audit_raw_wide_qc_data(root: Path, out_dir: Path) -> pd.DataFrame:
    """Audits the raw wide-format AQ_Sensors_Hourly_YYYY.csv files in QC Data/."""
    qc_raw_dir = root / "QC Data"
    if not qc_raw_dir.exists():
        return pd.DataFrame()

    yearly_files = sorted(qc_raw_dir.glob("AQ_Sensors_Hourly_*.csv"))
    if not yearly_files:
        return pd.DataFrame()

    audit_rows = []
    for fpath in yearly_files:
        try:
            df_wide = pd.read_csv(fpath)
            unnamed_cols = [c for c in df_wide.columns if c.startswith("Unnamed") or c.strip() == ""]
            df_wide = df_wide.drop(columns=unnamed_cols)

            time_col = df_wide.columns[0]
            sensor_cols = list(df_wide.columns[1:])

            has_split_commonwealth = (
                "Commonwealth_BH" in sensor_cols and "Commonwealth_BHall" in sensor_cols
            )
            if has_split_commonwealth:
                df_wide["Commonwealth_BHall"] = df_wide["Commonwealth_BHall"].combine_first(
                    df_wide["Commonwealth_BH"]
                )
                df_wide = df_wide.drop(columns=["Commonwealth_BH"])
                sensor_cols = [c for c in sensor_cols if c != "Commonwealth_BH"]

            ts = pd.to_datetime(df_wide[time_col], errors="coerce")
            numeric_vals = df_wide[sensor_cols].apply(pd.to_numeric, errors="coerce")

            total_cells = numeric_vals.size
            non_null_cells = int(numeric_vals.notna().sum().sum())
            completeness_pct = (non_null_cells / total_cells * 100.0) if total_cells > 0 else 0.0

            audit_rows.append({
                "file_name": fpath.name,
                "rows_hours": len(df_wide),
                "active_station_columns": len(sensor_cols),
                "dropped_trailing_unnamed_cols": len(unnamed_cols),
                "merged_commonwealth_split": has_split_commonwealth,
                "start_timestamp": ts.min(),
                "end_timestamp": ts.max(),
                "non_null_measurements": non_null_cells,
                "overall_cell_completeness_pct": round(completeness_pct, 2),
                "mean_pm25": round(float(numeric_vals.mean().mean()), 2),
                "max_pm25": round(float(numeric_vals.max().max()), 2),
            })
        except Exception as exc:
            print(f"[WARN] Could not audit {fpath.name}: {exc}")

    df_audit = pd.DataFrame(audit_rows)
    if not df_audit.empty:
        df_audit.to_csv(out_dir / "04_raw_wide_qc_yearly_audit.csv", index=False)
    return df_audit


def compute_station_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive station-level statistics and data quality metrics."""
    records = []
    for (source, loc_id, loc_name), grp in df.groupby(
        ["source", "location_id", "location_name"], dropna=False
    ):
        s = grp["pm25_value"].dropna()
        if s.empty:
            continue

        t_min = grp["datetime_local"].min()
        t_max = grp["datetime_local"].max()
        expected_hours = max(1, int((t_max - t_min).total_seconds() // 3600) + 1)
        observed_valid = len(s)
        completeness_pct = min(100.0, (observed_valid / expected_hours) * 100.0)

        daily_mean = grp.set_index("datetime_local")["pm25_value"].resample("1D").mean().dropna()
        who_24hr_exceed_pct = (
            (daily_mean > WHO_24HR_GUIDELINE).mean() * 100.0 if len(daily_mean) > 0 else np.nan
        )
        naaqs_24hr_exceed_pct = (
            (daily_mean > PH_NAAQS_24HR_GUIDELINE).mean() * 100.0 if len(daily_mean) > 0 else np.nan
        )

        q25, q50, q75, q95, q99 = np.percentile(s, [25, 50, 75, 95, 99])

        records.append({
            "source": source,
            "location_id": loc_id,
            "location_name": loc_name,
            "district": grp["district"].iloc[0],
            "classification": grp["classification"].iloc[0],
            "latitude": grp["latitude"].iloc[0],
            "longitude": grp["longitude"].iloc[0],
            "start_date": t_min.strftime("%Y-%m-%d %H:%M"),
            "end_date": t_max.strftime("%Y-%m-%d %H:%M"),
            "valid_hours": observed_valid,
            "expected_hours_in_span": expected_hours,
            "completeness_pct": round(completeness_pct, 2),
            "missing_in_span_pct": round(100.0 - completeness_pct, 2),
            "zero_or_negative_count": int((s <= 0).sum()),
            "extreme_spike_gt150_count": int((s > EXTREME_SPIKE_THRESHOLD).sum()),
            "mean_pm25": round(float(s.mean()), 2),
            "std_pm25": round(float(s.std()), 2),
            "median_pm25": round(float(q50), 2),
            "iqr_pm25": round(float(q75 - q25), 2),
            "p95_pm25": round(float(q95), 2),
            "p99_pm25": round(float(q99), 2),
            "max_pm25": round(float(s.max()), 2),
            "days_observed": len(daily_mean),
            "who_24hr_exceedance_days_pct": round(float(who_24hr_exceed_pct), 2),
            "ph_naaqs_24hr_exceedance_days_pct": round(float(naaqs_24hr_exceed_pct), 2),
        })

    summary_df = pd.DataFrame(records).sort_values(
        ["source", "mean_pm25"], ascending=[True, False]
    )
    return summary_df


def generate_plots(df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Generates and saves the 6 diagnostic EDA figures."""

    # -------------------------------------------------------------------------
    # FIG 1: Monthly Data Completeness Heatmap (Station x Year-Month)
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 1: Monthly Data Completeness Heatmap...")
    df["year_month"] = df["datetime_local"].dt.to_period("M").astype(str)
    monthly_counts = (
        df.dropna(subset=["pm25_value"])
        .groupby(["source", "location_name", "year_month"])["pm25_value"]
        .count()
        .unstack(fill_value=0)
    )
    monthly_pct = (monthly_counts / 730.0 * 100.0).clip(upper=100.0)
    monthly_pct.index = [f"[{src}] {name}" for src, name in monthly_pct.index]

    fig, ax = plt.subplots(figsize=(16, max(10, len(monthly_pct) * 0.28)))
    sns.heatmap(
        monthly_pct,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Monthly Hourly Completeness (%)"},
        ax=ax,
    )
    ax.set_title("Monthly Data Availability (%) Across QC Data & OpenAQ Stations", fontsize=14, pad=12)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Sensor Station")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_dir / "fig01_data_completeness_heatmap.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIG 2: PM2.5 Distributions by Network, Roadside Classification & District
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 2: PM2.5 Distributions by Network, Classification & District...")
    df_valid = df.dropna(subset=["pm25_value"]).copy()
    df_valid = df_valid[df_valid["pm25_value"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        data=df_valid,
        x="classification",
        y="pm25_value",
        hue="classification",
        legend=False,
        palette="Set2",
        showfliers=False,
        ax=axes[0],
    )
    axes[0].axhline(WHO_24HR_GUIDELINE, color="orange", linestyle="--", label="WHO 24h (15 µg/m³)")
    axes[0].axhline(PH_NAAQS_24HR_GUIDELINE, color="red", linestyle="--", label="PH NAAQS 24h (35 µg/m³)")
    axes[0].set_title("Hourly PM2.5 by Site Classification (Outliers Hidden)")
    axes[0].set_xlabel("Site Classification")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].legend(loc="upper right")

    qc_only = df_valid[df_valid["source"] == "QC Data Converted"].copy()
    district_order = sorted(qc_only["district"].unique())
    sns.boxplot(
        data=qc_only,
        x="district",
        y="pm25_value",
        order=district_order,
        hue="classification",
        palette="Set1",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].axhline(WHO_24HR_GUIDELINE, color="orange", linestyle="--", label="WHO 24h (15 µg/m³)")
    axes[1].set_title("Quezon City PM2.5 Across Legislative Districts (I–VI)")
    axes[1].set_xlabel("Quezon City District")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Classification", loc="upper right")

    fig.suptitle("PM2.5 Concentration Distributions Across Sensor Groups", fontsize=14, y=1.02)
    fig.savefig(out_dir / "fig02_pm25_distributions_by_source_and_class.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIG 3: Diurnal (Hourly) and Seasonal (Monthly) Profiles
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 3: Diurnal and Seasonal Cycles...")
    df["hour"] = df["datetime_local"].dt.hour
    df["month"] = df["datetime_local"].dt.month
    df["is_nye_window"] = (
        ((df["datetime_local"].dt.month == 1) & (df["datetime_local"].dt.day == 1) & (df["datetime_local"].dt.hour <= 6))
        | ((df["datetime_local"].dt.month == 12) & (df["datetime_local"].dt.day == 31) & (df["datetime_local"].dt.hour >= 21))
    )

    df_no_nye = df[~df["is_nye_window"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    hourly_stats = (
        df_no_nye.groupby(["classification", "hour"])["pm25_value"]
        .agg(["mean", "median"])
        .reset_index()
    )
    sns.lineplot(
        data=hourly_stats,
        x="hour",
        y="mean",
        hue="classification",
        marker="o",
        linewidth=2.2,
        ax=axes[0],
    )
    axes[0].axhline(WHO_24HR_GUIDELINE, color="orange", linestyle="--", alpha=0.8, label="WHO 24h (15 µg/m³)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_title("Diurnal PM2.5 Cycle (Local Time PHT, Excl. NYE Fireworks)")
    axes[0].set_xlabel("Hour of Day (00:00 – 23:00 PHT)")
    axes[0].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[0].legend(title="Classification")

    monthly_all = df.groupby(["source", "month"])["pm25_value"].mean().reset_index()
    monthly_clean = df_no_nye.groupby(["source", "month"])["pm25_value"].mean().reset_index()
    monthly_all["subset"] = monthly_all["source"] + " (All Data)"
    monthly_clean["subset"] = monthly_clean["source"] + " (Excl. NYE)"
    monthly_comp = pd.concat([monthly_all, monthly_clean], ignore_index=True)

    sns.lineplot(
        data=monthly_comp,
        x="month",
        y="pm25_value",
        hue="subset",
        style="subset",
        markers=True,
        linewidth=2.2,
        ax=axes[1],
    )
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axes[1].set_title("Monthly Seasonal PM2.5 Profile (Impact of Jan 1 Fireworks)")
    axes[1].set_xlabel("Month of Year")
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[1].legend(title="Dataset & Filter")

    fig.savefig(out_dir / "fig03_diurnal_and_seasonal_profiles.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIG 4: Geospatial Scatter Map of Metro Manila Sensor Network
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 4: Geospatial Station Map...")
    geo_df = summary_df.dropna(subset=["latitude", "longitude"]).copy()
    geo_df = geo_df[
        (geo_df["latitude"].between(14.30, 14.85))
        & (geo_df["longitude"].between(120.85, 121.20))
    ]

    if not geo_df.empty:
        fig, ax = plt.subplots(figsize=(10, 11))
        scatter = ax.scatter(
            geo_df["longitude"],
            geo_df["latitude"],
            c=geo_df["mean_pm25"],
            s=geo_df["completeness_pct"] * 2.2 + 40,
            cmap="inferno_r",
            alpha=0.88,
            edgecolors="black",
            linewidth=0.8,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.75)
        cbar.set_label("Station Mean PM2.5 (µg/m³)")

        top_annotate = pd.concat([
            geo_df.nlargest(8, "mean_pm25"),
            geo_df.nsmallest(5, "mean_pm25"),
        ]).drop_duplicates(subset=["location_id"])

        for _, row in top_annotate.iterrows():
            short_name = str(row["location_name"])[:22]
            ax.annotate(
                f"{short_name} ({row['mean_pm25']:.1f})",
                (row["longitude"], row["latitude"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0.4),
            )

        ax.set_title(
            "Spatial Distribution of Mean PM2.5 Across Metro Manila\n"
            "(Bubble Size = Data Completeness %, Labels = Extreme High/Low Nodes)",
            fontsize=12,
        )
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        fig.savefig(out_dir / "fig04_spatial_pm25_map.png")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # FIG 5: Inter-Station Spatiotemporal Correlation Matrix (Top 25 Complete)
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 5: Inter-Station Correlation Matrix...")
    top_stations = summary_df.nlargest(25, "valid_hours")["location_name"].tolist()
    pivot_hourly = (
        df[df["location_name"].isin(top_stations)]
        .pivot_table(index="datetime_local", columns="location_name", values="pm25_value")
    )
    if pivot_hourly.shape[1] >= 2:
        corr_matrix = pivot_hourly.corr(min_periods=100)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws={"label": "Pearson Correlation (r)"},
            ax=ax,
        )
        ax.set_title("Hourly PM2.5 Inter-Station Correlation Matrix (Top 25 Stations by Coverage)", fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        fig.savefig(out_dir / "fig05_interstation_correlation_matrix.png")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # FIG 6: Multi-Year Rolling Time Series & New Year's Eve Fireworks Zoom
    # -------------------------------------------------------------------------
    print("  -> Generating Fig 6: Multi-Year Time Series & NYE Spike Zoom...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))

    daily_source = (
        df.set_index("datetime_local")
        .groupby("source")["pm25_value"]
        .resample("1D")
        .mean()
        .unstack(level=0)
    )
    rolling_7d = daily_source.rolling(7, min_periods=2).mean()
    for col in rolling_7d.columns:
        axes[0].plot(rolling_7d.index, rolling_7d[col], label=f"{col} (7-Day Rolling Mean)", linewidth=1.8)
    axes[0].axhline(WHO_24HR_GUIDELINE, color="orange", linestyle="--", label="WHO 24h (15 µg/m³)")
    axes[0].axhline(PH_NAAQS_24HR_GUIDELINE, color="red", linestyle="--", label="PH NAAQS 24h (35 µg/m³)")
    axes[0].set_title("Multi-Year 7-Day Rolling Average PM2.5 Concentration")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].legend(loc="upper right")

    nye_window = df[
        (df["datetime_local"] >= "2023-12-30") & (df["datetime_local"] <= "2024-01-03")
    ]
    if not nye_window.empty:
        nye_hourly = nye_window.groupby(["datetime_local", "classification"])["pm25_value"].mean().unstack()
        for col in nye_hourly.columns:
            axes[1].plot(nye_hourly.index, nye_hourly[col], marker=".", label=col)
        axes[1].set_title("Zoomed Anomaly View: New Year's Eve Fireworks Spike (Dec 30, 2023 – Jan 3, 2024)")
        axes[1].set_xlabel("Local Timestamp (PHT)")
        axes[1].set_ylabel("Mean Hourly PM2.5 (µg/m³)")
        axes[1].legend(title="Classification", loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig06_timeseries_and_nye_spikes.png")
    plt.close(fig)


def write_markdown_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wide_audit_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Writes a clean Markdown summary report of all findings and modeling recommendations."""
    total_rows = len(df)
    valid_rows = int(df["pm25_value"].notna().sum())
    total_stations = summary_df["location_id"].nunique()
    qc_stations = len(summary_df[summary_df["source"] == "QC Data Converted"])
    openaq_stations = len(summary_df[summary_df["source"] == "OpenAQ"])

    nye_mask = (
        ((df["datetime_local"].dt.month == 1) & (df["datetime_local"].dt.day == 1) & (df["datetime_local"].dt.hour <= 6))
        | ((df["datetime_local"].dt.month == 12) & (df["datetime_local"].dt.day == 31) & (df["datetime_local"].dt.hour >= 21))
    )
    mean_all = df["pm25_value"].mean()
    mean_no_nye = df.loc[~nye_mask, "pm25_value"].mean()
    max_nye = df.loc[nye_mask, "pm25_value"].max()

    top5_polluted = summary_df.nlargest(5, "mean_pm25")[
        ["source", "location_name", "classification", "mean_pm25", "p95_pm25", "completeness_pct"]
    ]
    top5_complete = summary_df.nlargest(5, "valid_hours")[
        ["source", "location_name", "start_date", "end_date", "valid_hours", "completeness_pct"]
    ]

    report_lines = [
        "# Exploratory Data Analysis (EDA) Report: Metro Manila PM2.5 Sensor Network",
        "",
        "## 1. Executive Dataset Summary",
        f"- **Total Combined Hourly Records Loaded:** {total_rows:,} ({valid_rows:,} valid numeric PM2.5 readings)",
        f"- **Total Unique Sensor Stations:** {total_stations} ({qc_stations} in `QC Data Converted`, {openaq_stations} in `OpenAQ`)",
        f"- **Overall Temporal Span:** `{df['datetime_local'].min()}` to `{df['datetime_local'].max()}` (Asia/Manila PHT)",
        f"- **Global Mean PM2.5 (All Data):** {mean_all:.2f} µg/m³",
        f"- **Global Mean PM2.5 (Excluding NYE Fireworks Window):** {mean_no_nye:.2f} µg/m³",
        f"- **Peak New Year's Eve Hourly Spike:** {max_nye:.2f} µg/m³",
        "",
        "## 2. Key Data Quality Findings & Pre-Modeling Recommendations",
        "1. **New Year's Eve Fireworks Extreme Outliers:**",
        "   - Across both `QC Data Converted` and `QC Data` (particularly Jan 1, 2024 00:00–03:00 PHT and Jan 1, 2025 00:00–04:00 PHT), stations record extreme spikes between **300 and 529.06 µg/m³** (e.g., Pearl Drive Footbridge, Jose Rizal High School, Old Balara Elementary School).",
        "   - *Recommendation:* Mask or winsorize `[Dec 31 21:00 – Jan 1 06:00]` before training CNN-LSTM or XGBoost models so firework anomalies do not dominate MSE/RMSE loss.",
        "2. **Zero and Near-Zero Readings in OpenAQ:**",
        "   - Certain OpenAQ nodes (such as `Valenzuela` location `5975180` and `Las Pinas` location `5976206`) contain stretches of exact `0.0` or `< 0.1` µg/m³ readings indicative of sensor floor/calibration dropouts.",
        "   - *Recommendation:* Filter out `pm25_value <= 0.5` µg/m³ prior to log-transformation or feature lag generation.",
        "3. **Staggered Sensor Deployment Cohorts (Quezon City):**",
        "   - Cohort 1 (8 sensors, IDs `9000001`–`9000008`): Deployed July–October 2021.",
        "   - Cohort 2 (12 sensors, IDs `9000009`–`9000020`): Deployed July–November 2022.",
        "   - Cohort 3 (20 sensors, IDs `9000021`–`9000040`): Deployed November 2023–March 2024.",
        "   - *Recommendation:* When constructing spatiotemporal graphs or cross-station lag features, group by deployment cohort or restrict joint multi-station training windows to **April 2024 – 2025** where all 40 QC sensors operate concurrently.",
        "4. **Raw Wide-Format Formatting Quirks (`QC Data/`):**",
        "   - `AQ_Sensors_Hourly_2025.csv` has a trailing comma on every line, producing an empty `Unnamed: 41` column.",
        "   - `AQ_Sensors_Hourly_2022.csv` and `2023.csv` split Commonwealth into `Commonwealth_BH` and `Commonwealth_BHall`.",
        "   - `QCDATA_SensorLoc.csv` has trailing spaces in column headers (`'ID '` and `'Site name '`).",
        "",
        "## 3. Top 5 Highest Mean PM2.5 Stations",
        top5_polluted.to_markdown(index=False),
        "",
        "## 4. Top 5 Longest-Running Stations by Valid Hours",
        top5_complete.to_markdown(index=False),
        "",
    ]

    if not wide_audit_df.empty:
        report_lines.extend([
            "## 5. Raw Wide-Format Yearly File Audit (`QC Data/`)",
            wide_audit_df.to_markdown(index=False),
            "",
        ])

    report_path = out_dir / "EDA_Report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run EDA on Metro Manila Air Quality Data.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the downloaded Google Drive folder (auto-detected if omitted).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="eda_outputs",
        help="Directory where CSV summaries, plots, and Markdown report will be saved.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(" METRO MANILA AIR QUALITY (QC DATA & OPENAQ) - AUTOMATED EDA PIPELINE")
    print("=" * 72)

    root = locate_data_root(args.data_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Located data directory : {root}")
    print(f"[INFO] Saving outputs to      : {out_dir}")

    # 1. Load Sensor Metadata
    print("\n[STEP 1/5] Loading sensor coordinates, districts, and roadside classifications...")
    meta_df = load_sensor_metadata(root)
    print(f"  -> Loaded metadata for {len(meta_df)} unique sensor locations.")

    # 2. Load Long-Format Time Series (QC Data Converted + OpenAQ)
    print("\n[STEP 2/5] Loading station time-series CSVs...")
    df_qc = load_long_format_folder(root / "QC Data Converted", "QC Data Converted", "qcdata_")
    print(f"  -> QC Data Converted : {len(df_qc):,} rows across {df_qc['location_id'].nunique() if not df_qc.empty else 0} stations")

    df_openaq = load_long_format_folder(root / "OpenAQ", "OpenAQ", "openaq_")
    print(f"  -> OpenAQ            : {len(df_openaq):,} rows across {df_openaq['location_id'].nunique() if not df_openaq.empty else 0} stations")

    df_all = pd.concat([df_qc, df_openaq], ignore_index=True)
    if df_all.empty:
        print("[ERROR] No time-series CSV files were found in 'QC Data Converted' or 'OpenAQ'.")
        sys.exit(1)

    # Merge metadata
    df_all = df_all.merge(meta_df, on="location_id", how="left")
    df_all["district"] = df_all["district"].fillna("Unknown")
    df_all["classification"] = df_all["classification"].fillna(
        pd.Series(
            np.where(df_all["source"] == "OpenAQ", "OpenAQ Node", "Unknown"),
            index=df_all.index,
        )
    )

    # 3. Compute Summary Tables & Anomaly Audits
    print("\n[STEP 3/5] Computing station summary statistics and data quality audits...")
    summary_df = compute_station_summary(df_all)
    summary_df.to_csv(out_dir / "01_station_summary_stats.csv", index=False)

    df_all["year"] = df_all["datetime_local"].dt.year
    yearly_completeness = (
        df_all.dropna(subset=["pm25_value"])
        .groupby(["source", "location_id", "location_name", "classification", "year"])["pm25_value"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    yearly_completeness.to_csv(out_dir / "02_temporal_completeness_by_year.csv", index=False)

    df_all["is_nye_fireworks_window"] = (
        ((df_all["datetime_local"].dt.month == 1) & (df_all["datetime_local"].dt.day == 1) & (df_all["datetime_local"].dt.hour <= 6))
        | ((df_all["datetime_local"].dt.month == 12) & (df_all["datetime_local"].dt.day == 31) & (df_all["datetime_local"].dt.hour >= 21))
    )
    outlier_audit = (
        df_all.groupby(["source", "location_id", "location_name", "classification"])
        .agg(
            total_valid_hours=("pm25_value", "count"),
            overall_mean_pm25=("pm25_value", "mean"),
            mean_excl_nye=("pm25_value", lambda s: s[~df_all.loc[s.index, "is_nye_fireworks_window"]].mean()),
            max_nye_pm25=("pm25_value", lambda s: s[df_all.loc[s.index, "is_nye_fireworks_window"]].max()),
            hours_gt_150=("pm25_value", lambda s: int((s > 150.0).sum())),
            hours_zero_or_neg=("pm25_value", lambda s: int((s <= 0.0).sum())),
        )
        .reset_index()
        .round(2)
    )
    outlier_audit.to_csv(out_dir / "03_outlier_and_nye_spike_audit.csv", index=False)

    wide_audit_df = audit_raw_wide_qc_data(root, out_dir)

    # 4. Generate Visualizations
    print("\n[STEP 4/5] Generating diagnostic plots...")
    generate_plots(df_all, summary_df, out_dir)

    # 5. Write Markdown Summary Report
    print("\n[STEP 5/5] Writing Markdown EDA summary report...")
    write_markdown_report(df_all, summary_df, wide_audit_df, out_dir)

    print("\n" + "=" * 72)
    print(" EDA COMPLETE! Generated files in:", out_dir)
    print("  - 01_station_summary_stats.csv")
    print("  - 02_temporal_completeness_by_year.csv")
    print("  - 03_outlier_and_nye_spike_audit.csv")
    print("  - 04_raw_wide_qc_yearly_audit.csv")
    print("  - fig01_data_completeness_heatmap.png")
    print("  - fig02_pm25_distributions_by_source_and_class.png")
    print("  - fig03_diurnal_and_seasonal_profiles.png")
    print("  - fig04_spatial_pm25_map.png")
    print("  - fig05_interstation_correlation_matrix.png")
    print("  - fig06_timeseries_and_nye_spikes.png")
    print("  - EDA_Report.md")
    print("=" * 72)


if __name__ == "__main__":
    main()