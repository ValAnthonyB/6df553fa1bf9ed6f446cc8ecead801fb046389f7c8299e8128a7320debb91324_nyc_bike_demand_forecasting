from pathlib import Path
import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger

def detect_drift(reference_data_path: str, 
                 current_data_path: str,
                 threshold: float = 0.05
                 ):
    
    # Read the reference and drifted dataset
    reference_df = pd.read_csv(reference_data_path).drop(columns=["ride_date", "t+7d"])
    current_df = pd.read_csv(current_data_path).drop(columns=["ride_date", "t+7d"])

    # Run Evidently report
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_df, current_data=current_df)
    metrics_dict = json.loads(result.json())

    # Extract feature drift scores
    feature_drifts = {
        m["metric_id"].split("(")[1].split(")")[0].replace("column=", ""): m["value"]
        for m in metrics_dict["metrics"]
        if m["metric_id"].startswith("ValueDrift")
    }

    # Build drift summary
    drift_summary = {
        "drift_detected": any(val <= threshold for val in feature_drifts.values()),
        "feature_drifts": feature_drifts, # p-values
        "overall_drift_score": sum(feature_drifts.values()) / len(feature_drifts),
    }

    # Save report
    report_path = Path("reports") / "drift_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(drift_summary, indent=4))

    logger.info(f"Drift report saved to {report_path.resolve()}")

    return drift_summary