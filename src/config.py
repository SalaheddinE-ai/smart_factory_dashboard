"""Static configuration: logical column names, aliases and sensor grouping rules.

The dashboard never assumes the CSV uses these exact names. Each logical name is
resolved against the real header (exact -> normalised -> alias) and can be
overridden manually in the sidebar "Column mapping" panel.
"""

KEY_ALIASES = {
    "machine_id": ["machine_id", "machineid", "asset_id", "equipment_id", "device_id"],
    "timestamp": ["timestamp", "datetime", "date_time", "recorded_at", "measurement_time", "date"],
    "factory_id": ["factory_id", "factory", "plant_id", "plant", "site_id"],
    "production_line": ["production_line", "line_id", "prod_line", "line"],
    "machine_type": ["machine_type", "equipment_type", "asset_type"],
    "manufacturer": ["manufacturer", "vendor", "brand", "oem"],
    "machine_failure": ["machine_failure", "failure", "failed", "is_failure", "target"],
    "failure_type": ["failure_type", "failure_mode", "failure_category"],
    "maintenance_priority": ["maintenance_priority", "priority", "maint_priority"],
    "maintenance_type": ["maintenance_type", "last_maintenance_type", "maint_type"],
    "machine_health_score": ["machine_health_score", "health_score", "machine_health", "overall_health"],
    "remaining_useful_life_hours": ["remaining_useful_life_hours", "rul_hours", "remaining_useful_life", "rul"],
    "failure_probability": ["failure_probability", "predicted_failure_probability", "failure_prob", "prob_failure"],
    "machine_age_years": ["machine_age_years", "machine_age", "age_years", "equipment_age"],
    "days_since_maintenance": ["days_since_maintenance", "days_since_last_maintenance"],
    "maintenance_count": ["maintenance_count", "num_maintenance", "maintenance_events"],
    "maintenance_cost": ["maintenance_cost", "maintenance_cost_usd", "repair_cost"],
    "last_repair_duration_hours": ["last_repair_duration_hours", "repair_duration_hours", "last_repair_duration"],
    "replacement_parts_count": ["replacement_parts_count", "parts_replaced", "replacement_parts"],
    "technician_experience_years": ["technician_experience_years", "technician_experience"],
    "previous_failure_count": ["previous_failure_count", "prior_failures", "past_failure_count"],
    "days_since_previous_failure": ["days_since_previous_failure", "days_since_last_failure"],
    "failure_frequency": ["failure_frequency", "failure_rate_history"],
    "downtime_hours": ["downtime_hours", "downtime", "downtime_minutes", "downtime_mins"],
    "shift": ["shift", "factory_shift", "work_shift"],
    "operational_load": ["operational_load", "load_percentage", "load_pct", "machine_load", "load"],
    "energy_consumption": ["energy_consumption_kwh", "energy_consumption", "power_consumption", "power_kw", "energy_kwh"],
}

# Categorical columns converted to pandas 'category' (labels are never altered)
CAT_KEYS = ["machine_id", "factory_id", "production_line", "machine_type", "manufacturer",
            "failure_type", "maintenance_priority", "maintenance_type", "shift"]

# Normalised failure_type labels that mean "no failure"
NO_FAILURE = {"none", "nofailure", "normal", "ok", "nofault", "nodefect", "healthy", "0", "nan", "nonefailure"}

PRIORITY_ORDER = ["low", "medium", "high", "critical"]

# First matching group wins. Matching is done on name tokens (split on "_"), prefix match.
SENSOR_GROUPS = {
    "Thermal": ["temp"],
    "Vibration": ["vibr"],
    "Pressure": ["press"],
    "Mechanical": ["rpm", "speed", "torque", "rotation"],
    "Electrical": ["volt", "current", "amp", "power", "freq"],
    "Fluids & lubrication": ["oil", "coolant", "hydraulic", "flow", "viscos"],
    "Operational": ["load", "energy", "consum", "throughput", "cycle", "runtime", "operating", "production"],
    "Environment": ["humid", "ambient", "dust", "noise"],
}

# Tokens that mark a numeric column as NOT a raw sensor
NON_SENSOR_TOKENS = {"health", "anomaly", "probability", "prob", "remaining", "rul", "days", "maintenance",
                     "failure", "count", "cost", "age", "repair", "technician", "downtime", "id", "score",
                     "flag", "previous", "experience", "parts", "replacement", "row"}
