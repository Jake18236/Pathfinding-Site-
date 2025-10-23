import yaml
import json

# Load semester information
with open("../static/json/semester_info.json", "r") as f:
    semester_info = json.load(f)

# Extract TAs from semester_info.json
tas = semester_info.get("TAs", {})

# Define the office hours data structure
office_hours = {}

# Process each TA's office hours
for ta_id, ta_info in tas.items():
    name = ta_info.get("name")
    day = ta_info.get("OH Days", "TBD")
    time = ta_info.get("OH Times", "TBD")
    location = ta_info.get("OH Location", "TBD")
    
    # Add to office_hours
    if day not in office_hours:
        office_hours[day] = []
    office_hours[day].append({
        "name": name,
        "time": time,
        "location": location
    })

# Convert office hours to YAML format
office_hours_data = {"office_hours": []}
for day, slots in office_hours.items():
    office_hours_data["office_hours"].append({
        "day": day,
        "slots": slots
    })

# Save to office_hours.yaml
with open("office_hours.yaml", "w") as f:
    yaml.dump(office_hours_data, f, sort_keys=False)

print("Generated office_hours.yaml")