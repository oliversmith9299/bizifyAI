import json
from ProfileAnalysis import profile_analysis  # your function

# 1. Read questionnaire output
with open("../data/questionnaireOutput.json", "r") as f:
    data = json.load(f)

# 2. Run analysis
result = profile_analysis(data)

# 3. Save result to file
with open("../data/profileAnalysis.json", "w") as f:
    json.dump(result, f, indent=2)

print("✅ profileAnalysis.json created successfully")