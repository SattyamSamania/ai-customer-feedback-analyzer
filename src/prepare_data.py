import pandas as pd

# Path to your raw Kaggle file
RAW_FILE = "data/sample_reviews.csv"
OUTPUT_FILE = "data/sample_reviews_clean.csv"
SAMPLE_SIZE = 200  # change to 500, 1000, etc. if needed

cleaned_data = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("__label__2"):
            label = "POSITIVE"
            text = line.replace("__label__2", "", 1).strip()
        elif line.startswith("__label__1"):
            label = "NEGATIVE"
            text = line.replace("__label__1", "", 1).strip()
        else:
            continue
        cleaned_data.append([text, label])

# Convert to DataFrame
df = pd.DataFrame(cleaned_data, columns=["review_text", "label"])

# Randomly sample
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42)

# Save clean CSV
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Clean dataset saved to {OUTPUT_FILE} with {len(df)} reviews")
