from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_profile.txt"

def generate_report():
    df = load_dataset()
    
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("WORKFORCEIQ - DATASET PROFILE\n")
        file.write("="* 40 + "\n\n")
        
        file.write("1. Dataset Information\n")
        file.write(f"Rows: {df.shape[0]}\n")
        file.write(f"Columns: {df.shape[1]}\n\n")
        
        file.write("2. Data Types\n")
        file.write(df.dtypes.to_string() + "\n\n")
        
        file.write("3. Missing Values\n")
        file.write(df.isnull().sum().to_string() + "\n\n")
        
        file.write("4. Duplicate Records\n")
        file.write(f"{df.duplicated().sum()}\n\n")
        
        file.write("5. Target Distribution\n")
        file.write(df['Attrition'].value_counts().to_string() + "\n\n")
        
        file.write("6. Numerical Statistics\n")
        file.write(df.describe().to_string() + "\n\n")
        
        file.write("7. Categorical Statistics\n")
        file.write(df.describe(include=['object']).to_string() + "\n")
        
    print(f"Dataset report generated: {REPORT_PATH}")

if __name__=="__main__":
    generate_report()
        