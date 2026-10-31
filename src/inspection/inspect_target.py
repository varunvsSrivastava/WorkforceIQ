from src.data_loader import load_dataset

def main():
    df = load_dataset()
    
    print("--- Attrition Column Values ---")
    print(df["Attrition"].unique())
    
    print("\n--- Attrition Counts ---")
    print(df["Attrition"].value_counts())
    
    print("\n--- Attrition Percentages ---")
    print(
        (df["Attrition"].value_counts(normalize=True)*100).round(2)
    )
    
if __name__=="__main__":
    main()