from src.data_loader import load_dataset

def main():
    df=load_dataset()
    
    candidate_columns = [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]
    
    print("--- Candidate Administrative Features ---")
    
    for column in candidate_columns:
        print(f"\nColumn:",{column})
        print("Data type:", df[column].dtype)
        print("Unique values:", df[column].nunique())
        print("Sample values:", df[column].unique()[:10])
        
if __name__=="__main__":
    main()