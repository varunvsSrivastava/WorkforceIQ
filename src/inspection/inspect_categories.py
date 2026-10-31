from src.data_loader import load_dataset

def main():
    df=load_dataset()
    
    categorical_columns = df.select_dtypes(
        include=['object','category']
    ).columns
    
    for column in categorical_columns:
        print(f"\n--- {column} ---")
        print("Unique values:", df[column].nunique())
        print(df[column].unique())
        
if __name__=="__main__":
    main()