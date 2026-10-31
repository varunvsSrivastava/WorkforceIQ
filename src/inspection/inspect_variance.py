from src.data_loader import load_dataset

def main():
    df=load_dataset()
    
    print("--- Unique Values Per Column ---")
    unique_counts = df.nunique()
    print(unique_counts)
    
    constant_columns = unique_counts[
        unique_counts == 1
    ].index.tolist()
    
    print("\n--- Constant Columns ---")
    print(constant_columns)
    
    print("\n--- Low Variance Numerical Features ---")
    numerical_df = df.select_dtypes(include=['number'])
    
    variance = numerical_df.var()
    print(variance.sort_values().head(10))
    
if __name__=="__main__":
    main()