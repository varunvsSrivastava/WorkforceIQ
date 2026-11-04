from src.data_loader import load_dataset

def main():
    df=load_dataset()

    numerical_features = df.select_dtypes(
        include=['int64','float64']
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=['object','category','bool']
    ).columns.tolist()

    print("--- Numerical Features ---")
    print(numerical_features)
    print("Total:", len(numerical_features))

    print("\nCategorical Features ---")
    print(categorical_features)
    print("Total:", len(categorical_features))

if __name__=="__main__":
    main()
