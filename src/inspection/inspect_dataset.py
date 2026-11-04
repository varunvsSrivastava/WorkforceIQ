from src.data_loader import load_dataset

def main():
    df=load_dataset()

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- First Five Rows ---")
    print(df.head())

if __name__=="__main__":
    main()
