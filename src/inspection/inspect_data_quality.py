from src.data_loader import load_dataset

def main():
    df = load_dataset()

    print("--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Total Missing Values ---")
    print(df.isnull().sum().sum())

    print("\n--- Duplicate Records ---")
    print(df.duplicated().sum())

    print("\n--- Duplicate Employee IDs ---")
    print(df['EmployeeNumber'].duplicated().sum())

if __name__=="__main__":
    main()
