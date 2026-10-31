from src.data_loader import load_dataset

def main():
    df = load_dataset()
    
    print("--- Numerical Feature Statistics ---")
    print(df.describe().T)
    
    print("\n--- Categorical Feature Statistics ---")
    print(df.describe(include=['object']).T)
    
if __name__=="__main__":
    main()