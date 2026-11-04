from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features

TEST_SIZE = 0.20
RANDOM_STATE = 42

def split_dataset(X, y):
    """Split features and target into stratified training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

def main():
    df = load_dataset()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("--- Train-Test Split ---")

    print("\nTraining features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining target:", y_train.shape)
    print("Testing target:", y_test.shape)

    print("\n--- Training Target Distribution ---")
    print(y_train.value_counts(normalize=True).sort_index().round(4))

    print("\n--- Testing Target Distribution ---")
    print(y_test.value_counts(normalize=True).sort_index().round(4))

if __name__=="__main__":
    main()

