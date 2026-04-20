import pandas as pd
def main():
    df = pd.read_csv("orders.csv")
    print("Hello from datathon!")
    print(df.head())


if __name__ == "__main__":
    main()
