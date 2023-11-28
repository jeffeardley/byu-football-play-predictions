import pandas as pd

#### WE CAN PUT ANY USEFUL FUNCTIONS IN HERE ####

def read_data(start = 1, end = 9, folder="play-data"):
    big_df = pd.DataFrame()
    for i in range(start, end+1):
        df = pd.read_csv(f"{folder}/2022_wk{i}.csv")
        df['week'] = i
        big_df = big_df._append(df, ignore_index=True)
    return big_df





def main():
    df = read_data()
    # df = pd.DataFrame()
    # df2 = pd.read_csv('byu-football-play-predictions/play-data/2022_wk1.csv')
    # df = df._append(df2)
    print(df)



if __name__== "__main__":
    main()