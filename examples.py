from .utils import moving_avg
from .prob import smooth

def climateHockey():
    import pandas as pd
    import matplotlib.pyplot as plt

    # download the data from https://ourworldindata.org/explorers/climate-change
    # get data
    try:
        df = pd.read_csv('climate-change.csv', parse_dates=["Date"])
    except FileNotFoundError as e:
        print("Download the data from https://ourworldindata.org/explorers/climate-change and save the csv in the current directory")
        return
    temp = df[df["Entity"] == "Northern Hemisphere"].set_index("Date")["Temperature anomaly"]
    # calculate smoothing
    y = smooth(temp)
    smoothed = pd.Series(y,index=temp.index)
    # xticks for plotting (data starts at 1880-01-15)
    idx = pd.date_range(start = temp.index[0], end = temp.index[-1], freq = "5Y") + pd.DateOffset(days=15,years=-1)
    # plot
    plt.figure(figsize=(10,6))
    temp.plot(rot=45, xticks=idx, alpha=0.3)
    smoothed.plot(color='red', label="Smoothed") # for some reason also changes xticks to year-only
    plt.plot(temp.index[15:-15], moving_avg(temp,31), label="31-day moving average")
    plt.legend()
    plt.grid()
    plt.title("Temperature as deviation from the 1951-1980 mean")
    plt.ylabel("Temperature anomaly")
    plt.margins(0.01) # axis limits closer to graph
    plt.show()
