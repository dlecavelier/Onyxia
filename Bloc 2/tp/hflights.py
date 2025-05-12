import pandas as pd

hflights = pd.read_csv("~/work/Onyxia/Bloc 2/tp/hflights.csv")
print(hflights.dtypes)
print(hflights)

# hflights.CancellationCode.dropna().head()

hflights.CancellationCode.fillna("", inplace=True)
print(hflights.head(2))

cancel_map = {
    "A": "carrier",
    "B": "weather",
    "C": "national air system",
    "D": "security",
    "": "not cancelled",
}

hflights["CancellationCode"] = hflights.CancellationCode.apply(
    lambda cancel_code: cancel_map[cancel_code]
)

hflights.CancellationCode.head()

print(
    # Remarquer la syntaxe de cet exemple
    hflights
    .groupby(hflights.CancellationCode)
    .agg(
        # NamedAgg permet de nommer les agrégations
        FlightCount=pd.NamedAgg(column="FlightNum", aggfunc="count"),
        DelayMean=pd.NamedAgg(column="DepDelay", aggfunc="mean"),
        DelayStd=pd.NamedAgg(column="DepDelay", aggfunc="std")
        )
    .sort_values(by="FlightCount", ascending=False)
)

(
    hflights.DepDelay
    .groupby(hflights.UniqueCarrier)
    .agg("mean")
    .reset_index() # Index UniqueCarrier en colonne
    .plot.scatter(x="UniqueCarrier", y="DepDelay")
)
