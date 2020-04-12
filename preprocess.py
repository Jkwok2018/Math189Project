
import numpy as np
import pandas as pd

sampleDate = pd.read_csv("sampleDate.csv")
sp500  = pd.read_csv("SandPMarch31.csv") #Excel File

spDate = sp500['Date'].tolist()
spPriceChange = sp500['PriceChange'].tolist()
spVolumeChange = sp500['VolumeChange'].tolist()

standardDate = sampleDate['Date'].tolist()
export = []

for i in range(len(standardDate)):
    new = []
    new.append(standardDate[i])
    if (standardDate[i] in spDate):
        new.append(spPriceChange[spDate.index(standardDate[i])])
        new.append(spVolumeChange[spDate.index(standardDate[i])])
    else:
        new.append(0)
        new.append(0)
    export.append(new)    

df = pd.DataFrame(export, columns=['Date', 'Price Change', 'Volume Change'])    
df.to_csv(r'newProcessed.csv', index = False, header = True)
