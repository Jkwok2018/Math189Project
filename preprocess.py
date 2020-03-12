import numpy as np
import pandas as pd


#Read in data from Excel Sheets
sh  = pd.read_csv("ShanghaiTest.csv") #Excel File
hk = pd.read_csv("HKTest.csv")
sampleDate = pd.read_csv("SampleDate.csv")

SHANGHAI = sh['PercentChange'] #Associated with Column Name
SHANGHAIDate = sh['Date']

HK = hk['PercentChange']
HKdate = hk['Date']

standardDate = sampleDate["Date"]
export = []

# print(standardDate[3])
# print(SHANGHAIDate[0])
# print(standardDate[3]==SHANGHAIDate[0])

# for stdDate in standardDate:
#     print(stdDate in SHANGHAIDate)
#     new = []
#     if (stdDate in SHANGHAIDate):
#         new.append(SHANGHAI[SHANGHAIDate.index(stdDate)])
#     else:
#         new.append(300)
#     if (stdDate in HKdate):
#         new.append(HK[HKdate.index(stdDate)])
#     else:
#         new.append(300)
#     export.append(new)

# print(np.matrix(export))
# print(SHANGHAIDate)

for i in range(len(standardDate)):
    new = []
    # print(standardDate[i])
    if (standardDate[i] in SHANGHAIDate):
        new.append(SHANGHAI[SHANGHAIDate.index(standardDate[i])])
    else:
        new.append(300)
    if (standardDate[i] in HKdate):
        new.append(HK[HKdate.index(standardDate[i])])
    else:
        new.append(300)
    export.append(new)

# print( str(standardDate[0]) == "2/21/15")
# print( str(SHANGHAIDate[0]) == "2/25/15")

# print(standardDate.index(str(SHANGHAIDate[0])))

# x= sampleDate.index[sh.str.match(SHANGHAI[0])]
# print(x)

# print(SHANGHAIDate[0] in sampleDate.values)
# print(sampleDate.index(SHANGHAIDate[0]))

for item in SHANGHAIDate:
    item = str(item)
for item in standardDate:
    item = str(item)
print(SHANGHAIDate[0] in sampleDate.values)
print("2/25/15" in sampleDate.values)