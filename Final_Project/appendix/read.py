import pandas as pd
import numpy as np
import sys  
import matplotlib.pyplot as plt

filename = sys.argv[1]
df = pd.read_csv(filename) 
print(df)

leng = df.shape[0]-1



for i in range(leng):
    print(df['Index'][i],",")
