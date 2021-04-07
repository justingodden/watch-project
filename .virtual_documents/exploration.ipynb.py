import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

get_ipython().run_line_magic("matplotlib", " inline")
plt.style.use('seaborn')
mpl.rcParams['figure.figsize'] = (18, 8)
plt.rc('axes', titlesize=22) 
plt.rc('figure', titlesize=22)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)


df = pd.read_csv('wf_df_raw.csv').drop('Unnamed: 0', axis=1)


df.head()


df.shape


df['brand'].value_counts().plot.bar()
plt.show()


df['series'].value_counts().iloc[:50].plot.bar()
plt.show()


df['model'].value_counts().iloc[:50].plot.bar()
plt.show()


# Inspect missing Values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()


print(f"Rolex: {df['brand'].value_counts()[0]}\nRest: {len(df) - df['brand'].value_counts()[0]}")
