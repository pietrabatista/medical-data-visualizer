import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df['weight'] / (df['height'] ** 2) > 25, 1, 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else (1 if x > 1 else x))
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else (1 if x > 1 else x))

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, 
                     id_vars=['cardio'],  # Mantemos a coluna cardio
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='counts')
    
    # 7

    cat_plot = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count')
    
    # 8
    fig = plot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
