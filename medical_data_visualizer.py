import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df['weight'] / (df['height'] / 100)** 2 > 25, 1, 0)

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
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7

    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig
    
    # 8

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, mask=mask, cmap="coolwarm", square=True, linewidths=0.5, fmt='.1f', ax=ax)


    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
