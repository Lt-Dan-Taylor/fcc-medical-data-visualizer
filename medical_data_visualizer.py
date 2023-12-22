import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Draw Categorical Plot
def draw_cat_plot():

    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars='cardio', var_name='variable', value_name='value')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['variable', 'value', 'cardio']).size().reset_index(name='total')
    
    # Draw the catplot with 'sns.catplot'
    plt.figure(figsize=(10.58, 5))
    
    catplot = sns.catplot(
                x='variable', 
                y='total', 
                data=df_cat, 
                col='cardio', 
                kind='bar', 
                hue='value', 
                palette='tab10'
              )
    
    # Get the figure for the output
    fig = catplot.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():

    # Clean the data
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))) &
        (df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975)))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(
        ax=ax,
        data=corr,
        mask=mask,
        annot=True,
        cmap='icefire',
        fmt='.1f',
        linewidths='0.5',
        center=0.0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'boundaries': np.arange(-0.16, 0.32, 0.01), 'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24], 'shrink': 0.5}
    )

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
