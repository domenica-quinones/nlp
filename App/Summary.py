# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:26:31 2023

@author: mquinones
"""

############################# NLP Group Assignment ############################

#==============================================================================
# Import libraries
#==============================================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sb
import plotly.express as px
import matplotlib.colors as mcolors
#from gensim.corpora import Dictionary
#==============================================================================
# Tab 1: Summary
#==============================================================================

# Upload dataset initially
data = pd.read_csv(r'C:\Users\mquinones\Documents\MASTER BIG DATA ANALYTICS\15. FUNDAMENTALS OF NLP\Group Project\dataset.csv')

# Layout -----

st.set_page_config(layout="wide")

# Title -----

st.title("Song Success Based on Lyrics")

st.header("WordCloud")
st.header("Sentiment Analysis")
st.markdown('Sentiment Analysis is the process of determining the “sentiment” of a text, usually classified as positive, negative or neutral.'
           'For the lyrics analysis, the VADER sentiment analyzer was used as it returns a positive, neutral and negative score for each song, as well as a compound score, which determines the overall sentiment of the lyrics. This score obtains normalized sum of all the lexicon ratings, between -1 (negative) and +1 (positive).'
           'VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool usually used for determining sentiments in social media. VADER identifies as mix of sentiment lexicon which are identified according to their semantic orientation as either positive, neutral or negative.' 
)

# Treemap
sentiment_groups = data.groupby('Sentiment')['id'].agg(['count']).reset_index(drop=False)

# generate a list of colors in the coolwarm palette
colors = sb.color_palette("coolwarm_r", len(sentiment_groups))

# convert RGB tuples to hex strings
hex_colors = [mcolors.rgb2hex(color) for color in colors]

# create treemap
fig1 = px.treemap(sentiment_groups, path=['Sentiment'], values='count', 
                  color_discrete_sequence=hex_colors)

# Barplot
sent_pop_groups = data.groupby('Sentiment')['popularity'].agg(['mean']).reset_index()
sent_pop_groups['mean'] = round(sent_pop_groups['mean'], 0)

sb.set_style("darkgrid")
fig2, ax = plt.subplots()
sb.barplot(data=sent_pop_groups, x="Sentiment", y="mean", palette="coolwarm", ax=ax)
ax.bar_label(ax.containers[0])
ax.set_ylim(1, 100)
ax.set(ylabel='Popularity')

# display the figures side by side
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align: center;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig1)
with col2:
    st.markdown("<h3 style='text-align: center;'>Mean Popularity by Sentiment</h3>", unsafe_allow_html=True)
    st.pyplot(fig2)

st.header("Topic Modelling")

st.header("NER")

long_format=pd.melt(data, id_vars='id', value_vars=['SWEAR', 'TIME', 'SLANG', 'PRODUCT','GPE','PERSON', 'ORG'])
entities_grouped = long_format.groupby('variable').sum().reset_index(drop=False)
entities_grouped=entities_grouped.rename(columns={0: "entities", 1: "sum"}).reset_index(drop=True)
entities_grouped= entities_grouped.sort_values("value", ascending=False)

# Create a barplot using seaborn
sb.set(style="white")
ent = sb.barplot(x='value', y="variable", data=entities_grouped, palette='coolwarm',
                 ci=None)
ent.set(xticklabels=[])

# sb.despine()

# Add chart title and axis labels
plt.title("Number of entities")
plt.xlabel("Count")
plt.ylabel("Entities")


total = entities_grouped['value'].sum()
for p in ent.patches:
    ent.annotate(f'{p.get_width():.0f} ({p.get_width()/total*100:.1f}%)',
                 (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2),
                 ha='left', va='center')

# Display the chart
st.pyplot(ent.figure)