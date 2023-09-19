import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")
sns.set()

st.title('Mental Health Survey Dashboard')

data = pd.read_csv("survey.csv")
data_no_na = data[data['comments'].isna() == False]


# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

comments = data_no_na['comments']

# Perform sentiment analysis for each comment
sentiments = []

for comment in comments:
    sentiment_scores = sia.polarity_scores(comment)

    # Determine the sentiment label based on compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    sentiments.append(sentiment)
    

# Add the sentiments as a new column in your dataset
data_no_na['sentiments'] = sentiments


# # Plot 1 - Sentiments of Comments

# fig, ax = plt.subplots()
# bar_chart = data_no_na['sentiments'].groupby(data_no_na['sentiments']).count().plot(
#     kind='bar',ax=ax
# )

# plt.xticks(rotation=0)

# # Save the entire figure as an image
# fig.savefig('sentiments_bar_chart.png')  # Replace with your desired file path and format

# # Display the bar chart in Streamlit
# st.header("Sentiments of Comments")
# st.image('sentiments_bar_chart.png')


# # Plot 2 - WordCloud

# # Combine all comments into a single text
# all_comments_text = " ".join(comment for comment in comments if isinstance(comment, str))

# # Create a WordCloud object
# wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200).generate(all_comments_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")

# st.title("Word Cloud of Comments")
# st.pyplot(plt)


# Plot 3 - Most Common Words

tokens = []
stop_words = set(stopwords.words("english"))

for comment in data_no_na['comments']:
  # Check if the comment is a string
  words = word_tokenize(comment.lower())
  filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
  tokens.extend(filtered_words)
  
word_freq = nltk.FreqDist(tokens)
words = list(word_freq.keys())
counts = list(word_freq.values())
word_count = pd.DataFrame(counts,words)
word_count.reset_index(inplace=True)
word_count.sort_values(0, ascending = False, inplace=True)
word_count = word_count.head(20)

plt.figure(figsize=(10, 8))
chart = sns.barplot(data = word_count,x='index', y=0)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

st.title("Top 20 Most Common Words in Comments")
st.pyplot(plt)


# Plot 4 - Map

location = pd.DataFrame(data['Country'].groupby(data['Country']).count().sort_values(ascending=False))
location = location.rename_axis('country')
location.reset_index(inplace=True)
location = location.rename(columns={'Country':'Number of respondents'})

fig = px.choropleth(
    location,  # DataFrame with your data
    locations='country',  # Column containing country names
    locationmode='country names',  # Location mode for country names
    color='Number of respondents',  # Column for color density
    hover_name='country',  # Column for hover labels
    color_continuous_scale=px.colors.sequential.Plasma,  # Choose a color scale
)

# Customize the layout
fig.update_geos(projection_type="natural earth")
fig.update_layout(geo=dict(showcoastlines=True))

# Display the choropleth map in Streamlit
st.header("Respondents by Country")
st.plotly_chart(fig, use_container_width=True)


# Plot 5 - Employment Status

employment_status = pd.DataFrame(data['self_employed'].groupby(data['self_employed']).count())
employment_status = employment_status.rename_axis('Self_Employed')
employment_status.reset_index(inplace=True)
employment_status = employment_status.rename(columns={'self_employed':'count'})
employment_status.loc[employment_status['Self_Employed'] == 'No','Self_Employed'] = 'Employed'  
employment_status.loc[employment_status['Self_Employed'] == 'Yes','Self_Employed'] = 'Self-Employed'

plt.figure(figsize=(6, 6))
plt.pie(employment_status['count'], labels=employment_status['Self_Employed'], autopct='%1.1f%%', startangle=90)

st.header('Employment Status of Respondents')
st.pyplot(plt)