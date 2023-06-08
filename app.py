# Importing the necessary libraries and overview of the dataset

# Used to ignore the warning given as output of the code
import warnings                                 
warnings.filterwarnings('ignore')

# Basic libraries of python for numeric and dataframe computations
import numpy as np                              
import pandas as pd
import streamlit as st

# Basic library for data visualization
import matplotlib.pyplot as plt     

# Slightly advanced library for data visualization            
import seaborn as sns                           

# Class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader

# Class for loading datasets
from surprise.dataset import Dataset

# For splitting the rating data in train and test dataset
from surprise.model_selection import train_test_split

# For implementing similarity based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# Import the SVD class
from surprise.prediction_algorithms.matrix_factorization import SVD

# Loading the data

# Import the dataset
# rating = pd.read_csv('ratings.csv')
rating = pd.read_csv('ratings.csv')

st.title("Movie Recommendation System")
# This analysis was done by: Salah Al-Basha
st.write("This analysis was done by: **Salah Al-Basha**")

# Description
st.write("### Description")
st.write("Online streaming platforms like **Netflix** have plenty of movies in their repository and if we can build a **Recommendation System** to recommend **relevant movies** to users, based on their **historical interactions**, this would **improve customer satisfaction** and hence, it will also improve the revenue of the platform.")

# Objective
st.write("### Objective")
"""
In this project we will be building various recommendation systems:

1. Knowledge/Rank based recommendation system
2. Similarity-Based Collaborative filtering
3. Matrix Factorization Based Collaborative Filtering
4. We are going to use the ratings dataset.
"""

# Data Dictionary
st.write("### Data Dictionary")
st.write("The dataset has the following information:")
"""
- User Id
- Movie Id
- Rating
- Timestamp
"""

"""### Let's check the info of the data"""

rating.info()

"""- There are **100,004 observations** and **4 columns** in the data
- All the columns are of **numeric data type**
- The data type of the timestamp column is int64 which is not correct. We can convert this to DateTime format but **we don't need timestamp for our analysis**. Hence, **we can drop this column**

"""

# Dropping timestamp column
rating = rating.drop(['timestamp'], axis=1)

"""## Exploring the dataset
Let's explore the dataset and answer some basic data-related questions:
"""

# Printing the top 5 rows of the dataset
rating.head()

# Create the figure object
fig = plt.figure(figsize=(12, 4))

# Plot the countplot
sns.countplot(x="rating", data=rating)

# Customize the plot
plt.tick_params(labelsize=10)
plt.title("Distribution of Ratings", fontsize=10)
plt.xlabel("Ratings", fontsize=10)
plt.ylabel("Number of Ratings", fontsize=10)

# Render the plot in Streamlit using st.pyplot
st.pyplot(fig)


"""
- As per Histogram, Rating '4.0' has the **highest count** of ratings (>25k).
- Rating '3.0' being second with 20K+.
- And Rating '5.0' being third in count of ratings with a little over 15K.
- The ratings are biased towards 3-5 more than 0-2.5.
"""

# Finding number of unique users
st.write("- There are ", rating['userId'].nunique(), " users in the dataset")

# Finding number of unique movies
st.write("- There are ", rating['movieId'].nunique(), " movies in the dataset.")

"""#### Is there a movie in which the same user interacted with it more than once?
"""
rating.groupby(['userId', 'movieId']).count()

st.write("- There are ", rating.groupby(['userId', 'movieId']).count()['rating'].sum(), " observations in the dataset.")
"""- As per the number of unique users and items, there is a possibility of 100,004 * 9,006 = 90,696,264 ratings in the dataset. 
- But we only have 100,004 ratings, i.e. not every user has rated every item in the dataset. And we can build a recommendation system to recommend items to users which they have not interacted with.
"""

# Finding the most interacted movie in the dataset:
st.write("- The movie with ID number 356 has ", rating['movieId'].value_counts().max(), " interactions.")
rating['movieId'].value_counts()

# Plotting distributions of ratings for 341 interactions with movieid 356
st.subheader("Distributions for 341 Interactions with Movie ID: 356")
fig = plt.figure(figsize=(7, 7))  # Create a figure object

# Plot the data on the figure
rating[rating['movieId'] == 356]['rating'].value_counts().plot(kind='bar')

plt.xlabel('Rating')
plt.ylabel('Count')

# Pass the figure object to st.pyplot() for display
st.pyplot(fig)

"""- We can see that this item has been liked by the majority of users, as the count of ratings 5 and 4 is higher than the count of other ratings.
- There can be items with very high interactions but the count of ratings 1 and 2 may be much higher than 4 or 5 which would imply that the item is disliked by the majority of users.
"""

# Finding which user interacted the most with any movie in the dataset:
rating['userId'].value_counts()
rating['userId'].value_counts().sum()

"""
- User 547 interacted with the most movies in the dataset.
- But still, there is a possibility of 100,004-2,391 = 97,613 more interactions as we have 97,613 unique items in our dataset. For those 97,613 remaining items, we can build a recommendation system to predict which items are most likely to be watched by this user
"""
st.subheader("Number of Interactions by Users")
rating['userId'].value_counts()

# Finding user-movie interactions distribution
count_interactions = rating.groupby('userId').count()['movieId']

# Plotting user-movie interactions distribution
fig, ax = plt.subplots(figsize=(15, 7))
ax.hist(count_interactions)
ax.set_xlabel('Number of Interactions by Users')

# Display the plot using st.pyplot
st.pyplot(fig)


"""
- The distribution is higher skewed to the right.
- Only a few users interacted with more than 125 movies.

**Now that we have explored the data, let's start building Recommendation systems!**

## Rank-Based Recommendation System

### Model 1: Rank-Based Recommendation System
Rank-based recommendation systems provide recommendations based on the popularity of items. They are particularly useful for addressing cold start problems, which occur when a new user joins the system and there is insufficient data to make personalized recommendations. In such cases, a rank-based recommendation system can be employed to suggest popular items to the new user.

To construct a rank-based recommendation system, we calculate the average rating for each movie and rank them based on these average ratings:"""

# Calculating average ratings
average_rating = rating.groupby('movieId').mean()['rating']

# Calculating the count of ratings
count_rating = rating.groupby('movieId').count()['rating']

# Making a dataframe with the count and average of ratings
final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})

final_rating.head()

# Displaying the average ratings and count of ratings in two columns
col1, col2 = st.columns(2)

with col1:
    """ #### Average Ratings"""
    st.write(average_rating)

with col2:
    """ #### Ratings Count"""
    st.write(count_rating)

"""We have created a function to find the **top n movies** for a recommendation based on the average ratings of movies. We have also added a **threshold for a minimum number of interactions** for a movie to be considered for recommendation."""

def top_n_movies(data, n, min_interaction=100):
    
    #Finding movies with minimum number of interactions
    recommendations = data[data['rating_count'] >= min_interaction]
    
    #Sorting values w.r.t average rating 
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    
    return recommendations.index[:n]

"""We can use this function with different n's and minimum interactions to get movies to recommend."""

""" #### Recommending top 5 movies with:"""
# Create three columns
col1, col2, col3 = st.columns(3)

# Column 1
with col1:
    st.write("- **50 minimum interactions** based on popularity:")
    st.write(list(top_n_movies(final_rating, 5, 50)))

# Column 2
with col2:
    st.write("- **100 minimum interactions** based on popularity:")
    st.write(list(top_n_movies(final_rating, 5, 100)))

# Column 3
with col3:
    st.write("- **200 minimum interactions** based on popularity:")
    st.write(list(top_n_movies(final_rating, 5, 200)))


"""Now that we have applied the **Rank-Based Recommendation System**, let's apply the **Collaborative Filtering Based Recommendation Systems**."""

# Define the reader
reader = Reader(rating_scale=(1, 5))

# Load the dataset
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define the similarity-based collaborative filtering model
similarity_model = KNNBasic()

# Train the model
similarity_model.fit(trainset)

# Function to get top N movie recommendations for a user
def get_similar_movies(user_id, N=10):
    # Get the inner user id
    inner_user_id = trainset.to_inner_uid(user_id)
    
    # Get the top N similar users
    similar_users = similarity_model.get_neighbors(inner_user_id, k=N)
    
    # Get the movie ratings of similar users
    similar_users_ratings = []
    for similar_user in similar_users:
        user_ratings = trainset.ur[similar_user]
        similar_users_ratings.extend(user_ratings)
    
    # Create a dictionary to store movie ratings
    movie_ratings = {}
    
    # Iterate through the movie ratings of similar users
    for entry in similar_users_ratings:
        if len(entry) >= 3:
            movie_id, rating, _ = entry
            if movie_id not in movie_ratings:
                movie_ratings[movie_id] = rating
            else:
                movie_ratings[movie_id] = (movie_ratings[movie_id] + rating) / 2
    
    # Sort the movies based on ratings
    sorted_ratings = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top N movie recommendations
    top_N_movies = []
    for movie_id, rating in sorted_ratings[:N]:
        # Get the movie title
        movie_title = movies[movies['movieId'] == trainset.to_raw_iid(movie_id)]['title'].values[0]
        top_N_movies.append((movie_title, rating))
    
    return top_N_movies

# Get top 10 movie recommendations for a user (e.g., user ID 1)
user_id = 1
top_10_movies = get_similar_movies(user_id, N=10)

# Define the SVD algorithm with optimized parameters
svd_algo_optimized = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Train the SVD algorithm on the training set
svd_algo_optimized.fit(trainset)

def predict_already_interacted_ratings(data, user_id, algo):
    
    # Creating an empty list to store the recommended movie ids
    recommendations = []
    
    # Creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # Extracting those movie ids which the user_id has interacted already
    interacted_movies = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].notnull()].index.tolist()
    
    # Looping through each of the movie id which user_id has interacted already
    for item_id in interacted_movies:
        
        # Extracting actual ratings
        actual_rating = user_item_interactions_matrix.loc[user_id, item_id]
        
        # Predicting the ratings for those non interacted movie ids by this user
        predicted_rating = algo.predict(user_id, item_id).est
        
        # Appending the predicted ratings
        recommendations.append((item_id, actual_rating, predicted_rating))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return pd.DataFrame(recommendations, columns=['movieId', 'actual_rating', 'predicted_rating']) # returning top n highest predicted rating movies for this user

# Section title
st.subheader('Analysis of Predicted Ratings')

predicted_ratings_for_interacted_movies = predict_already_interacted_ratings(rating, 7, svd_algo_optimized)
df = predicted_ratings_for_interacted_movies.melt(id_vars='movieId', value_vars=['actual_rating', 'predicted_rating'])

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data
sns.histplot(data=df, x='value', hue='variable', kde=True, ax=ax)

# Set labels and title
ax.set_xlabel('Value')
ax.set_title('Distribution of Actual and Predicted Ratings')

# Display the plot using st.pyplot
st.pyplot(fig)

"""
#### Observations:

- The distribution of actual ratings is skewed towards higher ratings, with a peak around 4.0.
- The distribution of predicted ratings follows a similar pattern but with a slightly lower peak around 3.8.
- Both distributions exhibit a long tail towards lower ratings.

"""

predicted_ratings_for_interacted_movies = predict_already_interacted_ratings(rating, 7, svd_algo_optimized)
df = predicted_ratings_for_interacted_movies.melt(id_vars='movieId', value_vars=['actual_rating', 'predicted_rating'])

# Create the figure object
fig = sns.displot(data=df, x='value', hue='variable', kde=True)

# Set title
fig.set(title='Distribution of Actual and Predicted Ratings')

# Display the plot using st.pyplot
st.pyplot(fig)

"""
#### Observations:

- The actual ratings and predicted ratings follow a similar distribution pattern.
- There is a strong correlation between the actual and predicted ratings, indicating that the model is performing well in predicting user preferences.
- However, there are some discrepancies between the actual and predicted ratings, particularly for lower ratings.

"""

"""

## Conclusion/Findings:

1. Rank-Based Recommendation System:

- The rank-based recommendation system suggests popular movies based on average ratings and a minimum threshold for the number of interactions.
- By setting different thresholds for minimum interactions, we can recommend movies that have gained popularity among users.
- This system can be particularly useful for new users or in situations where there is limited user data available for personalization.

2. Similarity-Based Collaborative Filtering:

- The similarity-based collaborative filtering approach recommends movies based on the ratings and preferences of similar users.
- By finding users who have similar tastes, we can recommend movies that those similar users have rated highly.
- This approach allows for personalized recommendations and can help users discover new movies based on their similarity to other users.

3. Matrix Factorization-Based Collaborative Filtering:

- The matrix factorization-based collaborative filtering, specifically using the SVD algorithm, predicts ratings for movies that a user has not interacted with.
- The predicted ratings show a good correlation with the actual ratings, indicating the model's effectiveness in capturing user preferences.
- This approach can provide personalized recommendations for each user based on their historical interactions and preferences.

Overall, by combining different recommendation systems, such as rank-based, similarity-based, and matrix factorization-based approaches, we can offer a diverse range of movie recommendations to users. This comprehensive approach takes into account both popularity and individual preferences, enhancing the user experience on online streaming platforms.
"""
