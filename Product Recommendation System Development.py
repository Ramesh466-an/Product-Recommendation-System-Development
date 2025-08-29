import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample data representing user ratings for products
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'item_id': ['A', 'B', 'C', 'A', 'D', 'E', 'B', 'C', 'F', 'C', 'E', 'F'],
    'rating': [5, 3, 4, 4, 5, 3, 4, 5, 2, 5, 4, 3]
}
ratings_df = pd.DataFrame(data)

# Create a user-item matrix from the data
user_item_matrix = ratings_df.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0) # Fill NaN values with 0 for unrated items

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Convert the similarity matrix to a DataFrame for easier indexing
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

def get_recommendations(target_user_id, k=2, n_recommendations=3):
    """
    Recommends products to a target user using a user-based collaborative filtering approach.

    Args:
        target_user_id (int): The ID of the user to get recommendations for.
        k (int): The number of most similar users to consider.
        n_recommendations (int): The number of products to recommend.
    """
    # Find the k most similar users to the target user
    similar_users = user_similarity_df.loc[target_user_id].sort_values(
        ascending=False
    ).drop(target_user_id).head(k).index

    # Get the items the target user has already rated
    rated_items = user_item_matrix.loc[target_user_id][
        user_item_matrix.loc[target_user_id] > 0
    ].index.tolist()

    # Calculate the average rating for each item from the similar users
    predicted_ratings = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False)

    # Filter out items the target user has already rated
    recommendations = predicted_ratings.drop(rated_items, errors='ignore').head(n_recommendations)

    return recommendations

# Example usage
target_user = 1
recommendations = get_recommendations(target_user)

print(f"Top recommendations for User {target_user}:")
print(recommendations)