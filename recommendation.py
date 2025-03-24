import pandas as pd
import os
import joblib
import random
from surprise import SVD, Dataset, Reader

# Load dataset
full_df = pd.read_csv("ratings_Electronics.csv", header=None, names=['user_id', 'product_id', 'rating', 'timestamp'])
full_df['user_id'] = full_df['user_id'].astype(str)
full_df['product_id'] = full_df['product_id'].astype(str)

# Prepare Surprise dataset
sample_df = full_df[['user_id', 'product_id', 'rating']]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(sample_df, reader)
trainset = data.build_full_trainset()

# Train or load model
model_path = "model.pkl"
if os.path.exists(model_path):
    print(" Loading existing model...")
    model = joblib.load(model_path)
else:
    print("⏳ Training model... Please wait.")
    model = SVD()
    model.fit(trainset)
    joblib.dump(model, model_path)
    print("Model trained and saved as model.pkl")

#Check if user is in training set
def is_user_in_training(user_id):
    return str(user_id) in trainset._raw2inner_id_users

# Popular products (shuffled)
def get_top_popular_products(n=5):
    top_products = (
        sample_df.groupby('product_id')['rating']
        .agg(['count', 'mean'])
        .sort_values(by=['count', 'mean'], ascending=False)
        .head(20)
        .index.tolist()
    )
    random.shuffle(top_products)
    return top_products[:n]

# Hidden gems (shuffled)
def get_top_rated_hidden_gems(n=5):
    hidden_gems = (
        sample_df.groupby('product_id')['rating']
        .agg(['count', 'mean'])
        .query("count < 10 and mean >= 4.5")
        .sort_values(by='mean', ascending=False)
        .head(20)
    )
    hidden_list = hidden_gems.index.tolist()
    random.shuffle(hidden_list)
    return hidden_list[:n]

# Recommendation logic with shuffle 
def smart_recommend(user_id, style="popular", num_recommendations=5, shuffle=False):
    user_id = str(user_id)

    if is_user_in_training(user_id):
        # Rated products wihhout shuffle
        rated_products = sample_df[sample_df['user_id'] == user_id]['product_id'].tolist()

        # Recommended products  optional shuffle
        all_products = sample_df['product_id'].unique()
        unseen_products = [item for item in all_products if item not in rated_products]

        predictions = [model.predict(user_id, item) for item in unseen_products]
        predictions.sort(key=lambda x: x.est, reverse=True)

        top_k_predictions = predictions[:50]
        top_k_items = [pred.iid for pred in top_k_predictions]

        if shuffle:
            random.shuffle(top_k_items)

        top_recommendations = top_k_items[:num_recommendations]
        return rated_products, top_recommendations

    else:
        #  Cold-start fallback
        if style == "gems":
            return [], get_top_rated_hidden_gems(num_recommendations)
        else:
            return [], get_top_popular_products(num_recommendations)

# Optional: Run test locally
if __name__ == "__main__":
    test_user = "A3SGXH7AUHU8GW"
    if is_user_in_training(test_user):
        rated, recommended = smart_recommend(test_user, shuffle=True)
        print("Rated products:", rated)
        print("Recommended (shuffled):", recommended)
    else:
        _, recommended = smart_recommend(test_user)
        print("Cold-start recommendations:", recommended)
