import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from recommendation import smart_recommend

app = Flask(__name__)

# Load dataset only when needed (to avoid startup timeout)
def load_ratings():
    try:
        ratings = pd.read_csv(
            "ratings_Electronics.csv",
            header=None,
            names=['user_id', 'product_id', 'rating', 'timestamp']
        )
        ratings = ratings[['user_id', 'product_id', 'rating']]
        ratings['user_id'] = ratings['user_id'].astype(str)
        ratings['product_id'] = ratings['product_id'].astype(str)
        return ratings
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return pd.DataFrame(columns=['user_id', 'product_id', 'rating'])

# Call recommend with shuffle=True
def get_recommendations(user_id, style="popular", num_recommendations=5, shuffle=True):
    try:
        ratings = load_ratings()
        if ratings.empty:
            return [], ["Error: Failed to load ratings dataset."]
        return smart_recommend(user_id, style=style, num_recommendations=num_recommendations, shuffle=shuffle)
    except Exception as e:
        return [], [f"Error: {str(e)}"]

@app.route('/')
def home():
    return redirect(url_for('recommendation'))

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        style = request.form.get('style', 'popular')
        rated, recommended = get_recommendations(user_id, style=style, shuffle=True)
        return render_template('index.html',
                               recommendations=recommended,
                               rated_products=rated,
                               user_id=user_id,
                               selected_style=style)
    return render_template('index.html', recommendations=[], rated_products=[], user_id=None, selected_style='popular')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
