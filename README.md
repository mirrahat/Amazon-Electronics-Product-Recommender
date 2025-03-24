
# Amazon Electronics Product Recommender

This project implements a product recommendation system using collaborative filtering techniques. It is based on user-product rating data from the Amazon Electronics dataset and explores multiple algorithms to identify the most accurate recommendation model.

## Project Overview

The goal of this project is to build and evaluate different recommendation models that predict user preferences based on historical product ratings. The system supports both personalized recommendations for existing users and fallback logic for new users using popularity-based suggestions.

## Features

- Data cleaning and preprocessing of raw Amazon Electronics ratings
- Exploratory data analysis (EDA) to understand user and product behavior
- Implementation of collaborative filtering models using the Surprise library:
  - Singular Value Decomposition (SVD)
  - K-Nearest Neighbors (KNNBasic)
  - Non-negative Matrix Factorization (NMF)
  - BaselineOnly
- Evaluation of model performance using RMSE
- Visualization of model comparison results
- Functions for generating personalized recommendations for existing users
- Fallback logic using most-rated products for new users

## Project Structure

- `Product_Recommendation.ipynb`: Full notebook including data processing, EDA, modeling, and evaluation
- `recommendation.py`: Python module containing recommendation logic
- `app.py`: Flask application for serving the model
- `model.pkl`: Serialized version of the trained model (excluded from GitHub due to size)
- `requirements.txt`: List of Python dependencies
- `templates/` and `static/`: HTML and CSS assets for the Flask frontend

## Dataset

The dataset used is the Amazon Electronics Ratings dataset, which contains user reviews and ratings for electronic products. The original data is available from public sources and is not included in this repository due to GitHub file size restrictions.

You can download the dataset from:
https://nijianmo.github.io/amazon/index.html

## Getting Started

1. Clone the repository:

   ```
   git clone https://github.com/your-username/Amazon-Electronics-Product-Recommender.git
   cd Amazon-Electronics-Product-Recommender
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app locally:

   ```
   python app.py
   ```

## Notes

- `model.pkl` and large datasets have been excluded from this repository. You can retrain the model using the notebook or add your own dataset.
- The project is intended for educational and demonstration purposes.
## Author

Mir Hasibul Hasan Rahat  
GitHub: https://github.com/mirrahat  
LinkedIn: www.linkedin.com/in/mir-rahat-2b2108147

## License

This project is open-source and available under the MIT License.
