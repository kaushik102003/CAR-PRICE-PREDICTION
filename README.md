# CAR-PRICE-PREDICTION
Problem: The code seems to be exploring the dataset to understand the factors that influence the price of cars and potentially build machine learning models for predicting car prices.

Analysis Steps:

Data Import: The code imports the dataset from a CSV file using Pandas and displays the first 5 rows of the dataset.

Correlation Visualization: It creates a heatmap using Seaborn to visualize the correlation between numerical columns. This helps identify which features have the strongest correlation with the price of cars.

Price Distribution: It creates a distribution plot (histogram) for the 'price' column to understand the distribution of car prices in the dataset.

Bar Plots: It creates bar plots to visualize the relationship between different features and car prices:

One bar plot is for 'carbody' vs. 'price'.
Another bar plot is for 'horsepower' vs. 'price'.
Two more bar plots are for 'citympg' and 'cylindernumber' vs. 'price'.
Violin Plot: It creates a violin plot to visualize the relationship between 'price' and 'CarName' with different categories. This helps understand how car prices vary across different car names.

Scatter Plot with Regression Line: It creates a scatter plot with a regression line to visualize the correlation between 'horsepower' and 'price'. This can help understand how the car's horsepower affects its price.

Machine Learning Preparation:

The code prepares the data for machine learning by selecting specific columns, including 'horsepower', 'citympg', 'highwaympg', and 'price'.
It splits the data into training and testing sets.
Machine Learning Models:

The code creates and fits a Decision Tree Regressor model to predict car prices.
It also creates and fits a Linear Regression model to predict car prices.
Model Evaluation: While the models are created, it appears there may be some issues with how the models are evaluated. The code calculates and displays some evaluation metrics, but the results of these evaluations are not being stored or shown. For instance, the model.score() function should be used to calculate the model's score.

