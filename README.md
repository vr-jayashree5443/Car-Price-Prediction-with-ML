# Car-Price-Prediction-with-ML
Let's break down the code and explain the workflow step by step:

1. **Import Libraries**:
   - Import the necessary Python libraries, including Pandas, NumPy, Matplotlib, scikit-learn modules for data manipulation, visualization, and machine learning.

2. **Load the Dataset**:
   - Use `pd.read_csv` to load the car price dataset from the specified file path into a Pandas DataFrame.

3. **Data Preprocessing**:
   - Use one-hot encoding to convert categorical variables (Fuel_Type, Selling_type, Transmission) into binary format (0 or 1) so that they can be used in the machine learning model. The `drop_first=True` argument avoids multicollinearity.

4. **Data Splitting**:
   - Split the data into features (X) and the target variable (y). `X` contains all columns except "Car_Name" and "Selling_Price," and `y` contains the "Selling_Price" column, which is what we want to predict.

5. **Train-Test Split**:
   - Split the data into a training set (80%) and a testing set (20%) using `train_test_split`. This allows us to evaluate the model's performance on unseen data.

6. **Model Selection**:
   - Create a Random Forest Regressor model. This is a machine learning model for regression tasks. The `n_estimators` parameter specifies the number of decision trees in the ensemble, and `random_state` ensures reproducibility.

7. **Model Training**:
   - Fit the Random Forest Regressor model to the training data using `model.fit(X_train, y_train)`.

8. **Model Prediction**:
   - Use the trained model to make predictions on the test data. The predicted values are stored in `y_pred`.

9. **Model Evaluation**:
   - Calculate various regression metrics to assess the model's performance on the test data, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared (R2) score. These metrics provide information about how well the model's predictions match the actual car prices.

10. **Visualization**:
    
    ![Figure_1](https://github.com/vr-jayashree5443/Car-Price-Prediction-with-ML/assets/128161257/de48b06e-0859-40b5-bcae-36b58fa61ea2)
   - Create a scatter plot that compares the actual car prices (y-axis) to the predicted prices (x-axis). This visualization helps you understand how well the model's predictions align with the true values.

11. **Print Results**:
    
    ![Screenshot 2023-11-05 112237](https://github.com/vr-jayashree5443/Car-Price-Prediction-with-ML/assets/128161257/a6ca333f-3437-46c6-9e56-3ede4a195500)
   - Create a DataFrame called `results` that contains the actual prices and the model's predictions. This DataFrame is printed to the console, allowing you to examine the individual predictions and their corresponding actual values.

The code follows a structured workflow from data loading and preprocessing to model training, evaluation, and visualization. It provides key performance metrics and visual representations to assess how well the machine learning model predicts car prices based on the dataset.
