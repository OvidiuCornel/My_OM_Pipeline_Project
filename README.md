# My_OM_Pipeline_Project

This project implements a machine learning pipeline to predict whether a customer recommends a product based on their review. 
The dataset includes a combination of numerical, categorical, and text features. 
The goal is to build a well-structured and modular pipeline for feature processing and model training.

Objective

To classify product reviews into recommended (1) or not recommended (0) using features such as age, product category, and review text.

Dataset

The dataset is pre-cleaned and contains 8 features plus the target:

- Clothing ID: Categorical ID of the reviewed item
- Age: Age of the reviewer
- Title: Review title
- Review Text: Full review content
- Positive Feedback Count: Number of helpful votes received
- Division Name: Product's division
- Department Name: Product's department
- Class Name: Product's class
- Recommended IND: Target variable (1 = recommended, 0 = not recommended)

Approach

1. Data Splitting: Train/test split (90% / 10%)
2. Feature Engineering:
   - Numerical features: Imputation + scaling
   - Categorical features: Imputation + encoding
   - Text features: Lemmatization (using spaCy) + TF-IDF vectorization
   - Custom text features: Count of spaces, exclamations, and question marks
3. Model: RandomForestClassifier trained on the processed features
4. Evaluation: Accuracy score and confusion matrix

Technologies

- Python
- pandas, numpy
- scikit-learn
- spaCy
- matplotlib, seaborn


## Getting Started (on Powershell Prompt)

1.	Clone the repository:
	git clone https://github.com/OvidiuCornel/My_OM_Pipeline_Project.git
  cd My_OM_Pipeline_Project
3.	Install dependencies:
  pip install -r requirements.txt
4.	Download the English language model for spaCy:
  python -m spacy download en_core_web_sm
5.	Run the notebook:
   Open 'starter/My_OM_pipeline_project.ipynb' in Jupyter or VS Code


## How to Use This Project on New Data

This project can be used to predict whether new product reviews will lead to a recommendation (1) or not (0).

### Step 1: Prepare New Input Data

- Create a CSV file (e.g., `new_product_reviews.csv`) with the **same structure** as the original `reviews.csv`:

  Columns required:
  ```
  Clothing ID, Age, Title, Review Text, Positive Feedback Count, Division Name, Department Name, Class Name
  ```

- Do NOT include the target column `Recommended IND` — the model will predict it.

### Step 2: Run the Trained Pipeline on New Data

```python
import pandas as pd

# Load new review data
new_data = pd.read_csv('new_product_reviews.csv')

# Predict with the trained pipeline
predictions = model_pipeline.predict(new_data)

# Save predictions to a new file
new_data['Predicted Recommendation'] = predictions
new_data.to_csv('scored_reviews.csv', index=False)
```

Each row will be labeled with:
- `1`: Customer is likely to recommend the product
- `0`: Customer is unlikely to recommend the product

### Optional: Save/Load the Model

You can also export the model with joblib for future use:
```python
import joblib
joblib.dump(model_pipeline, 'recommendation_model.pkl')
```

And reload it later:
```python
model_pipeline = joblib.load('recommendation_model.pkl')
```
## License

[License](LICENSE.txt)
