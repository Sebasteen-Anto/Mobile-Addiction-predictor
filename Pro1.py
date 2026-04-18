#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df=pd.read_csv('/content/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
df

df.head()
df.tail()
df.info()
df.describe()
df.isnull().sum()
df.shape
df.dtypes
df.nunique()
df['sleep_hours'].unique()
df['sleep_hours'].value_counts()
df.sample(100)
df.duplicated()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['addiction_level']=le.fit_transform(df['addiction_level'])

plt.scatter(df['daily_screen_time_hours'], df['sleep_hours'], alpha=0.6)
plt.xlabel("Screen Time ")
plt.ylabel("Sleep")
plt.title("Mobile Usage")
plt.grid()
plt.show()
plt.hist(df['sleep_hours'], bins=5)
plt.xlabel("social_media_hours")
plt.ylabel("work_study_hours")
plt.title("Mobile Usage")
plt.grid()
plt.show()
plt.boxplot(df['addiction_level'])
plt.title("Box Plot of Mobile Usage")
plt.show()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title("Mobile Usage")
plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
X = df[['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']]
y = df['addiction_level']
rf=RandomForestClassifier(random_state=42)
param_dist ={
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,5, 7]
}
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=5,
                                   scoring = 'accuracy',
                                   n_jobs=-1)
random_search.fit(X_train, y_train)
print("Best parameters" , random_search.best_params_)
print("Best score", random_search.best_score_)
with open('sales.pkl', 'wb') as f:
    pickle.dump(model, f)





# In[19]:


get_ipython().system('pip install streamlit')


# In[20]:


get_ipython().run_cell_magic('writefile', 'app.py', '\nimport streamlit as st\nimport pickle\nimport numpy as np\n\n# Title\nst.title("E-Commerce Sales Prediction")\n\nst.write("Predict Product Price using Marketing Spend and Discount")\n\n# Load the trained model\nmodel = pickle.load(open(\'sales.pkl\', \'rb\'))\n\n# User Inputs\nmarketing_spend = st.number_input("Enter Marketing Spend")\ndiscount = st.number_input("Enter Discount")\n\n# Prediction button\nif st.button("Predict Price"):\n\n    input_data = np.array([[marketing_spend, discount]])\n\n    prediction = model.predict(input_data)\n\n    st.success(f"Predicted Price: {prediction[0]:.2f}")\n')

