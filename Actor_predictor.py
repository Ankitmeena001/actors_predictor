#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv('actors_data_modified.csv')


# In[52]:


df.info()


# In[160]:


X_train = df.iloc[:,1:]
y_train = df['name']


# In[161]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 20)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth = 5)
from sklearn.neighbors  import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[162]:


clf.fit(X_train,y_train)


# In[163]:


lr.fit(X_train,y_train)


# In[164]:


rfc.fit(X_train,y_train)


# In[165]:


knn.fit(X_train,y_train)


# In[166]:


X_testb = df.sample(n=15)


# In[167]:


X_test = X_testb.iloc[:,1:]
y_test = X_testb['name']


# In[168]:


y_pred_clf = clf.predict(X_test)
y_pred_clf


# In[169]:


y_test


# In[170]:


y_pred_lr = lr.predict(X_test)
y_pred_lr


# In[171]:


y_pred_rfc = rfc.predict(X_test)
y_pred_rfc


# In[106]:


y_pred_knn = knn.predict(X_test)
y_pred_knn


# In[159]:


df.info()


# In[173]:


user_input = {}

# Ask gender first
ans = input("Is the actor a female? (yes/no): ").strip().lower()
user_input['is_female'] = 1 if ans in ['yes', 'y'] else 0

# Ask common questions
user_input['from_south_india'] = 1 if input("Is the actor from South India? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
user_input['is_alive'] = 1 if input("Is the actor alive? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
user_input['is_married'] = 1 if input("Is the actor married? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
user_input['works_in_hollywood'] = 1 if input("Is the actor works in famous hollywood movies? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
user_input['age_grattter_then_60'] = 1 if input("Is the actor's age is greater then 60? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
user_input['parents_are_famous'] = 1 if input("Is the actor have famous parents ? (yes/no): ").strip().lower() in ['yes', 'y'] else 0

# Conditional questions for male actors
if user_input['is_female'] == 0:
    user_input['is_romantic_hero'] = 1 if input("Is the actor known for romantic roles? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
    if user_input['is_romantic_hero'] == 1:
         user_input['owns_cricket_team'] = 1 if input("Is the actor have a team in ipl? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
    else:
         user_input['is_action_hero'] = 1 if input("Is the actor an action hero? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
         user_input['is_comedy_actor'] = 1 if input("Is the actor known for comedy roles? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
         user_input['is_good_dancer'] = 1 if input("Is the actor a good dancer? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
         user_input['is_good_singer'] = 1 if input("Is the actor a good singer? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
         user_input['is_ott_star'] = 1 if input("Is the actor famous in OTT platforms? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
    if user_input['from_south_india'] == 1:
        user_input['works_in_both_south_and_bollywood'] = 1 if input("Is the actor works in both south and bollywood films? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        if user_input['is_ott_star'] == 1:
            user_input['famous_movie_Jai Bhim'] = 1 if input("Is the actor starred in Jai Bhim? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        else:
            user_input['famous_movie_RRR'] = 1 if input("Is the actor starred in RRR? (yes/no): ").strip().lower() in ['yes', 'y'] else 0    
            user_input['famous_movie_Pushpa'] = 1 if input("Is the actor starred in Pushpa? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            user_input['famous_movie_Kalki 2898 AD'] = 1 if input("Is the actor starred in Kalki 2898 AD? (yes/no): ").strip().lower() in ['yes', 'y'] else 0

    else: 
        if user_input['is_ott_star'] == 1:
            user_input['famous_movie_ludo'] = 1 if input("Is the actor starred in Ludo? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            user_input['famous_movie_sacred_games'] = 1 if input("Is the actor starred in Sacred Games? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            user_input['famous_movie_fami_man'] = 1 if input("Is the actor starred in Family Man? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            user_input['famous_movie_mirzapur'] = 1 if input("Is the actor starred in Mirzapur? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        else:
            user_input['wife_is_actoress'] = 1 if input("Is the actor's wife is acting in movies? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            if user_input['wife_is_actoress'] == 1:
                user_input['famous_movie_Gully Boy'] = 1 if input("Is the actor starred in Gully Boy? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['famous_movie_Kalki 2898 AD'] = 1 if input("Is the actor starred in Bhool Bhuliya Part 1 or 2? (yes/no): ").strip().lower() in ['yes', 'y'] else 0 
                user_input['is_part_of_golmaal_series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['is_partof_Housfull Series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
            else:
                user_input['kissing'] = 1 if input("Is the actor known for kissing scenes? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['is_part_of_golmaal_series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['is_partof_Housfull Series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['famous_movie_Bhool Bhuliya Part 1 or 2'] = 1 if input("Is the actor starred in Bhool Bhuliya Part 1 or 2? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['famous_movie_Judwaa Part 1 or 2'] = 1 if input("Is the actor starred in Judwaa Part 1 or 2? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
                user_input['famous_movie_Chup Chup Ke'] = 1 if input("Is the actor starred in Chup Chup Ke? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
else:
    # For female-specific traits
    user_input['is_ott_star'] = 1 if input("Is the actor famous in OTT platforms? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
    user_input['is_good_dancer'] = 1 if input("Is the actor a good dancer? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
    user_input['is_good_singer'] = 1 if input("Is the actor a good singer? (yes/no): ").strip().lower() in ['yes', 'y'] else 0 
    if user_input['from_south_india'] == 1:
        user_input['works_in_both_south_and_bollywood'] = 1 if input("Is the actor works in both south and bollywood films? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
   
        user_input['famous_movie_Pushpa'] = 1 if input("Is the actor starred in Pushpa? (yes/no): ").strip().lower() in ['yes', 'y'] else 0

    else:
        user_input['won_miss_india'] = 1 if input("Is the actoress won Miss India? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        user_input['husband_is_actor'] = 1 if input("Is the actoress's husband is actor? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        user_input['is_part_of_golmaal_series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        user_input['is_partof_Housfull Series'] = 1 if input("Is the actor starred in Golmaal Series? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        user_input['famous_movie_Bhool Bhuliya Part 1 or 2'] = 1 if input("Is the actor starred in Bhool Bhuliya Part 1 or 2? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        user_input['famous_movie_Judwaa Part 1 or 2'] = 1 if input("Is the actor starred in Judwaa Part 1 or 2? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        if user_input['famous_movie_Judwaa Part 1 or 2'] == 1:
            user_input['from_Sri Lanka'] = 1 if input("Is the actoress from Sri Lanaka? (yes/no): ").strip().lower() in ['yes', 'y'] else 0
        else:
            user_input['famous_movie_Chup Chup Ke'] = 1 if input("Is the actor starred in Chup Chup Ke? (yes/no): ").strip().lower() in ['yes', 'y'] else 0

# Conditional questions for south actors
    
# Add your random movie questions here as before...
# Fill missing columns with 0
for feature in X_train.columns:
    if feature not in user_input:
        user_input[feature] = 0


# Convert to DataFrame
user_df = pd.DataFrame([user_input])
user_df = user_df[X_train.columns]


# Predict the actor
predicted_actor_index = lr.predict(user_df)[0]
predicted_actor = predicted_actor_index
print("🎬 I think the actor you're thinking of is:", predicted_actor)


# In[46]:


# List of actors known for good singing
good_singers = [
    "Ayushmann Khurrana",
    "Farhan Akhtar",
    "Parineeti Chopra",
    "Dhanush"
]

# List of actors known for dancing talent
good_dancers = [
    "Hrithik Roshan",
    "Tiger Shroff",
    "Shahid Kapoor",
    "Varun Dhawan",
    "Nora Fatehi",
    "Ranbir Kapoor",
    "Katrina Kaif",
    "Alia Bhatt",
    "Allu Arjun",
    "Tamannaah Bhatia",
    "Ram Charan",
    "Jr NTR",
    "Vijay",
    "Samantha Ruth Prabhu",
    "Sai Pallavi"
]

# Mark singers
df.loc[df["name"].isin(good_singers), "is_good_singer"] = 1

# Mark dancers
df.loc[df["name"].isin(good_dancers), "is_good_dancer"] = 1


# In[43]:


# Define the column names
columns = df.columns.tolist()

# Create a new row with 0s for all columns
new_row = dict.fromkeys(columns, 0)

# Update values for Ananya Pandey
new_row["name"] = "Sai Pallavi"
new_row["from_south_india"] = 1
new_row["is_female"] = 1
new_row["is_alive"] = 1
new_row["is_married"] = 0
new_row["parents_are_famous"] = 0


# Append new row to dataframe
new_row_df = pd.DataFrame([new_row])
df = pd.concat([df, new_row_df], ignore_index=True)


# In[60]:


df.loc[70:,'name']


# In[49]:


# Add the column with default value 0
# List of popular OTT stars
ott_stars = [
    "Pankaj Tripathi",
    "Manoj Bajpayee",
    "Nawazuddin Siddiqui",
    "Huma Qureshi",
    "Radhika Apte",
    "Abhishek Bachchan",  # Breathe
    "Bobby Deol",
    "Samantha Ruth Prabhu",
    "Suriya",
    "Dhanush"
]

# Update values in column based on name
df.loc[df["name"].isin(ott_stars), "is_ott_star"] = 1


# In[51]:


# Mapping of actors to the movies they have worked in
movie_actor_map = {
    'ludo': ['Pankaj Tripathi', 'Rajkummar Rao', 'Abhishek Bachchan'],
    'mirzapur': ['Pankaj Tripathi'],
    'sita_ramam': ['Dulquer Salmaan','Rashmika Mandanna'],  # optional: cameo or related
    'sacred_games': ['Pankaj Tripathi', 'Nawazuddin Siddiqui'],
    'family_man': ['Manoj Bajpayee', 'Samantha Ruth Prabhu']
}

# Loop through each movie and create a binary column
for movie, actors in movie_actor_map.items():
    col_name = f'famous_movie_{movie}'  # e.g., famous_movie_ludo
    df[col_name] = df['name'].apply(lambda x: 1 if x in actors else 0)


# In[157]:


# Add new column with default value 0
#df['famous_movie_Gully Boy'] = 0

# Mark specific actresses
df.loc[df['name'] == 'Akshay Kumar', 'is_comedy_actor'] = 0
df.loc[df['name'] == 'Rajkummar Rao', 'is_comedy_actor'] = 0


# In[61]:


df['age_gratter_then_60'] = 0
actors_above_55 = [
    # Bollywood Actors
    "Amitabh Bachchan",
    "Anil Kapoor",
    "Jackie Shroff",
    "Sunny Deol",
    "Sanjay Dutt",
    "Boman Irani",
    "Mithun Chakraborty",
    "Paresh Rawal",

    # Bollywood Actresses
    "Hema Malini",
    "Rekha",
    # South Indian Actors
    "Rajinikanth",
    "Kamal Haasan",
    "Chiranjeevi",
    "Nagarjuna"
]
for x in actors_above_55:
    df.loc[df['name'] == x, 'age_gratter_then_60'] = 1


# In[174]:


df.loc[df['is_good_dancer'] == 1,'name']


# In[150]:


df.iloc[50:,:]


# In[116]:


df=df.drop(index=96,axis=0)


# In[177]:


import joblib

# Assuming your trained model is called lr, and your training data is X_train
joblib.dump(lr, "lr.pkl")
joblib.dump(X_train.columns.tolist(), "X_train_columns.pkl")


# In[178]:


import streamlit as st
import pandas as pd
import joblib

# Load the trained model and training columns
lr = joblib.load("lr.pkl")
X_train = joblib.load("X_train_columns.pkl")  # This should be a list of feature names

st.title("🎭 Indian Actor Akinator")
st.write("Answer a few questions, and I'll try to guess the actor you're thinking of!")

user_input = {}

# Gender
is_female = st.radio("Is the actor a female?", ["Yes", "No"]) == "Yes"
user_input['is_female'] = 1 if is_female else 0

# Common Questions
user_input['from_south_india'] = 1 if st.radio("Is the actor from South India?", ["Yes", "No"]) == "Yes" else 0
user_input['is_alive'] = 1 if st.radio("Is the actor alive?", ["Yes", "No"]) == "Yes" else 0
user_input['is_married'] = 1 if st.radio("Is the actor married?", ["Yes", "No"]) == "Yes" else 0
user_input['works_in_hollywood'] = 1 if st.radio("Has the actor worked in Hollywood movies?", ["Yes", "No"]) == "Yes" else 0
user_input['age_grattter_then_60'] = 1 if st.radio("Is the actor older than 60?", ["Yes", "No"]) == "Yes" else 0
user_input['parents_are_famous'] = 1 if st.radio("Are the actor's parents famous?", ["Yes", "No"]) == "Yes" else 0

if not is_female:
    user_input['is_romantic_hero'] = 1 if st.radio("Is the actor known for romantic roles?", ["Yes", "No"]) == "Yes" else 0
    if user_input['is_romantic_hero']:
        user_input['owns_cricket_team'] = 1 if st.radio("Does the actor own an IPL team?", ["Yes", "No"]) == "Yes" else 0
    else:
        user_input['is_action_hero'] = 1 if st.radio("Is the actor an action hero?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_comedy_actor'] = 1 if st.radio("Is the actor known for comedy roles?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_good_dancer'] = 1 if st.radio("Is the actor a good dancer?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_good_singer'] = 1 if st.radio("Is the actor a good singer?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_ott_star'] = 1 if st.radio("Is the actor famous on OTT platforms?", ["Yes", "No"]) == "Yes" else 0
    
    if user_input['from_south_india']:
        user_input['works_in_both_south_and_bollywood'] = 1 if st.radio("Works in both South and Bollywood?", ["Yes", "No"]) == "Yes" else 0
        if user_input.get('is_ott_star', 0):
            user_input['famous_movie_Jai Bhim'] = 1 if st.radio("Starred in Jai Bhim?", ["Yes", "No"]) == "Yes" else 0
        else:
            user_input['famous_movie_RRR'] = 1 if st.radio("Starred in RRR?", ["Yes", "No"]) == "Yes" else 0
            user_input['famous_movie_Pushpa'] = 1 if st.radio("Starred in Pushpa?", ["Yes", "No"]) == "Yes" else 0
            user_input['famous_movie_Kalki 2898 AD'] = 1 if st.radio("Starred in Kalki 2898 AD?", ["Yes", "No"]) == "Yes" else 0
    else:
        if user_input.get('is_ott_star', 0):
            user_input['famous_movie_ludo'] = 1 if st.radio("Starred in Ludo?", ["Yes", "No"]) == "Yes" else 0
            user_input['famous_movie_sacred_games'] = 1 if st.radio("Starred in Sacred Games?", ["Yes", "No"]) == "Yes" else 0
            user_input['famous_movie_fami_man'] = 1 if st.radio("Starred in Family Man?", ["Yes", "No"]) == "Yes" else 0
            user_input['famous_movie_mirzapur'] = 1 if st.radio("Starred in Mirzapur?", ["Yes", "No"]) == "Yes" else 0
        else:
            user_input['wife_is_actoress'] = 1 if st.radio("Is his wife an actress?", ["Yes", "No"]) == "Yes" else 0
            if user_input['wife_is_actoress']:
                user_input['famous_movie_Gully Boy'] = 1 if st.radio("Starred in Gully Boy?", ["Yes", "No"]) == "Yes" else 0
                user_input['famous_movie_Kalki 2898 AD'] = 1 if st.radio("Starred in Bhool Bhulaiyaa 1 or 2?", ["Yes", "No"]) == "Yes" else 0
                user_input['is_part_of_golmaal_series'] = 1 if st.radio("Starred in Golmaal series?", ["Yes", "No"]) == "Yes" else 0
                user_input['is_partof_Housfull Series'] = 1 if st.radio("Starred in Housefull series?", ["Yes", "No"]) == "Yes" else 0
            else:
                user_input['kissing'] = 1 if st.radio("Known for kissing scenes?", ["Yes", "No"]) == "Yes" else 0
                user_input['is_part_of_golmaal_series'] = 1 if st.radio("Starred in Golmaal series?", ["Yes", "No"]) == "Yes" else 0
                user_input['is_partof_Housfull Series'] = 1 if st.radio("Starred in Housefull series?", ["Yes", "No"]) == "Yes" else 0
                user_input['famous_movie_Bhool Bhuliya Part 1 or 2'] = 1 if st.radio("Starred in Bhool Bhulaiyaa 1 or 2?", ["Yes", "No"]) == "Yes" else 0
                user_input['famous_movie_Judwaa Part 1 or 2'] = 1 if st.radio("Starred in Judwaa 1 or 2?", ["Yes", "No"]) == "Yes" else 0
                user_input['famous_movie_Chup Chup Ke'] = 1 if st.radio("Starred in Chup Chup Ke?", ["Yes", "No"]) == "Yes" else 0
else:
    user_input['is_ott_star'] = 1 if st.radio("Is the actress famous on OTT?", ["Yes", "No"]) == "Yes" else 0
    user_input['is_good_dancer'] = 1 if st.radio("Is she a good dancer?", ["Yes", "No"]) == "Yes" else 0
    user_input['is_good_singer'] = 1 if st.radio("Is she a good singer?", ["Yes", "No"]) == "Yes" else 0
    if user_input['from_south_india']:
        user_input['works_in_both_south_and_bollywood'] = 1 if st.radio("Works in both South and Bollywood?", ["Yes", "No"]) == "Yes" else 0
        user_input['famous_movie_Pushpa'] = 1 if st.radio("Starred in Pushpa?", ["Yes", "No"]) == "Yes" else 0
    else:
        user_input['won_miss_india'] = 1 if st.radio("Won Miss India?", ["Yes", "No"]) == "Yes" else 0
        user_input['husband_is_actor'] = 1 if st.radio("Husband is an actor?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_part_of_golmaal_series'] = 1 if st.radio("Starred in Golmaal series?", ["Yes", "No"]) == "Yes" else 0
        user_input['is_partof_Housfull Series'] = 1 if st.radio("Starred in Housefull series?", ["Yes", "No"]) == "Yes" else 0
        user_input['famous_movie_Bhool Bhuliya Part 1 or 2'] = 1 if st.radio("Starred in Bhool Bhulaiyaa 1 or 2?", ["Yes", "No"]) == "Yes" else 0
        user_input['famous_movie_Judwaa Part 1 or 2'] = 1 if st.radio("Starred in Judwaa 1 or 2?", ["Yes", "No"]) == "Yes" else 0
        if user_input['famous_movie_Judwaa Part 1 or 2']:
            user_input['from_Sri Lanka'] = 1 if st.radio("Is the actress from Sri Lanka?", ["Yes", "No"]) == "Yes" else 0
        else:
            user_input['famous_movie_Chup Chup Ke'] = 1 if st.radio("Starred in Chup Chup Ke?", ["Yes", "No"]) == "Yes" else 0

# Fill missing features with 0
for feature in X_train:
    if feature not in user_input:
        user_input[feature] = 0

user_df = pd.DataFrame([user_input])[X_train]

# Predict
predicted_actor_index = lr.predict(user_df)[0]
st.success(f"🎬 I think the actor you're thinking of is: **{predicted_actor_index}**")


# In[ ]:




