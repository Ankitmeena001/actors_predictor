import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("lr.pkl")
    feature_names = joblib.load("X_train_columns.pkl")
    return model, feature_names

model, feature_names = load_model()

st.title("🎬 Indian Actor Predictor")
st.write("Answer the questions one by one to find out which actor you're thinking of!")

if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.gender_branch = None
    st.session_state.south_branch = None
    st.session_state.romantic_branch = None
    st.session_state.ott_branch = None
    st.session_state.wife_actor_branch = None
    st.session_state.judwaa_branch = None

def yes_no_input(question, key):
    return st.radio(question, ['Yes', 'No'], key=key) == 'Yes'

questions = []

# Ask gender first
questions.append(('is_female', "Is the actor a female?"))
questions.append(('from_south_india', "Is the actor from South India?"))
questions.append(('is_alive', "Is the actor alive?"))
questions.append(('is_married', "Is the actor married?"))
questions.append(('works_in_hollywood', "Is the actor working in famous Hollywood movies?"))
questions.append(('age_grattter_then_60', "Is the actor's age greater than 60?"))
questions.append(('parents_are_famous', "Does the actor have famous parents?"))
questions.append(('is_muslim', "Is the actor Muslim?"))

user_input = st.session_state.answers

if st.session_state.step < len(questions):
    feat, q = questions[st.session_state.step]
    if yes_no_input(q, feat):
        user_input[feat] = 1
    else:
        user_input[feat] = 0
    if st.button("Next"):
        st.session_state.step += 1
        st.rerun()

elif st.session_state.gender_branch is None:
    # Gender branching
    if user_input['is_female'] == 0:
        st.session_state.gender_branch = 'male'
    else:
        st.session_state.gender_branch = 'female'
    st.rerun()

elif st.session_state.gender_branch == 'male':
    if 'is_ott_star' not in user_input:
        user_input['is_ott_star'] = int(yes_no_input("Is the actor famous in OTT platforms?", 'is_ott_star'))
        st.rerun()

    if 'is_romantic_hero' not in user_input:
        user_input['is_romantic_hero'] = int(yes_no_input("Is the actor known for romantic roles?", 'is_romantic_hero'))
        st.rerun()

    if user_input['is_romantic_hero'] == 1 and 'owns_cricket_team' not in user_input:
        user_input['owns_cricket_team'] = int(yes_no_input("Does the actor own an IPL team?", 'owns_cricket_team'))
        st.rerun()

    if user_input['is_romantic_hero'] == 0:
        for qid, qtext in [
            ('is_action_hero', "Is the actor an action hero?"),
            ('is_comedy_actor', "Is the actor known for comedy roles?"),
            ('is_good_dancer', "Is the actor a good dancer?"),
            ('is_good_singer', "Is the actor a good singer?")]:
            if qid not in user_input:
                user_input[qid] = int(yes_no_input(qtext, qid))
                st.rerun()

    # South branching
    if user_input['from_south_india'] == 1:
        if 'works_in_both_south_and_bollywood' not in user_input:
            user_input['works_in_both_south_and_bollywood'] = int(yes_no_input("Does the actor work in both South and Bollywood films?", 'works_in_both'))
            st.rerun()

        if user_input['is_ott_star'] == 1:
            if 'famous_movie_Jai Bhim' not in user_input:
                user_input['famous_movie_Jai Bhim'] = int(yes_no_input("Did the actor star in Jai Bhim?", 'jai_bhim'))
                st.rerun()
        else:
            for qid, qtext in [
                ('famous_movie_RRR', "Did the actor star in RRR?"),
                ('famous_movie_Pushpa', "Did the actor star in Pushpa?"),
                ('famous_movie_Kalki 2898 AD', "Did the actor star in Kalki 2898 AD?")]:
                if qid not in user_input:
                    user_input[qid] = int(yes_no_input(qtext, qid))
                    st.rerun()
    else:
        if user_input['is_ott_star'] == 1:
            for qid, qtext in [
                ('famous_movie_ludo', "Did the actor star in Ludo?"),
                ('famous_movie_sacred_games', "Did the actor star in Sacred Games?"),
                ('famous_movie_fami_man', "Did the actor star in Family Man?"),
                ('famous_movie_mirzapur', "Did the actor star in Mirzapur?")]:
                if qid not in user_input:
                    user_input[qid] = int(yes_no_input(qtext, qid))
                    st.rerun()
        else:
            if 'wife_is_actoress' not in user_input:
                user_input['wife_is_actoress'] = int(yes_no_input("Is the actor's wife an actress?", 'wife_actor'))
                st.rerun()

            if user_input['wife_is_actoress'] == 1:
                for qid, qtext in [
                    ('famous_movie_Gully Boy', "Did the actor star in Gully Boy?"),
                    ('famous_movie_Kalki 2898 AD', "Did the actor star in Bhool Bhuliya 1 or 2?"),
                    ('is_part_of_golmaal_series', "Is the actor part of the Golmaal series?"),
                    ('is_partof_Housfull Series', "Is the actor part of the Housefull series?")]:
                    if qid not in user_input:
                        user_input[qid] = int(yes_no_input(qtext, qid))
                        st.rerun()
            else:
                for qid, qtext in [
                    ('kissing', "Is the actor known for kissing scenes?"),
                    ('is_part_of_golmaal_series', "Is the actor part of the Golmaal series?"),
                    ('is_partof_Housfull Series', "Is the actor part of the Housefull series?"),
                    ('famous_movie_Bhool Bhuliya Part 1 or 2', "Did the actor star in Bhool Bhuliya 1 or 2?"),
                    ('famous_movie_Judwaa Part 1 or 2', "Did the actor star in Judwaa 1 or 2?"),
                    ('famous_movie_Chup Chup Ke', "Did the actor star in Chup Chup Ke?")]:
                    if qid not in user_input:
                        user_input[qid] = int(yes_no_input(qtext, qid))
                        st.rerun()

elif st.session_state.gender_branch == 'female':
    for qid, qtext in [
        ('is_ott_star', "Is the actress famous on OTT platforms?"),
        ('is_good_dancer', "Is the actress a good dancer?"),
        ('is_good_singer', "Is the actress a good singer?")]:
        if qid not in user_input:
            user_input[qid] = int(yes_no_input(qtext, qid))
            st.rerun()

    if user_input['from_south_india'] == 1:
        for qid, qtext in [
            ('works_in_both_south_and_bollywood', "Does the actress work in both South and Bollywood films?"),
            ('famous_movie_Pushpa', "Did the actress star in Pushpa?")]:
            if qid not in user_input:
                user_input[qid] = int(yes_no_input(qtext, qid))
                st.rerun()
    else:
        for qid, qtext in [
            ('won_miss_india', "Did the actress win Miss India?"),
            ('husband_is_actor', "Is the actress's husband an actor?"),
            ('is_part_of_golmaal_series', "Is the actress part of the Golmaal series?"),
            ('is_partof_Housfull Series', "Is the actress part of the Housefull series?"),
            ('famous_movie_Bhool Bhuliya Part 1 or 2', "Did the actress star in Bhool Bhuliya 1 or 2?"),
            ('famous_movie_Judwaa Part 1 or 2', "Did the actress star in Judwaa 1 or 2?")]:
            if qid not in user_input:
                user_input[qid] = int(yes_no_input(qtext, qid))
                st.rerun()

        if user_input['famous_movie_Judwaa Part 1 or 2'] == 1:
            if 'from_Sri Lanka' not in user_input:
                user_input['from_Sri Lanka'] = int(yes_no_input("Is the actress from Sri Lanka?", 'from_sri_lanka'))
                st.rerun()
        else:
            if 'famous_movie_Chup Chup Ke' not in user_input:
                user_input['famous_movie_Chup Chup Ke'] = int(yes_no_input("Did the actress star in Chup Chup Ke?", 'chup_chup_ke'))
                st.rerun()

# Fill missing values with 0
final_input = {}
for feat in feature_names:
    final_input[feat] = user_input.get(feat, 0)

# Predict
input_df = pd.DataFrame([final_input])
pred = model.predict(input_df)[0]
st.success(f"🎬 I think you're thinking of: **{pred}**")
