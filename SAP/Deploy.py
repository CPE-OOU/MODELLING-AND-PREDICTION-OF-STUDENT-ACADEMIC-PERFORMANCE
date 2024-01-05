import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('RFC_class_mdl.pkl','rb'))
scaler = pickle.load(open('scal_cls.pkl','rb'))
encoder = pickle.load(open('enc_cls.pkl','rb'))



st.header("Student Academic Performance")

from PIL import Image

img = Image.open('Abbey-Library-Austria.jpeg')
img = img.resize((1000,400))
st.image(img)

def data1():
    gender = st.selectbox('What is the child gender',['M', 'F'])
        
    NationalITy = st.selectbox('What is the child Nationality',['KW', 'lebanon', 'Egypt', 'SaudiArabia', 'USA', 'Jordan',
           'venzuela', 'Iran', 'Tunis', 'Morocco', 'Syria', 'Palestine',
           'Iraq', 'Lybia'])
    PlaceofBirth = st.selectbox('What is the child Place of birth', ['KuwaIT', 'lebanon', 'Egypt', 'SaudiArabia', 'USA', 'Jordan',
           'venzuela', 'Iran', 'Tunis', 'Morocco', 'Syria', 'Iraq',
           'Palestine', 'Lybia'])
    StageID = st.selectbox('educational level student belongs  ',['lowerlevel', 'MiddleSchool', 'HighSchool'])
    GradeID = st.selectbox('What is the grade student belongs ',['G-04', 'G-07', 'G-08', 'G-06', 'G-05', 'G-09', 'G-12', 'G-11',
           'G-10', 'G-02'])
    SectionID = st.selectbox('What is the classroom student belongs ',['A', 'B', 'C'])
    Topic = st.selectbox('What Topic are you grading for : ',['IT', 'Math', 'Arabic', 'Science', 'English', 'Quran', 'Spanish',
           'French', 'History', 'Biology', 'Chemistry', 'Geology'] )
    Semester = st.selectbox('What is the school year semester : ', ['F', 'S'])
    Relation = st.selectbox('Who raised the child : ',['Father', 'Mum'])
    raisedhands = st.number_input('how many times the student raises his/her hand in  the classroom :' , step = 1)
    VisITedResources = st.number_input('How many times has he/she use the library ', step = 1)
    AnnouncementsView = st.number_input('How many times has he/she viewed the notice board ', step =1)
    Discussion = st.number_input('How many times did he/her contribute to discussion ', step = 1 )
    ParentAnsweringSurvey = st.selectbox('Did the Parent Answer the Survey Sent',['Yes', 'No'])
    ParentschoolSatisfaction = st.selectbox('the Degree of parent satisfaction from school',['Good', 'Bad'])
    StudentAbsenceDays = st.selectbox('the number of absence days for each student',['Under-7', 'Above-7'])


    feat = np.array([gender, NationalITy, PlaceofBirth, StageID, GradeID,
           SectionID, Topic, Semester, Relation, raisedhands,
           VisITedResources, AnnouncementsView, Discussion,
           ParentAnsweringSurvey, ParentschoolSatisfaction,
           StudentAbsenceDays]).reshape(1,-1)
    cols =['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays']

    feat1 = pd.DataFrame(feat, columns=cols)

    return feat1
frame = data1() 
 
def prepare(df):
    enc_data = pd.DataFrame(encoder.transform(df[['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
       'SectionID', 'Topic', 'Semester', 'Relation',
       'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
       'StudentAbsenceDays']]).toarray())
    enc_data.columns = encoder.get_feature_names(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
       'SectionID', 'Topic', 'Semester', 'Relation',
       'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
       'StudentAbsenceDays'])
    df = df.join(enc_data)

    df_dropped = df.drop(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays'], axis=1)

    df_scaled = scaler.transform(df_dropped)
    df_final = pd.DataFrame(df_scaled, columns=df_dropped.columns)
    return df_final


 
frame2= prepare(frame)
 
if st.button('predict'):
    
    #frame2= prepare(frame)
    pred = model.predict(frame2)
    if pred[0] == 'M':
        st.write('Middle-Level student')
    if pred[0] == 'L':
        st.write('Low-Level student')
    if pred[0] == 'H':
         st.write('High-Level student')
 
 
st.markdown(
"""
<style>
body {
    background-color: #d4c8a7;
}
.secondaryBackgroundColor {
    background-color: #d8d6c7;
}
.textColor {
    color: #292b35;
}
</style>
""",
unsafe_allow_html=True)



