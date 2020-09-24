from flask import Flask,render_template,request

import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("training.csv")


# In[3]:





# In[4]:


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']


# In[5]:


disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']


# In[6]:


df=pd.read_csv("training.csv")
DF= pd.read_csv('training.csv', index_col='prognosis')
#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)
#df.head()



# In[7]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[8]:


X= df[l1]
y = df[["prognosis"]]
np.ravel(y)


# In[9]:


tr=pd.read_csv("testing.csv")

#Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)


# In[10]:


X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)



# In[11]:





# In[13]:


from sklearn import tree
clf3 = tree.DecisionTreeClassifier() 
clf3 = clf3.fit(X,y)
l2 = []
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_pred=clf3.predict(X_test)
for o in range(95):
    l2.append(0)
app=Flask(__name__,template_folder="templates")
@app.route("/",methods=['GET','POST'])
def main():
    errors=[]
    vals=list()
    results={}
    if request.method == 'POST':
        try:
            check=request.form.get("check")
            vals.append(check.split(","))
            for x in vals:
                x = x.replace(' ','_')
            
            for x in range(len(l1)):
                if l1[x] in vals:
                   l2[x]=1
                else:
                    l2[x]=0
        except:
            errors.append("LOL try again")
    return render_template('main.html',errors=errors,results=results)

dictionary = {0:'Fungal infection',1:'Allergy',2:'GERD',3:'Chronic cholestasis',4:'Drug Reaction',
    5:'Peptic ulcer diseae',6:'AIDS',7:'Diabetes ',8:'Gastroenteritis',9:'Bronchial Asthma',10:'Hypertension ',
    11:'Migraine',12:'Cervical spondylosis',
    13:'Paralysis (brain hemorrhage)',14:'Jaundice',15:'Malaria',16:'Chicken pox',17:'Dengue',18:'Typhoid',19:'hepatitis A',
    20:'Hepatitis B',21:'Hepatitis C',22:'Hepatitis D',23:'Hepatitis E',24:'Alcoholic hepatitis',25:'Tuberculosis',
    26:'Common Cold',27:'Pneumonia',28:'Dimorphic hemmorhoids(piles)',29:'Heart attack',30:'Varicose veins',31:'Hypothyroidism',
    32:'Hyperthyroidism',33:'Hypoglycemia',34:'Osteoarthristis',35:'Arthritis',
    36:'(vertigo) Paroymsal  Positional Vertigo',37:'Acne',38:'Urinary tract infection',39:'Psoriasis',
    40:'Impetigo'}



inputtest = [l2]
predict = clf3.predict(inputtest)
predicted=dictionary.get(predict[0])


@app.route('/your_disease',methods=['POST'])
def disease():
    # put return from model here
    return f"{predicted}"









if __name__=="__main__":
    app.run()
