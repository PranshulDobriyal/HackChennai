#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tkinter as tk
from time import sleep

height = 800
width = 1500


class symp:
    def __init__(self,a):
        import pandas as pd
        import numpy as np

        df = pd.read_csv("training.csv")

        l1=['backpain','constipation','abdominalpain','diarrhoea','mildfever','yellowurine',
        'yellowingofeyes','acuteliverfailure','fluidoverload','swellingofstomach',
        'swelledlymphnodes','malaise','blurredanddistortedvision','phlegm','throatirritation',
        'rednessofeyes','sinuspressure','runnynose','congestion','chestpain','weaknessinlimbs',
        'fastheartrate','painduringbowelmovements','paininanalregion','bloodystool',
        'irritationinanus','neckpain','dizziness','cramps','bruising','obesity','swollenlegs',
        'swollenbloodvessels','puffyfaceandeyes','enlargedthyroid','brittlenails',
        'swollenextremeties','excessivehunger','extramaritalcontacts','dryingandtinglinglips',
        'slurredspeech','kneepain','hipjointpain','muscleweakness','stiffneck','swellingjoints',
        'movementstiffness','spinningmovements','lossofbalance','unsteadiness',
        'weaknessofonebodyside','lossofsmell','bladderdiscomfort','foulsmellofurine',
        'continuousfeelofurine','passageofgases','internalitching','toxiclook(typhos)',
        'depression','irritability','musclepain','alteredsensorium','redspotsoverbody','bellypain',
        'abnormalmenstruation','dischromicpatches','wateringfromeyes','increasedappetite','polyuria','familyhistory','mucoidsputum',
        'rustysputum','lackofconcentration','visualdisturbances','receivingbloodtransfusion',
        'receivingunsterileinjections','coma','stomachbleeding','distentionofabdomen',
        'historyofalcoholconsumption','fluidoverload','bloodinsputum','prominentveinsoncalf',
        'palpitations','painfulwalking','pusfilledpimples','blackheads','scurring','skinpeeling',
        'silverlikedusting','smalldentsinnails','inflammatorynails','blister','redsorearoundnose',
        'yellowcrustooze']


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


        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt


        X= df[l1]
        y = df[["prognosis"]]
        np.ravel(y)


      


        dictionary = {0:'Fungal infection',1:'Allergy',2:'GERD',3:'Chronic cholestasis',4:'Drug Reaction',
            5:'Peptic ulcer diseae',6:'AIDS',7:'Diabetes ',8:'Gastroenteritis',9:'Bronchial Asthma',10:'Hypertension ',
            11:'Migraine',12:'Cervical spondylosis',
            13:'Paralysis (brain hemorrhage)',14:'Jaundice',15:'Malaria',16:'Chicken pox',17:'Dengue',18:'Typhoid',19:'hepatitis A',
            20:'Hepatitis B',21:'Hepatitis C',22:'Hepatitis D',23:'Hepatitis E',24:'Alcoholic hepatitis',25:'Tuberculosis',
            26:'Common Cold',27:'Pneumonia',28:'Dimorphic hemmorhoids(piles)',29:'Heart attack',30:'Varicose veins',31:'Hypothyroidism',
            32:'Hyperthyroidism',33:'Hypoglycemia',34:'Osteoarthristis',35:'Arthritis',
            36:'(vertigo) Paroymsal  Positional Vertigo',37:'Acne',38:'Urinary tract infection',39:'Psoriasis',
            40:'Impetigo'}


      


        from sklearn import tree
        clf3 = tree.DecisionTreeClassifier() 
        clf3 = clf3.fit(X,y)
        a = list(str(a).split(','))
        for x in range(len(a)):
            a[x] = a[x].lower()
        l2 = []
        for x in range(len(l1)):
            if l1[x] in a:
                l2.append(1)
            else:
                l2.append(0)
        predict = clf3.predict([l2])
        predicted=dictionary.get(predict[0])
        plabel = tk.Label(mainframe,text = 'DISEASE: ',bg = '#2e2a2a',fg = 'white',font = ('Arial',20))
        plabel.place(relx = .4,rely = .6,relheight = .05,relwidth = .15)
        outputlabel = tk.Label(mainframe,text = predicted,bg = '#2e2a2a',fg = 'white',font = ('Arial',20))
        outputlabel.place(relx = .55, rely = .6, relheight = .05, relwidth = .3)
            
            
root = tk.Tk()

root.title("DISEASE PREDICTOR")

pic = tk.PhotoImage(file = "back.PNG")
canvas = tk.Canvas(root,height = height, width = width)
canvas.pack()

mainframe = tk.Frame(canvas, bg = 'white')
mainframe.place(relheight = 1, relwidth = 1)

label = tk.Label(mainframe, image = pic)
label.place(relheight=1, relwidth=1)

listbox = tk.Listbox(mainframe,fg = 'red', font = ('Arial',20))
listbox.insert(1,'backpain')
listbox.insert(2,'constipation')
listbox.insert(3,'abdominalpain')
listbox.insert(4,'diarrhoea')
listbox.insert(5,'mildfever')
listbox.insert(6,'yellowurine')
listbox.insert(7,'yellowingofeyes')
listbox.insert(8,'acuteliverfailure')
listbox.insert(9,'fluidoverload')
listbox.insert(10,'swellingofstomach')
listbox.insert(11,'swelledlymphnodes')
listbox.insert(12,'malaise')
listbox.insert(13,'blurredanddistortedvision')
listbox.insert(14,'phlegm')
listbox.insert(15,'throatirritation')
listbox.insert(16,'rednessofeyes')
listbox.insert(17,'sinuspressure')
listbox.insert(18,'runnynose')
listbox.insert(19,'congestion')
listbox.insert(20,'chestpain')
listbox.insert(21,'weaknessinlimbs')
listbox.insert(22,'fastheartrate')
listbox.insert(23,'painduringbowelmovements')
listbox.insert(24,'paininanalregion')
listbox.insert(25,'bloodystool')
listbox.insert(26,'irritationinanus')
listbox.insert(27,'neckpain')
listbox.insert(28,'dizziness')
listbox.insert(29,'cramps')
listbox.insert(30,'bruising')
listbox.insert(31,'obesity')
listbox.insert(32,'swollenlegs')
listbox.insert(33,'swollenbloodvessels')
listbox.insert(34,'puffyfaceandeyes')
listbox.insert(35,'enlargedthyroid')
listbox.insert(36,'brittlenails')
listbox.insert(37,'swollenextremeties')
listbox.insert(38,'excessivehunger')
listbox.insert(39,'extramaritalcontacts')
listbox.insert(40,'dryingandtinglinglips')
listbox.insert(41,'slurredspeech')
listbox.insert(42,'kneepain')
listbox.insert(43,'hipjointpain')
listbox.insert(44,'muscleweakness')
listbox.insert(45,'stiffneck')
listbox.insert(46,'swellingjoints')
listbox.insert(47,'chestpain')
listbox.insert(48,'movementstiffness')
listbox.insert(49,'spinningmovements')
listbox.insert(50,'lossofbalance')
listbox.insert(51,'unsteadiness')
listbox.insert(52,'weaknessofonebodyside')
listbox.insert(53,'lossofsmell')
listbox.insert(54,'bladderdiscomfort')
listbox.insert(55,'foulsmellofurine')
listbox.insert(56,'continuousfeelofurine')
listbox.insert(57,'passageofgases')
listbox.insert(58,'internalitching')
listbox.insert(59,'toxiclook(typhos)')
listbox.insert(60,'depression')
listbox.insert(61,'irritability')
listbox.insert(62,'musclepain')
listbox.insert(63,'alteredsensorium')
listbox.insert(64,'redspotsoverbody')
listbox.insert(65,'bellypain')
listbox.insert(66,'abnormalmenstruation')
listbox.insert(67,'dischromicpatches')
listbox.insert(68,'wateringfromeyes')
listbox.insert(69,'increasedappetite')
listbox.insert(70,'polyuria')
listbox.insert(71,'familyhistory')
listbox.insert(72,'mucoidsputum')
listbox.insert(73,'rustysputum')
listbox.insert(74,'lackofconcentration')
listbox.insert(75,'visualdisturbances')
listbox.insert(76,'receivingbloodtransfusion')
listbox.insert(77,'receivingunsterileinjections')
listbox.insert(78,'coma')
listbox.insert(79,'stomachbleeding')
listbox.insert(80,'distentionofabdomen')
listbox.insert(81,'historyofalcoholconsumption')
listbox.insert(82,'fluidoverload')
listbox.insert(83,'bloodinsputum')
listbox.insert(84,'prominentveinsoncalf')
listbox.insert(85,'palpitations')
listbox.insert(86,'painfulwalking')
listbox.insert(87,'blackheads')
listbox.insert(88,'scurring')
listbox.insert(89,'skinpeeling')
listbox.insert(90,'silverlikedusting')
listbox.insert(91,'smalldentsinnails')
listbox.insert(92,'redsorearoundnose')
listbox.insert(93,'inflammatorynails')
listbox.insert(94,'blister')
listbox.insert(95,'yellowcrustooze')

listbox.place(relx=.05,rely = .3,relheight = .5,relwidth = .3)
llabel = tk.Label(mainframe,text = 'List Of Symptoms(scroll to see more)', font = ('Arial',20),bg = '#2e2a2a',fg = 'white')
llabel.place(relx = .05,rely = .82,relheight = .05,relwidth = .3)

klabel = tk.Label(mainframe, text = 'SYMPTOMS: ',font = ('Arial',20),bg = '#2e2a2a',fg = 'white')
klabel.place(relx = .4,rely = .3,relwidth = .15,relheight = .05)

sbox = tk.Entry(mainframe, font = ('Gabriola',20))
sbox.place(relx = .55,rely = .3,relwidth = .3, relheight = .05)

note = tk.Label(mainframe,text = 'NOTE -> Enter symptoms without space and if multiple symptoms seperate by comma',font = ('Arial',20),bg = '#2e2a2a',fg = 'white')
note.place(relx = .2,rely = .2,relwidth = .7,relheight = .05)

sbutton = tk.Button(mainframe,text = 'Search',bg = '#2e2a2a',fg = 'white',font = ('Arial',20),command = lambda : symp(sbox.get()))
sbutton.place(relx = .75,rely = .4,relwidth = .1,relheight = .05)
root.mainloop()


# In[ ]:




