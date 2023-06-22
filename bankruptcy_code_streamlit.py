import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

#To stop deprecation warnings from matplotlib.pyplot
st.set_option('deprecation.showPyplotGlobalUse',False)

st.header('Bankruptcy Prediction App')
st.write('''
This app predicts whether the **Company goes bankrupt or not**!
''')

# #### **Importing data**

df = pd.read_csv('bankruptcy-prevention.csv',sep=";")

df_1 = df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_1[' class'] = LE.fit_transform(df_1[' class'])

X= df_1.iloc[:,0:6]
Y = df_1.iloc[:,6]

#sidebar
#Header of specify input parameters
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    industrial_risk=st.sidebar.selectbox('INDUATRIAL_RISK',('0.0','0.5','1.0'))
    management_risk=st.sidebar.selectbox('MANAGEMENT_RISK',('0.0','0.5','1.0'))
    financial_flexibility=st.sidebar.selectbox('FINANCIAL_FLEXIBILITY',('0.0','0.5','1.0'))
    credibility=st.sidebar.selectbox('CREDIBILITY',('0.0','0.5','1.0'))
    competitiveness=st.sidebar.selectbox('COMPETITIVENESS',('0.0','0.5','1.0'))
    operating_risk=st.sidebar.selectbox('OPERATING_RISK',('0.0','0.5','1.0'))
    model = st.sidebar.selectbox('Model',('Logistic Regression','Naive Bayes Classifier','K- Nearest Neighbors Classifier','SVC','Decision Tree Classifier','Bagging Classifier','Random Forest Classifier'))

    data = {'industrial_risk':industrial_risk,
            ' management_risk':management_risk,
            ' financial_flexibility': financial_flexibility,
            ' credibility':credibility,
            ' competitiveness':competitiveness,
            ' operating_risk': operating_risk}
    features = pd.DataFrame(data,index = [0])
    return (features,model)

df_uif,model = user_input_features()
st.subheader('User Input parameters')
st.write(df_uif,model)


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=3)
dbscan.fit(X)
y=dbscan.labels_
y=pd.DataFrame(y)


c1 = pd.DataFrame(dbscan.labels_,columns=["clusters"])
clustered = pd.concat([df_1,c1],axis = 1)

noisedata = clustered[clustered['clusters']==-1]
df_2 = clustered[clustered['clusters']>=0]

if model=='Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    LR=LogisticRegression()
    LR.fit(X,Y)
    Prediction = LR.predict(df_uif)
    
elif model == 'Naive Bayes Classifier':
    from sklearn.naive_bayes import MultinomialNB
    MNB = MultinomialNB()
    MNB.fit(X,Y)
    Prediction = MNB.predict(df_uif)

elif model =='K- Nearest Neighbors Classifier':
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=7, p=2)
    KNN.fit(X,Y)
    Prediction = KNN.predict(df_uif)
elif model=='SVC':
    from sklearn.svm import SVC
    svm = SVC(kernel='poly',degree=6)
    svm.fit(X,Y)
    Prediction = svm.predict(df_uif)
elif model=='Decision Tree Classifier':
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=(4)) 
    dt.fit(X, Y)
    Prediction = dt.predict(df_uif)
elif model =='Bagging Classifier':
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=(4)) 
    bag = BaggingClassifier(base_estimator=(dt),n_estimators=100,max_samples=0.7,max_features=0.7,random_state=4) 
    bag.fit(X, Y)
    Prediction = bag.predict(df_uif)
else:
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=100,max_samples=0.9,max_features=0.5,random_state=5,max_depth=(3)) 
    RFC.fit(X,Y)
    Prediction = RFC.predict(df_uif)

st.header('Prediction')
if Prediction == 0:
    st.write('Bankrupt')
else:   
    st.write('Non Bankrupt')




