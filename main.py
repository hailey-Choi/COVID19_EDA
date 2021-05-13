import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import math
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash.dependencies as dependencies
import matplotlib.pyplot as plt
from sklearn.impute   import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import statsmodels.api as sm
from itertools import combinations
import flask
import html2text
import glob
import os
import json

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#------------------------------------------------------------------------------------------------------------------------
#* Before Running this app
"""
1) Make sure that you have all the files in the same working directory as this main.py
2) The required Python version is 3.7.X
3) It make take some time to execute the programme file
4) Once you run this app, the app runs on http://127.0.0.1:4544/
"""
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Data Loading
globe_viz_data = pd.read_csv("./covid_19_data.csv")
Clinical_viz_data = pd.read_csv('./covid_analytics_clinical_data.csv')
ICU_pred_viz_data = pd.read_csv('./Kaggle_Sirio_Libanes_ICU_Prediction.csv')

#* Images
image_directory = './'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
# Globe --> Visualizing Current Covid Cases Globally
def globe():
    cnf = '#393e46'
    dth = '#ff2e63'
    rec = '#21bf73'
    act = '#fe9801'
    grp = globe_viz_data.groupby(['Country/Region'])[['Confirmed', 'Deaths', 'Recovered']].max()
    grp = grp.reset_index()
    grp[['Country']] =  grp[['Country/Region']]

    fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                        color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths],projection="orthographic",
                        color_continuous_scale='Plasma')
    fig.update(layout_coloraxis_showscale=True)
    return fig
fig_globe = globe()
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Function for callback for globe
def timeSeries(df, country):
    df = df[df["Country/Region"]==country]
    df= df.groupby("ObservationDate").sum()
    df = df.reset_index()
    for i in range(df.shape[0]):
        df["ObservationDate"][i] = datetime.strptime(df["ObservationDate"][i], '%m/%d/%Y').date()
    df = df.sort_values(by=['ObservationDate'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ObservationDate"], y=df["Confirmed"],
                        mode='lines',
                        name='Confirmed Cases',
                        line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=df["ObservationDate"], y=df["Deaths"],
                        mode='lines',
                        name='Deaths',line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=df["ObservationDate"], y=df["Recovered"],
                        mode='lines', name='Recovered Cases',
                        line=dict(color='green', width=4)))
    fig.update_layout(title='Cumulative COVID-19 Statistics in '+country,
                       xaxis_title='Date',
                       yaxis_title='Number of Cases')
    return fig
fig_ts = timeSeries(globe_viz_data, "Hong Kong")
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* ICU Overall Statistics
def ICU_preprocessing(df):
    is_above_12 = df['WINDOW'] == 'ABOVE_12'
    df_new = df[is_above_12]
    return df_new
df_ICU_new = ICU_preprocessing(ICU_pred_viz_data)

def ICU_age(df) :
    demo_lst = [i for i in df.columns if "AGE_" in i]
    demo_lst.append("GENDER")

    age = pd.DataFrame(df[[demo_lst[0],"ICU"]])
    age = age.loc[age['ICU'] == 1]
    age.reset_index(drop=True,inplace=True)
    del age["ICU"]
    age = pd.DataFrame(age["AGE_ABOVE65"].value_counts())
    age["Age"] = ["Above 65","Equal to or below 65"]

    import plotly.graph_objects as go

    colors = ['lightpink','lightblue']

    fig = go.Figure(data=[go.Bar(
        x=age["Age"],
        y=age["AGE_ABOVE65"],
        text=age["AGE_ABOVE65"],
        textposition='auto',
        width=0.5,
        marker_color=colors
    )])
    fig.update_traces(marker_line_color='rgb(8,48,107)',marker_line_width=1.5,
    opacity=1)
    return fig
fig_ICU_age = ICU_age(df_ICU_new)

def ICU_age_percentile(df) :
    demo_lst = [i for i in df.columns if "AGE_" in i]
    demo_lst.append("GENDER")

    age_per = pd.DataFrame(df[[demo_lst[1],"ICU"]])
    age_per = age_per.loc[age_per['ICU'] == 1]
    age_per.reset_index(drop=True,inplace=True)
    count = age_per["AGE_PERCENTIL"].value_counts()
    count = pd.DataFrame(count)
    colors = px.colors.sequential.haline
    fig = go.Figure(data=[go.Bar(
        x=count.index,
        y=count["AGE_PERCENTIL"],
        text=count["AGE_PERCENTIL"],
        textposition='auto',
        width=0.5,
        marker_color=colors)])
    return fig
fig_ICU_age_percentile = ICU_age_percentile(df_ICU_new)

def gender(df) :
    demo_lst = [i for i in df.columns if "AGE_" in i]
    demo_lst.append("GENDER")

    gender = pd.DataFrame(df[[demo_lst[2],"ICU"]])
    gender = gender.loc[gender['ICU'] == 1]
    gender.reset_index(drop=True,inplace=True)
    gender["GENDER"].replace({0: "Male", 1: "Female"}, inplace=True)
    gender_count=gender["GENDER"].value_counts()
    gender_count = pd.DataFrame(gender_count)

    colors = ['lightblue','lightpink']

    fig = go.Figure(data=[go.Bar(
        x=gender_count.index,
        y=gender_count["GENDER"],
        text=gender_count["GENDER"],
        textposition='auto',
        width=0.5,
        marker_color=colors)])
    fig.update_traces(marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=1)

    return fig
fig_ICU_gender = gender(df_ICU_new)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Vital Signs
def is_one_to_one(df1, cols):
    if len(cols) == 1:
        return True
        # You can define you own rules for 1 column check, Or forbid it

    # MAIN THINGs: for 2 or more columns check!
    res = df1.groupby(cols).count()
    uniqueness = [res.index.get_level_values(i).is_unique
                for i in range(res.index.nlevels)]
    return all(uniqueness)

#PLOT VITAL SIGN
def vitalsign(df):
    vitalSigns_lst = df.iloc[:,193:-2].columns.tolist()

    # Getting combinations of all the colmns
    combos = list(combinations(df.columns,2))
    # Running to see if any of them are identical
    identical_cols = []
    for col in np.arange(0,len(combos),1):
        x = [combos[col][0],combos[col][1]]
        if is_one_to_one(df,x) == True:
             identical_cols.append(combos[col][0])
    all_cols = [x for x in df.columns if x not in identical_cols]
    df = df.loc[:, all_cols]

    # missing values
    df = df\
        .sort_values(by=['PATIENT_VISIT_IDENTIFIER', 'WINDOW'])\
        .groupby('PATIENT_VISIT_IDENTIFIER', as_index=False)\
        .fillna(method='ffill')\
        .fillna(method='bfill')
    #df = df.set_index('PATIENT_VISIT_IDENTIFIER')
    w02 = df[df.WINDOW == '0-2']

    #PLOT
    fig = px.box(pd.melt(w02[vitalSigns_lst]), x="value", y="variable",
    points="outliers", title="Variables Boxplots")
    fig.update_layout(yaxis = dict( tickfont = dict(size=4))) 
    return fig
fig_vital = vitalsign(ICU_pred_viz_data)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* For Logistic Regression
def lr_preprocessing(ICU_pred_viz_data):
    ICU_pred_viz_data['AGE_PERCENTIL'] = ICU_pred_viz_data['AGE_PERCENTIL'].str.replace('Above ','').str.extract(r'(.+?)th')
    selected10 = ["RESPIRATORY_RATE_MEAN","AGE_PERCENTIL","BLOODPRESSURE_DIASTOLIC_MEAN","BLOODPRESSURE_SISTOLIC_MEAN",
        "RESPIRATORY_RATE_MAX","BLOODPRESSURE_DIASTOLIC_MAX","BLOODPRESSURE_SISTOLIC_MAX","RESPIRATORY_RATE_DIFF",
        "OXYGEN_SATURATION_MAX","HEART_RATE_DIFF_REL"]
    df_ICU = ICU_pred_viz_data[selected10]
    mean_impute  = SimpleImputer(strategy='mean')
    impute = mean_impute.fit_transform(df_ICU)
    df_lr = pd.DataFrame(impute, columns = df_ICU.columns)
    return df_lr
df_lr = lr_preprocessing(ICU_pred_viz_data)

def logreg_plot(df, df_lr, feature):
    X= df_lr[feature].values.reshape(-1,1)
    y= df["ICU"].values
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, y)
    b0 = clf.intercept_[0]
    b1 = clf.coef_[0][0]

    xlist = np.linspace(X.min(),X.max(),100)
    ylist=[]
    for i in range(len(xlist)):
        X_temp=xlist[i]*b1+b0
        y_temp = 1 / (1 + math.exp(-X_temp))
        ylist.append(y_temp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xlist, y=ylist,
                        mode='markers+lines',
                        name='Logistic Regression Line'))
    fig.add_trace(go.Scatter(x=df_lr[feature], y=y,
                        mode='markers',
                        name='Data'))
    fig.update_layout(yaxis_range=[-0.25,1.25])
    fig.update_layout(title="Logistic Regression Model",
                     xaxis_title=feature,
                     yaxis_title="ICU Admission")
    fig.update_layout(yaxis = dict(tickmode = 'array', tickvals=[0,1], ticktext=["False", "True"]))
    return fig

html_2_text = html2text.HTML2Text()
def regression_summary(df_lr, df, feature):
    x= df_lr[[feature]]
    y= df[["ICU"]]
    log_reg = sm.Logit(y, x).fit()
    html = log_reg.summary().as_html()
    result = html_2_text.handle(html)
    return result
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Treatment --> Visualizing what kinds of treatments were mostly used in the studies
def treatment(df):
    # 1. PREPROCESSING
    df_new = df[df["Peer-Reviewed? (As of Date Added)"]=="Yes"]
    #df_new = df_new[df_new["Study Pop Size (N)"]>30]
    df_new = df_new[df_new['Positive/negative cases']=="Positive only"]
    #df_new = df_new[df_new['Mortality'].notna()]
    df_new = df_new.reset_index(drop=True)
    # 2. TREATMENTS
    trt = ["Antibiotic", "Antiviral (Any)", "Uses Kaletra (lopinavir–ritonavir)",
           "Uses Favipiravir", "Uses Tamiflu (oseltamivir)","Uses Remdesivir",
            "Uses umif+lop-rit","Uses Arbidol (umifenovir)","Uses hydroxychloroquine and/or chloroquine",
            "Corticosteroid (including Glucocorticoid, Methylprednisolone)",
            "Intravenous immunoglobin","Nasal cannula","High-flow nasal cannula oxygen therapy",
            "Oxygen therapy","Noninvasive mechanical ventilation","Invasive mechanical ventilation",
            "ECMO","Renal replacement therapy","Interferon Alpha-1b","Thymalfasin and/or Thymosin","Sepsis"]
    df_trt = df_new[trt]
    for i in range(21):
        df_trt.iloc[:,i] = df_trt.iloc[:,i].str.replace('%', '').astype(float)
    df_trt.fillna(0, inplace=True)
    #df_trt = df_trt.reset_index()
    # 3. Bar chart
    df_size = pd.concat([df_new.iloc[:,2:5],df_trt], axis=1)
    rows = []
    for i in range(3,24):
        rows.append(np.round(df_size.iloc[:,2] * df_size.iloc[:,i] / 100).astype(int).values)
    df_sizeFin = pd.DataFrame(rows)
    df_sizeFin= df_sizeFin.T
    df_sizeFin.columns = list(df_size.columns)[3:]
    df_sizeFin = pd.concat([df_new.iloc[:,2:5],df_sizeFin], axis=1)
    trt_sum=pd.DataFrame(df_sizeFin.iloc[:,3:].sum(axis=0), columns=["Total number of patients"])
    trt_sum = trt_sum.sort_values(by=['Total number of patients'],ascending=False)
    trt_sum = trt_sum.reset_index()
    trt_sum.columns = ["Treatment Description","Total number of patients"]
    # Rename
    trt_name = ['Antiviral', 'Antibiotic', 'IV', 'Corticosteroid', 'Kaletra',
     'IFN Alpha-1b', 'Oxygen therapy', 'NC','HFNC oxygen therapy', 'Arbidol', 'IVIg', 'NIV', 'Tamiflu',
     'umif+lop-rit', 'Sepsis', 'HCQ and/or CQ','RRT', 'Remdesivir', 'ECMO','Thymalfasin', 'Favipiravir']
    trt_sum["Treatment"]=trt_name
    fig = px.bar(trt_sum, y="Treatment", x="Total number of patients"
            ,color_discrete_sequence =['green']*len(df),
            hover_data= ["Treatment Description"])
    fig = fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    return fig

fig_treatment = treatment(Clinical_viz_data)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* PCA --> PCA on above data
def pca(df):
        # 1. PREPROCESSING
        df_new = df[df["Peer-Reviewed? (As of Date Added)"]=="Yes"]
        #df_new = df_new[df_new["Study Pop Size (N)"]>30]
        df_new = df_new[df_new['Positive/negative cases']=="Positive only"]
        #df_new = df_new[df_new['Mortality'].notna()]
        df_new = df_new.reset_index(drop=True)
        # 2. TREATMENTS
        trt = ["ID","Antibiotic", "Antiviral (Any)", "Uses Kaletra (lopinavir–ritonavir)",
               "Uses Favipiravir", "Uses Tamiflu (oseltamivir)",
                "Uses Remdesivir", "Uses umif+lop-rit", "Uses Arbidol (umifenovir)",
                "Uses hydroxychloroquine and/or chloroquine",
                "Corticosteroid (including Glucocorticoid, Methylprednisolone)",
                "Intravenous immunoglobin", "Nasal cannula", "High-flow nasal cannula oxygen therapy",
                "Oxygen therapy", "Noninvasive mechanical ventilation",
                "Invasive mechanical ventilation", "ECMO", "Renal replacement therapy",
                "Interferon Alpha-1b", "Thymalfasin and/or Thymosin", "Sepsis"]
        df_trt = df_new[trt]
        for i in range(21):
            df_trt.iloc[:,i+1] = df_trt.iloc[:,i+1].str.replace('%', '').astype(float)

        df_trt.fillna(0, inplace=True)
        #df_trt = df_trt.reset_index()

        # 3. PCA & k-means clustering
        np.random.seed(2021)
        pca = PCA(n_components=2, random_state=0)
        PCs = pca.fit_transform(df_trt.iloc[:,1:].to_numpy())

        kmeans = KMeans(n_clusters=3, random_state=0).fit(df_trt.iloc[:,1:].to_numpy())
        cluster = kmeans.labels_

        df_PC = pd.DataFrame(PCs, columns=["PC1","PC2"])
        df_PC["Cluster"]=cluster
        df_PC["Cluster"] = df_PC["Cluster"].astype(str)
        df_final = pd.concat([df_trt, df_PC], axis=1)

        fig=px.scatter(df_final, x="PC1", y="PC2", color="Cluster", hover_data=
        ["ID","Noninvasive mechanical ventilation","Invasive mechanical ventilation"
         ,"Antibiotic", "Antiviral (Any)"])
        return fig
fig_PCA = pca(Clinical_viz_data)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Layout
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

tabs_styles = {'height': '44px','align-items': 'center','font-family':'Tahoma'}
tab_style = {'borderBottom': '1px solid #d6d6d6','padding': '6px','fontWeight': 'bold',
    'border-radius': '15px','background-color': '#F2F2F2','box-shadow': '4px 4px 4px 4px lightgrey','font-family':'Tahoma'}
tab_selected_style = {'borderTop': '1px solid #d6d6d6','borderBottom': '1px solid #d6d6d6','backgroundColor': '#646464',
    'color': 'white','padding': '6px','border-radius': '15px','font-family':'Tahoma'}

app.layout = html.Div(children=[
    html.H1(
        'Holistic Visual Analysis of COVID-19 Treatments',
        style={'margin-bottom':'40px','margin-top':'40px',"text-align": "center",'font-family':'Tahoma'}),
    html.H5(
        'Team Covid19 Treatments',
        style={'margin-bottom':'30px', "text-align": "center",'font-family':'Tahoma'}),
    html.Div(id='main-tab', children=[
        dcc.Tabs(id="main_tabs", value='main_tab-1', children=[
            dcc.Tab(label='Section 1', value='main_tab-1',style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label='Section 2', value='main_tab-2',style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label='Section 3', value='main_tab-3',style = tab_style, selected_style = tab_selected_style)],style = tabs_styles),
        html.Div(id='result',children=[])
    ])
])
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* [MAIN LAYOUT] Callback for main tab
@app.callback(
    dependencies.Output('result', 'children'),
    [dependencies.Input('main_tabs', 'value')]
)
def figure_main(value):
    if value == 'main_tab-1':
        return html.Div(id='section1', children=[
            html.H3(children='Section 1: Introduction',
            style={'margin-bottom':'30px','margin-top':'30px',"text-align": "center",'font-family':'Tahoma'}),
            dcc.Markdown("""
            Coronavirus disease 2019 (COVID-19) is a contagious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) 
            and the disease has since spread worldwide, leading to an ongoing pandemic. Section 1 offers an interactive map plot that allows you to click on the 
            desired region or country and provides the cumulative COVID-19 statistics in the corresponding area. This analysis is based on the time series 
            COVID-19 dataset by Johns Hopkins University Center for Systems and Science and Engineering (JHU CSSE), 
            supported by ESRI Living Atlas Team and the Johns Hopkins University Applied Physics Lab (JHU APL). 

            [Dataset Available HERE](https://github.com/CSSEGISandData/COVID-19)
            """),
            html.Div(children=[
                html.Div(children=[
                    html.H5(children='Covid19 Global Statistics (Last Update: 2021 Feb 27)',
                    style={'margin-top':'30px',"text-align": "center",'font-family':'Tahoma'}),
                    dcc.Graph(id='globe', figure=fig_globe),]),
                ]),
                # html.Div(id='debugging',children=[]),
                html.Div(children=[
                    dcc.Graph(id='time_series', figure=fig_ts),
                ],),
            ],),
             
    elif value == 'main_tab-2':
        return html.Div(id='section2', children=[
            html.H3(children='Section 2.1: Covid-19 ICU Predictive Analysis',
            style={'margin-bottom':'30px','margin-top':'30px',"text-align": "center",'font-family':'Tahoma'}),
            dcc.Markdown("""
            Exponentially increasing COVID-19 cases around the world is overwhelming medical systems, causing lack of hospitalization capacity 
            for Intensive Care Unit patients, professionals and medical appliances and resources. Section 2.1 provides healthcare professionals with better understanding of 
            relationship between demographical and clinical data of patients with ICU admission. In this section, you can have a view of overall information statistics of ICU patients, 
            vital sign distributions and feature contribution to prediction of ICU admission using LightGBM model.

            Analysis in Section 2 is built upon the public dataset provided by Kaggle which contains anonymized data from Hospital Sírio-Libanês, São Paulo and Brasilia. 
            
            [Dataset Available HERE]( https://www.kaggle.com/S%C3%ADrio-Libanes/covid19)
            """),
            html.Div(children=[
                html.Label('Select Chart :'),
                dcc.Dropdown(
                    id='sec2-chart',
                    options=[{'label': 'Overall Statistics', 'value': 'overall'},
                            {'label': 'Variables Boxplots', 'value':'vital'},
                            {'label': 'Feature Importance', 'value':'FI'}],
                    value='overall'),
                html.Div(id='Dropdown-contents', children=[]),]),
            html.H3(children='Section 2.2: Logistic Regression Analysis',
            style={'margin-bottom':'30px','margin-top':'50px',"text-align": "center",'font-family':'Tahoma'}),    
            dcc.Markdown("""
                Logistic regression is a widely used statistical model that in its basic form uses a logistic function to model a binary dependent variable. 
                Section 2.2 provides the logistic regression graph of one-to-one relationship of the desired feature and ICU admission and the corresponding summary of fitted model. 
                The ten features include demographic and medical data of patients, selected based on the LGBM feature importance in section 2.1. 
                Result of model summary includes coefficient of logistic regression line, standard error of the coefficient, z-value (the regression coefficient divided by standard error), Pr(>|z|) and confidence intervals. 
                """),
            html.Label('Select Variable :'),
            dcc.Dropdown(
                id = 'select-variable',
                options=[{'label': var, 'value':var} for var in df_lr.columns],
                value = 'RESPIRATORY_RATE_MEAN'
            ),
            dcc.Graph(id='logreg', figure={}),
            html.H5('Regression Summary',
            style={'font-family':'Tahoma'}),
            html.Div(id='reg_summary', children=[]),
        ])

    elif value == 'main_tab-3':
        return html.Div(id='section3',children=[
            html.Div(children=[
                html.H3(children='Section 3.1: Visual Analysis on Covid19 Clinical Studies',
                style={'margin-bottom':'30px','margin-top':'30px',"text-align": "center",'font-family':'Tahoma'}),
                dcc.Markdown("""
                A great number of clinical trials and researches around the world are tackling the COVID-19 pandemic and thereby we have annalyzed the current trend of clinical studies, 
                aiming to contribute to effective decision making of global level researches. Section 3.1 provides cumulative frequency of treatments used in the clinical studies around the world and an 
                interactive Principal Component Analysis (PCA) of clinical studies which some information of each study and their corresponding cluster based on their veiled nature. 
                
                **You can have a more detailed view of the research information if you use 'Box Select' to select a group of studies in the interactive plot.**

                The dataset used in this analysis aggregates data from over 160 published clinical studies and preprints released between December 2019 and April 2020. It is provided by MIT Operations Research Center, led by Professor Dimitris Bertsimas. 
                
                [Dataset Available HERE]( https://covidanalytics.io/home)
                """),
                html.Div(children=[
                    html.H5(children='Treatments used in the Clinical Studies\t\t',
                    style={"text-align": "center",'font-family':'Tahoma'}),
                    dcc.Graph(id='treatment', figure=fig_treatment, style={'height': '550px', 'width': '700px'}),
                ], style={'display': 'inline-block'}),
                html.Div(children=[
                    html.H5(children='PCA of the Clinical Studies',
                    style={"text-align": "center",'font-family':'Tahoma'}),
                    dcc.Graph(id='pca', figure=fig_PCA, style={'height': '550px', 'width': '700px'})
                ], style={'display': 'inline-block'}),
            ], style={'display': 'inline-block'}),
            html.Div(children=[
                html.H5('Datatable of Clinical Studies',
                style={'font-family':'Tahoma'}),
                # html.Div(id='debug',children=[]),
                dash_table.DataTable(
                    id = 'datatable',
                    columns = [{'name':i, 'id':i, 'deletable':True} for i in Clinical_viz_data.columns[:13]],
                    style_table = {'overflowX': 'scroll'},
                    style_header={'fontWeight': 'bold'},
                    style_cell={'height': '90','minWidth': '140px', 'width': '140px', 'maxWidth': '140px','whiteSpace': 'normal','textAlign': 'left'},
                    page_current= 0,
                    page_size= 5,
                    page_action='custom',
                    sort_action='custom',
                    sort_mode='multi',
                    sort_by=[])]),
            html.Div(children=[
                html.H3(children='Section 3.2: Meta-Analysis of Clinical Studies Using Hydroxychloroquine',
                style={'margin-bottom':'30px','margin-top':'80px',"text-align": "center",'font-family':'Tahoma'}),
                dcc.Markdown("""
                The COVID-19 pandemic has caused an ongoing research for possible treatments, with almost 700 clinical trials commenced in the first quarter of 2020 and one in five of these trials 
                target hydroxychloroquine (HCQ) or chloroquine (CQ). HCQ is a medication used to treat malaria or rheumatic diseases. Many studies focused on HCQ treatment primarily due to in vitro data, 
                immunomodulatory capacities, and the oral formulation and well-documented safety profiles. In this section, we present a meta-analysis of Randomized Controlled Trials to estimate the effects of 
                hydroxychloroquine on survival in COVID-19. The analysis includes a summary of result, a forest plot and a funnel plot.

                The dataset used in this analysis was shared within an article *Mortality outcomes with hydroxychloroquine and chloroquine in COVID-19 from an international collaborative meta-analysis of randomized trials*, 
                published by *Axfors, C., Schmitt, A.M., Janiaud, P.* in 2021. 
                
                [Dataset Available HERE](https://www.nature.com/articles/s41467-021-22446-z#citeas)
                """),
                dcc.Tabs(id="tabs", value='tab-1', children=[
                    dcc.Tab(label='Meta Analysis', value='tab-1'),
                    dcc.Tab(label='Forest Plot', value='tab-2'),
                    dcc.Tab(label='Funnel Plot', value='tab-3'),
                ]),
                html.Div(children=[
                    html.Img(id='image', style={'height':'60%', 'width':'60%'})
                ], style={'textAlign': 'center'})
            ])
        ])
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Section 1 - Select Country
@app.callback(
    dependencies.Output('time_series', 'figure'),
    [dependencies.Input('globe', 'clickData')]
)
def update_graph(clickData):
    df_ts = globe_viz_data
    country = clickData['points'][0]['location']
    return timeSeries(df_ts, country)

# @app.callback(
#     dependencies.Output('debugging', 'children'),
#     [dependencies.Input('globe', 'clickData')]
# )
# def debugging(clickData):
#     print(clickData['points'][0]['location'])
#     return html.H6('Output: {}'.format(clickData))
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Section 2 - Dropdown
@app.callback(
    dependencies.Output('Dropdown-contents', 'children'),
    [dependencies.Input('sec2-chart', 'value')]
)
def update_content(value):
    if value == 'vital':
        # return (html.H1('Please de-comment the lines for vital sign!!'))
        return html.Div(children=[
            dcc.Graph(figure=fig_vital, style={'height': '700px', 'width': '1200px'})
        ])
    elif value == 'overall':
        return html.Div(children=[
            html.Div(children=[
                html.H5(children='ICU Patients Age over 65\t\t',
                style={"margin_top": "50px", "text-align": "center",'font-family':'Tahoma'}),
                dcc.Graph(figure=fig_ICU_age, style={'height': '380px', 'width': '455px'}),
            ], style={'display': 'inline-block'}),
            html.Div(children=[
                html.H5(children='ICU Patients by Age Percentile\t\t',
                style={"margin_top": "50px","text-align": "center",'font-family':'Tahoma'}),
                dcc.Graph(figure=fig_ICU_age_percentile, style={'height': '380px', 'width': '455px'}),
            ], style={'display': 'inline-block'}),
            html.Div(children=[
                html.H5(children='ICU Patients by Gender\t\t',
                style={"margin_top": "50px","text-align": "center",'font-family':'Tahoma'}),
                dcc.Graph(figure=fig_ICU_gender, style={'height': '380px', 'width': '455px'}),
            ], style={'display': 'inline-block'}),
        ], style={'display': 'inline-block'})
    elif value == 'FI':
        return html.Img(id='image-dropdown', style={'height':'60%', 'width':'60%'})
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Section 2 - Dropdown - Image
@app.callback(
    dependencies.Output('image-dropdown', 'src'),
    [dependencies.Input('sec2-chart', 'value')]
)
def update_image(value):
    if value == 'FI':
        return static_image_route + 'average_contribution.png'
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for LogReg
@app.callback(
    dependencies.Output('logreg', 'figure'),
    [dependencies.Input('select-variable', 'value')]
)
def update_figure(value):
    df = ICU_pred_viz_data
    return logreg_plot(df, df_lr, value)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Regression Summary
@app.callback(
    dependencies.Output('reg_summary', 'children'),
    [dependencies.Input('select-variable', 'value')]
)
def update_reg_summary(value):
    df = ICU_pred_viz_data
    return dcc.Markdown('{}'.format(regression_summary(df_lr, df, value)))
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Section 3.1 - PCA --> Changing Datatable

@app.callback(
    dependencies.Output('datatable', 'data'),
    [dependencies.Input('pca', 'selectedData'),
    dependencies.Input('datatable', 'page_size'),
    dependencies.Input('datatable', 'page_current'),
    dependencies.Input('datatable', 'sort_by')]
)
def render_table(selectedData, page_size, page_current, sort_by):
    dff = Clinical_viz_data
    selectedpoints = dff.ID
    for selected_data in [selectedData]:
        if selected_data and selectedData['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                                            [p['customdata'][0] for p in selectedData['points']])
    flag = [True if item in selectedpoints else False for item in dff.ID]
    filtered_df = dff[flag]
    if len(sort_by):
        filtered_df = filtered_df.sort_values([col['column_id'] for col in sort_by],ascending=[col['direction'] == 'asc' for col in sort_by],inplace=False)
    page = page_current
    size = page_size
    return filtered_df.iloc[page * size: (page + 1) * size].to_dict('records')

# @app.callback(
#     dependencies.Output('debug', 'children'),
#     [dependencies.Input('pca', 'hoverData')]
# )
# def debugging(hoverData):
#     return html.H6('Output: {}'.format(hoverData))
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Callback for Section 3.2 - Tabs
@app.callback(
    dependencies.Output('image', 'src'),
    [dependencies.Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return static_image_route + 'summary.png'
    elif tab == 'tab-2':
        return static_image_route + 'forest_plot.png'
    elif tab == 'tab-3':
        return static_image_route + 'funnel_plot.png'

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#* Running the App
if __name__ == '__main__':
    app.run_server(debug=False,port=int(os.getenv('PORT', '4544'))) #debug=True
#------------------------------------------------------------------------------------------------------------------------