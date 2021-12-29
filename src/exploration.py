

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
font = {'size': 12}
plt.rc('font', **font)


def funcPie(values):
    val = iter(values)
    return lambda pct: f"{pct:.1f}% ({next(val)})"

def explo_plot(datos):
        datos.describe()
        datos.info()
        datos.isna().sum()
        ax=sns.countplot(x='EXITUS', hue='EXITUS', palette='Set1', data=datos)
        ax.set(title='Patient survival state (died/survived)', xlabel='Died?', ylabel='Total')
        plt.show()
        datos['EXITUS'] = datos['EXITUS'].fillna('NO')
        datos['EXITUS'] = datos['EXITUS'].map({'NO': 0, 'YES': 1}).astype(int)
        corr_df = datos.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True)
        plt.show()
        intervaloEdad1=datos[datos['AGE']<50].pivot_table(values='AGE', index='EXITUS', aggfunc='count')
        intervaloEdad2=datos[(datos['AGE']>50) & (datos['AGE']<=80)].pivot_table(values='AGE', index='EXITUS', aggfunc='count')
        intervaloEdad3=datos[datos['AGE']>80].pivot_table(values='AGE', index='EXITUS', aggfunc='count')
        fig, ax=plt.subplots(1, 3, figsize = (16, 7))
        ax[0].pie(intervaloEdad1['AGE'].to_list(), labels=intervaloEdad1.index.to_list(), autopct=funcPie(intervaloEdad1['AGE'].to_list()), shadow=True, startangle=90)
        ax[0].axis('equal')
        ax[0].set_title('Under 50 years old')  
        ax[1].pie(intervaloEdad2['AGE'].to_list(), labels=intervaloEdad2.index.to_list(), autopct=funcPie(intervaloEdad2['AGE'].to_list()), shadow=True, startangle=90)
        ax[1].axis('equal') 
        ax[1].set_title('Between 50 and 80 years old')  
        ax[2].pie(intervaloEdad3['AGE'].to_list(), labels=intervaloEdad3.index.to_list(),autopct=funcPie(intervaloEdad3['AGE'].to_list()), shadow=True, startangle=90)
        ax[2].axis('equal')  
        ax[2].set_title('Over 80 years old') 
        plt.legend()
        plt.show()
        sns.boxplot(x=datos['AGE'])
        sns.boxplot(x=datos['TEMP'])




  


