from IPython.display import display, HTML
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)

#https://amitness.com/2019/07/identify-text-language-python/#:~:text=Fasttext%20is%20an%20open-source,subword%20information%20and%20model%20compression.



def my_df_describe(df,name = 'dataset',show = True,path='',save=False):
    
    path_analisis    = ''
    path_description = ''
    if save:
        path_analisis        = os.path.join(path,'analysis')
        path_description = os.path.join(path_analisis,'description') 
        for folder in [path,path_analisis,path_description]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

    print(20*'*'+name+20*'*')
    objects = []
    numerics = []
    for c in df:
        if (str(df[c].dtype) in ['str','object','category']):
            objects.append(c)
        else:
            numerics.append(c)
    
    df = df.replace(to_replace=['',' ','None','NaN'], value=np.nan)
    numeric_desc, categorical_desc = None,None

    if len(numerics)>0:
        numeric_desc = df[numerics].describe().transpose()
        counting = df.nunique().transpose()
        numeric_desc['unique'] = counting[numerics]
        numeric_desc['nulls'] = df[numerics].isna().sum().values
        numeric_desc['nulls_perc'] = df[numerics].isna().sum().values/df.shape[0]
        if save:
            filename = os.path.join(path_description,name+'_numeric_variables_description.csv')
            print('saving ',filename)
            numeric_desc.to_csv(filename)#,index=None

    if len(objects)>0:
        categorical_desc = df[objects].describe().transpose()
        #print(df[objects].isna().sum())
        #print(categorical_desc.shape)
        categorical_desc['nulls'] = df[objects].isna().sum().values
        categorical_desc['nulls_perc'] = df[objects].isna().sum().values/df.shape[0]
        if save:
            filename = os.path.join(path_description,name+'_categorical_variables_description.csv')
            print('saving',filename)
            categorical_desc.to_csv(filename)
    
    print('shape ',df.shape)
    if show:
        if len(numerics)>0:
            print(10*'*'+'numerics'+10*'*')
            display(numeric_desc)
        if len(objects)>0:
            print(10*'*'+'categorical'+10*'*')
            display(categorical_desc)
    
    return numeric_desc, categorical_desc

def graph_numerical_distribution(data,name="target",figsize=(6,4),title_name=None,show=False,save=False,name_file=None,path='',suffix='',prefix='',rotation=None):
    

    if name_file is None:
        name_file = name+'_distribution'
    plt.figure(figsize=figsize)
    total = float(len(data)) # one person per row 
    title_name = "Num. "+name+" distribution of "+str(int(total))+" elements" if title_name is None else title_name+" of "+str(int(total))+" users"
    ax = sns.histplot(data=data, x=name, kde=True)
    if rotation != None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    plt.title(title_name)
    if show:
        plt.show()
    figure = ax.get_figure()    
    if save:    
        path_images        = os.path.join(path,'images')
        path_distributions = os.path.join(path_images,'distributions') 
        for folder in [path,path_images,path_distributions]:
            if not os.path.isdir(folder):
                os.mkdir(folder)
        filename_img = os.path.join(path_distributions,suffix+name_file+prefix+'.png')
        print('saving image',filename_img)
        figure.savefig(filename_img,dpi=400, bbox_inches = 'tight')
    plt.close()
    return figure

def graph_categorical_distribution(data,name="target",figsize=(6,4),title_name=None,color_text="black",show=False,save=False,name_file=None,path='',suffix='',prefix='',rotation=None):
    
    if name_file is None:
        name_file = name+'_distribution'
    plt.figure(figsize=figsize)
    total = float(len(data)) # one person per row 
    title_name = "Cat. "+name+" distribution of "+str(int(total))+" elements" if title_name is None else title_name+" of "+str(int(total))+" users"
    ax = sns.countplot(x=name, data=data) # for Seaborn version 0.7 and more
    if rotation != None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height/3,
                '{:.2f}%\n{:d}'.format(100*height/total,height),
                ha="center",color=color_text,fontweight='bold')#fontsize=10
    plt.title(title_name)
    if show:
        plt.show()
    figure = ax.get_figure()    
    if save:    
        path_images        = os.path.join(path,'images')
        path_distributions = os.path.join(path_images,'distributions') 
        for folder in [path,path_images,path_distributions]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        filename_img = os.path.join(path_distributions,suffix+name_file+prefix+'.png')
        print('saving image',filename_img)
        figure.savefig(filename_img,dpi=400, bbox_inches = 'tight')
    plt.close()
    return figure


def graph_histogram_xy(data,x,y,figsize=(6,4),title_name=None,color_text="black",fontsize=10,show=False,save=False,name_file=None,path='',suffix='',prefix='',rotation=None):
    
    title_name = "distribution of "+y+" with labels "+x if title_name is None else title_name

    if name_file is None:
        name_file = title_name
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, data=data,
                    palette="Blues_d")
    if rotation != None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height,
                str(height),
                ha="center",color=color_text,fontweight='bold',fontsize=fontsize)#
    plt.title(title_name)
    if show:
        plt.show()

    figure = ax.get_figure()    
    if save:    
        path_images        = os.path.join(path,'images')
        path_distributions = os.path.join(path_images,'distributions') 
        for folder in [path,path_images,path_distributions]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        filename_img = os.path.join(path_distributions,suffix+name_file+prefix+'.png')
        print('saving image',filename_img)
        figure.savefig(filename_img,dpi=400, bbox_inches = 'tight')
    plt.close()
    return figure

def get_perceptiles(values,mini=25,maxi=75):
    Q1,Q3=np.percentile(values,[mini,maxi])
    return Q1,Q3

def graph_missing_values(data,title='Missing values in Data',figsize=(25,11),show=False,save=False,path=''):

    plt.figure(figsize = figsize)
    ax = sns.heatmap(data.isna().values, xticklabels=data.columns)
    plt.title(title, size=20)
    if show:
        plt.show()

    figure = ax.get_figure()    
    if save:    
        path_images        = os.path.join(path,'images')
        path_distributions = os.path.join(path_images,'distributions') 
        for folder in [path,path_images,path_distributions]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        filename_img = os.path.join(path_distributions,title+'.png')
        print('saving image',filename_img)
        figure.savefig(filename_img,dpi=400, bbox_inches = 'tight')
    plt.close()
    return figure

def compare_train_test_values(data_train,data_test,column_name):

    train_uniques = data_train[column_name].unique()
    test_uniques  = data_test[column_name].unique()
    test_missing  = set(test_uniques)-set(train_uniques)
    train_missing  = set(train_uniques)-set(test_uniques)

    print("train uniques values in "+column_name,len(train_uniques))
    print("test  uniques values in "+column_name,len(test_uniques))
    print("train "+column_name+" doesn't have values of test :",list(test_missing))
    print("test  "+column_name+" doesn't have values of train :",list(train_missing))

    return train_uniques,test_uniques,test_missing,train_missing


def mean_value_by_cat_value(data,column_name,target_name):
    # target_name's mean by column_name
    
    mean_values = data[[column_name,target_name]].groupby([column_name]).mean().reset_index()
    mean_values.columns = [column_name,target_name]
    mean_values = mean_values.sort_values(by=target_name, ascending=False)
    return mean_values


def graph_boxplot_value_by_cat(data,column_name,target_name,title=None,figsize=(25,11),show=False,save=False,path='',rotation=None):

    if title is None:
        title = 'Box plot of '+target_name+' by '+column_name

    fig = plt.figure(figsize = figsize)
    ax = sns.boxplot(x=data[column_name], y=data[target_name])

    if rotation != None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    plt.title(" Boxplot of "+target_name+": by "+column_name)
    plt.show()

    if show:
        plt.show()

    figure = ax.get_figure()    
    if save:    
        path_images        = os.path.join(path,'images')
        path_distributions = os.path.join(path_images,'distributions') 
        for folder in [path,path_images,path_distributions]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        filename_img = os.path.join(path_distributions,title+'.png')
        print('saving image',filename_img)
        figure.savefig(filename_img,dpi=400, bbox_inches = 'tight')
    plt.close()
    return figure

