# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:31:25 2020

@author: kevinwang
"""
#%%1.Calculate the distance
#Calculate the Euclidean distance between all test samples and training samples（201*490）(dataframe/list/dict)
import pandas as pd
def Distance(train,test,column):
    dict_diff = {}
    for i in range(len(train_data),len(dataset)):
        list_diff = []
        for j in range(len(train_data)):
            test_dis = 0
            for col in column:
                #Calculate the distance between the current test sample and each training sample
                test_dis += (float(test.loc[i,col]) - float(train.loc[j,col]))**2
            diff = test_dis**0.5
            list_diff.append(diff)
        #Put the result in the dictionary
        dict_list_diff = {i:list_diff}
        dict_diff.update(dict_list_diff)
    return dict_diff
#2.Sort the distance of each test sample to get the index of the nearest k points
def Neighbor(distance,k):
    label_index = {}
    for itme in distance:
        #Sort the distance of all test samples
        value = distance[itme]
        value_sort = sorted(value)
        index_1 = []
        #Select K points closest to the current test sample
        for i in range(k):
            for j in range(len(value)):
                if value_sort[i] == value[j]:
                    index_1.append(j)
        #Put the result in the dictionary
        dict_index = {itme:index_1}
        label_index.update(dict_index)
    return label_index
#3.Determine which label the current test belongs to based on the labels of the nearest k points
def Get_K_label(train,index,target):
    final_label_dict = {}
    for key in index:
        value = index[key]
        label_list = []
        #Determine which label the current test sample belongs to according to the label with the highest frequency of the k nearest selected points
        for i in range(len(value)):
            label = train.loc[value[i],target]
            label_list.append(label)
            final_label = max(label_list, key=label_list.count)
        #Put the result in the dictionary
        dict_label = {key:final_label}
        final_label_dict.update(dict_label)
    return final_label_dict
#4.Estimation of accuracy
def Acc(dict_label,test):
    df = pd.DataFrame(pd.Series(dict_label), columns=['check_label']).reset_index()
    check_data = test.reset_index()
    #Link test data set and prediction data set through index
    check_data = check_data.merge(df,on = 'index')
    #Calculate the accuracy rate between the original label and the predicted label
    check_data['check'] = check_data[['label','check_label']].apply(lambda x : 1 if x[0] == x[1] else 0,axis =1)
    df_acc = check_data['check'].value_counts().reset_index()
    print(df_acc)
    #Count the number of times that the prediction is correct, and divide it by all the test sample sizes to get the correct rate
    accuracy = int(df_acc[df_acc['index'] == 1]['check'])/len(test_data)
    return accuracy
#%%
if __name__ == '__main__':
    #Import data
    dataset = pd.read_excel(r'member.xlsx')
    #Divide training set and test set, and specify explanatory variables and predictor variables
    train_data = dataset[0:490]
    test_data = dataset[490:]
    cal_cloumns = ['air','game','Travel']
    target_columns = 'label'
    #Calculate distance (Euclidean distance)
    distance = Distance(train_data,test_data,cal_cloumns)
    dict_accuracy = {}
    #Generally, the number of K in the KNN setting does not exceed 20
    for i in range(1,21):
        #Sort the distance of each test sample to get the index of the nearest k points
        index_1 = Neighbor(distance,i)
        #Determine which label the current sample belongs to based on the labels of the nearest k points
        dict_label = Get_K_label(train_data,index_1,target_columns)
        #A new column is generated in the test set as a prediction category, and the classification accuracy rate is estimated
        accuracy = Acc(dict_label,test_data)
        dict_acc = {i:accuracy}
        dict_accuracy.update(dict_acc)
    # max_accuracy = max(zip(dict_accuracy.values(), dict_accuracy.keys()))
    # print('rate is {}, k is {}'.format(max_accuracy[0], max_accuracy[1]))
    list_value = []
    for item in dict_accuracy:
        list_value.append(dict_accuracy[item])
    max_accuracy = max(list_value)
    max_accuracy_k = list_value.index(max(list_value))+1
    print('the maximum accuracy rate is %s, and the number of K is %s'%(max_accuracy,max_accuracy_k))
    #%%
    index_max = Neighbor(distance,max_accuracy_k)
    dict_label = Get_K_label(train_data,index_max,target_columns)