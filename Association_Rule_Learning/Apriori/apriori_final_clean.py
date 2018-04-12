# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:48:28 2018

@author: Hemanth kumar
"""
import itertools    
#Function to convert binary data to item set comma sepersted (I1,I2,I3....) 
def conv_data(data):
    label=[]
    new_data=[]
    for i in range(1,len(data[0])+1):
        label.append('I'+str(i))
    for i in range(len(data)):
        tmp=""
        for j in range(len(data[0])):
            if(data[i][j]==1):
                    tmp+=label[j]+','
        tmp=tmp[:-1]
        new_data.append(tmp)
    return (new_data,label)

#Function to generate Frequent item set
def generate_freq_set(set_itm,data,min_sup,flag):
    count_list=[]
    tmp=[]
    for i in range(len(set_itm)):
        count=0
        if(flag):
            y=set([set_itm[i]])
        else:
            y=set(set_itm[i].split(','))
        for j in range(len(data)):
            x=set(data[j].split(','))
            if(y==x.intersection(y)):
                count+=1
            
        if(count<min_sup):
            tmp.append(i)
        else:
            count_list.append(count)
            
    new_set=set_itm.copy()
    for i in range(len(tmp)):
        new_set.remove(set_itm[tmp[i]])
    return set_itm,new_set,count_list

#Function to compute confidence
def compute_conf(item_set,itm,supp):
    y=set(itm.split(','))
    count=0
    
    for i in range(len(item_set)):         
        x=set(item_set[i].split(','))
        if(y==x.intersection(y)):
            count+=1
            
    if(count==0):
        return 0
        
    return (supp/count)*100
            
       
#Function to generate all possible combinations of a Item set (string)
def generate_combinations(data,leng):
    itm_set=[]
    for itm in itertools.combinations(data,leng):
        itm_set.append(",".join(itm))
    return itm_set

#Function to print Association Rule
def print_assoc_rule(final_res,min_conf=70):
    freq_set=final_res[1]
    support=final_res[2]
    count=0
    
    for p in range(len(freq_set)):
        label=set()
        label=label.union(set(freq_set[p].split(",")))
        label=list(label)
        print("\n")
        for j in range(1,len(label)):
            combi=sorted(generate_combinations(label,j))
            
            for i in range(len(combi)):
                right=set(label)-set(combi[i].split(","))
                conf=compute_conf(data_new,combi[i],support[p])
                if(conf>=min_conf):
                    count+=1
                    print(count,")")
                    print("Confidence:",conf)
                    print(str(combi[i])+"-->"+str(right))
                    print("\n")
                    
    print("Total Association Rules obtained :",count)
    

#Function to generate item set
def generate_itm_set(data,leng):
    if(leng==1):
        return data
    if(leng==(len(data[0])-1)):
        return data
        
    itm_set=[]
    
    if(leng==2):
        for itm in itertools.combinations(data,leng):
            itm_set.append(",".join(itm))
        return itm_set
    
    if(leng>2):
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                x=set(data[i].split(','))
                y=set(data[j].split(','))
                if(x.intersection(y)==set(data[i].split(',')[0:leng-2])):
                    itm_set.append(",".join(sorted(list(x.union(y)))))
                else:
                    break
        return itm_set

#Driver function for Apriori Algorithm
def apriori_algo(data,itm,min_supp):
    leng=1
    res={}
    itm_feed=itm.copy()
    while True:
        if(leng==1):
            flag=True
        else:
            flag=False
        
        item_set=generate_itm_set(itm_feed,leng)
        if(len(item_set)==0):
            break
        
        itm_set,freq_set,supp=generate_freq_set(item_set,data,min_supp,flag)
        
        if(len(freq_set)==0):
            break
        else:
            itm_feed=freq_set.copy()
            res['set'+str(leng)]=(itm_set,freq_set,supp)
            leng+=1
    return res

#function to print association rule (I1,I2...In)
def print_assoc_rule(final_res,min_conf=70):
    freq_set=final_res[1]
    support=final_res[2]
    count=0
    
    for p in range(len(freq_set)):
        label=set()
        label=label.union(set(freq_set[p].split(",")))
        label=list(label)
        print("\n")
        for j in range(1,len(label)):
            combi=sorted(generate_combinations(label,j))
            
            for i in range(len(combi)):
                right=set(label)-set(combi[i].split(","))
                conf=compute_conf(data_new,combi[i],support[p])
                if(conf>=min_conf):
                    count+=1
                    print(count,")")
                    print("Confidence:",conf)
                    print(str(combi[i])+"-->"+str(right))
                    print("\n")
                    
    print("Total Association Rules obtained :",count)
    
#function to print association rule with names    
def print_assoc_rule1(final_res,min_conf=70):
    freq_set=final_res[1]
    support=final_res[2]
    count=0
    
    for p in range(len(freq_set)):
        #p=0
        label=set()
        label=label.union(set(freq_set[p].split(",")))
        label=list(label)
        
        print("\n")
        for j in range(1,len(label)):
            #j=1
            combi=sorted(generate_combinations(label,j))
            
            for i in range(len(combi)):
                #i=0
                left=set(combi[i].split(","))
                right=set(label)-left
                right_list=list(right)
                left_list=list(left)
                left_names=conv_item_names(left_list,item)
                right_names=conv_item_names(right_list,item)
                conf=compute_conf(data_new,combi[i],support[p])
                
                if(conf>=min_conf):
                    count+=1
                    print(count,")")
                    print("Confidence:",conf,"%\n")
                    print(str(set(left_names))+"-->"+str(set(right_names))+"\n")
                    #print(str(combi[i])+"-->"+str(right))
                    #print("\n")
                    
    print("Total Association Rules obtained :",count)
    
#convert Item I to name
def conv_item_names(data,names):
    label_name=[]
    for i in range(len(data)):
        label_name.insert(i,names[int(data[i][1:])])
    return label_name


#Importing dataset
import pandas as pd
data_set=pd.read_csv("1000-out2.csv",header=None)
data=data_set.iloc[:].values
data=data[:,1:]
data_new,label=conv_data(data)
items_data_set=pd.read_csv("bakery_items.csv",header=None)
item_flav=items_data_set.iloc[:,[0,1]].values.tolist()
item=[]
for i in range(len(item_flav)):
    item.append("".join(item_flav[i]))


#Applying Apriori
    
#Confidence=70% Support count=15
print("-----------------------------------------------------\n")
print("Confidence= 90% \nSupport count= 35")
res=apriori_algo(data_new,label,min_supp=35)
final_res=res['set'+str(len(res))]
print_assoc_rule1(final_res,min_conf=90)



print("-----------------------------------------------------\n")
#Confidence=70% Support count=30
print("Confidence= 90% \nSupport count= 40")
res=apriori_algo(data_new,label,min_supp=40)
final_res=res['set'+str(len(res))]
print_assoc_rule1(final_res,min_conf=90)


print("-----------------------------------------------------\n")
print("Confidence= 95% \nSupport count= 35")
res=apriori_algo(data_new,label,min_supp=35)
final_res=res['set'+str(len(res))]
print_assoc_rule1(final_res,min_conf=95)

print("-----------------------------------------------------\n")
print("Confidence= 95% \nSupport count= 40")
res=apriori_algo(data_new,label,min_supp=40)
final_res=res['set'+str(len(res))]
print_assoc_rule1(final_res,min_conf=95)

