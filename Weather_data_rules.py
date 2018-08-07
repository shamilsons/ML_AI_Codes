# -*- coding: utf-8 -*-
from random import *

corrIdentified=0
uncorrIdentified = 0

#This is the rule-base engine for the classifier
#Accepts instances from weather dataset
def ruleBaseModel(record):
    #Rule-based classifier for weather dataset (2 rules)
    result=''
    global corrIdentified
    global uncorrIdentified

    if record[0]=='overcast':
        result='yes'
    if record[0]=='sunny' and record[2]=='high':
        result='no'

    if result==record[4]:
        corrIdentified+=1
    else:
        uncorrIdentified+=1


#Where the story begins
def main():
    #Features definitions and declarations
    #0->Outlook, 1->Temperature, 2->Humidity, 3->Windi, 4->CLASS
    weather_dataset=(('sunny', 'hot',  'high',   'false',  'no'),
                     ('sunny', 'hot',  'high',   'true',   'no'),
                     ('sunny', 'mild', 'high',   'false',  'no'),
                     ('sunny', 'mild', 'normal', 'true',   'yes'),
                     ('sunny', 'cool', 'normal', 'false',  'yes'),
                     ('rainy', 'mild', 'high',   'false',  'yes'),
                     ('rainy', 'mild', 'high',   'true',   'no'),
                     ('rainy', 'mild', 'normal', 'false',  'yes'),
                     ('rainy', 'cool', 'normal', 'false',  'yes'),
                     ('rainy', 'cool', 'normal', 'true',   'no'),
                     ('overcast', 'cool', 'normal', 'true',  'yes'),
                     ('overcast', 'mild', 'high', 'true',    'yes'),
                     ('overcast', 'hot',  'high', 'false',   'yes'),
                     ('overcast', 'hot',  'normal', 'false', 'yes'))

    outlook=['sunny', 'overcast', 'rainy']
    temperature=['hot', 'mild', 'cool']
    humidity=['normal', 'high']
    windy=['true', 'false']
    counter=0
    numLine=1
    for out in outlook:
        for tmp in temperature:
            for hmd in humidity:
                for wnd in windy:
                    #if(weather_dataset[counter][0]!=out and weather_dataset[counter][1]!=tmp and weather_dataset[counter][2]!=hmd and weather_dataset[counter][3]==wnd):
                    print str(numLine)+'.'+out+'-'+tmp+'-'+hmd+'-'+wnd
                    numLine+=1
                    #counter+=1
    '''
    #print range(0,5,1)
    recNum=7
    for num in range(0,recNum,1):
        index = randint(0, 13)
        #print 'Index is:', index
        ruleBaseModel(weather_dataset[index])

    print 'Accuracy:',(corrIdentified/float(recNum))*100,'%'
    print 'OUTPUT: Correctly identified:' + str(corrIdentified) + ' Uncorrenctly identified:' + str(uncorrIdentified)
    '''
main()