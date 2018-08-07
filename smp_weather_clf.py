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
    weather_dataset=(('sunny', 'hot', 'high', 'false', 'no'),
                     ('sunny', 'hot', 'high', 'true', 'no'),
                     ('overcast', 'hot', 'high', 'false', 'yes'),
                     ('rainy', 'mild', 'high', 'false', 'yes'),
                     ('rainy', 'cool', 'normal', 'false', 'yes'))

    #print range(0,5,1)

    for num in range(0,5,1):
        index = randint(0, 4)
        print 'Index is:', index
        ruleBaseModel(weather_dataset[index])

    print 'Accuracy:',(corrIdentified/5.0)*100,'%'
    print 'OUTPUT: Correctly identified:' + str(corrIdentified) + ' Uncorrenctly identified:' + str(uncorrIdentified)

main()
