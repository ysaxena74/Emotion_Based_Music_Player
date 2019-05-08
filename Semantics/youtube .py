# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:26:50 2019

@author: dell
"""

import webbrowser
import urllib.request
from bs4 import BeautifulSoup
import random
import re
import Project_1
import os
import random
import numpy as np
import csv
import requests



mx=np.argmax(Project_1.emoji)
mx
base='C:/Users\dell\Desktop\musicmood-master\musicmood-master\code\classify_lyrics\make_folder'
if(mx==2 or mx==3):
    pre='happy'
elif(mx==1 or mx==0):
    pre='sad'
x=random.randint(0,9)
#print(x)
path_dir=os.path.join(base,pre)
links = []
listr=os.listdir(path_dir)
print(len(listr))
for i in range(0,len(listr)):
    deep=open(os.path.join(path_dir,listr[i]),'r')
    s=""
    c=0
    for i in deep.readlines():
        c=c+1
        s+=i[:-1]+' '
    value1=s[0:100]
    links = []
    textToSearch = value1
    print(value1)
    query = urllib.parse.quote(textToSearch)
    url = "https://www.youtube.com/results?search_query=" + query 
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    
    for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
        links.append('https://www.youtube.com' + vid['href'])
    print(links[0])    
    filesInChannel = [ links[0] ]
    def getStats(link):
        
        page = requests.get(link)
        likes = re.search("with (\d*.\d*.\d*)", page.text).group(1)
        title = re.search("property=\"og:title\" content=\"([^\n]*)", page.text).group(1)
        return (likes, title)           
    
        
    for link in filesInChannel:
        stats = getStats(link)
        #print (stats[0].encode("utf-8") + " " + stats[1].encode("utf-8"))
        row = [stats[0],stats[1]]

    with open('abcde.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()  
        
a = []  
max1        = -1
valuee = ''   
with open('abcde.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if (row):
            #print(row[0])
            if(type(row[0]) == int):
                print(row[0])
                if (row[0]>max1):
                    max1=row[0]
                    valuee =row[1]
            
                
            #a.append([row[0],row[1]])

    
#for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
 #   links.append('https://www.youtube.com' + vid['href'])
#print(links)

print("")
#print(value1)
#p = random.randint(0,len(links))
url1 = links[0]
filesInChannel = [
"https://www.youtube.com/watch?v=-Ox9MvottBI"
]
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chrome_path).open(url1)
