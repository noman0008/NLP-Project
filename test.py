import requests
import json

def retPageid(cat):

    title='Indiana'
    count=5
    #url='https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&titles={}&rvprop=user|comment|timestamp&rvlimit={}'.format(title,count)
    
    url='https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&format=json&cmtitle=Category:{}&cmlimit=500'.format(cat)
    
    
    
    #data=requests.get(url).json()
    data=requests.get(url).json()
    
    #print(data['query']['categorymembers'])   
    return [item['pageid'] for item in data['query']['categorymembers']]
        

def analyzeRevisionHist(page_id):
    list_of_ids=''
    for item in page_id:
        list_of_ids=list_of_ids+str(item)+'|'
        
    list_of_ids = list_of_ids[:-1]
     
    url='https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&pageids={}&rvprop=user|comment|timestamp'.format(list_of_ids)
    data=requests.get(url).json()
    
    list_of_users=[]
    for item in page_id: 
        while True:
            try:
                list_of_users=[item['user'] for item in data['query']['pages'][str(item)]['revisions']]
                break
            except:
                pass
            
            
    print(list_of_users)
    #print(list_of_ids)
    #return [item['user'] for item in data['query']['pages']['18839']['revisions']]

def detectGender(user):
    url='api.php?action=query&list=users&ususers={}&usprop=gender'.format()
    



 