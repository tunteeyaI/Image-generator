import requests
from config import APIKEY
API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers={"Authorization":f"Bearer {APIKEY}"}
def analyze_sentiment(text: str):
    payload={"inputs":text} #what is sent to the URL
    r=requests.post(API_URL,headers=headers,json=payload,timeout=30)
    if not r.ok:
        raise Exception(f'{r.status_code}-{r.text}')
    result=r.json()
    #display output
    if isinstance(result,list) and isinstance(result[0],list):
        preds=result[0]
    elif isinstance(result,list) and isinstance(result[0],dict):
        preds=result
    else:
        raise Exception(f" Unexpected Error:{result}")
    best=max(preds,key=lambda X: X["score"])
    return best["label"] , float(best["score"]),preds
while True:
    text=input("Enter text(exit to quit):")
    if text.lower()=="exit":
        break
    if not text:
        continue
    try:
        label,score,all_pred=analyze_sentiment(text)
        print("\n Sentiment:",label.upper())
    except Exception as e:
        print("Error",e)