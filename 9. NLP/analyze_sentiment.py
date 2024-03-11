from fastapi import FastAPI, HTTPException, Request

from transformers import pipeline

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

def compute_sentiment(text):
    # 5 (average #letters in a word) * 256 gave a tensor of size 647
    if type(text) == str:
        text = text[:2 * 256]
        s = distilled_student_sentiment_classifier (text)[0][0]['score']
        #print (s)
        if s < 0.2:
            return 1
        elif s < 0.4:
            return 2
        elif s < 0.6:
            return 3
        elif s < 0.8:
            return 4
        else:
            return 5
    # text was of type float in some cases
    return 1

app = FastAPI()

@app.post("/analyze")
async def summarize_text(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Content not provided")
    score = compute_sentiment(text)
    return {"content": text, "sentiment_score": score}

@app.get("/")
async def root():
    return {"message": "Sentiment Analyer"}