from transformers import pipeline

model = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

model ("I love this movie and i would watch it again and again!")

# returns somethine like
# [[{'label': 'positive', 'score': 0.9731044769287109},
#  {'label': 'neutral', 'score': 0.016910076141357422},
#  {'label': 'negative', 'score': 0.00998548325151205}]]
