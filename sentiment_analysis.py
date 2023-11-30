#AUDIO PREPROCESSING
import wave
import numpy as np
import pyaudio
import os


channels = 2


seconds = input('Enter the number of seconds you want to record: ')
seconds = int(seconds)

sample_format = pyaudio.paInt32

#record in Frams_per_buffers of 1024 samples
Frams_per_buffer = 1024
fs = 44100 #record at 44100 samples per second

p = pyaudio.PyAudio()

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate = fs,
                frames_per_buffer=Frams_per_buffer,
                input = True)

frames = []

for i in range(0,int((fs/Frams_per_buffer)*seconds)):
    data = stream.read(Frams_per_buffer)
    frames.append(data)

#stop and close the stream
stream.stop_stream()
stream.close()
#terminate the portaudio interface
p.terminate()

output = "recording.wav"

wf = wave.open(output,'wb')
wf.setnchannels(channels)
wf.setsampwidth((p.get_sample_size(sample_format)))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close







#AUDIO TO TEXT
import speech_recognition as sr
import os
r = sr.Recognizer()

recording = sr.AudioFile('recording.wav')
with recording as source:
    audio = r.record(source)
r.recognize_google(audio)


recognized_audio = r.recognize_google(audio)
print(recognized_audio)








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#data visualization

import nltk
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('stopwords')
# nltk.download('corpus')
# nltk.download('averaged_perceptron_tagger')

# Read the data
#The . read_csv() function takes a path to a CSV file and reads the data into a Pandas DataFrame object.
df = pd.read_csv('Reviews.csv')
print(df.shape)

#Since we've got a huge sample, we're cutting down. Using the 5000values.
df = df.head(500)
print(df.shape)

# ax = df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of Reviews by Stars',
#           figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()
#We see that our dataset is biased towards positive reviews
example = df['Text'][40] #Picking the 50th entry
#print(example)
tokens = nltk.word_tokenize(example) #Splits the words of the sentence
tokens[:10] #Showing the First 10 tokens

tagged = nltk.pos_tag(tokens) #Part of Speech. There's a table where you'll see what these mean
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
#pprint -> pretty print

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('This is the best ML project')

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text) #Dictionary to score polarity score
    
#Storing it in the form of PandaFrame object since it's easier to work with.
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('Compound Score by Amazon Star Review')
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(12, 3))
# sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# plt.tight_layout()
# plt.show()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#This model was trained on Twitter comments
L = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(L)
model = AutoModelForSequenceClassification.from_pretrained(L)

# VADER results on example
print(example)
sia.polarity_scores(example)

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)  

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
        
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")
print(sent_pipeline(recognized_audio))