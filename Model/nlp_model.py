
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# initialize the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

#Generate Summary
def abstractive_summary(text):
  # preprocess input
  preprocessed_text = text.strip().replace('\n','') # menghapus line kosong
  t5_input_text = 'summarize: ' + preprocessed_text
  tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True).to(device) # mengubah text menjadi bentuk token

  # mengenrate summary menggunakan transformer
  summary_ids = model.generate(tokenized_text, min_length=30, max_length=120) # generate summary
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) # mengubah kembali dari bentuk token ke bentuk text

  return summary

"""##Extractive"""

# import the required libraries
from nltk.tokenize import sent_tokenize # Sentence tokenizer untuk split text menjadi kalimat, word tokenizer untuk split kalimat menjadi kata
from sklearn.feature_extraction.text import TfidfVectorizer # Agar dapat convert dokumen menjadi matrix TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # Untuk menghitung cosine similarity dari dua vektor
from heapq import nlargest # mengembalikan n element terbesar secara urutan descending

def extractive_summary(text):
  sentences = sent_tokenize(text) # Tokenize text menjadi kalimat
  
  # Membuat matriks TF-IDF
  vectorizer = TfidfVectorizer(stop_words='english')
  tfidf_matrix = vectorizer.fit_transform(sentences)

  # Menghitung cosine similarity dari setiap kata dalam dokumen
  sentence_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

  # Mengambil kalimat dengan skor paling tinggi
  summary_sentences = nlargest(1, range(len(sentence_scores)), key=sentence_scores.__getitem__)
  summary_tfidf = ' '.join([sentences[i] for i in sorted(summary_sentences)])

  return summary_tfidf