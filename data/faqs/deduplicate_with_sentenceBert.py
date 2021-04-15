from haystack.retriever.elasticsearch import ElasticsearchRetriever
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import covid_nlp.language.detect_language as detect
import covid_nlp.language.ms_translate as translate
import pandas as pd

# loading questions and calculating similarities based of sentence bert embeddings
df = pd.read_csv("200416_englishFAQ.csv",sep=",")
if df.columns[0] != "question":
    df = df.iloc[:,1:]

#df = pd.concat((df.loc[df.name == "CDC General FAQ"],df.loc[df.name != "CDC General FAQ"]),ignore_index=True)
df = df.loc[df.name == "CDC General FAQ"]
df = df.loc[df.category != "School Dismissals and Children"]

df.reset_index(inplace=True,drop=True)


questions = [{"text": v} for v in df.question.values]
retriever = ElasticsearchRetriever(document_store=None, embedding_model="deepset/sentence_bert", gpu=False)
res1 = retriever.embedding_model.extract_vectors(
    dicts=questions,
    extraction_strategy="reduce_mean",
    extraction_layer=-1)
res1 = np.array([i["vec"] for i in res1])
sims = cosine_similarity(res1,res1)

threshold = 0.85
indices = [0]
for i in range(1,len(questions)):
    if (sims[:i,i] < threshold).all():
        indices.append(i)
    else:
        print(df.question[i])
        idxs = np.nonzero(sims[:i,i] > threshold)[0]
        print(df.iloc[idxs,1])
        print("newexample \n")


newdf = df.iloc[indices,:]
print(newdf.shape)
print(df.shape)
newdf.to_csv("200416_CDCGen_dedup.csv",index=True,sep=",")


##Creational Pattern
class QuestionSet ():

  def __init__(self):
    pass

  def getInstance():
    if (isinstance == None):
        instance = QuestionSet


  def getQuestions():
    ret = ''
    dir = 'COVID-QA/data/faqs'
    with open(dir, 'faq_covidbert.csv') as csvFile:
        text = pd.reader(csvFile, delimiter=',')
        for row in text:
            return (row[1], row[2])                             #questions are in the second column of the csv file


  def query(self, S):
    questionLang = detect.detect_lang(self, S)                                 #detect language using method imported from covidnlp/lang/detectlanguage/import
    ques = translate.MSTranslator(self, None, None, "en").translate(self, S)   #translate input question to English so it can be detected in the .csv file
    dir = 'COVID-QA/data/faqs/'                                                #obtain directory for .csv file containing questions and answers
    with open(dir, 'faq_covidbert.csv') as csvFile:                            #open file
        #search for corresponding answer to question
        text = pd.reader(csvFile, delimiter=",")                               #questions and answers were delimited by commas
        for row in text:                #search the file for the question
                if ques in row:         #question was found
                    ans = row[2]        #answers were in the 3rd column of the text, split the string to return the answer only
    if(detect.LanguageDetector().detect_lang(ans, ans) != questionLang):
        ans = translate.MSTranslator(self, None, None, "en").translate(self, ans)      #translate answer to user's desired language (i.e. language of the input question) if necesseary

    return ans



