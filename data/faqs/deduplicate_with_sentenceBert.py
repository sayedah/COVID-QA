from haystack.retriever.elasticsearch import ElasticsearchRetriever
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Aggregate Pattern
Class Question:
	def init(self, QuestID: str, question: str language: str):
		self.QuestID = QuestID
		self.quest = question
       	self.lang = language
        self.source = null
        self.ans = null 

     #assign an answer to the question  
	def answer(self, source : DataSource, answer : str) 
		#after scrapers/ElasticSearch are used, assign 'answer' to Question along with the data source the information was pulled from
        self.source = DataSource
        self.ans = answer

     #translate the object to another language
	def translate(self, QuestID, language):
        if self.lang != language 
           self.lang = language  #translate to parameterized language 
		return self.quest, self.ans
##  
    
    
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


