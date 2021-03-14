package com.theapache64.cs.models.rest

import com.google.gson.annotations.SerializedName

class CoronaQuestion(
    @SerializedName("questions")
    val questions: Array<String>, // How does corona spread?
    @SerializedName("top_k_retriever")
    val resultCount: Int = 1
)


//Aggregate Pattern
Class Question:
	def init(self, QuestID: str, question: str language):
		self.QuestID = QuestID
		self.quest = question
       	self.lang = language
        self.source = null       //question initially has no answer and source of information for the answer
        self.ans = null 
		self.embedding = get.embedding()   //store embeddings of question when it's time to match user's input 

        def answer(self, source : DataSource, answer : str) 
		//after scrapers/ElasticSearch are used, assign 'answer' to Question along with the 'source' the information was pulled from
                self.source = DataSource
                self.ans = answer

        def translate(self, QuestID, language):
                if self.lang != language 
                self.lang = language  //call on translators to translate to parameterized language 

        def match(self) 
                

        

        

