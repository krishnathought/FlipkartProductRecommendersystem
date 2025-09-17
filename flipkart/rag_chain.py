from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory # below 2 are responsibile for chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model=ChatGroq(model=Config.RAG_MODEL,temperature=0.5)
        self.history_store={} # to store chat history for different session
    def _get_history(self,session_id:str)->BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id]=ChatMessageHistory()
        return self.history_store[session_id]
    
    # "krisha" - str
    # chatmessagehistory - basechatmessagehistory
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles in readable table format with product name, price , other details. 
                          Stick to context. Do not give any content other than available in flipkart website, Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""), # input  id standalone
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        # context prompt 
        #User input + chat history -> standalone question

        history_aware_retriever = create_history_aware_retriever(
           
           self.model,retriever,context_prompt
       )
        
        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )
          
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"

        )      
    
       # 👇 new helper
    def get_history(self, session_id: str):
        return self._get_history(session_id).messages