from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config

class DataIngestor:
    def __init__(self):
        self.embeddding=HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL_NAME)

        self.vector_store=AstraDBVectorStore(
            embedding=self.embeddding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, load_existing=True): # Default to True to load existing data if available
        if load_existing:
            return self.vector_store
            
        docs=DataConverter("data/flipkart_product.csv").convert()

        self.vector_store.add_documents(docs)
        return self.vector_store
        
if __name__ == "__main__":
    ingestor=DataIngestor()
    ingestor.ingest(load_existing=True)  # Set to False to ingest new data
    print("Data ingestion completed. Vector store is ready for use.")