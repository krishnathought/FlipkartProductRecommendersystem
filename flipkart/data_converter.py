import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def convert(self):
        df=pd.read_csv(self.filepath)[["product_title","review"]]
        
        docs=[

            Document(page_content=row['review'], metadata={"product_name": row['product_title']}) 
            for _, row in df.iterrows()
        ]
        return docs
    
if __name__ == "__main__":
    converter = DataConverter("/Users/krishnamurthy/Deeplearning/GenAI/FlipKart Product Recommender/data/flipkart_product_review.csv")
    documents = converter.convert()
    print(f"Converted {len(documents)} documents.")
    print(documents[0])  # Print the first document as a sample


