import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import shutil

class VectorDatabaseManager:
    def __init__(self, api_key: str, db_path: str = "crustdata_db"):
        """
        Initialize the Vector Database Manager.
        
        Args:
            api_key (str): OpenAI API key
            db_path (str): Path where the vector database will be stored
        """
        self.api_key = api_key
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )

    def _load_documents(self, data_folder_path: str) -> list:
        """
        Load all documents from the data folder.
        
        Args:
            data_folder_path (str): Path to the data folder
            
        Returns:
            list: List of processed documents
        """
        all_texts = []
        
        for filename in os.listdir(data_folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_folder_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                
                document = Document(page_content=file_content)
                texts = self.text_splitter.split_documents([document])
                all_texts.extend(texts)
                
        return all_texts

    def update_database(self, data_folder_path: str) -> None:
        """
        Update the vector database with new documents.
        
        Args:
            data_folder_path (str): Path to the data folder
        """
        # Remove existing database if it exists
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            
        # Load and process all documents
        all_texts = self._load_documents(data_folder_path)
        
        # Create new FAISS vector store
        if all_texts:
            faiss_db = FAISS.from_documents(all_texts, self.embeddings)
            faiss_db.save_local(self.db_path)
            print(f"Vector database updated successfully at {self.db_path}")
        else:
            print("No documents found to process")

def update_vector_db():
    """
    Utility function to update the vector database.
    Should be called after new files are added to the data folder.
    """
    data_folder_path = os.path.join(os.path.dirname(__file__), 'data')
    db_manager = VectorDatabaseManager(api_key=os.environ['OPENAI_API_KEY'])
    db_manager.update_database(data_folder_path)

if __name__ == "__main__":
    update_vector_db()