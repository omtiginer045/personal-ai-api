from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Veriyi Yükle
# docs klasöründeki lore.txt dosyamızı okuyoruz
loader = TextLoader("docs/lore.txt", encoding="utf-8")
documents = loader.load()

# 2. Metni Parçalara Böl (Chunking)
# LLM'ler tek seferde koca bir kitabı okuyamaz, metni mantıklı küçük paragraflara bölüyoruz
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Embedding Modeli
# Metinleri sayılara çevirecek çok hızlı ve hafif bir model kullanıyoruz (RTX 3050'yi hiç yormaz)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vektör Veritabanına (ChromaDB) Kaydet
# "chroma_db" adında bir klasör oluşturup vektörleri oraya kaydedecek
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Veriler başarıyla parçalandı ve vektör veritabanına (ChromaDB) kaydedildi!")