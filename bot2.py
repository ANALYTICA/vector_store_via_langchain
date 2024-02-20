
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain.chains import LLMChain, RetrievalQA
from textgen import TextGen
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
import langchain
langchain.verbose = True

# many different types of loaders, pdf, etc.
loaders = [
    TextLoader("m1.txt",autodetect_encoding=True),
    TextLoader("m2.txt",autodetect_encoding=True),
    TextLoader("m3.txt",autodetect_encoding=True),
    TextLoader("m4.txt",autodetect_encoding=True)
]

# notice the splitting order
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

# run splitting
docs = []
for loader in loaders:
    docs.extend(loader.load())
splits = r_splitter.split_documents(docs)

# set up vector store, need a model to perform embeddings on each chunk; can use hf, for example
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

# set up api using text-generation-web-ui
# https://github.com/oobabooga/text-generation-webui#one-click-installers
# start in api mode using --api flag. still need to manually load model on web interface
model_url = "http://localhost:5000"

def raq_question(question,retriever):
    # TextGen is a openai "equivalent" to the openai api interface
    llm = TextGen(model_url=model_url,max_new_tokens=2048, ban_eos_token=True)
    # langchain tool RetrievalQA formats questions and answers using context. 
    # takes an llm to query and a retriever to supply context, automatically formats questions and "stuffs" with context
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True
    )
    result = qa_chain({"query": question,"max_tokens": 1024})
    return result

# query/api call
result = raq_question("can I eat a death cap",retriever)

# print response 
print("\n\n\n Bot Response: \n", result['result'])
for d in result["source_documents"]:
    print("\n",d,"\n")
