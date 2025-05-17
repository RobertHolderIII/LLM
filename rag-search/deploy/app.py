import requests
from googlesearch import search
import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import os
import gradio as gr

load_dotenv('../../.env')
client = Groq(api_key = os.getenv('GROQ_API_KEY'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def summarize(client, transcript):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text."
            },
            {
                "role": "user",
                "content": f"Please summarize the following articles:\n\n{transcript}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content


def web_search(query="US tariff news, analysis, and predictions"):
    search_res = list(search(query, num_results=10, unique=True))
    return search_res


def get_text(url):
    print(f'downloading {url}...')
    response = requests.get(url)
    html = response.text
    text = BeautifulSoup(html, features="html.parser").get_text().strip()
    return text

def create_vector_store(urls):
    documents = [get_text(url) for url in urls]
    ids = [f'id{i}' for i in range(len(documents))]
    metadatas = [{'source-url': url} for url in urls]

    chroma_client = chromadb.Client()
    collection_name = "webpages"

    # get rid of old collection
    try:
        chroma_client.get_collection(collection_name)
    except:
        # Collection does not exist
        pass
    else:
        chroma_client.delete_collection(collection_name)
        
    collection = chroma_client.create_collection(name=collection_name)
        
    # TODO need to handle the case where search_res has a URL that we did not use
    # `add` uses Chroma's default sentence embedding model
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f'collection has {collection.count()} documents')
    return collection

def retrieve_documents(collection):
    retrieval = collection.query(
        query_texts=["what are the most important bits of news to know about US tariff policy and analysis of future actions?"],
        n_results=5,
        include=["documents", "metadatas"]
    )
    return retrieval

def generate_summaries(user_question, history):
    search_res = web_search()
    collection = create_vector_store(search_res)
    retrieval = retrieve_documents(collection)
    relevant_docs = retrieval['documents'][0]
    print(f'summarizing {len(relevant_docs)} documents')
    summaries = [summarize(client,doc[:5000]) for doc in relevant_docs]

    # format summaries
    out = ''
    metadatas = retrieval['metadatas'][0]
    urls = [data['source-url'] for data in metadatas]
    for idx, (url, summary) in enumerate(zip(urls, summaries)):
        out += (f'{idx+1}: {url}\n---------------------\n{summary}\n\n')

    return out

with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## Tarrif Intel Tool

        Instructions go here
        """
        )
    gr.ChatInterface(fn=generate_summaries, type="messages")
    gr.Markdown(
        """
        [2025-05-17] initial deployment<br>
        """
        )
    
if __name__ == "__main__":
    print('v2025-05-17--1148')
    print('use `gradio app.py` to run in reload mode')
    demo.launch()
