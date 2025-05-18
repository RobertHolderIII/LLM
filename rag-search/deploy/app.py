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
os.environ["TOKENIZERS_PARALLELISM"] = "false" #prevents a warning message from langchain
chroma_client = None

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


def create_vector_store(search_terms):
    urls = web_search(search_terms)
    documents = []
    for i,url in enumerate(urls):
        documents.append(get_text(url))
        yield f'downloading {i+1}/{len(urls)}: {url}'
    ids = [f'id{i}' for i in range(len(documents))]
    metadatas = [{'source-url': url} for url in urls]

    chromadb.api.client.SharedSystemClient.clear_system_cache() #prevents client getting stale
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
    yield f'Document store recreated with {collection.count()} documents'
    

def retrieve_documents(collection):
    retrieval = collection.query(
        query_texts=["what are the most important bits of news to know about US tariff policy and analysis of future actions?"],
        n_results=5,
        include=["documents", "metadatas"]
    )
    return retrieval

def generate_summaries():
    collection = chroma_client.get_collection("webpages")
    retrieval = retrieve_documents(collection)
    relevant_docs = retrieval['documents'][0]

    summaries = []
    for i, doc in enumerate(relevant_docs):
        yield f'summarizing {i+1}/{len(relevant_docs)} documents','In progress...'
        summaries.append(summarize(client,doc[:5000]))

    
    # format summaries
    out = ''
    metadatas = retrieval['metadatas'][0]
    urls = [data['source-url'] for data in metadatas]
    for idx, (url, summary) in enumerate(zip(urls, summaries)):
        out += (f'{idx+1}: {url}\n---------------------\n{summary}\n\n')

    yield f'summarized {len(relevant_docs)} documents', out

with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## Tarrif Intel Tool

        Instructions go here
        """
        )
    search_terms = gr.Dropdown(label='search query', show_label=True, allow_custom_value=True,
                               choices=["US tariff news, analysis, and predictions",
                                        "world reaction to US tariff policy"
                                        ]

                               )
    update_btn = gr.Button('Update documents')

    status = gr.Textbox(show_label=False)
    
    sum_btn = gr.Button('Generate summaries')
    sum_status = gr.Textbox(show_label=False)
    out = gr.TextArea(label='Summarizations', show_label=True)

    update_btn.click(fn=create_vector_store, inputs=search_terms, outputs=status)
    
    sum_btn.click(fn=generate_summaries, outputs=[sum_status, out])
    gr.Markdown(
        """
        [2025-05-17] initial deployment<br>
        """
        )
    
if __name__ == "__main__":
    print('v2025-05-17--2102')
    print('use `gradio app.py` to run in reload mode')
    demo.launch()
