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

def web_search(query="US tariff news, analysis, and predictions", max_search_res=10):
    search_res = list(search(query, num_results=max_search_res, unique=True))
    return search_res


def get_text(url):
    print(f'downloading {url}...')
    response = requests.get(url)
    html = response.text
    text = BeautifulSoup(html, features="html.parser").get_text().strip()
    return text


def create_vector_store(search_terms, max_search_res):
    running_status = ''
    urls = web_search(search_terms, int(max_search_res))
    documents = []
    for i,url in enumerate(urls):
        documents.append(get_text(url))
        status = f'downloading {i+1}/{len(urls)}: {url}'
        running_status += '\n' + status
        yield status, running_status
    ids = [f'id{i}' for i in range(len(documents))]
    metadatas = [{'source-url': url} for url in urls]

    # prevents client getting stale
    # https://github.com/langchain-ai/langchain/issues/26884
    chromadb.api.client.SharedSystemClient.clear_system_cache() 

    global chroma_client
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
    status = f'Indexing documents...'
    running_status += '\n' + status
    yield status, running_status

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    status = f'Document store recreated with {collection.count()} documents'
    running_status += '\n' + status
    yield status, running_status

def retrieve_documents(prompt, num_docs):
    collection = chroma_client.get_collection("webpages")
    retrieval = collection.query(
        query_texts=["what are the most important bits of news to know about US tariff policy and analysis of future actions?"],
        n_results=int(num_docs),
        include=["documents", "metadatas"]
    )
    return retrieval

def generate_summaries(prompt, num_docs):
    retrieval = retrieve_documents(prompt, num_docs)
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


def comprehensive_summary(prompt, num_docs):
    retrieval = retrieve_documents(prompt, num_docs)
    relevant_docs = retrieval['documents'][0]

    # limit is 12000 tokens ~ 48000 characters
    char_per_doc = int(12000 / len(relevant_docs) * 0.7)
    
    relevant_docs = [doc[:char_per_doc] for doc in relevant_docs]

    transcript = '\n------\n'.join(relevant_docs)

    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text."
            },
            {
                "role": "user",
                "content": f"Please provide an overall summary of the points from the following articles:\n\n{transcript}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    status_msg = f'generated single summary of {len(relevant_docs)} documents'
    
    return status_msg, chat_completion.choices[0].message.content

    
def disable_btn(b):
    return gr.update(interactive=False)
def enable_btn(b):
    return gr.update(interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## Tarrif Intel Tool

        Tool will
        - search for documents
        - pick the best ones
        - generate summaries
        """
        )


    with gr.Tab('Analysis'):
        #
        # search and vector store
        #
        with gr.Row('Search'):
            search_terms = gr.Dropdown(label='search query',
                                       show_label=True,
                                       allow_custom_value=True,
                                       choices=["US tariff news, analysis, and predictions",
                                                "world reaction to US tariff policy"
                                                ]
                                       )
            num_search_res = gr.Dropdown(label='maximum search results',
                                         show_label=True,
                                         allow_custom_value=True,
                                         choices=[10, 25, 50, 100]
                                         )
        update_btn = gr.Button('Search for documents')
            
        status = gr.Textbox(show_label=False)
            
            
        #
        # summarization
        #
        with gr.Row():
            num_relevant_docs = gr.Dropdown(label='extract relevant documents',
                                            show_label=True,
                                            allow_custom_value=True,
                                            choices=[5, 10, 25]
                                            )
            prompt = gr.Dropdown(label='prompt',
                                 show_label=True,
                                 allow_custom_value=True,
                                 choices=["what are the most important bits of news to know about US tariff policy and analysis of future actions?"
                                          ]
                                 )
        sum_btn = gr.Button('Generate summaries', interactive=False)
                
                
        sum_status = gr.Textbox(show_label=False)
        out = gr.TextArea(label='Summarizations', show_label=True)


    #
    # Tab for logging information
    #
    with gr.Tab('Log'):
        all_status = gr.TextArea(lines=25, show_label=False)
        

    #
    # event listeners 
    #
    update_btn.click(
        fn=disable_btn,
        inputs=update_btn,
        outputs=sum_btn
    ).then(
        fn=create_vector_store,
        inputs=[search_terms, num_search_res],
        outputs=[status, all_status]
    ).then(
        fn=enable_btn,
        inputs=update_btn,
        outputs=sum_btn
    )
    
    sum_btn.click(
        #fn=generate_summaries,
        fn=comprehensive_summary,
        inputs=[prompt, num_relevant_docs],
        outputs=[sum_status, out]
    )
    
    gr.Markdown(
        """
        [2025-05-17] initial deployment<br>
        """
        )
    
if __name__ == "__main__":
    print('v2025-05-17--2102')
    print('use `gradio app.py` to run in reload mode')
    demo.launch()
