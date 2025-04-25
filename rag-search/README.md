## search + RAG + summarization

Just wanted to do an exercise using different tools.  

In short, this
- does a Google search
- parses out the text with BeautifulSoup
- populates a Chroma vector store using Chroma's default embedding function
- prompts for the most important stories via Chroma's default model
- prompts for the summarization of each story using Llama (via Groq inference)

Of course, a prompt such as `Do a search for the 5 most recent articles on XXX and provide summaries` submitted to ChatGPT does essentially the same thing.

One advantage of this approach, however, is that is does allow for the possibility of automating the inclusion of articles not publicly available on the web.  These may consist of propriatary documents or documents behind a paywall, for example.  One can manually upload articles to ChatGPT, but that may not be feasible with a large number of documents.
