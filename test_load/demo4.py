from langchain_community.document_loaders import UnstructuredMarkdownLoader


loader = UnstructuredMarkdownLoader(
    file_path="D:\ipython\RAG_PROJECT\datas\md\overview.md",
    mode="elements"
)
docs = loader.load()