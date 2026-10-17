from langchain_community.document_loaders import UnstructuredMarkdownLoader


loader = UnstructuredMarkdownLoader(
    file_path="D:\ipython\RAG_PROJECT\datas\md\overview.md",
    mode="elements"
)
docs = loader.load()

print(f'doc的数量是：{len(docs)}')
for i, doc in enumerate(docs[:20]):
    print(f"\n第 {i} 个元素")
    print("element_id:", doc.metadata.get("element_id"))
    print("parent_id:", doc.metadata.get("parent_id"))
    print("category:", doc.metadata.get("category"))
    print("text:", doc.page_content[:100])

