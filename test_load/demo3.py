import json
from langchain_core.documents import Document





def load_doc_from_json(json_file):
    """从json文件加载为Document对象"""
    with open(json_file,'r',encoding='utf-8') as file:
        doce = json.load(file)
        return Document(page_content=doce['page_content'],metadata=doce['metadata'])


if __name__=='__main__':
    file_path = 'D:\ipython\RAG_PROJECT\datas\output\output1_2.json'
    res = load_doc_from_json(file_path)
    print(res)