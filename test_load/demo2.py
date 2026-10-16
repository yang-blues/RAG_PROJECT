import json

from langchain_unstructured import UnstructuredLoader

from untils.env_utils import UNSTRUCTURED_KEY

pdf_file = r'D:\ipython\RAG_PROJECT\datas\layout-parser-paper.pdf'
def write_json(data,file_name):
    with open('D:\\ipython\\RAG_PROJECT\\datas\\output\\'+file_name,'w',encoding='utf-8') as file:
        #将 Python 中的 data 数据转换成 JSON 格式，并写入 file 文件中，同时保留中文字符，并使用 4 个空格进行缩进，让 JSON 文件更容易阅读。
        json.dump(data,file,ensure_ascii=False,indent=4)

#高精度的解析pdf
loader = UnstructuredLoader(
    file_path=pdf_file,#文件的路径
    strategy='hi_res',#解析的精度
    partition_via_api=True,#是本地还是api
    api_key=UNSTRUCTURED_KEY
)
docs = []
counter = 0
#load处理完一次性返回所有的文档，lazy_load边处理边返回
for doc in loader.lazy_load():
    docs.append(doc)
    json_file_name = str(doc.metadata.get('page_number')) + '_' + str(counter) + '.json'
    counter += 1
    write_json(doc.model_dump(),json_file_name)
print(f'doc的数量是：{len(docs)}')
print(docs[0].metadata)
print(docs[0].page_content)