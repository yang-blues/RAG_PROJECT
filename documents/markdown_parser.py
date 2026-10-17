from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from untils.log_utils import log

class MarkdownParser:
    """
    专门负责markdown文件的解析和切片
    """



    def parse_markdown_to_documents(self,md_file:str,encoding='utf-8') -> List[Document]:
        docs = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度：{len(docs)}')
        merged_document = self.merge_title_content(docs)
        log.info(f'文件合并后的docs长度：{len(merged_document)}')
        return merged_document

    def parse_markdown(self,md_file:str) -> List[Document]:
        """做MD的初始解析切片封装成Document"""
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self,datas:List[Document]) -> List[Document]:
        '''收集符合特定条件的元素，合并整理'''
        merged_data = [] #最终整合好的数据
        parent_dict = {}#是一个字典，保存所有的父document,key为当前父document
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            if category == 'NarrativeText' and parent_id is None:
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + '->' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id :
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + '->' + document.page_content
                parent_dict[parent_id].metadata['category'] = 'content'
        #处理字典
        if parent_dict is not None:
            #先单独收集没有父标题的NarrativeText,再将字典中合并完成的标题Document全部追加到结果列表
            merged_data.extend(parent_dict.values())
        return merged_data

if __name__ == '__main__':
    file_path = "D:\ipython\RAG_PROJECT\datas\md\overview.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for doc in docs:
        print("text:", doc.page_content[:100])


