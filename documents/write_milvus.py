import multiprocessing
import os
import time
from multiprocessing.queues import Queue

from documents.milvus_db import MilvusVectorSave
from untils.log_utils import log
from documents.markdown_parser import MarkdownParser


# 采用分布式，多进程的方式把海量的数据写入到Milvus数据库
def file_parse_process(dir_path:str,output_queue:Queue,batch_size:int = 50):
    """
    进程1：解析目录下所有的md并分批放入队列
    :param dir_path: 需要保存到向量知识库的路径
    :param output_queue: #输出队列
    :param batch_size: #批处理数量（按Document条数计，不是按文件数）
    :return:
    """
    log.info(f"解析进程开始扫描目录：{dir_path}")
    # 获取目录下所有的.md文件
    md_files = [
        os.path.join(dir_path,f)
        for f in os.listdir(dir_path)
        if f.endswith(".md")
    ]

    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None) #发送终止信号
        return
    parser = MarkdownParser()
    doc_batch = []
    total_docs = 0
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if not docs:
                continue

            # 注意：这里必须 extend 展平成 Document 列表。
            # 用 append 会把「一个文件的文档列表」当作一个元素，
            # 导致 batch_size 数的是文件数，且下游要再嵌套遍历一次。
            doc_batch.extend(docs)
            total_docs += len(docs)

            #达到批次大小时发送到队列中
            while len(doc_batch) >= batch_size:
                output_queue.put(doc_batch[:batch_size])
                doc_batch = doc_batch[batch_size:]
        except Exception as e:
            log.error(f"解析失败{file_path}:{str(e)}")
            log.exception(e)
    #发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    #发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件，产出{total_docs}个文档片段")

def milvus_writer_process(input_queue:Queue,max_retry:int = 3):
    """进程2：从队列读取并写入Milvus"""
    log.info("milvus写入进程启动")

    mv = MilvusVectorSave()
    mv.connection(is_first=True)
    total_count = 0
    fail_count = 0
    while True:
        datas = input_queue.get() #阻塞的函数
        if datas is None: #收到了终止信号
            break
        if not isinstance(datas,list) or not datas:
            continue

        # 失败重试：embedding 接口偶发 400/限流，直接吞掉会静默丢数据
        for attempt in range(1,max_retry + 1):
            try:
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计写入:{total_count}个文档")
                break
            except Exception as e:
                log.error(f"第{attempt}/{max_retry}次写入失败（{len(datas)}条）:{str(e)}")
                if attempt == max_retry:
                    fail_count += len(datas)
                    log.exception(e)
                else:
                    time.sleep(2 ** attempt) #指数退避

    if fail_count:
        log.error(f"写入进程结束：成功{total_count}个文档，失败{fail_count}个文档")
    else:
        log.info(f"写入进程结束，总计写入{total_count}个文档")

if __name__ == "__main__":
    #配置参数
    md_dir = r'D:\ipython\RAG_PROJECT\datas\md'
    queue_maxsize = 20 #队列最大容量（防止内存溢出）

    #创建进程间通信队列
    docs_queue = multiprocessing.Queue(maxsize=queue_maxsize)

    #启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parse_process,
        args=(md_dir,docs_queue)
    )

    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()


    #等待进程结束
    parser_proc.join()
    writer_proc.join()
    print("系统提示：所有任务完成")

