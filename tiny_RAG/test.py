from tiny_RAG.vector_base import VectorStore
from tiny_RAG.embeddings import JinaEmbedding
from tiny_RAG.llm import InternLMChat
from tiny_RAG.utils import ReadFiles

# 建立向量数据库
# docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)  # 读取data文件下的所有内容并分割
# vector = VectorStore(docs)
# embedding_model = JinaEmbedding(path='../jinaai/jina-embeddings-v2-base-zh', is_api=False)
# vector.get_vector(EmbeddingModel=embedding_model)
# vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# 连续问答
vector = VectorStore()
vector.load_vector(path='storage')  # 从本地加载数据库

embedding_model = JinaEmbedding(path='../jinaai/jina-embeddings-v2-base-zh', is_api=False)
chat = InternLMChat(path='../Shanghai_AI_Laboratory/internlm2-chat-7b')

history = []

while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    content = vector.query(question, EmbeddingModel=embedding_model, k=1)[
        0]  # Retrieve relevant document from the database
    answer = chat.chat(prompt=question, history=history, content=content)  # Get the answer using InternLM
    print(f"Answer: {answer}")

    # 将用户问题和模型回答组成一个 tuple，再添加到 history 中
    history.append((question, answer))