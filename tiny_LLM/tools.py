import os, json
import requests

"""
工具函数

- 首先要在 tools 中添加工具的描述信息
- 然后在 tools 中添加工具的具体实现

- https://serper.dev/dashboard
"""


class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
        self.api_key = self._load_api_key()

    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def _load_api_key(self):
        with open('agent_api_key.json', 'r') as file:
            config = json.load(file)
        return config['api_key']

    def google_search(self, search_query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": search_query})
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()

        # 汇聚所有有机搜索结果的标题和摘要，并添加序号
        result = ' '.join([f"线索{str(index + 1).zfill(2)}: {item['title']}: {item['snippet']}"
                           for index, item in enumerate(response['organic'])])
        return result
        # result = response['organic'][0]['snippet']
        # return result
