import time
from typing import Dict, List, Optional, Tuple, Union
import json5

from agent_base_model import InternLM2Chat
from tools import Tools

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**Important**: Always use the {tool_names} tool for any question that requires external information.

Begin!
"""


class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = InternLM2Chat(path)

    def build_system_input(self):
        """
        构造上文中所说的系统提示词
        """
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        """
        解析第一次大模型返回选择的工具和工具参数
        """
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')

        print(f"Debug: Full response text:\n{text}")
        print(f"Debug: Action index: {i}, Action Input index: {j}, Observation index: {k}")

        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:'): j].strip()
            plugin_args = text[j + len('\nAction Input:'): k].strip()
            text = text[:k]

        print(f"Debug: Parsed plugin name: {plugin_name}, plugin args: {plugin_args}")

        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        """
        调用选择的工具
        Args:
            plugin_name: 插件名称
            plugin_args: 插件参数

        Returns:
            str: 插件返回结果
        """
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)

    def text_completion(self, text, history=[]):
        """
        调用大模型，整合两次调用
        Args:
            text: 输入文本
            history: 对话历史

        Returns:
            str: 补全后的文本
        """
        start_time = time.time()
        text = "\nQuestion:" + text
        response, his = self.model.chat(text, history, self.system_prompt)
        print(f"Model chat 1 took {time.time() - start_time:.2f} seconds")
        print(response)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            plugin_start_time = time.time()
            response += self.call_plugin(plugin_name, plugin_args)
            print(f"Plugin call took {time.time() - plugin_start_time:.2f} seconds")
        else:
            print(f"No plugin call detected, with response:{response}")

        start_time = time.time()
        response, his = self.model.chat(response, history, self.system_prompt)
        print(f"Model chat 2 took {time.time() - start_time:.2f} seconds")

        # Convert history to string and calculate its character length
        history_str = ''.join([str(item) for item in his])
        print(f"History character length: {len(history_str)}")
        print(f"Current history: {history_str}")

        return response, his


# if __name__ == '__main__':
#     agent = Agent('../Shanghai_AI_Laboratory/internlm2-chat-7b')
#     prompt = agent.build_system_input()
#     print(prompt)
if __name__ == '__main__':
    agent = Agent('../Shanghai_AI_Laboratory/internlm2-chat-7b')
    response, _ = agent.text_completion(text='你好', history=[])
    print(response)
    print('=====================')

    response, _ = agent.text_completion(text='你知道月涌大江流这句诗吗？', history=_)
    print(response)
    print('=====================')

    response, _ = agent.text_completion(text='这句诗的作者是？', history=_)
    print(response)
    print('=====================')
