import os
import sys
import datetime
import hashlib
import logging
import json
from typing import Iterator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    BaseMessageChunk,
)
from langchain.output_parsers.json import parse_json_markdown
from langchain_openai import ChatOpenAI


class Agent:
    def __init__(self, prompt="", name="", log_level=logging.INFO, **model_args):
        self.prompt = prompt
        self.log_level = log_level
        if not name:
            self.name = f"Agent-{hashlib.md5(self.prompt.encode()).hexdigest()}"

        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)

        self.logger = logging.getLogger(self.name)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(self.log_level)

        self.usage_logger = logging.getLogger(f"usage.{self.name}")
        self.usage_logger.setLevel(self.log_level)
        self.usage_logger.addHandler(logging.FileHandler(f"logs/usage.{self.name}.log"))

        self.model = ChatOpenAI(
            model=model_args.pop("llm_model", os.getenv("LLM_MODEL", "qwen-plus")),
            temperature=0.2,
            max_tokens=2000,
            api_key=model_args.pop("llm_api_key", os.getenv("API_KEY")),
            base_url=model_args.pop(
                "llm_base_url",
                os.getenv(
                    "LLM_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            ),
            **model_args,
        )

    def __invoke(self, query: str, history=None, stream=False, **prompt_kwargs) -> str:
        if history is None:
            history: list = []

        today = str(datetime.datetime.now().date())
        system_msg = SystemMessage(self.prompt.format(today=today, **prompt_kwargs))
        messages = [system_msg]
        messages.extend(history)
        messages.append(HumanMessage(query))

        self.logger.debug(f"{self.name} __invoke messages: {messages}")

        if stream:
            return self.model.stream(messages)
        return self.model.invoke(messages)

    def __log_usage(self, msg: BaseMessage, **_):
        data = {
            **msg.response_metadata["token_usage"],
            "time": str(datetime.datetime.now()),
            "agent": self.name,
            "model_name": msg.response_metadata.get("model_name", "unknown")
        }
        self.usage_logger.info(json.dumps(data))

    def invoke(self, query: str, history=None, **kwargs) -> str:
        if history is None:
            history = []
        msg: BaseMessage = self.__invoke(query, history, **kwargs)

        self.logger.debug(f"{self.name} invoke return msg: {msg}")
        self.__log_usage(msg)
        return msg.content

    def invoke_json(self, query: str, history=None, retry_count: int = 1, **kwargs) -> dict[str, any]:
        if history is None:
            history = []

        count = 0
        while count < retry_count:
            try:
                msg: BaseMessage = self.__invoke(query, history, **kwargs)

                self.logger.debug(f"{self.name} invoke_json return msg: {msg}")
                self.__log_usage(msg)

                parsed = parse_json_markdown(msg.content)
                return parsed

            except Exception as e:
                self.logger.error(f"{self.name} invoke_json error: {e}")
            finally:
                count += 1
        return {}

    def stream(self, query: str, history=None, **kwargs) -> Iterator[BaseMessageChunk]:
        if history is None:
            history = []
        return self.__invoke(query, history, stream=True, **kwargs)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    agent = Agent()
    print(agent.invoke("你好"))
