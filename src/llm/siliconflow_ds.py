import os
import json
import asyncio
import warnings
from typing import Dict, List, Optional, Tuple, Iterator, AsyncIterator

import aiohttp
import requests
from dotenv import load_dotenv
from pydantic import Field, SecretStr, SkipValidation

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ChatMessage,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
    agenerate_from_stream,
)
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk

# Exception raised for SiliconFlow API errors.
class SiliconFlowAPIError(Exception):
    """Exception raised for errors from the SiliconFlow API."""


class ChatSiliconFlow(BaseChatModel):
    """
    SiliconFlow chat model integration.

    To use, set the SILICONFLOW_API_KEY environment variable.
    """

    # Primary parameters
    api_key: SecretStr = Field(..., alias="api_key")
    base_url: str = Field(default="https://api.siliconflow.com/v1")
    model: str = Field(default="deepseek-ai/DeepSeek-V3")
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    include_response_headers: bool = False

    # Additional generation parameters
    top_p: float = 0.7
    top_k: int = 50
    frequency_penalty: float = 0.5
    n: int = 1
    stop: Optional[List[str]] = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "siliconflow-ds-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "SILICONFLOW_API_KEY"}

    @property
    def _default_params(self) -> Dict[str, str]:
        """Get the default parameters used when calling the SiliconFlow API."""
        params = {
            "model": self.model,
            "stream": self.streaming,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 512,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
        }
        if self.stop is not None:
            params["stop"] = self.stop
        return params

    def _create_message_dicts(
        self, messages: List[str], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Convert messages into a list of dicts, and return request parameters."""
        params = self._default_params.copy()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _convert_message_to_dict(self, message: str) -> Dict[str, str]:
        """
        Convert a LangChain message into a dict acceptable by the SiliconFlow API.

        For instance, HumanMessage maps to {'role': 'user', ...},
        AIMessage maps to {'role': 'assistant', ...}, etc.
        """
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            msg = {"role": "assistant", "content": message.content}
            if message.additional_kwargs.get("function_call"):
                msg["function_call"] = message.additional_kwargs["function_call"]
                if msg["content"] == "":
                    msg["content"] = None
            return msg
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            return {"role": "function", "content": message.content, "name": message.name}
        elif isinstance(message, ChatMessage):
            return {"role": message.role, "content": message.content}
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

    def _convert_dict_to_message(self, data: Dict[str, str]) -> str:
        """
        Convert a SiliconFlow API response dict into a LangChain message.

        This helper ensures that "user" returns a HumanMessage, "assistant" returns an AIMessage, etc.
        """
        role = data.get("role", "")
        if role == "user":
            return HumanMessage(content=data.get("content", ""))
        elif role == "assistant":
            additional_kwargs = {}
            if "function_call" in data:
                additional_kwargs["function_call"] = data["function_call"]
            return AIMessage(content=data.get("content", ""), additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=data.get("content", ""))
        elif role == "function":
            return FunctionMessage(content=data.get("content", ""), name=data.get("name", ""))
        else:
            return ChatMessage(content=data.get("content", ""), role=role)

    def _create_chat_result(
        self, response: Dict[str, str], generation_info: Optional[Dict[str, str]] = None
    ) -> ChatResult:
        """Convert the API response into a ChatResult object."""
        generations = []
        token_usage = response.get("usage", {})
        choices = response.get("choices", [])
        for choice in choices:
            message_data = choice.get("message", {})
            message = self._convert_dict_to_message(message_data)
            gen_info = {"finish_reason": choice.get("finish_reason")}
            if "logprobs" in choice:
                gen_info["logprobs"] = choice.get("logprobs")
            generation = ChatGeneration(message=message, generation_info=gen_info)
            generations.append(generation)
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self, messages: List[str], stop: Optional[List[str]] = None, **kwargs: str
    ) -> ChatResult:
        """Synchronous generation using the SiliconFlow API."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, "messages": message_dicts}
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self.streaming:
            stream_iter = self._stream(messages, stop=stop, **kwargs)
            return generate_from_stream(stream_iter)
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
        except Exception as e:
            raise SiliconFlowAPIError(f"Error during API call: {str(e)}") from e
        return self._create_chat_result(response_json)

    async def _agenerate(
        self, messages: List[str], stop: Optional[List[str]] = None, **kwargs: str
    ) -> ChatResult:
        """Asynchronous generation using the SiliconFlow API."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, "messages": message_dicts}
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
        except Exception as e:
            raise SiliconFlowAPIError(f"Error during async API call: {str(e)}") from e
        return self._create_chat_result(result)

    def _stream(
        self, messages: List[str], stop: Optional[List[str]] = None, **kwargs: str
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream synchronous responses from the SiliconFlow API.

        This method yields ChatGenerationChunk objects as chunks of data arrive.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, "messages": message_dicts, "stream": True}
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            with requests.post(self.endpoint, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        if not isinstance(chunk_data, dict):
                            continue
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        text = delta.get("content", "")
                        chunk_message = AIMessage(content=text)
                        generation_info = {}
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            generation_info["finish_reason"] = finish_reason
                        yield ChatGenerationChunk(message=chunk_message, generation_info=generation_info)
        except Exception as e:
            raise SiliconFlowAPIError(f"Error during streaming API call: {str(e)}") from e

    async def _astream(
        self, messages: List[str], stop: Optional[List[str]] = None, **kwargs: str
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Stream asynchronous responses from the SiliconFlow API.

        Yields ChatGenerationChunk objects as soon as chunks arrive.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, "messages": message_dicts, "stream": True}
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.content:
                        if line:
                            try:
                                chunk_data = json.loads(line.decode("utf-8"))
                            except Exception:
                                continue
                            if not isinstance(chunk_data, dict):
                                continue
                            choices = chunk_data.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            text = delta.get("content", "")
                            chunk_message = AIMessage(content=text)
                            generation_info = {}
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                generation_info["finish_reason"] = finish_reason
                            yield ChatGenerationChunk(message=chunk_message, generation_info=generation_info)
        except Exception as e:
            raise SiliconFlowAPIError(f"Error during async streaming API call: {str(e)}") from e


# Example usage:
if __name__ == "__main__":
    async def main():
        load_dotenv()
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable not set")

        # Wrap api_key in SecretStr for Pydantic.
        from pydantic import SecretStr
        secret_api_key = SecretStr(api_key)

        chat = ChatSiliconFlow(api_key=secret_api_key)
        messages = [HumanMessage(content="Hello! How are you?")]

        # Synchronous invocation:
        response = chat.invoke(messages)
        print("Synchronous response:")
        print(response)

        # Asynchronous invocation:
        async_response = await chat.ainvoke(messages)
        print("Asynchronous response:")
        print(async_response)

    asyncio.run(main())
