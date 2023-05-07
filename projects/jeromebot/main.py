from pathlib import Path
from typing import List

from pydantic import BaseModel


"""
Facebook data dir is in the structure:
1/
    ...
    messages/
        inbox/
            ...
            <thread_id>/
                message_1.json
                message_2.json
2/
    ...
    messages/
        inbox/
            ...
            <thread_id>/
                audio/
                files/
                gifs/
                photos/
                videos/
"""
_FACEBOOK_DATA_ROOT_DIR = "~/source/bigdata/data/facebook/"
_FACEBOOK_DATA_ROOT_DIR = Path(_FACEBOOK_DATA_ROOT_DIR).expanduser().resolve()

_USER_NAME = "<|user|>"
_BOT_NAME = "<|assistant|>"
_END_TOKEN = "<|endoftext|>"

_BLOCKED_USER = "Facebook User"
_BLOCKED_USER_IN_MESSAGE = "Other user"


class Message(BaseModel):
    sender_name: str
    timestamp_ms: int
    content: str = None
    type: str
    is_unsent: bool


class User(BaseModel):
    name: str


class Conversation(BaseModel):
    participants: list[User]
    messages: list[Message]
