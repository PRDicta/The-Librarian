


from typing import Dict, List, Optional, runtime_checkable
from typing import Protocol

from .types import ContentModality, Message


@runtime_checkable
class LLMAdapter(Protocol):


    async def extract(
        self,
        chunk: str,
        modality: ContentModality,
    ) -> List[Dict]:


        ...

    async def predict_topics(
        self,
        messages: List[Message],
    ) -> List[Dict]:


        ...
