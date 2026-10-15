import codecs
from collections.abc import AsyncIterator

import httpx

from portfolio_client.domain.errors import DomainError


async def bounded_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Bound network bytes before a server-controlled line can grow in memory."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, size = "", 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > 200_000:
            raise DomainError("AI_OUTPUT_BUDGET", "模型响应超过本地大小限制")
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")
