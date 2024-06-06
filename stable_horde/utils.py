from typing import Any, Dict, Optional

import aiohttp
from urllib.parse import urljoin


class HordeRequestSession(aiohttp.ClientSession):
    def __init__(self, base_url: str, base_headers: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.base_url = base_url
        self.base_headers = base_headers or {}

    def _set_base_url(self, base_url: str):
        self.base_url = base_url

    async def request(self, method: str, url: str, *args: Any, **kwargs: Any):
        joined_url = urljoin(self.base_url, url)
        headers = kwargs.setdefault("headers", {})
        headers.update(self.base_headers)
        return await super().request(method, joined_url, *args, **kwargs)
