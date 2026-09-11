from __future__ import annotations

from qdrant_client import QdrantClient


class QdrantConnection:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        timeout: float = 30.0,
        location: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.location = location
        self._client: QdrantClient | None = None

    def client(self) -> QdrantClient:
        if self._client is None:
            if self.location is not None:
                self._client = QdrantClient(location=self.location, timeout=int(self.timeout))
            else:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=int(self.timeout),
                )
        return self._client

    def health(self) -> bool:
        try:
            self.client().get_collections()
        except Exception:
            return False
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
