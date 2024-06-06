import aiohttp

from typing import List, Optional


class HordeWorker:
    def __init__(self, worker_id: str, name: str, maintenance_mode: bool):
        self.id = worker_id
        self.name = name
        self.maintenance_mode = maintenance_mode

    @classmethod
    async def from_api(cls, session: aiohttp.ClientSession, worker_id: str):
        async with session.get(f"/api/v2/workers/{worker_id}") as response:
            json = await response.json()
            return cls(
                worker_id=worker_id,
                name=json["name"],
                maintenance_mode=json["maintenance_mode"],
            )


class HordeUser:
    def __init__(
        self, user_id: str, username: str, kudos: int, workers: List[HordeWorker]
    ):
        self.id = user_id
        self.username = username
        self.kudos = kudos
        self.workers = workers

    @classmethod
    async def from_api(cls, session: aiohttp.ClientSession):
        async with session.get("/api/v2/find_user") as response:
            json = await response.json()
            workers = []

            for worker_id in json["worker_ids"]:
                worker = await HordeWorker.from_api(session, worker_id)
                workers.append(worker)

            return cls(
                user_id=json["id"],
                username=json["username"],
                kudos=json["kudos"],
                workers=workers,
            )

    def get_worker(self, worker_id: str) -> Optional[HordeWorker]:
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
