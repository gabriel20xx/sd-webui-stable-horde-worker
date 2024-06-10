import aiohttp


class HordeUser:
    async def get_user_info(self, session: aiohttp.ClientSession, apikey: str):
        """
        Get user info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }

        r = await session.get(
            "https://stablehorde.net/api/v2/find_user", headers=headers
        )
        json = await r.json()
        if r.status == 200:
            return json
        else:
            raise Exception(f"Error: {json.get('message')}")


class HordeWorker:
    async def get_worker_info(
        self, session: aiohttp.ClientSession, apikey: str, worker_id: str
    ):
        """
        Get worker info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }
        r = await session.get(
            f"https://stablehorde.net/api/v2/workers/{worker_id}", headers=headers
        )
        json = await r.json()
        if r.status == 200:
            return json
        else:
            raise Exception(f"Error: {json.get('message')}")
