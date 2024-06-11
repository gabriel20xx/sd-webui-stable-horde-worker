import requests


class HordeUser:
    def get_user_info(session: requests.Session, apikey: str):
        """
        Get user info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }

        r = session.get("https://stablehorde.net/api/v2/find_user", headers=headers)
        json = r.json()
        if r.status == 200:
            return json
        else:
            raise Exception(f"Error: {json.get('message')}")


class HordeWorker:
    def get_worker_info(session: requests.Session, apikey: str, worker_id: str):
        """
        Get worker info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }
        r = session.get(
            f"https://stablehorde.net/api/v2/workers/{worker_id}", headers=headers
        )
        json = r.json()
        if r.status == 200:
            return json
        else:
            raise Exception(f"Error: {json.get('message')}")
