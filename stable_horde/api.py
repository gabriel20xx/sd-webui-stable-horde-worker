import requests


class HordeUser:
    @staticmethod
    def get_user_info(session: requests.Session, apikey: str):
        """
        Get user info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }

        r = session.get("https://stablehorde.net/api/v2/find_user", headers=headers)
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")


class HordeWorker:
    @staticmethod
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
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")


class HordeNews:
    @staticmethod
    def get_horde_news(session: requests.Session):
        """
        Get horde news
        """
        headers = {"accept": "application/json"}
        r = session.get("https://stablehorde.net/api/v2/status/news", headers=headers)
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")


class HordeStatus:
    @staticmethod
    def get_horde_status(session: requests.Session):
        """
        Get horde status
        """
        headers = {"accept": "application/json"}
        r = session.get("https://stablehorde.net/api/v2/status/modes", headers=headers)
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")


class KudoTransfer:
    @staticmethod
    def transfer_kudos(
        session: requests.Session, apikey: str, username: str, amount: int
    ):
        """
        Transfer kudos
        """
        payload = {"username": username, "amount": amount}
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }
        r = session.get(
            "https://stablehorde.net/api/v2/kudos/transfer",
            json=payload,
            headers=headers,
        )
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")
