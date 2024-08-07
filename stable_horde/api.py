import requests


class API:
    @staticmethod
    def get_user_info(session: requests.Session, apikey: str) -> dict:
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

    @staticmethod
    def get_worker_info(session: requests.Session, apikey: str, worker_id: str) -> dict:
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

    @staticmethod
    def get_team_info(session: requests.Session, apikey: str, team_id: str) -> dict:
        """
        Get worker info
        """
        headers = {
            "accept": "application/json",
            "apikey": apikey,
        }
        r = session.get(
            f"https://stablehorde.net/api/v2/workers/{team_id}", headers=headers
        )
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")

    @staticmethod
    def get_news_info(session: requests.Session) -> list:
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

    @staticmethod
    def get_stats_info(session: requests.Session) -> dict:
        headers = {
            "accept": "application/json",
        }
        r = session.get(
            "https://stablehorde.net/api/v2/stats/img/totals", headers=headers
        )
        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")

    @staticmethod
    def get_status_info(session: requests.Session) -> dict:
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
        elif r.status_code == 400:
            return "ValidationError"
        elif r.status_code == 401:
            return "InvalidAPIKeyError"
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")
