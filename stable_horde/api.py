import requests

from stable_horde import (
    StableHordeConfig,
)

from modules import scripts

basedir = scripts.basedir()
config = StableHordeConfig(basedir)
session = requests.Session()


class API:
    @staticmethod
    def get_request(
        session: requests.Session = session,
        mode: str = None,
        apikey: str = config.apikey,
        arg: str = None,
    ) -> dict:
        """
        Make API requests
        """
        match mode:
            case "News" | "Stats" | "Status":
                headers = {
                    "accept": "application/json",
                }

            case "User" | "Worker" | "Team" | "Kudos":
                headers = {
                    "accept": "application/json",
                    "apikey": apikey,
                }

        match mode:
            case "News":
                r = session.get(
                    "https://stablehorde.net/api/v2/status/news", headers=headers
                )
            case "Stats":
                r = session.get(
                    "https://stablehorde.net/api/v2/stats/img/totals", headers=headers
                )
            case "Status":
                r = session.get(
                    "https://stablehorde.net/api/v2/status/modes", headers=headers
                )
            case "User" | "Kudos":
                r = session.get(
                    "https://stablehorde.net/api/v2/find_user", headers=headers
                )
            case "Worker":
                r = session.get(
                    f"https://stablehorde.net/api/v2/workers/{arg}", headers=headers
                )
            case "Team":
                r = session.get(
                    f"https://stablehorde.net/api/v2/workers/{arg}", headers=headers
                )

        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")

    @staticmethod
    def post_request(
        session: requests.Session = session,
        apikey: str = config.apikey,
        username: str = None,
        amount: int = None,
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
