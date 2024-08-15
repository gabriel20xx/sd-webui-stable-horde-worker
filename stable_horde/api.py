import requests


class API:
    @staticmethod
    def api_get_request(
        session: requests.Session, mode: str, arg1: str = None, arg2: str = None
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
                    "apikey": arg1,
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
                    f"https://stablehorde.net/api/v2/workers/{arg2}", headers=headers
                )
            case "Team":
                r = session.get(
                    f"https://stablehorde.net/api/v2/workers/{arg2}", headers=headers
                )

        data = r.json()
        if r.status_code == 200:
            return data
        else:
            raise Exception(f"Error: {data.get('message'), 'Unknown API error'}")

    @staticmethod
    def api_post_request(
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
