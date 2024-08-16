import requests


class API:
    @staticmethod
    def get_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
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
            case "Worker" | "Team":
                r = session.get(
                    f"https://stablehorde.net/api/v2/workers/{arg}", headers=headers
                )

        data = r.json()
        if r.status_code == 200:
            return data
        else:
            return None

    @staticmethod
    def post_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg1: str = None,
        arg2: str | int = None,
    ):
        """
        Transfer kudos
        """
        match mode:
            case "TransferKudos" | "CreateTeam":
                headers = {
                    "accept": "application/json",
                    "apikey": apikey,
                }

        match mode:
            case "TransferKudos":
                payload = {
                    "username": arg1,
                    "amount": arg2,
                }
            case "CreateTeam":
                payload = {
                    "name": arg1,
                    "info": arg2,
                }

        match mode:
            case "TransferKudos":
                r = session.get(
                    "https://stablehorde.net/api/v2/kudos/transfer",
                    json=payload,
                    headers=headers,
                )
            case "CreateTeam":
                r = session.get(
                    "/v2/teams",
                    json=payload,
                    headers=headers,
                )

        data = r.json()
        if r.status_code == 200:
            return data
        elif r.status_code == 400:
            # Validation Error
            return None
        elif r.status_code == 401:
            # Invalid API Key Error
            return None
        elif r.status_code == 403:
            # Access Denied
            return None
        else:
            return None

    @staticmethod
    def patch_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg1: str = None,
        arg2: str = None,
        arg3: str = None,
    ):
        pass

    @staticmethod
    def put_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg1: str = None,
        arg2: str = None,
    ):
        pass

    @staticmethod
    def delete_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg1: str = None,
        arg2: str = None,
    ):
        pass
