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
                r = session.post(
                    "https://stablehorde.net/api/v2/kudos/transfer",
                    json=payload,
                    headers=headers,
                )
            case "CreateTeam":
                r = session.post(
                    "https://stablehorde.net/api/v2/teams",
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
        arg0: str = None,
        arg1: str = None,
        arg2: str = None,
        arg3: str = None,
        arg4: str = None,
        arg5: str = None,
        arg6: str = None,
    ):
        match mode:
            case "UpdateTeam" | "ModifySharedKey":
                headers = {
                    "accept": "application/json",
                    "apikey": apikey,
                }

        match mode:
            case "UpdateTeam":
                payload = {
                    "name": arg1,
                    "info": arg2,
                }
            case "ModifySharedKey":
                payload = {
                    "kudos": arg1,
                    "expiry": arg2,
                    "name": arg3,
                    "max_image_pixels": arg4,
                    "max_image_steps": arg5,
                    "max_text_tokens": arg6,
                }

        match mode:
            case "UpdateTeam":
                r = session.patch(
                    f"https://stablehorde.net/api/v2/teams/{arg0}",
                    json=payload,
                    headers=headers,
                )
            case "ModifySharedKey":
                r = session.patch(
                    f"https://stablehorde.net/api/v2/sharedkeys/{arg0}",
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
        elif r.status_code == 404:
            # Team Or Shared Key Not Found
            return None
        else:
            return None

    @staticmethod
    def put_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg0: str = None,
        arg1: str = None,
        arg2: str = None,
        arg3: str = None,
        arg4: str = None,
        arg5: str = None,
        arg6: str = None,
    ):
        match mode:
            case "CreateSharedKey" | "ModifyWorker":
                headers = {
                    "accept": "application/json",
                    "apikey": apikey,
                }

        match mode:
            case "CreateSharedKey":
                payload = {
                    "kudos": arg1,
                    "expiry": arg2,
                    "name": arg3,
                    "max_image_pixels": arg4,
                    "max_image_steps": arg5,
                    "max_text_tokens": arg6,
                }
            case "ModifyWorker":
                payload = {
                    "maintenance": arg1,
                    "maintenance_msg": arg2,
                    "paused": arg3,
                    "info": arg4,
                    "name": arg5,
                    "team": arg6,
                }

        match mode:
            case "CreateSharedKey":
                r = session.put(
                    "https://stablehorde.net/api/v2/sharedkeys",
                    json=payload,
                    headers=headers,
                )
            case "ModifyWorker":
                r = session.put(
                    f"https://stablehorde.net/api/v2/workers/{arg0}",
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
        elif r.status_code == 404:
            # Team Or Shared Key Not Found
            return None
        else:
            return None

    @staticmethod
    def delete_request(
        session: requests.Session = requests.Session(),
        mode: str = None,
        apikey: str = None,
        arg: str = None,
    ):
        match mode:
            case "DeleteSharedKey" | "DeleteTeam" | "DeleteWorker":
                headers = {
                    "accept": "application/json",
                    "apikey": apikey,
                }

        match mode:
            case "DeleteSharedKey":
                r = session.delete(
                    f"https://stablehorde.net/api/v2/sharedkeys/{arg}", headers=headers
                )
            case "DeleteTeam":
                r = session.delete(
                    f"https://stablehorde.net/api/v2/teams/{arg}", headers=headers
                )
            case "DeleteWorker":
                r = session.delete(
                    f"https://stablehorde.net/api/v2/workers/{arg}", headers=headers
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
        elif r.status_code == 404:
            # Team, Worker Or Shared Key Not Found
            return None
        else:
            return None
