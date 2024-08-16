import requests

horde_url = "https://stablehorde.net/api/v2/"


class API:
    @staticmethod
    def request(
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
    ) -> dict:
        """
        Make API requests
        """
        match mode:
            case "News" | "Stats" | "Status":
                headers = {
                    "accept": "application/json",
                }

            case (
                "User"
                | "Worker"
                | "Team"
                | "Kudos"
                | "TransferKudos"
                | "CreateTeam"
                | "UpdateTeam"
                | "ModifySharedKey"
                | "CreateSharedKey"
                | "ModifyWorker"
                | "DeleteSharedKey"
                | "DeleteTeam"
                | "DeleteWorker"
            ):
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
            case "UpdateTeam":
                payload = {
                    "name": arg1,
                    "info": arg2,
                }
            case "ModifySharedKey" | "CreateSharedKey":
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
            case "News":
                r = session.get(f"{horde_url}status/news", headers=headers)
            case "Stats":
                r = session.get(f"{horde_url}stats/img/totals", headers=headers)
            case "Status":
                r = session.get(f"{horde_url}status/modes", headers=headers)
            case "User" | "Kudos":
                r = session.get(f"{horde_url}find_user", headers=headers)
            case "Worker" | "Team":
                r = session.get(f"{horde_url}workers/{arg0}", headers=headers)
            case "TransferKudos":
                r = session.post(
                    f"{horde_url}kudos/transfer",
                    json=payload,
                    headers=headers,
                )
            case "CreateTeam":
                r = session.post(
                    f"{horde_url}teams",
                    json=payload,
                    headers=headers,
                )
            case "UpdateTeam":
                r = session.patch(
                    f"{horde_url}teams/{arg0}",
                    json=payload,
                    headers=headers,
                )
            case "ModifySharedKey":
                r = session.patch(
                    f"{horde_url}sharedkeys/{arg0}",
                    json=payload,
                    headers=headers,
                )
            case "CreateSharedKey":
                r = session.put(
                    f"{horde_url}sharedkeys",
                    json=payload,
                    headers=headers,
                )
            case "ModifyWorker":
                r = session.put(
                    f"{horde_url}workers/{arg0}",
                    json=payload,
                    headers=headers,
                )
            case "DeleteSharedKey":
                r = session.delete(f"{horde_url}sharedkeys/{arg0}", headers=headers)
            case "DeleteTeam":
                r = session.delete(f"{horde_url}teams/{arg0}", headers=headers)
            case "DeleteWorker":
                r = session.delete(f"{horde_url}workers/{arg0}", headers=headers)

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
