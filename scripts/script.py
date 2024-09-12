from typing import Optional

from fastapi import FastAPI
import gradio as gr
import asyncio
import requests
from threading import Thread

from modules import scripts, script_callbacks, sd_models, shared

from stable_horde import (
    StableHorde,
    StableHordeConfig,
    API,
)

basedir = scripts.basedir()
config = StableHordeConfig(basedir)
horde = StableHorde(basedir, config)
session = requests.Session()
api = API()


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    started = False

    @app.on_event("startup")
    @app.get("/horde/startup-events")
    async def startup_event():
        nonlocal started
        if not started:
            thread = Thread(daemon=True, target=horde_thread)
            thread.start()
            started = True

    # This is a hack to make sure the startup event is
    # called even it is not in an async scope
    # fix https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/issues/109
    if demo is None:
        # flake8: noqa: E501
        local_url = f"http://localhost:{shared.cmd_opts.port if shared.cmd_opts.port else 7861}/"
    else:
        local_url = demo.local_url
    requests.get(f"{local_url}horde/startup-events")


def horde_thread():
    asyncio.run(horde.run())


def apply_stable_horde_apikey(apikey: str):
    config.apikey = apikey
    config.save()
    gr.Info("API Key Saved")


# Apply Settings
def apply_stable_horde_settings(
    enable: bool,
    name: str,
    apikey: str,
    allow_img2img: bool,
    allow_painting: bool,
    allow_unsafe_ipaddr: bool,
    allow_post_processing,
    restore_settings: bool,
    nsfw: bool,
    interval: int,
    max_pixels: str,
    endpoint: str,
    show_images: bool,
    save_images: bool,
    save_source_images: bool,
    save_images_folder: str,
):
    config.enabled = enable
    config.allow_img2img = allow_img2img
    config.allow_painting = allow_painting
    config.allow_unsafe_ipaddr = allow_unsafe_ipaddr
    config.allow_post_processing = allow_post_processing
    config.restore_settings = restore_settings
    config.interval = interval
    config.endpoint = endpoint
    config.apikey = apikey
    config.name = name
    config.max_pixels = int(max_pixels)
    config.nsfw = nsfw
    config.show_image_preview = show_images
    config.save_images = save_images
    config.save_source_images = save_source_images
    config.save_images_folder = save_images_folder
    config.save()
    gr.Info("Config Saved")

    return (
        f'Status: {"Running" if config.enabled else "Stopped"}',
        "Running Type: Image Generation",
    )


tab_prefix = "stable-horde-"


# Dictionary transformation
def transform_key(key, type=None):
    """Transform a key by replacing underscores with spaces, capitalizing each word, and replacing the id."""
    if type == "user" and key == "id":
        key = "user_id"
    if type == "worker" and key == "id":
        key = "worker_id"
    return key


def transform_dict(d, type=None):
    """Recursively transform the keys of a dictionary."""
    if isinstance(d, dict):
        return {transform_key(k, type): transform_dict(v, type) for k, v in d.items()}
    elif isinstance(d, list):
        return [transform_dict(i, type) for i in d]
    else:
        return d


# Fetch the api info
def fetch_api_info(mode: str, arg=None):
    match mode:
        case "News" | "Stats" | "Status":
            data = api.request(session, mode)
        case "User" | "Kudos":
            data = api.request(session, mode, config.apikey)
        case "Worker" | "Team":
            data = api.request(session, mode, config.apikey, arg)

    if data:
        transformed_dict = transform_dict(data, mode)
        return transformed_dict
    else:
        return None


# Generator UI
def get_generator_ui():
    tab_prefix = "stable-horde-"
    with gr.Blocks() as generator_ui:
        with gr.Row():
            gr.Markdown(
                "## Generations",
                elem_id="kudos_title",
            )
        with gr.Row():
            with gr.Column(elem_id="stable-horde"):
                current_id = gr.Textbox(
                    "Current ID: ",
                    label="",
                    elem_id=tab_prefix + "current-id",
                    readonly=True,
                    lines=1,
                    max_lines=1,
                )

                # Running type and state is currently not in use
                running_type = gr.Textbox(
                    "Image Generation",
                    label="Running Type",
                    elem_id=tab_prefix + "running-type",
                    readonly=True,
                    visible=False,
                    lines=1,
                    max_lines=1,
                )

                state = gr.HTML(
                    "",
                    label="State",
                    elem_id=tab_prefix + "state",
                    visible=False,
                    readonly=True,
                )

                log = gr.HTML(elem_id=tab_prefix + "log")

            with gr.Column(elem_id="stable-horde"):
                refresh = gr.Button(
                    "Refresh",
                    visible=False,
                    elem_id=tab_prefix + "refresh",
                )
                refresh_image = gr.Button(
                    "Refresh Image",
                    visible=False,
                    elem_id=tab_prefix + "refresh-image",
                )

                preview = gr.Gallery(
                    label="Preview",
                    elem_id=tab_prefix + "preview",
                    visible=config.show_image_preview,
                    readonly=True,
                    columns=4,
                )

                def on_refresh(image=False, show_images=config.show_image_preview):
                    cid = f"Current ID: {horde.state.id}"
                    html = "".join(
                        map(
                            lambda x: f"<p>{x[0]}: {x[1]}</p>",
                            horde.state.to_dict().items(),
                        )
                    )
                    images = (
                        [horde.state.image] if horde.state.image is not None else []
                    )
                    if image and show_images:
                        return cid, html, horde.state.status, images
                    return cid, html, horde.state.status

                # Click functions
                refresh.click(
                    fn=lambda: on_refresh(),
                    outputs=[current_id, log, state],
                    show_progress=False,
                )

                refresh_image.click(
                    fn=lambda: on_refresh(True),
                    outputs=[
                        current_id,
                        log,
                        state,
                        preview,
                    ],
                    show_progress=False,
                )

    return generator_ui


# Worker UI
def get_worker_ui(worker):
    with gr.Blocks() as worker_ui:
        if worker:
            worker_info = fetch_api_info("Worker", worker)

        gr.Markdown("## Worker Details")
        worker_update = gr.Button("Update Worker Details", elem_id="worker-update")

        name = gr.Textbox(
            value=worker_info.get("name"),
            label="Name",
            elem_id="name",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        id = gr.Textbox(
            value=worker_info.get("id"),
            label="Worker ID",
            elem_id="id",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        type = gr.Textbox(
            value=worker_info.get("type"),
            label="Type",
            elem_id="type",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        online = gr.Textbox(
            value=worker_info.get("online"),
            label="Online",
            elem_id="online",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        requests_fulfilled = gr.Textbox(
            value=worker_info.get("requests_fulfilled"),
            label="Requests Fulfilled",
            elem_id="requests_fulfilled",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        kudos_rewards = gr.Textbox(
            value=worker_info.get("kudos_rewards"),
            label="Kudos Rewards",
            elem_id="kudos_rewards",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        performance = gr.Textbox(
            value=worker_info.get("performance"),
            label="Performance",
            elem_id="performance",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        maintenance_mode = gr.Textbox(
            value=worker_info.get("maintenance_mode"),
            label="Maintenance Mode",
            elem_id="maintenance_mode",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        nsfw = gr.Textbox(
            value=worker_info.get("nsfw"),
            label="NSFW",
            elem_id="nsfw",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        trusted = gr.Textbox(
            value=worker_info.get("trusted"),
            label="Trusted",
            elem_id="trusted",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        flagged = gr.Textbox(
            value=worker_info.get("flagged"),
            label="Flagged",
            elem_id="flagged",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        uncompleted_jobs = gr.Textbox(
            value=worker_info.get("uncompleted_jobs"),
            label="Uncompleted Jobs",
            elem_id="uncompleted_jobs",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        models_value = ", ".join(str(model) for model in worker_info.get("models", []))
        models = gr.Textbox(
            value=models_value,
            label="Models",
            elem_id="models",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        max_pixels = gr.Textbox(
            value=worker_info.get("max_pixels"),
            label="mMax Pixels",
            elem_id="max_pixels",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        megapixelsteps_generated = gr.Textbox(
            value=worker_info.get("megapixelsteps_generated"),
            label="Megapixelsteps Generated",
            elem_id="megapixelsteps_generated",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        img2img = gr.Textbox(
            value=worker_info.get("img2img"),
            label="Img 2 Img",
            elem_id="img2img",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        painting = gr.Textbox(
            value=worker_info.get("painting"),
            label="Painting",
            elem_id="painting",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        post_processing = gr.Textbox(
            value=worker_info.get("post-processing"),
            label="Post-Processing",
            elem_id="post_processing",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        lora = gr.Textbox(
            value=worker_info.get("lora"),
            label="LoRa",
            elem_id="lora",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        controlnet = gr.Textbox(
            value=worker_info.get("controlnet"),
            label="ControlNet",
            elem_id="controlnet",
            interactive=False,
            lines=1,
            max_lines=1,
        )
        sdxl_controlnet = gr.Textbox(
            value=worker_info.get("sdxl_controlnet"),
            label="SDXL ControlNet",
            elem_id="sdxl_controlnet",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        def update_worker_info():
            worker_info_updated = fetch_api_info("Worker", worker)
            keys = [
                "name",
                "id",
                "type",
                "online",
                "requests_fulfilled",
                "kudos_rewards",
                "performance",
                "maintenance_mode",
                "nsfw",
                "trusted",
                "flagged",
                "uncompleted_jobs",
                "models",
                "max_pixels",
                "megapixelsteps_generated",
                "img2img",
                "painting",
                "post_processing",
                "lora",
                "controlnet",
                "sdxl_controlnet",
            ]
            return [worker_info_updated.get(key) for key in keys]

        worker_update.click(
            fn=update_worker_info,
            inputs=[],
            outputs=[
                name,
                id,
                type,
                online,
                requests_fulfilled,
                kudos_rewards,
                performance,
                maintenance_mode,
                nsfw,
                trusted,
                flagged,
                uncompleted_jobs,
                models,
                max_pixels,
                megapixelsteps_generated,
                img2img,
                painting,
                post_processing,
                lora,
                controlnet,
                sdxl_controlnet,
            ],
        )

    return worker_ui


# User UI
def get_user_ui():
    """Creates and returns Gradio UI components based on the user info."""
    with gr.Blocks() as user_ui:
        user_info = fetch_api_info("User")

        gr.Markdown("## User Details")
        user_update = gr.Button("Update User Details", elem_id="user-update")

        username = gr.Textbox(
            value=user_info.get("username"),
            label="Username",
            elem_id="username",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        id = gr.Textbox(
            value=user_info.get("id"),
            label="ID",
            elem_id="id",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        kudos = gr.Textbox(
            value=user_info.get("kudos"),
            label="Kudos",
            elem_id="kudos",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        concurrency = gr.Textbox(
            value=user_info.get("concurrency"),
            label="Concurrency",
            elem_id="concurrency",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        worker_count = gr.Textbox(
            value=user_info.get("worker_count"),
            label="user_info Count",
            elem_id="worker_count",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        worker_ids = gr.Textbox(
            value=user_info.get("worker_ids"),
            label="Worker IDs",
            elem_id="worker_ids",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        trusted = gr.Textbox(
            value=user_info.get("trusted"),
            label="Trusted",
            elem_id="trusted",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        flagged = gr.Textbox(
            value=user_info.get("flagged"),
            label="Flagged",
            elem_id="flagged",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        vpn = gr.Textbox(
            value=user_info.get("vpn"),
            label="VPN",
            elem_id="vpn",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        image_fulfillment = gr.Textbox(
            value=user_info.get("image_fulfillment"),
            label="Image Fulfillment",
            elem_id="image_fulfillment",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        def update_user_info():
            user_info_updated = fetch_api_info("User")
            keys = [
                "username",
                "id",
                "kudos",
                "concurrency",
                "worker_count",
                "worker_ids",
                "trusted",
                "flagged",
                "vpn",
                "image_fulfillment",
            ]
            # Extract values for the keys
            user_info_values = [user_info_updated.get(key) for key in keys]

            # Process the worker_ids to remove brackets and single quotes
            worker_ids_index = keys.index("worker_ids")
            worker_ids = user_info_values[worker_ids_index]

            if isinstance(worker_ids, list):
                # Convert list to a string, remove brackets, and remove single quotes
                user_info_values[worker_ids_index] = ", ".join(worker_ids)

            return user_info_values

        user_update.click(
            fn=update_user_info,
            inputs=[],
            outputs=[
                username,
                id,
                kudos,
                concurrency,
                worker_count,
                worker_ids,
                trusted,
                flagged,
                vpn,
                image_fulfillment,
            ],
        )
    return user_ui


# Team UI
def get_team_ui():
    with gr.Blocks() as team_ui:
        gr.Markdown("## Team Details")
        team_id = gr.Textbox(
            label="Team ID",
            placeholder="Enter Team ID",
            elem_id="team_id",
            interactive=True,
            lines=1,
            max_lines=1,
        )
        team_update = gr.Button("Update Team Details", elem_id="team-update")

        name = gr.Textbox(
            value=None,
            label="Name",
            elem_id="name",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        id = gr.Textbox(
            value=None,
            label="ID",
            elem_id="id",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        info = gr.Textbox(
            value=None,
            label="Info",
            elem_id="info",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        requests_fulfilled = gr.Textbox(
            value=None,
            label="Requests Fulfilled",
            elem_id="requests_fulfilled",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        kudos = gr.Textbox(
            value=None,
            label="Kudos",
            elem_id="kudos",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        worker_count = gr.Textbox(
            value=None,
            label="Worker Count",
            elem_id="worker_count",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        def update_team_info(team_id):
            if not team_id or team_id == None:
                gr.Error("Please provide a Team ID")
                return [None] * 6

            team_info_updated = fetch_api_info("Team", team_id)
            keys = [
                "name",
                "id",
                "info",
                "requests_fulfilled",
                "kudos",
                "worker_count",
            ]
            return [team_info_updated.get(key) for key in keys]

        team_update.click(
            fn=update_team_info,
            inputs=[team_id],
            outputs=[
                name,
                id,
                info,
                requests_fulfilled,
                kudos,
                worker_count,
            ],
        )

    return team_ui


# Kudos UI
def get_kudos_ui():
    with gr.Blocks() as kudos_ui:

        def update_kudos_info():
            kudos_info_updated = fetch_api_info("User")
            keys = [
                "kudos",
            ]
            return [kudos_info_updated.get(key) for key in keys]

        # Kudos functions
        user_info = fetch_api_info("Kudos")

        # Kudos UI
        with gr.Row():
            # Transfer Kudos Title
            gr.Markdown(
                "## Transfer Kudos",
                elem_id="kudos_title",
            )

        with gr.Row():
            update_kudos = gr.Button(
                "Update Kudos",
                variant="secondary",
                elem_id="update_kudos_button",
            )
        with gr.Row():
            with gr.Column():
                # Username
                recipient = gr.Textbox(
                    label="Recipient Username",
                    placeholder="Enter recipient username",
                    elem_id="kudos_recipient",
                    interactive=True,
                    lines=1,
                    max_lines=1,
                )

            with gr.Column():
                validate = gr.Button(
                    "Validate",
                    variant="secondary",
                    elem_id="kudos_validate_button",
                )
                validate_output = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                # Kudo amount display
                kudos = int(str(user_info["kudos"]).strip("[]").split(".")[0])
                your_kudos = gr.Textbox(
                    value=kudos,
                    label="Your Kudos",
                    placeholder="0",
                    elem_id="kudos_display",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )

            with gr.Column():
                # Transfer Kudo amount
                kudos_amount = gr.Slider(
                    label="Kudos",
                    minimum=0,
                    maximum=user_info["kudos"],
                    step=1,
                    value=10,
                    elem_id="kudos_amount",
                    interactive=True,
                )

        with gr.Row():
            # Transfer Button
            transfer = gr.Button(
                "Transfer",
                variant="primary",
                elem_id="kudos_transfer_button",
            )

        def validate_username(username):
            # Todo
            result = api.request(session, "TransferKudos", config.apikey, username, 0)
            if result == "ValidationError":
                return "User does not exist"
            elif result == "InvalidAPIKeyError":
                return "Invalid API Key"
            else:
                return "Success"

        def transfer_kudos_wrapper(username, kudos_amount):
            if username:
                validation = validate_username(username)
                if validation == "Success":
                    if kudos_amount and kudos_amount != 0:
                        result = api.request(
                            session, "Transfer", config.apikey, username, kudos_amount
                        )
                        return result
                    else:
                        return "Can't transfer 0 Kudos"
                else:
                    return validation
            else:
                return "No username specified"

        update_kudos.click(
            fn=lambda: update_kudos_info(),
            inputs=[],
            outputs=your_kudos,
        )

        validate.click(
            fn=validate_username, inputs=[recipient], outputs=validate_output
        )

        transfer.click(
            fn=transfer_kudos_wrapper, inputs=[recipient, kudos_amount], outputs=None
        )

    return kudos_ui


# News UI
def get_news_ui():
    """Creates and returns the Gradio UI with an update button."""
    with gr.Blocks() as news_ui:

        def update_news_info():
            news_info_updated = fetch_api_info("News")

            # Prepare to store the values for later use
            extracted_data = {
                "titles": [],
                "publish_dates": [],
                "importances": [],
                "newspieces": [],
                "tags": [],
            }

            # Process the first five news items (or fewer if not available)
            for i in range(min(5, len(news_info_updated))):
                news_item = news_info_updated[i]
                extracted_data["titles"].append(news_item.get("title"))
                extracted_data["publish_dates"].append(news_item.get("date_published"))
                extracted_data["importances"].append(news_item.get("importance"))
                extracted_data["newspieces"].append(news_item.get("newspiece"))
                extracted_data["tags"].append(news_item.get("tags"))

            # Ensure there are always 5 items by filling with default values
            while len(extracted_data["titles"]) < 5:
                extracted_data["titles"].append("No title available")
                extracted_data["publish_dates"].append("No date available")
                extracted_data["importances"].append("No importance available")
                extracted_data["newspieces"].append("No newspiece available")
                extracted_data["tags"].append([])

            # Return the data, separating what you need externally and what Gradio will display
            return extracted_data

        # Function that Gradio will use to display newspiece and tags
        def update_gradio_outputs():
            news_data = update_news_info()

            # Flatten the lists for Gradio outputs (up to 5 items)
            gradio_outputs = []
            for i in range(5):
                gradio_outputs.append(
                    news_data["newspieces"][i]
                    if i < len(news_data["newspieces"])
                    else None
                )
                gradio_outputs.append(
                    news_data["tags"][i] if i < len(news_data["tags"]) else None
                )

            return gradio_outputs

        news_info = update_news_info()

        gr.Markdown("## News", elem_id="news_title")
        news_update = gr.Button("Update News", elem_id="news-update")

        titles = news_info["titles"]
        publish_dates = news_info["publish_dates"]
        importances = news_info["importances"]

        with gr.Accordion(f"{publish_dates[0]} - {importances[0]} - {titles[0]}"):
            first_newspiece = gr.Textbox(
                value=news_info["newspieces"][0],
                label="Newspiece",
                elem_id="first_newspiece",
                interactive=False,
                lines=4,
            )
            first_tags = gr.Textbox(
                value=news_info["tags"][0],
                label="Tags",
                elem_id="first_tags",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion(f"{publish_dates[1]} - {importances[1]} - {titles[1]}"):
            second_newspiece = gr.Textbox(
                value=news_info["newspieces"][1],
                label="Newspiece",
                elem_id="second_newspiece",
                interactive=False,
                lines=4,
            )
            second_tags = gr.Textbox(
                value=news_info["tags"][1],
                label="Tags",
                elem_id="second_tags",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion(f"{publish_dates[2]} - {importances[2]} - {titles[2]}"):
            third_newspiece = gr.Textbox(
                value=news_info["newspieces"][2],
                label="Newspiece",
                elem_id="third_newspiece",
                interactive=False,
                lines=4,
            )
            third_tags = gr.Textbox(
                value=news_info["tags"][2],
                label="Tags",
                elem_id="third_tags",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion(f"{publish_dates[3]} - {importances[3]} - {titles[3]}"):
            fourth_newspiece = gr.Textbox(
                value=news_info["newspieces"][3],
                label="Newspiece",
                elem_id="fourth_newspiece",
                interactive=False,
                lines=4,
            )
            fourth_tags = gr.Textbox(
                value=news_info["tags"][3],
                label="Tags",
                elem_id="fourth_tags",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion(f"{publish_dates[4]} - {importances[4]} - {titles[4]}"):
            fifth_newspiece = gr.Textbox(
                value=news_info["newspieces"][4],
                label="Newspiece",
                elem_id="fifth_newspiece",
                interactive=False,
                lines=4,
            )
            fifth_tags = gr.Textbox(
                value=news_info["tags"][4],
                label="Tags",
                elem_id="fifth_tags",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        # Update button action
        news_update.click(
            fn=update_gradio_outputs,
            inputs=[],
            outputs=[
                first_newspiece,
                first_tags,
                second_newspiece,
                second_tags,
                third_newspiece,
                third_tags,
                fourth_newspiece,
                fourth_tags,
                fifth_newspiece,
                fifth_tags,
            ],
        )

    return news_ui


# Status UI
def get_status_ui():
    """Sets up the status UI with Gradio."""
    with gr.Blocks() as status_ui:

        def update_status_info():
            status_info_updated = fetch_api_info("Status")
            maintenance_mode = status_info_updated.get("maintenance_mode", "Unknown")
            invite_only_mode = status_info_updated.get("invite_only_mode", "Unknown")
            return maintenance_mode, invite_only_mode

        maintenance_mode, invite_only_mode = update_status_info()

        gr.Markdown("## Status", elem_id="status_title")
        status_update = gr.Button("Update Status", elem_id="status-update")

        maintenance_mode = gr.Textbox(
            value=maintenance_mode,
            label="Maintenance Mode",
            elem_id="maintenance_mode",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        invite_only_mode = gr.Textbox(
            value=invite_only_mode,
            label="Invite Only Mode",
            elem_id="invite_only_mode",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        status_update.click(
            fn=update_status_info,
            inputs=[],
            outputs=[
                maintenance_mode,
                invite_only_mode,
            ],
        )

    return status_ui


# Stats UI
def get_stats_ui():
    """Sets up the stats UI with Gradio."""
    with gr.Blocks() as stats_ui:

        def update_stats_info():
            stats_info_updated = fetch_api_info("Stats")
            keys = [
                ("minute", "images"),
                ("minute", "ps"),
                ("hour", "images"),
                ("hour", "ps"),
                ("day", "images"),
                ("day", "ps"),
                ("month", "images"),
                ("month", "ps"),
                ("total", "images"),
                ("total", "ps"),
            ]
            return [
                stats_info_updated.get(period, {}).get(stat) for period, stat in keys
            ]

        stats_info = update_stats_info()

        gr.Markdown("## Stats", elem_id="stats_title")
        stats_update = gr.Button("Update Stats", elem_id="stats-update")

        with gr.Accordion("Minute"):
            minute_images = gr.Textbox(
                value=stats_info[0],
                label="Images",
                elem_id="minute_images",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            minute_ps = gr.Textbox(
                value=stats_info[1],
                label="Pixelsteps",
                elem_id="minute_ps",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion("Hour"):
            hour_images = gr.Textbox(
                value=stats_info[2],
                label="Images",
                elem_id="hour_images",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            hour_ps = gr.Textbox(
                value=stats_info[3],
                label="Pixelsteps",
                elem_id="hour_ps",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion("Day"):
            day_images = gr.Textbox(
                value=stats_info[4],
                label="Images",
                elem_id="day_images",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            day_ps = gr.Textbox(
                value=stats_info[5],
                label="Pixelsteps",
                elem_id="day_ps",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion("Month"):
            month_images = gr.Textbox(
                value=stats_info[6],
                label="Images",
                elem_id="month_images",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            month_ps = gr.Textbox(
                value=stats_info[7],
                label="Pixelsteps",
                elem_id="month_ps",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        with gr.Accordion("Total"):
            total_images = gr.Textbox(
                value=stats_info[8],
                label="Images",
                elem_id="total_images",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            total_ps = gr.Textbox(
                value=stats_info[9],
                label="Pixelsteps",
                elem_id="total_ps",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        stats_update.click(
            fn=update_stats_info,
            inputs=[],
            outputs=[
                minute_images,
                minute_ps,
                hour_images,
                hour_ps,
                day_images,
                day_ps,
                month_images,
                month_ps,
                total_images,
                total_ps,
            ],
        )
    return stats_ui


# Settings UI
def get_settings_ui(status):
    with gr.Blocks() as settings_ui:
        # Settings UI
        gr.Markdown(
            "## Settings",
            elem_id="settings_title",
        )

        apply_settings = gr.Button(
            "Apply Settings",
            visible=True,
            elem_id=tab_prefix + "apply-settings",
        )

        enable = gr.Checkbox(
            config.enabled,
            label="Enable",
            elem_id=tab_prefix + "enable",
            visible=False,
        )
        name = gr.Textbox(
            config.name,
            label="Worker Name",
            elem_id=tab_prefix + "name",
            lines=1,
            max_lines=1,
        )
        apikey = gr.Textbox(
            config.apikey,
            label="Stable Horde API Key",
            elem_id=tab_prefix + "apikey",
            visible=False,
        )
        allow_img2img = gr.Checkbox(config.allow_img2img, label="Allow img2img")
        allow_painting = gr.Checkbox(config.allow_painting, label="Allow Painting")
        allow_unsafe_ipaddr = gr.Checkbox(
            config.allow_unsafe_ipaddr,
            label="Allow Unsafe IP Address",
        )
        allow_post_processing = gr.Checkbox(
            config.allow_post_processing,
            label="Allow Post Processing",
        )
        restore_settings = gr.Checkbox(
            config.restore_settings,
            label="Restore settings after rendering a job",
        )
        nsfw = gr.Checkbox(config.nsfw, label="Allow NSFW")
        interval = gr.Slider(
            0,
            60,
            config.interval,
            step=1,
            label="Duration Between Generations (seconds)",
        )
        max_pixels = gr.Textbox(
            str(config.max_pixels),
            label="Max Pixels",
            elem_id=tab_prefix + "max-pixels",
            lines=1,
            max_lines=1,
        )
        endpoint = gr.Textbox(
            config.endpoint,
            label="Stable Horde Endpoint",
            elem_id=tab_prefix + "endpoint",
            lines=1,
            max_lines=1,
        )
        save_images_folder = gr.Textbox(
            config.save_images_folder,
            label="Folder to Save Generation Images",
            elem_id=tab_prefix + "save-images-folder",
            lines=1,
            max_lines=1,
        )
        show_images = gr.Checkbox(config.show_image_preview, label="Show Images")
        save_images = gr.Checkbox(config.save_images, label="Save Images")
        save_source_images = gr.Checkbox(
            config.save_source_images, label="Save Source Images"
        )

        running_type = gr.Textbox(
            "",
            label="Running Type",
            visible=False,
        )

        def on_apply_selected_models(local_selected_models):
            status.update(
                f'Status: \
            {"Running" if config.enabled else "Stopped"}, \
            Updating selected models...'
            )
            selected_models = horde.set_current_models(local_selected_models)
            local_selected_models_dropdown.update(value=list(selected_models.values()))
            gr.Info("Models Applied")
            return f'Status: \
            {"Running" if config.enabled else "Stopped"}, \
            Selected models \
            {list(selected_models.values())} updated'

        local_selected_models_dropdown = gr.Dropdown(
            [model.name for model in sd_models.checkpoints_list.values()],
            value=[
                model.name
                for model in sd_models.checkpoints_list.values()
                if model.name in list(config.current_models.values())
            ],
            label="Selected models for sharing",
            elem_id=tab_prefix + "local-selected-models",
            multiselect=True,
            interactive=True,
        )

        local_selected_models_dropdown.change(
            on_apply_selected_models,
            inputs=[local_selected_models_dropdown],
            outputs=[status],
        )
        gr.Markdown("Once you select a model it will take some time to load.")

        # Settings click
        apply_settings.click(
            fn=apply_stable_horde_settings,
            inputs=[
                enable,
                name,
                apikey,
                allow_img2img,
                allow_painting,
                allow_unsafe_ipaddr,
                allow_post_processing,
                restore_settings,
                nsfw,
                interval,
                max_pixels,
                endpoint,
                show_images,
                save_images,
                save_source_images,
                save_images_folder,
            ],
            outputs=[status, running_type],
        )

    return settings_ui


# General UI
def on_ui_tabs():
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="green", secondary_hue="red")
    ) as ui_tabs:
        # General UI
        with gr.Row():

            def save_apikey_value(apikey_value: str):
                apply_stable_horde_apikey(apikey_value)

            with gr.Column():
                apikey = gr.Textbox(
                    config.apikey,
                    label="Stable Horde API Key",
                    elem_id=tab_prefix + "apikey",
                    resizeable=False,
                    lines=1,
                    max_lines=1,
                )
                save_apikey = gr.Button("Save", elem_id=f"{tab_prefix}apikey-save")

            with gr.Column():
                def toggle_running_fn(status, running_type, toggle_running):
                    if config.enabled:
                        config.enabled = False
                        status.value = "Status: Stopped"
                        running_type.value = "Running Type: Image Generation"
                        toggle_running.update(value="Enable", variant="primary")
                        gr.Info("Generation Disabled")
                    else:
                        config.enabled = True
                        status.value = "Status: Running"
                        toggle_running.update(value="Disable", variant="secondary")
                        gr.Info("Generation Enabled")
                    config.save()
                    return status, running_type, toggle_running
            
                status = gr.Textbox(
                    f'{"Running" if config.enabled else "Stopped"}',
                    label="Status",
                    elem_id=tab_prefix + "status",
                    readonly=True,
                    lines=1,
                    max_lines=1,
                )

                running_type = gr.Textbox(
                    "Running Type: Image Generation",
                    label="",
                    elem_id=tab_prefix + "running-type",
                    readonly=True,
                    visible=False,
                )

                toggle_running = gr.Button(
                    "Disable",
                    variant="secondary",
                    elem_id=f"{tab_prefix}disable",
                )

                save_apikey.click(fn=save_apikey_value, inputs=[apikey])
                toggle_running.click(
                    fn=toggle_running_fn,
                    inputs=[status, running_type, toggle_running],
                    outputs=[status, running_type, toggle_running],
                )

                # Get Worker
                def get_worker() -> str:
                    user_info = fetch_api_info("User")
                    worker_ids = user_info["worker_ids"]
                    if worker_ids:
                        for worker in worker_ids:
                            worker_info = fetch_api_info("Worker", worker)
                            worker_name = worker_info["name"]
                            if worker_name == config.name:
                                return worker
                    else:
                        worker = "Unavailable"
                        return worker

                worker = get_worker()

        # General tabs
        with gr.Row():
            with gr.Tab("Generation"):
                get_generator_ui()
            with gr.Tab("Worker"):
                get_worker_ui(worker)
            with gr.Tab("User"):
                get_user_ui()
            with gr.Tab("Team"):
                get_team_ui()
            with gr.Tab("Kudos"):
                get_kudos_ui()
            with gr.Tab("News"):
                get_news_ui()
            with gr.Tab("Status"):
                get_status_ui()
            with gr.Tab("Stats"):
                get_stats_ui()
            with gr.Tab("Settings"):
                get_settings_ui(status)

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
