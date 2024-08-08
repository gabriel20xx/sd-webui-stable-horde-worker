from typing import Optional

from fastapi import FastAPI
import gradio as gr
import asyncio
import requests
import json
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


# Settings
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
                    visible=True,
                    readonly=True,
                )

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

                log = gr.HTML(elem_id=tab_prefix + "log")

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
def fetch_worker_info(worker):
    """Fetches the latest worker info."""
    worker_info = api.get_worker_info(session, config.apikey, worker)
    return worker_info


def create_worker_ui(worker_info):
    """Creates and returns Gradio UI components based on the worker info."""
    details = []

    for key in worker_info.keys():
        if key.capitalize() in ["Kudos_details", "Team"]:
            with gr.Accordion(key.capitalize()):
                for secondkey in worker_info[key].keys():
                    value = worker_info[key][secondkey]
                    detail = gr.Textbox(
                        label=secondkey.capitalize(),
                        elem_id=tab_prefix + "worker-info",
                        value=value,
                        interactive=False,
                        lines=1,
                        max_lines=1,
                    )
                    details.append(detail)
        elif key.capitalize() in ["Models"]:
            pre_value = worker_info[key]
            worker_string = ", ".join(map(str, pre_value))
            stripped_worker_info = (
                worker_string.replace("'", "").replace("[", "").replace("]", "")
            )
            value = stripped_worker_info
            detail = gr.Textbox(
                label=key.capitalize(),
                value=value,
                elem_id=tab_prefix + "worker-info",
                interactive=False,
                lines=1,
                max_lines=1,
            )
        else:
            if worker_info[key] == "id":
                worker_info[key] = "worker_id"
            value = worker_info[key]
            detail = gr.Textbox(
                label=key.capitalize(),
                value=value,
                elem_id=tab_prefix + "worker-info",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            details.append(detail)
    return details


def update_worker_ui(worker):
    """Fetches and updates the worker UI."""
    worker_info = fetch_worker_info(worker)
    updated_values = []
    for key in worker_info.keys():
        if key.capitalize() in ["Kudos_details", "Team"]:
            for secondkey in worker_info[key].keys():
                value = worker_info[key][secondkey]
                updated_values.append(str(value))
        elif key.capitalize() in ["Models"]:
            pre_value = worker_info[key]
            worker_string = ", ".join(map(str, pre_value))
            stripped_worker_info = (
                worker_string.replace("'", "").replace("[", "").replace("]", "")
            )
            value = stripped_worker_info
            updated_values.append(str(value))
        else:
            value = worker_info[key]
            updated_values.append(str(value))
    # Return a list of updated components
    return updated_values


def get_worker_ui(worker):
    with gr.Blocks() as worker_ui:
        details = []

        worker_info = fetch_worker_info(worker)

        gr.Markdown("## Worker Details")
        worker_update = gr.Button("Update Worker Details", elem_id="worker-update")

        # Create the initial UI components
        details = create_worker_ui(worker_info)

        worker_update.click(
            fn=lambda: update_worker_ui(worker),
            inputs=[],
            outputs=details,
        )

    return worker_ui


# User UI
def fetch_user_info():
    """Fetches the latest user info."""
    user_info = api.get_user_info(session, config.apikey)
    return user_info


def create_user_ui(user_info):
    """Creates and returns Gradio UI components based on the user info."""
    details = []

    for key in user_info.keys():
        # Handle nested dictionaries
        if isinstance(user_info[key], dict):
            if key.capitalize() in ["Records"]:
                with gr.Accordion(key.capitalize()):
                    for secondkey in user_info[key].keys():
                        if isinstance(user_info[key][secondkey], dict):
                            with gr.Accordion(secondkey.capitalize()):
                                for thirdkey in user_info[key][secondkey].keys():
                                    value = user_info[key][secondkey][thirdkey]
                                    detail = gr.Textbox(
                                        label=thirdkey.capitalize(),
                                        value=value,
                                        elem_id=tab_prefix + "user-info",
                                        interactive=False,
                                        lines=1,
                                        max_lines=1,
                                    )
                                    details.append(detail)
                        else:
                            value = user_info[key][secondkey]
                            detail = gr.Textbox(
                                label=secondkey.capitalize(),
                                value=value,
                                elem_id=tab_prefix + "user-info",
                                interactive=False,
                                lines=1,
                                max_lines=1,
                            )
                            details.append(detail)

            elif key.capitalize() in [
                "Kudos_details",
                "Worker_ids",
                "Sharedkey_ids",
                "Usage",
                "Contributions",
            ]:
                with gr.Accordion(key.capitalize()):
                    for secondkey in user_info[key].keys():
                        value = user_info[key][secondkey]
                        detail = gr.Textbox(
                            label=secondkey.capitalize(),
                            value=value,
                            elem_id=tab_prefix + "user-info",
                            interactive=False,
                            lines=1,
                            max_lines=1,
                        )
                        details.append(detail)

        # Handle lists or other data structures
        elif isinstance(user_info[key], list):
            with gr.Accordion(key.capitalize()):
                for i, item in enumerate(user_info[key]):
                    value = item
                    detail = gr.Textbox(
                        label=f"Item {i+1}",
                        value=value,
                        elem_id=tab_prefix + "user-info",
                        interactive=False,
                        lines=1,
                        max_lines=1,
                    )
                    details.append(detail)

        # Handle other data types
        else:
            if user_info[key] == "id":
                user_info[key] = "user_id"
            value = user_info[key]
            detail = gr.Textbox(
                label=key.capitalize(), value=value, interactive=False, lines=1
            )
            details.append(detail)
    return details


def update_user_ui():
    """Fetches and updates the user UI."""
    user_info = fetch_user_info()
    updated_values = []
    for key in user_info.keys():
        # Handle nested dictionaries
        if isinstance(user_info[key], dict):
            if key.capitalize() in ["Records"]:
                for secondkey in user_info[key].keys():
                    if isinstance(user_info[key][secondkey], dict):
                        for thirdkey in user_info[key][secondkey].keys():
                            value = user_info[key][secondkey][thirdkey]
                            updated_values.append(str(value))
                    else:
                        value = user_info[key][secondkey]
                        updated_values.append(str(value))
            elif key.capitalize() in [
                "Kudos_details",
                "Worker_ids",
                "Sharedkey_ids",
                "Usage",
                "Contributions",
            ]:
                for secondkey in user_info[key].keys():
                    value = user_info[key][secondkey]
                    updated_values.append(str(value))
        elif isinstance(user_info[key], list):
            with gr.Accordion(key.capitalize()):
                for i, item in enumerate(user_info[key]):
                    value = item
                    updated_values.append(str(value))
        else:
            value = user_info[key]
            updated_values.append(str(value))
    # Return a list of updated components
    return updated_values


def get_user_ui():
    """Creates and returns the Gradio UI with an update button."""
    with gr.Blocks() as user_ui:
        user_info = fetch_user_info()

        gr.Markdown("## User Details", elem_id="user_title")

        user_update = gr.Button(
            "Update User Details", elem_id=f"{tab_prefix}user-update"
        )

        # Create the initial UI components
        details = create_user_ui(user_info)

        user_update.click(
            fn=lambda: update_user_ui(),
            inputs=[],
            outputs=details,
        )
    return user_ui


# Team UI
def fetch_team_info(team_id):
    """Fetches the latest team info."""
    if not team_id:
        return {}
    team_info = api.get_team_info(session, config.apikey, team_id)
    return team_info


def create_team_ui(team_info):
    """Creates and returns Gradio UI components based on the team info."""
    details = []

    for key in team_info.keys():
        if key.capitalize() in ["Kudos_details", "Team"]:
            with gr.Accordion(key.capitalize()):
                for secondkey in team_info[key].keys():
                    value = team_info[key][secondkey]
                    detail = gr.Textbox(
                        label=secondkey.capitalize(),
                        elem_id=tab_prefix + "team-info",
                        value=value,
                        interactive=False,
                        lines=1,
                        max_lines=1,
                    )
                    details.append(detail)
        elif key.capitalize() in ["Models"]:
            pre_value = team_info[key]
            team_string = ", ".join(map(str, pre_value))
            stripped_team_info = (
                team_string.replace("'", "").replace("[", "").replace("]", "")
            )
            value = stripped_team_info
            detail = gr.Textbox(
                label=key.capitalize(),
                value=value,
                elem_id=tab_prefix + "team-info",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            details.append(detail)
        else:
            if team_info[key] == "id":
                team_info[key] = "team_id"
            value = team_info[key]
            detail = gr.Textbox(
                label=key.capitalize(),
                value=value,
                elem_id=tab_prefix + "team-info",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            details.append(detail)
    return details


def update_team_ui(team_id):
    """Fetches and updates the team UI."""
    team_info = fetch_team_info(team_id)
    if not team_info:
        return []  # Return an empty list if team_info is empty
    return create_team_ui(team_info)


def get_team_ui():
    with gr.Blocks() as team_ui:
        gr.Markdown("## Team Details")

        with gr.Column():
            # Team ID
            team_id_input = gr.Textbox(
                label="Team ID",
                placeholder="Enter Team ID",
                elem_id="team_id",
                interactive=True,
                lines=1,
                max_lines=1,
            )
            team_update = gr.Button("Update Team Details", elem_id="team-update")
            details = gr.Column()  # Placeholder for dynamic team details

        team_update.click(
            fn=lambda team_id: update_team_ui(team_id),
            inputs=[team_id_input],
            outputs=[details],
        )

    return team_ui



# Kudos UI
def fetch_and_update_kudos():
    user_info = api.get_user_info(session, config.apikey)
    if user_info["kudos"]:
        kudos = user_info["kudos"]
        return kudos


def get_kudos_ui():
    with gr.Blocks() as kudos_ui:
        details = []

        # Kudos functions
        user_info = api.get_user_info(session, config.apikey)

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
                your_kudos = gr.Textbox(
                    user_info["kudos"],
                    label="Your Kudos",
                    placeholder="0",
                    elem_id="kudos_display",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )
                details.append(your_kudos)

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
            result = api.transfer_kudos(session, config.apikey, username, 0)
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
                        result = api.transfer_kudos(
                            session, config.apikey, username, kudos_amount
                        )
                        return result
                    else:
                        return "Can't transfer 0 Kudos"
                else:
                    return validation
            else:
                return "No username specified"

        update_kudos.click(
            fn=lambda: fetch_and_update_kudos(),
            inputs=[],
            outputs=details,
        )

        validate.click(
            fn=validate_username, inputs=[recipient], outputs=validate_output
        )

        transfer.click(
            fn=transfer_kudos_wrapper, inputs=[recipient, kudos_amount], outputs=None
        )

    return kudos_ui


# News UI
def fetch_news_info():
    """Fetches the latest news info."""
    news_info = api.get_news_info(session)
    return news_info


def create_news_ui(news_info):
    """Creates and returns Gradio UI components based on the news info."""
    details = []

    for news_item in news_info:
        if isinstance(news_item, dict):
            importance = news_item.get("importance", "No importance available")
            title = news_item.get("title", "No title available")
            date_published = news_item.get(
                "date_published", "No published date available"
            )
            with gr.Accordion(f"{importance} - {title} - {date_published}"):
                value = news_item.get("newspiece", "No message available")
                message = gr.TextArea(
                    label="Message",
                    value=value,
                    interactive=False,
                )
                details.append(message)

                tags_value = news_item.get("tags", [])
                if not isinstance(tags_value, list):
                    tags_value = []

                tags_string = ", ".join(map(str, tags_value))
                tags = gr.Textbox(
                    label="Tags",
                    value=tags_string,
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )
                details.append(tags)

    return details


def update_news_ui():
    """Fetches and updates the news UI."""
    news_info = fetch_news_info()
    updated_values = []
    for news_item in news_info:
        if isinstance(news_item, dict):
            value = news_item.get("newspiece", "No message available")
            updated_values.append(str(value))

            tags_value = news_item.get("tags", [])
            if not isinstance(tags_value, list):
                tags_value = []

            tags_string = ", ".join(map(str, tags_value))
            updated_values.append(str(tags_string))

    return updated_values


def get_news_ui():
    """Creates and returns the Gradio UI with an update button."""
    with gr.Blocks() as news_ui:
        news_info = fetch_news_info()

        gr.Markdown("## News", elem_id="news_title")

        with gr.Row():
            news_update = gr.Button("Update News", elem_id="news-update")

        # Create the initial UI components
        details = create_news_ui(news_info)

        # Update button action
        news_update.click(
            fn=lambda: update_news_ui(),
            inputs=[],
            outputs=details,
        )

    return news_ui


# Stats UI
def fetch_status_info():
    """Fetches the latest status info."""
    status_info = api.get_status_info(session)
    return status_info


def create_status_ui(status_info):
    """Creates UI components for the status info."""
    details = []
    for key in status_info.keys():
        with gr.Accordion(key.capitalize()):
            value = status_info[key]
            textbox = gr.Textbox(
                label=f"{key.capitalize()}",
                value=value,
                elem_id=f"{tab_prefix}_{key}",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            details.append(textbox)
    return details


def update_status_ui():
    """Fetches the latest status info and returns updated values for UI components."""
    status_info = fetch_status_info()
    # Extract values for each stat in each period and return them as a list
    updated_values = []
    for key in status_info.keys():
        value = status_info[key]
        updated_values.append(str(value))
    return updated_values


def get_status_ui():
    """Sets up the status UI with Gradio."""
    with gr.Blocks() as status_ui:
        status_info = fetch_status_info()

        gr.Markdown("## Stats", elem_id="status_title")
        with gr.Row():
            status_update = gr.Button("Update Stats", elem_id="status-update")

        # Create the initial UI components
        details = create_status_ui(status_info)

        status_update.click(
            fn=lambda: update_status_ui(),
            inputs=[],
            outputs=details,
        )
    return status_ui


# Stats UI
def fetch_stats_info():
    """Fetches the latest stats info."""
    stats_info = api.get_stats_info(session)
    return stats_info


def create_stats_ui(stats_info):
    """Creates UI components for the stats info."""
    details = []
    for period, stats in stats_info.items():
        with gr.Accordion(period.capitalize()):
            for stat_type, value in stats.items():
                value = str(value)
                textbox = gr.Textbox(
                    label=f"{stat_type.capitalize()}",
                    value=value,
                    elem_id=f"{tab_prefix}_{period}_{stat_type}",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )
                details.append(textbox)
    return details


def update_stats_ui():
    """Fetches the latest stats info and returns updated values for UI components."""
    stats_info = fetch_stats_info()
    # Extract values for each stat in each period and return them as a list
    updated_values = []
    for period, stats in stats_info.items():
        for stat_type, value in stats.items():
            value = str(value)
            updated_values.append(str(value))
    return updated_values


def get_stats_ui():
    """Sets up the stats UI with Gradio."""
    with gr.Blocks() as stats_ui:
        stats_info = fetch_stats_info()

        gr.Markdown("## Stats", elem_id="stats_title")
        with gr.Row():
            stats_update = gr.Button("Update Stats", elem_id="stats-update")

        # Create the initial UI components
        details = create_stats_ui(stats_info)

        stats_update.click(
            fn=lambda: update_stats_ui(),
            inputs=[],
            outputs=details,
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
        # General functions
        def save_apikey_value(apikey_value: str):
            apply_stable_horde_apikey(apikey_value)

        def toggle_running_fn():
            if config.enabled:
                config.enabled = False
                status.update("Status: Stopped")
                running_type.update("Running Type: Image Generation")
                toggle_running.update(value="Enable", variant="primary")
                gr.Info("Generation Disabled")
            else:
                config.enabled = True
                status.update("Status: Running")
                toggle_running.update(value="Disable", variant="secondary")
                gr.Info("Generation Enabled")
            config.save()

        # General UI
        with gr.Row():
            with gr.Column():
                apikey = gr.Textbox(
                    config.apikey,
                    label="Stable Horde API Key",
                    elem_id=tab_prefix + "apikey",
                    resizeable=False,
                )
                save_apikey = gr.Button("Save", elem_id=f"{tab_prefix}apikey-save")

            with gr.Column():
                status = gr.Textbox(
                    f'{"Running" if config.enabled else "Stopped"}',
                    label="Status",
                    elem_id=tab_prefix + "status",
                    readonly=True,
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

                def get_worker(session, apikey, worker_ids):
                    for worker in worker_ids:
                        worker_info = api.get_worker_info(session, apikey, worker)
                        worker_name = worker_info["name"]
                        if worker_name == config.name:
                            return worker

                # TODO Move this somewhere else
                user_info = api.get_user_info(session, config.apikey)
                worker_ids = user_info["worker_ids"]
                if worker_ids:
                    worker = get_worker(session, config.apikey, worker_ids)
                else:
                    worker = "Unavailable"

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

        save_apikey.click(fn=save_apikey_value, inputs=[apikey])
        toggle_running.click(fn=toggle_running_fn)

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
