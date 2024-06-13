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
    config.save_images_folder = save_images_folder
    config.save()

    return (
        f'Status: {"Running" if config.enabled else "Stopped"}',
        "Running Type: Image Generation",
    )


# Worker
def fetch_and_update_worker_info(worker):
    worker_info = api.get_worker_info(session, config.apikey, worker)
    return [
        (
            worker_info[key]
            if isinstance(worker_info[key], str)
            else (
                ", ".join(worker_info[key])
                if isinstance(worker_info[key], list)
                else str(worker_info[key])
            )
        )
        for key in worker_info.keys()
    ]


# User
def fetch_and_update_user_info():
    user_info = api.get_user_info(session, config.apikey)
    return [
        (
            user_info[key]
            if isinstance(user_info[key], str)
            else (
                ", ".join(user_info[key])
                if isinstance(user_info[key], list)
                else str(user_info[key])
            )
        )
        for key in user_info.keys()
    ]


# Stats
def fetch_and_update_stats_info():
    stats_info = api.get_horde_stats(session)
    return [
        (
            stats_info[key]
            if isinstance(stats_info[key], str)
            else (
                ", ".join(stats_info[key])
                if isinstance(stats_info[key], list)
                else str(stats_info[key])
            )
        )
        for key in stats_info.keys()
    ]


# News
def fetch_and_update_news_info():
    news_info = api.get_horde_stats(session)
    return [
        (
            news_info[key]
            if isinstance(news_info[key], str)
            else (
                ", ".join(news_info[key])
                if isinstance(news_info[key], list)
                else str(news_info[key])
            )
        )
        for key in news_info.keys()
    ]


tab_prefix = "stable-horde-"


# Generator UI
def get_generator_ui():
    with gr.Blocks() as generator_ui:
        with gr.Row():
            with gr.Column(elem_id="stable-horde"):
                gr.Markdown(
                    "## Generations",
                    elem_id="kudos_title",
                )

                current_id = gr.Textbox(
                    "Current ID: ",
                    label="",
                    elem_id=tab_prefix + "current-id",
                    readonly=True,
                )

                running_type = gr.Textbox(
                    "Running Type: Image Generation",
                    label="",
                    elem_id=tab_prefix + "running-type",
                    readonly=True,
                    visible=False,
                )

                state = gr.HTML(
                    "",
                    label="State",
                    elem_id=tab_prefix + "state",
                    visible=True,
                    readonly=True,
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

            with gr.Column():
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

            # Click functions
        if generator_ui.current_id and generator_ui.log and generator_ui.state:
            generator_ui.refresh.click(
                fn=lambda: generator_ui.on_refresh(),
                outputs=[generator_ui.current_id, generator_ui.log, generator_ui.state],
                show_progress=False,
            )

        if (
            generator_ui.current_id
            and generator_ui.log
            and generator_ui.state
            and generator_ui.preview
        ):
            generator_ui.refresh_image.click(
                fn=lambda: generator_ui.on_refresh(True),
                outputs=[
                    generator_ui.current_id,
                    generator_ui.log,
                    generator_ui.state,
                    generator_ui.preview,
                ],
                show_progress=False,
            )

    return generator_ui


# Worker UI
def get_worker_ui(worker):
    with gr.Blocks() as worker_ui:
        worker_info = api.get_worker_info(session, config.apikey, worker)

        gr.Markdown("## Worker Details")
        worker_update = gr.Button("Update Worker Details", elem_id="worker-update")

        details = []
        for key in worker_info.keys():
            detail = gr.Textbox(
                value=f"{key.capitalize()}: {worker_info[key]}",
                interactive=False,
                lines=1,
            )
            details.append(detail)

        worker_update.click(
            fn=lambda: fetch_and_update_worker_info(worker),
            inputs=[],
            outputs=details,
        )

    return worker_ui


# User UI
def get_user_ui():
    with gr.Blocks() as user_ui:
        user_info = api.get_user_info(session, config.apikey)

        gr.Markdown("## User Details", elem_id="user_title")
        user_update = gr.Button(
            "Update User Details", elem_id=f"{tab_prefix}user-update"
        )

        details = []
        for key in user_info.keys():
            detail = gr.Textbox(
                value=f"{key.capitalize()}: {user_info[key]}",
                interactive=False,
                lines=1,
            )
            details.append(detail)

        user_update.click(
            fn=lambda: fetch_and_update_user_info(),
            inputs=[],
            outputs=details,
        )
    return user_ui


# Kudos UI
def get_kudos_ui():
    with gr.Blocks() as kudos_ui:
        # Kudos functions
        horde_user = API()
        user_info = horde_user.get_user_info(session, config.apikey)

        # Kudos UI
        with gr.Row():
            with gr.Column():
                # Transfer Kudos Title
                gr.Markdown(
                    "## Transfer Kudos",
                    elem_id="kudos_title",
                )

                # Username
                username = gr.Textbox(
                    label="Username",
                    placeholder="Enter username",
                    elem_id="kudos_username",
                    interactive=True,
                )

            with gr.Column():
                # Kudo amount display
                gr.Textbox(
                    user_info["kudos"],
                    label="Your Kudos",
                    placeholder="0",
                    elem_id="kudos_display",
                    interactive=False,
                )

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
        kudos_ui.transfer.click(
            fn=api.transfer_kudos(kudos_ui.username, kudos_ui.kudos_amount)
        )

    return kudos_ui


# News UI
def get_news_ui():
    with gr.Blocks() as news_ui:
        news_info = api.get_news_info(session)
        gr.Markdown(
            "## News",
            elem_id="news_title",
        )
        with gr.Row():
            news_update = gr.Button("Update News", elem_id=f"{tab_prefix}news-update")

        details = []
        for key in news_info.keys():
            detail = gr.Textbox(
                value=f"{key.capitalize()}: {news_info[key]}",
                interactive=False,
                lines=1,
            )
            details.append(detail)

        news_update.click(
            fn=lambda: fetch_and_update_news_info(),
            inputs=[],
            outputs=details,
        )

    return news_ui


def get_stats_ui():
    with gr.Blocks() as stats_ui:
        stats_info = api.get_stats_info(session, config.apikey)
        gr.Markdown("## Stats", elem_id="stats_title")
        with gr.Row():
            stats_update = gr.Button("Update Stats", elem_id="stats-update")

        details = []
        for key in stats_info.keys():
            detail = gr.Textbox(
                value=f"{key.capitalize()}: {stats_info[key]}",
                interactive=False,
                lines=1,
            )
            details.append(detail)

        stats_update.click(
            fn=lambda: fetch_and_update_stats_info(),
            inputs=[],
            outputs=details,
        )


# Settings UI
def get_settings_ui(status):
    with gr.Blocks() as settings_ui:
        # Settings UI
        with gr.Column():
            gr.Markdown(
                "## Settings",
                elem_id="settings_title",
            )
            with gr.Box():
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
                )
                apikey = gr.Textbox(
                    config.apikey,
                    label="Stable Horde API Key",
                    elem_id=tab_prefix + "apikey",
                    visible=False,
                )
                allow_img2img = gr.Checkbox(config.allow_img2img, label="Allow img2img")
                allow_painting = gr.Checkbox(
                    config.allow_painting, label="Allow Painting"
                )
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
                )
                endpoint = gr.Textbox(
                    config.endpoint,
                    label="Stable Horde Endpoint",
                    elem_id=tab_prefix + "endpoint",
                )
                save_images_folder = gr.Textbox(
                    config.save_images_folder,
                    label="Folder to Save Generation Images",
                    elem_id=tab_prefix + "save-images-folder",
                )
                show_images = gr.Checkbox(
                    config.show_image_preview, label="Show Images"
                )
                save_images = gr.Checkbox(config.save_images, label="Save Images")

            with gr.Box():

                def on_apply_selected_models(local_selected_models):
                    status.update(
                        f'Status: \
                    {"Running" if config.enabled else "Stopped"}, \
                    Updating selected models...'
                    )
                    selected_models = horde.set_current_models(local_selected_models)
                    local_selected_models_dropdown.update(
                        value=list(selected_models.values())
                    )
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

            apply_settings = gr.Button(
                "Apply Settings",
                visible=True,
                elem_id=tab_prefix + "apply-settings",
            )

    return settings_ui


# General UI
def on_ui_tabs():
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="green", secondary_hue="red")
    ) as ui_tabs:
        # General functions
        def save_apikey_fn(apikey: str):
            config.apikey = apikey
            config.save()

        def toggle_running_fn():
            if config.enabled:
                config.enabled = False
                status.update("Status: Stopped")
                running_type.update("Running Type: Image Generation")
                toggle_running.update(value="Enable", variant="primary")
            else:
                config.enabled = True
                status.update("Status: Running")
                toggle_running.update(value="Disable", variant="secondary")
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

                # TODO Move this somewhere else
                api = API()
                user_info = api.get_user_info(session, config.apikey)
                worker_ids = user_info["worker_ids"]
                for worker in worker_ids:
                    worker_info = api.get_worker_info(session, config.apikey, worker)
                    worker_name = worker_info["name"]
                    if worker_name == config.name:
                        print(f"Current Worker: {worker_name}")
                        break

            # General tabs
            with gr.Tab("Generation"):
                get_generator_ui()
            with gr.Tab("Worker"):
                get_worker_ui(worker)
            with gr.Tab("User"):
                get_user_ui()
            with gr.Tab("Kudos"):
                get_kudos_ui()
            with gr.Tab("News"):
                get_news_ui()
            with gr.Tab("Stats"):
                stats_ui, stats_update, stats_outputs = get_stats_ui()
            with gr.Tab("Settings"):
                settings_ui, settings_update = get_settings_ui(status)

        save_apikey.click(fn=save_apikey_fn(apikey))
        toggle_running.click(fn=toggle_running_fn)

        # Stats click
        stats_update.click(
            fn=lambda: horde_stats(api, session),
            outputs=stats_outputs,
        )

        # Settings click
        settings_ui.apply_settings.click(
            fn=apply_stable_horde_settings,
            inputs=[
                settings_ui.enable,
                settings_ui.name,
                apikey,
                settings_ui.allow_img2img,
                settings_ui.allow_painting,
                settings_ui.allow_unsafe_ipaddr,
                settings_ui.allow_post_processing,
                settings_ui.restore_settings,
                settings_ui.nsfw,
                settings_ui.interval,
                settings_ui.max_pixels,
                settings_ui.endpoint,
                settings_ui.show_images,
                settings_ui.save_images,
                settings_ui.save_images_folder,
            ],
            output=[status, running_type],
        )

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
