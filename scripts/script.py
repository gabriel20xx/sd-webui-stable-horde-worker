from typing import Optional

from fastapi import FastAPI
import gradio as gr
import asyncio
import aiohttp
import requests
import datetime
from threading import Thread

from modules import scripts, script_callbacks, sd_models, shared

from stable_horde import (
    StableHorde,
    StableHordeConfig,
    HordeUser,
    HordeWorker,
    HordeNews,
    HordeStatus,
    KudoTransfer,
    HordeStats,
)

basedir = scripts.basedir()
config = StableHordeConfig(basedir)
horde = StableHorde(basedir, config)
session: Optional[aiohttp.ClientSession] = None


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


tab_prefix = "stable-horde-"


def get_generator_ui(state):
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

                if current_id and log and state:
                    refresh.click(
                        fn=lambda: on_refresh(),
                        outputs=[current_id, log, state],
                        show_progress=False,
                    )

                if current_id and log and state and preview:
                    refresh_image.click(
                        fn=lambda: on_refresh(True),
                        outputs=[current_id, log, state, preview],
                        show_progress=False,
                    )

    return generator_ui


def get_worker_ui(worker_info):
    with gr.Blocks() as worker_ui:
        gr.Markdown("## Worker Details")
        for key, value in worker_info.items():
            if value is not None:
                gr.Textbox(value, label=key.capitalize(), interactive=False, lines=1)

    return worker_ui


def get_user_ui(user_info):
    with gr.Blocks() as user_ui:
        with gr.Row():
            with gr.Column(scale=1):
                user_update = gr.Button("Update", elem_id=f"{tab_prefix}user-update")
            with gr.Column(scale=4):
                user_welcome = gr.Markdown(
                    "**Try click update button to fetch the user info**",
                    elem_id=f"{tab_prefix}user-webcome",
                )
        with gr.Column():
            gr.Markdown(
                "## User Details", 
                elem_id="user_title"
            )
            for key, value in user_info.items():
                if value is not None:
                    gr.Textbox(
                        value, label=key.capitalize(), interactive=False, lines=1
                    )

    return user_ui


def get_kudos_ui(user_info):
    with gr.Blocks() as kudos_ui:
        with gr.Row():
            with gr.Column():
                # Transfer Kudos Title
                gr.Markdown(
                    "## Transfer Kudos",
                    elem_id="kudos_title",
                )

                # Username
                gr.Textbox(
                    label="Username",
                    placeholder="Enter username",
                    elem_id="kudos_username",
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
                gr.Slider(
                    label="Kudos",
                    minimum=0,
                    maximum=user_info["kudos"],
                    step=1,
                    value=10,
                    elem_id="kudos_amount",
                )

            # Transfer Button
            gr.Button(
                "Transfer",
                variant="primary",
                elem_id="kudos_transfer_button",
            )

    return kudos_ui


def get_news_ui(news_info, horde_status):
    with gr.Blocks() as news_ui:
        gr.Markdown(
                "## News",
                elem_id="news_title",
            )
        with gr.Box(scale=2):
            with gr.Column():
                if "maintenance_mode" in horde_status:
                    gr.Textbox(
                        horde_status["maintenance_mode"],
                        label="Maintenance mode",
                        elem_id=tab_prefix + "status_maintenance_mode",
                        visible=True,
                        interactive=False,
                    )
                if "invite_only_mode" in horde_status:
                    gr.Textbox(
                        horde_status["invite_only_mode"],
                        label="Invite only mode",
                        elem_id=tab_prefix + "status_invite_only_mode",
                        visible=True,
                        interactive=False,
                    )
                if "raid_mode" in horde_status:
                    gr.Textbox(
                        horde_status["raid_mode"],
                        label="Raid mode",
                        elem_id=tab_prefix + "status_raid_mode",
                        visible=True,
                        interactive=False,
                    )
        with gr.Box(scale=2):
            with gr.Column():
                for news_item in news_info[:3]:
                    if "title" and "newspiece" in news_item:
                        gr.Textbox(
                            news_item["newspiece"],
                            label=news_item["title"],
                            elem_id=tab_prefix + "news_title",
                            visible=True,
                            interactive=False,
                        )
    return news_ui


def get_stats_ui(stats_info):
    with gr.Blocks() as stats_ui:
        gr.Markdown(
            "## Stats",
            elem_id="stats_title",
        )
        with gr.Box(scale=2):
            with gr.Column():
                for period, data in stats_info.items():
                    for metric, value in data.items():
                        gr.Textbox(
                            value,
                            label=f"{period.capitalize()} {metric.capitalize()}",
                            interactive=False,
                            lines=1,
                        )

    return stats_ui


def get_settings_ui(status, running_type):
    with gr.Blocks() as settings_ui:
        with gr.Column():
            gr.Markdown(
                "## Settings",
                elem_id="settings_title",
            )
            with gr.Box(scale=2):
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
                )
                allow_img2img = gr.Checkbox(
                    config.allow_img2img, label="Allow img2img"
                )
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

            with gr.Box(scale=2):
                def on_apply_selected_models(local_selected_models):
                    status.update(
                        f'Status: \
                    {"Running" if config.enabled else "Stopped"}, \
                    Updating selected models...'
                    )
                    selected_models = horde.set_current_models(
                        local_selected_models
                    )
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
                gr.Markdown(
                    "Once you select a model it will take some time to load."
                )

            apply_settings = gr.Button(
                "Apply Settings",
                visible=True,
                elem_id=tab_prefix + "apply-settings",
            )

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
                save_images_folder,
            ],
            outputs=[status, running_type],
        )

    return settings_ui


def on_ui_tabs():
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="green", secondary_hue="red")
    ) as ui_tabs:
        with gr.Row():
            with gr.Column():
                apikey = gr.Textbox(
                    config.apikey,
                    label="Stable Horde API Key",
                    elem_id=tab_prefix + "apikey",
                    resizeable=False,
                )
                save_apikey = gr.Button("Save", elem_id=f"{tab_prefix}apikey-save")

                def save_apikey_fn(apikey: str):
                    config.apikey = apikey
                    config.save()

                save_apikey.click(fn=save_apikey_fn, inputs=[apikey])

            with gr.Column():
                status = gr.Textbox(
                    f'{"Running" if config.enabled else "Stopped"}',
                    label="Status",
                    elem_id=tab_prefix + "status",
                    readonly=True,
                )

                toggle_running = gr.Button(
                    "Disable",
                    variant="secondary",
                    elem_id=f"{tab_prefix}disable",
                )

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

                toggle_running.click(fn=toggle_running_fn)
        with gr.Row():
            running_type = gr.Textbox(
                "Running Type: Image Generation",
                label="",
                elem_id=tab_prefix + "running-type",
                readonly=True,
                visible=False,
            )

            def call_apis(session, apikey):
                horde_user = HordeUser()
                horde_worker = HordeWorker()
                horde_news = HordeNews()
                horde_status = HordeStatus()
                horde_stats = HordeStats()
                user_info = horde_user.get_user_info(session, apikey)
                # Get worker id from user info
                worker_ids = user_info["worker_ids"]
                for worker in worker_ids:
                    worker_info = horde_worker.get_worker_info(session, apikey, worker)

                    worker_name = worker_info["name"]
                    if worker_name == config.name:
                        print(f"Current Worker: {worker_name}")
                        break

                news_info = horde_news.get_horde_news(session)
                horde_status = horde_status.get_horde_status(session)
                stats_info = horde_stats.get_horde_stats(session)

                return user_info, worker_info, news_info, horde_status, stats_info

            session = requests.Session()
            user_info, worker_info, news_info, horde_status, stats_info = call_apis(
                session, config.apikey
            )

            try:
                with gr.Tab("Generation"):
                    get_generator_ui(status)
            except Exception as e:
                print(f"Error: Generator UI not found, {e}")

            try:
                with gr.Tab("Worker"):
                    get_worker_ui(worker_info)
            except Exception as e:
                print(f"Error: Worker UI not found, {e}")

            try:
                with gr.Tab("User"):
                    get_user_ui(user_info)
            except Exception as e:
                print(f"Error: User UI not found, {e}")

            try:
                with gr.Tab("Kudos"):
                    get_kudos_ui()
            except Exception as e:
                print(f"Error: Kudos UI not found,  {e}")

            try:
                with gr.Tab("News"):
                    get_news_ui(news_info, horde_status)
            except Exception as e:
                print(f"Error: News UI not found, {e}")

            try:
                with gr.Tab("Stats"):
                    get_stats_ui(stats_info)
            except Exception as e:
                print(f"Error: Stats UI not found, {e}")

            try:
                with gr.Tab("Settings"):
                    get_settings_ui(status, running_type)
            except Exception as e:
                print(f"Error: Settings UI not found, {e}")

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
