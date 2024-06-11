from typing import Optional

from fastapi import FastAPI
import gradio as gr
import asyncio
import aiohttp
import requests
from threading import Thread

from modules import scripts, script_callbacks, sd_models, shared

from stable_horde import StableHorde, StableHordeConfig, HordeUser, HordeWorker

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
        with gr.Column(elem_id="stable-horde"):
            current_id = gr.Textbox(
                "Current ID: ",
                label="",
                elem_id=tab_prefix + "current-id",
                readonly=True,
            )
            with gr.Row(equal_height=False):
                with gr.Column() as refresh_column:
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

            with gr.Row(equal_height=False):
                with gr.Column():
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

                with gr.Column():
                    log = gr.HTML(elem_id=tab_prefix + "log")

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
        if "type" in worker_info:
            gr.Label(f"Type: {worker_info['type']}")
        if "name" in worker_info:
            gr.Label(f"Name: {worker_info['name']}")
        if "id" in worker_info:
            gr.Label(f"ID: {worker_info['id']}")
        if "online" in worker_info:
            gr.Label(f"Online: {worker_info['online']}")
        if "requests_fulfilled" in worker_info:
            gr.Label(f"Requests Fulfilled: {worker_info['requests_fulfilled']}")
        if "kudos_rewards" in worker_info:
            gr.Label(f"Kudos Rewards: {worker_info['kudos_rewards']}")
        if (
            "kudos_details" in worker_info
            and "generated" in worker_info["kudos_details"]
        ):
            gr.Label(
                f"Kudos Generated: {worker_info['kudos_details']['generated']}"
            )
        if (
            "kudos_details" in worker_info
            and "uptime" in worker_info["kudos_details"]
        ):
            gr.Label(f"Kudos Uptime: {worker_info['kudos_details']['uptime']}")
        if "performance" in worker_info:
            gr.Label(f"Performance: {worker_info['performance']}")
        if "threads" in worker_info:
            gr.Label(f"Threads: {worker_info['threads']}")
        if "uptime" in worker_info:
            gr.Label(f"Uptime: {worker_info['uptime']}")
        if "maintenance_mode" in worker_info:
            gr.Label(f"Maintenance Mode: {worker_info['maintenance_mode']}")
        if "paused" in worker_info:
            gr.Label(f"Paused: {worker_info['paused']}")
        if "info" in worker_info:
            gr.Label(f"Info: {worker_info['info']}")
        if "nsfw" in worker_info:
            gr.Label(f"NSFW: {worker_info['nsfw']}")
        if "owner" in worker_info:
            gr.Label(f"Owner: {worker_info['owner']}")
        if "ipaddr" in worker_info:
            gr.Label(f"IP Address: {worker_info['ipaddr']}")
        if "trusted" in worker_info:
            gr.Label(f"Trusted: {worker_info['trusted']}")
        if "flagged" in worker_info:
            gr.Label(f"Flagged: {worker_info['flagged']}")
        if "suspicious" in worker_info:
            gr.Label(f"Suspicious: {worker_info['suspicious']}")
        if "uncompleted_jobs" in worker_info:
            gr.Label(f"Uncompleted Jobs: {worker_info['uncompleted_jobs']}")
        if "models" in worker_info:
            gr.Label(f"Models: {', '.join(worker_info['models'])}")
        if "forms" in worker_info:
            gr.Label(f"Forms: {', '.join(worker_info['forms'])}")
        if "team" in worker_info and "name" in worker_info["team"]:
            gr.Label(f"Team Name: {worker_info['team']['name']}")
        if "team" in worker_info and "id" in worker_info["team"]:
            gr.Label(f"Team ID: {worker_info['team']['id']}")
        if "contact" in worker_info:
            gr.Label(f"Contact: {worker_info['contact']}")
        if "bridge_agent" in worker_info:
            gr.Label(f"Bridge Agent: {worker_info['bridge_agent']}")
        if "max_pixels" in worker_info:
            gr.Label(f"Max Pixels: {worker_info['max_pixels']}")
        if "megapixelsteps_generated" in worker_info:
            gr.Label(
                f"Megapixelsteps Generated: {worker_info['megapixelsteps_generated']}"
            )
        if "img2img" in worker_info:
            gr.Label(f"Img2Img: {worker_info['img2img']}")
        if "painting" in worker_info:
            gr.Label(f"Painting: {worker_info['painting']}")
        if "post-processing" in worker_info:
            gr.Label(f"Post-Processing: {worker_info['post-processing']}")
        if "lora" in worker_info:
            gr.Label(f"Lora: {worker_info['lora']}")
        if "controlnet" in worker_info:
            gr.Label(f"Controlnet: {worker_info['controlnet']}")
        if "sdxl_controlnet" in worker_info:
            gr.Label(f"SDXL Controlnet: {worker_info['sdxl_controlnet']}")
        if "max_length" in worker_info:
            gr.Label(f"Max Length: {worker_info['max_length']}")
        if "max_context_length" in worker_info:
            gr.Label(f"Max Context Length: {worker_info['max_context_length']}")
        if "tokens_generated" in worker_info:
            gr.Label(f"Tokens Generated: {worker_info['tokens_generated']}")

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
            workers = gr.HTML("No Worker")

        def update_user_info():
            if horde.state.user is None:
                return (
                    "**Try click update button to fetch the user info**",
                    "No Worker",
                )

            def map_worker_detail(worker: HordeWorker):
                return "\n".join(
                    map(
                        lambda x: f"<td>{x}</td>",
                        [
                            worker.id,
                            worker.name,
                            worker.maintenance_mode,
                            '<button onclick="'
                            + f"stableHordeSwitchMaintenance('{worker.id}')\">"
                            + "Switch Maintenance</button>",
                        ],
                    )
                )

            workers_table_cells = map(
                lambda x: f"<tr>{map_worker_detail(x)}</tr>",
                horde.state.user.workers,
            )

            workers_html = (
                """
                <table>
                <thead>
                <tr>
                <th>Worker ID</th>
                <th>Worker Name</th>
                <th>Maintenance Mode ?</th>
                <th>Actions</th>
                </tr>
                </thead>
                <tbody>
                """
                + "".join(workers_table_cells)
                + """
                </tbody>
                </table>
                """
            )

            return (
                f"Welcome Back, **{horde.state.user.username}** !",
                workers_html,
            )

        user_update.click(fn=update_user_info, outputs=[user_welcome, workers])

        return user_ui


def get_settings_ui(status, running_type):
    with gr.Blocks() as settings_ui:
        with gr.Row():
            with gr.Column():
                with gr.Box(scale=2):
                    enable = gr.Checkbox(
                        config.enabled,
                        label="Enable",
                        elem_id=tab_prefix + "enable",
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
    with gr.Blocks() as ui_tabs:
        with gr.Column():
            apikey = gr.Textbox(
                config.apikey,
                label="Stable Horde API Key",
                elem_id=tab_prefix + "apikey",
            )
            save_apikey = gr.Button("Save", elem_id=f"{tab_prefix}apikey-save")

            def save_apikey_fn(apikey: str):
                config.apikey = apikey
                config.save()

            save_apikey.click(fn=save_apikey_fn, inputs=[apikey])

        with gr.Column():
            status = gr.Textbox(
                f'Status: {"Running" if config.enabled else "Stopped"}',
                label="",
                elem_id=tab_prefix + "status",
                readonly=True,
            )

            toggle_running = gr.Button("Disable", elem_id=f"{tab_prefix}disable")

            def toggle_running_fn():
                if config.enabled:
                    config.enabled = False
                    status.update("Status: Stopped")
                    running_type.update("Running Type: Image Generation")
                else:
                    config.enabled = True
                    status.update("Status: Running")
                config.save()

            toggle_running.click(fn=toggle_running_fn)

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
            user_info = horde_user.get_user_info(session, apikey)
            # Get worker id from user info
            worker_ids = user_info["worker_ids"]
            for worker in worker_ids:
                worker_info = horde_worker.get_worker_info(session, apikey, worker)

                worker_name = worker_info["name"]
                if worker_name == config.name:
                    print(f"Current Worker: {worker_name}")
                    break

            return user_info, worker_info

        session = requests.Session()
        user_info, worker_info = call_apis(session, config.apikey)

        with gr.Tab("Generation"):
            get_generator_ui(status)

        with gr.Tab("Worker"):
            get_worker_ui(worker_info)

        with gr.Tab("User"):
            get_user_ui(user_info)

        with gr.Tab("Settings"):
            get_settings_ui(status, running_type)

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
