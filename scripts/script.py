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

async def get_generator_ui(state):
    with gr.Blocks() as generator_ui:
        with gr.Column(elem_id="stable-horde"):
            with gr.Row(equal_height=False):
                with gr.Column():
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

                    with gr.Row():
                        log = gr.HTML(elem_id=tab_prefix + "log")

                    refresh.click(
                        fn=lambda: on_refresh(),
                        outputs=[current_id, log, state],
                        show_progress=False,
                    )
                    refresh_image.click(
                        fn=lambda: on_refresh(True),
                        outputs=[current_id, log, state, preview],
                        show_progress=False,
                    )

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

                    current_id = gr.Textbox(
                        "Current ID: ",
                        label="",
                        elem_id=tab_prefix + "current-id",
                        readonly=True,
                    )
                    preview = gr.Gallery(
                        label="Preview",
                        elem_id=tab_prefix + "preview",
                        visible=config.show_image_preview,
                        readonly=True,
                        columns=4,
                    )

    return (generator_ui)

async def get_worker_ui():
    async with aiohttp.ClientSession() as session:
        user_info = await HordeUser.get_user_info(session, config.apikey)
        worker_info = await HordeWorker.get_worker_info(session, config.apikey, user_info["id"])
        with gr.Blocks() as worker_ui:
            with gr.Row():
                gr.Markdown("## Worker Details")
            with gr.Row():
                gr.Label(f"Type: {worker_info['type']}")
                gr.Label(f"Name: {worker_info['name']}")
                gr.Label(f"ID: {worker_info['id']}")
                gr.Label(f"Online: {worker_info['online']}")
                gr.Label(f"Requests Fulfilled: {worker_info['requests_fulfilled']}")
                gr.Label(f"Kudos Rewards: {worker_info['kudos_rewards']}")
                gr.Label(f"Kudos Generated: {worker_info['kudos_details']['generated']}")
                gr.Label(f"Kudos Uptime: {worker_info['kudos_details']['uptime']}")
                gr.Label(f"Performance: {worker_info['performance']}")
                gr.Label(f"Threads: {worker_info['threads']}")
                gr.Label(f"Uptime: {worker_info['uptime']}")
                gr.Label(f"Maintenance Mode: {worker_info['maintenance_mode']}")
                gr.Label(f"Paused: {worker_info['paused']}")
                gr.Label(f"Info: {worker_info['info']}")
                gr.Label(f"NSFW: {worker_info['nsfw']}")
                gr.Label(f"Owner: {worker_info['owner']}")
                gr.Label(f"IP Address: {worker_info['ipaddr']}")
                gr.Label(f"Trusted: {worker_info['trusted']}")
                gr.Label(f"Flagged: {worker_info['flagged']}")
                gr.Label(f"Suspicious: {worker_info['suspicious']}")
                gr.Label(f"Uncompleted Jobs: {worker_info['uncompleted_jobs']}")
                gr.Label(f"Models: {', '.join(worker_info['models'])}")
                gr.Label(f"Forms: {', '.join(worker_info['forms'])}")
                gr.Label(f"Team Name: {worker_info['team']['name']}")
                gr.Label(f"Team ID: {worker_info['team']['id']}")
                gr.Label(f"Contact: {worker_info['contact']}")
                gr.Label(f"Bridge Agent: {worker_info['bridge_agent']}")
                gr.Label(f"Max Pixels: {worker_info['max_pixels']}")
                gr.Label(f"Megapixelsteps Generated: {worker_info['megapixelsteps_generated']}")
                gr.Label(f"Img2Img: {worker_info['img2img']}")
                gr.Label(f"Painting: {worker_info['painting']}")
                gr.Label(f"Post-Processing: {worker_info['post-processing']}")
                gr.Label(f"Lora: {worker_info['lora']}")
                gr.Label(f"Controlnet: {worker_info['controlnet']}")
                gr.Label(f"SDXL Controlnet: {worker_info['sdxl_controlnet']}")
                gr.Label(f"Max Length: {worker_info['max_length']}")
                gr.Label(f"Max Context Length: {worker_info['max_context_length']}")
                gr.Label(f"Tokens Generated: {worker_info['tokens_generated']}")
    
    return worker_ui


async def get_user_ui():
    user_info = await HordeUser.get_user_info(session, config.apikey)
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
                        [
                            model.name
                            for model in sd_models.checkpoints_list.values()
                        ],
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
                    visible=False,
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
    with gr.Blocks() as demo:
        with gr.Row():
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
        
        with gr.Row():
            status = gr.Textbox(
                f'Status: {"Running" if config.enabled else "Stopped"}',
                label="",
                elem_id=tab_prefix + "status",
                readonly=True,
            )
            running_type = gr.Textbox(
                "Running Type: Image Generation",
                label="",
                elem_id=tab_prefix + "running-type",
                readonly=True,
                )
        with gr.Row():
            state = gr.Textbox("", label="", readonly=True)

        with gr.Tab("Generation"):
            get_generator_ui(state)

        with gr.Tab("Worker"):
            get_worker_ui()

        with gr.Tab("User"):
            get_user_ui()

        with gr.Tab("Settings"):
            get_settings_ui(status, running_type)

    return ((demo, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
