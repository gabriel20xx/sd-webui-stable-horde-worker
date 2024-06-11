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

            state = gr.HTML(
                "",
                label="State",
                elem_id=tab_prefix + "state",
                visible=True,
                readonly=True,
            )

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
            gr.Textbox(f"Type: {worker_info['type']}", interactive=False, lines=1)
        if "name" in worker_info:
            gr.Textbox(f"Name: {worker_info['name']}", interactive=False, lines=1)
        if "id" in worker_info:
            gr.Textbox(f"ID: {worker_info['id']}", interactive=False, lines=1)
        if "online" in worker_info:
            gr.Textbox(f"Online: {worker_info['online']}", interactive=False, lines=1)
        if "requests_fulfilled" in worker_info:
            gr.Textbox(f"Requests Fulfilled: {worker_info['requests_fulfilled']}", interactive=False, lines=1)
        if "kudos_rewards" in worker_info:
            gr.Textbox(f"Kudos Rewards: {worker_info['kudos_rewards']}", interactive=False, lines=1)
        if "kudos_details" in worker_info and "generated" in worker_info["kudos_details"]:
            gr.Textbox(f"Kudos Generated: {worker_info['kudos_details']['generated']}", interactive=False, lines=1)
        if "kudos_details" in worker_info and "uptime" in worker_info["kudos_details"]:
            gr.Textbox(f"Kudos Uptime: {worker_info['kudos_details']['uptime']}", interactive=False, lines=1)
        if "performance" in worker_info:
            gr.Textbox(f"Performance: {worker_info['performance']}", interactive=False, lines=1)
        if "threads" in worker_info:
            gr.Textbox(f"Threads: {worker_info['threads']}", interactive=False, lines=1)
        if "uptime" in worker_info:
            gr.Textbox(f"Uptime: {worker_info['uptime']}", interactive=False, lines=1)
        if "maintenance_mode" in worker_info:
            gr.Textbox(f"Maintenance Mode: {worker_info['maintenance_mode']}", interactive=False, lines=1)
        if "paused" in worker_info:
            gr.Textbox(f"Paused: {worker_info['paused']}", interactive=False, lines=1)
        if "info" in worker_info:
            gr.Textbox(f"Info: {worker_info['info']}", interactive=False, lines=1)
        if "nsfw" in worker_info:
            gr.Textbox(f"NSFW: {worker_info['nsfw']}", interactive=False, lines=1)
        if "owner" in worker_info:
            gr.Textbox(f"Owner: {worker_info['owner']}", interactive=False, lines=1)
        if "ipaddr" in worker_info:
            gr.Textbox(f"IP Address: {worker_info['ipaddr']}", interactive=False, lines=1)
        if "trusted" in worker_info:
            gr.Textbox(f"Trusted: {worker_info['trusted']}", interactive=False, lines=1)
        if "flagged" in worker_info:
            gr.Textbox(f"Flagged: {worker_info['flagged']}", interactive=False, lines=1)
        if "suspicious" in worker_info:
            gr.Textbox(f"Suspicious: {worker_info['suspicious']}", interactive=False, lines=1)
        if "uncompleted_jobs" in worker_info:
            gr.Textbox(f"Uncompleted Jobs: {worker_info['uncompleted_jobs']}", interactive=False, lines=1)
        if "models" in worker_info:
            gr.Textbox(f"Models: {', '.join(worker_info['models'])}", interactive=False, lines=1)
        if "forms" in worker_info:
            gr.Textbox(f"Forms: {', '.join(worker_info['forms'])}", interactive=False, lines=1)
        if "team" in worker_info and "name" in worker_info["team"]:
            gr.Textbox(f"Team Name: {worker_info['team']['name']}", interactive=False, lines=1)
        if "team" in worker_info and "id" in worker_info["team"]:
            gr.Textbox(f"Team ID: {worker_info['team']['id']}", interactive=False, lines=1)
        if "contact" in worker_info:
            gr.Textbox(f"Contact: {worker_info['contact']}", interactive=False, lines=1)
        if "bridge_agent" in worker_info:
            gr.Textbox(f"Bridge Agent: {worker_info['bridge_agent']}", interactive=False, lines=1)
        if "max_pixels" in worker_info:
            gr.Textbox(f"Max Pixels: {worker_info['max_pixels']}", interactive=False, lines=1)
        if "megapixelsteps_generated" in worker_info:
            gr.Textbox(f"Megapixelsteps Generated: {worker_info['megapixelsteps_generated']}", interactive=False, lines=1)
        if "img2img" in worker_info:
            gr.Textbox(f"Img2Img: {worker_info['img2img']}", interactive=False, lines=1)
        if "painting" in worker_info:
            gr.Textbox(f"Painting: {worker_info['painting']}", interactive=False, lines=1)
        if "post-processing" in worker_info:
            gr.Textbox(f"Post-Processing: {worker_info['post-processing']}", interactive=False, lines=1)
        if "lora" in worker_info:
            gr.Textbox(f"Lora: {worker_info['lora']}", interactive=False, lines=1)
        if "controlnet" in worker_info:
            gr.Textbox(f"Controlnet: {worker_info['controlnet']}", interactive=False, lines=1)
        if "sdxl_controlnet" in worker_info:
            gr.Textbox(f"SDXL Controlnet: {worker_info['sdxl_controlnet']}", interactive=False, lines=1)
        if "max_length" in worker_info:
            gr.Textbox(f"Max Length: {worker_info['max_length']}", interactive=False, lines=1)
        if "max_context_length" in worker_info:
            gr.Textbox(f"Max Context Length: {worker_info['max_context_length']}", interactive=False, lines=1)
        if "tokens_generated" in worker_info:
            gr.Textbox(f"Tokens Generated: {worker_info['tokens_generated']}", interactive=False, lines=1)

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
            gr.Markdown("## User Details")
            if "username" in user_info:
                gr.Textbox(f"Username: {user_info['username']}", interactive=False, lines=1)
            if "id" in user_info:
                gr.Textbox(f"ID: {user_info['id']}", interactive=False, lines=1)
            if "kudos" in user_info:
                gr.Textbox(f"Kudos: {user_info['kudos']}", interactive=False, lines=1)
            if "evaluating_kudos" in user_info:
                gr.Textbox(f"Evaluating Kudos: {user_info['evaluating_kudos']}", interactive=False, lines=1)
            if "concurrency" in user_info:
                gr.Textbox(f"Concurrency: {user_info['concurrency']}", interactive=False, lines=1)
            if "worker_invited" in user_info:
                gr.Textbox(f"Worker Invited: {user_info['worker_invited']}", interactive=False, lines=1)
            if "moderator" in user_info:
                gr.Textbox(f"Moderator: {user_info['moderator']}", interactive=False, lines=1)
            if "kudos_details" in user_info:
                kudos_details = user_info["kudos_details"]
                if "accumulated" in kudos_details:
                    gr.Textbox(f"Accumulated Kudos: {kudos_details['accumulated']}", interactive=False, lines=1)
                if "gifted" in kudos_details:
                    gr.Textbox(f"Gifted Kudos: {kudos_details['gifted']}", interactive=False, lines=1)
                if "donated" in kudos_details:
                    gr.Textbox(f"Donated Kudos: {kudos_details['donated']}", interactive=False, lines=1)
                if "admin" in kudos_details:
                    gr.Textbox(f"Admin Kudos: {kudos_details['admin']}", interactive=False, lines=1)
                if "received" in kudos_details:
                    gr.Textbox(f"Received Kudos: {kudos_details['received']}", interactive=False, lines=1)
                if "recurring" in kudos_details:
                    gr.Textbox(f"Recurring Kudos: {kudos_details['recurring']}", interactive=False, lines=1)
                if "awarded" in kudos_details:
                    gr.Textbox(f"Awarded Kudos: {kudos_details['awarded']}", interactive=False, lines=1)
            if "worker_count" in user_info:
                gr.Textbox(f"Worker Count: {user_info['worker_count']}", interactive=False, lines=1)
            if "worker_ids" in user_info:
                gr.Textbox(f"Worker IDs: {', '.join(user_info['worker_ids'])}", interactive=False, lines=1)
            if "sharedkey_ids" in user_info:
                gr.Textbox(f"Shared Key IDs: {', '.join(user_info['sharedkey_ids'])}", interactive=False, lines=1)
            if "monthly_kudos" in user_info:
                monthly_kudos = user_info["monthly_kudos"]
                if "amount" in monthly_kudos:
                    gr.Textbox(f"Monthly Kudos Amount: {monthly_kudos['amount']}", interactive=False, lines=1)
                if "last_received" in monthly_kudos:
                    gr.Textbox(f"Last Monthly Kudos Received: {monthly_kudos['last_received']}", interactive=False, lines=1)
            if "trusted" in user_info:
                gr.Textbox(f"Trusted: {user_info['trusted']}", interactive=False, lines=1)
            if "flagged" in user_info:
                gr.Textbox(f"Flagged: {user_info['flagged']}", interactive=False, lines=1)
            if "vpn" in user_info:
                gr.Textbox(f"VPN: {user_info['vpn']}", interactive=False, lines=1)
            if "service" in user_info:
                gr.Textbox(f"Service: {user_info['service']}", interactive=False, lines=1)
            if "education" in user_info:
                gr.Textbox(f"Education: {user_info['education']}", interactive=False, lines=1)
            if "customizer" in user_info:
                gr.Textbox(f"Customizer: {user_info['customizer']}", interactive=False, lines=1)
            if "special" in user_info:
                gr.Textbox(f"Special: {user_info['special']}", interactive=False, lines=1)
            if "suspicious" in user_info:
                gr.Textbox(f"Suspicious: {user_info['suspicious']}", interactive=False, lines=1)
            if "pseudonymous" in user_info:
                gr.Textbox(f"Pseudonymous: {user_info['pseudonymous']}", interactive=False, lines=1)
            if "contact" in user_info:
                gr.Textbox(f"Contact: {user_info['contact']}", interactive=False, lines=1)
            if "admin_comment" in user_info:
                gr.Textbox(f"Admin Comment: {user_info['admin_comment']}", interactive=False, lines=1)
            if "account_age" in user_info:
                gr.Textbox(f"Account Age: {user_info['account_age']}", interactive=False, lines=1)
            if "usage" in user_info:
                usage = user_info["usage"]
                if "megapixelsteps" in usage:
                    gr.Textbox(f"Usage Megapixelsteps: {usage['megapixelsteps']}", interactive=False, lines=1)
                if "requests" in usage:
                    gr.Textbox(f"Usage Requests: {usage['requests']}", interactive=False, lines=1)
            if "contributions" in user_info:
                contributions = user_info["contributions"]
                if "megapixelsteps" in contributions:
                    gr.Textbox(f"Contribution Megapixelsteps: {contributions['megapixelsteps']}", interactive=False, lines=1)
                if "fulfillments" in contributions:
                    gr.Textbox(f"Contribution Fulfillments: {contributions['fulfillments']}", interactive=False, lines=1)
            if "records" in user_info:
                records = user_info["records"]
                if "usage" in records:
                    usage_records = records["usage"]
                    if "megapixelsteps" in usage_records:
                        gr.Textbox(f"Record Usage Megapixelsteps: {usage_records['megapixelsteps']}", interactive=False, lines=1)
                    if "tokens" in usage_records:
                        gr.Textbox(f"Record Usage Tokens: {usage_records['tokens']}", interactive=False, lines=1)
                if "contribution" in records:
                    contribution_records = records["contribution"]
                    if "megapixelsteps" in contribution_records:
                        gr.Textbox(f"Record Contribution Megapixelsteps: {contribution_records['megapixelsteps']}", interactive=False, lines=1)
                    if "tokens" in contribution_records:
                        gr.Textbox(f"Record Contribution Tokens: {contribution_records['tokens']}", interactive=False, lines=1)
                if "fulfillment" in records:
                    fulfillment_records = records["fulfillment"]
                    if "image" in fulfillment_records:
                        gr.Textbox(f"Fulfillment Image: {fulfillment_records['image']}", interactive=False, lines=1)
                    if "text" in fulfillment_records:
                        gr.Textbox(f"Fulfillment Text: {fulfillment_records['text']}", interactive=False, lines=1)
                    if "interrogation" in fulfillment_records:
                        gr.Textbox(f"Fulfillment Interrogation: {fulfillment_records['interrogation']}", interactive=False, lines=1)
                if "request" in records:
                    request_records = records["request"]
                    if "image" in request_records:
                        gr.Textbox(f"Request Image: {request_records['image']}", interactive=False, lines=1)
                    if "text" in request_records:
                        gr.Textbox(f"Request Text: {request_records['text']}", interactive=False, lines=1)
                    if "interrogation" in request_records:
                        gr.Textbox(f"Request Interrogation: {request_records['interrogation']}", interactive=False, lines=1)

        """ def update_user_info():
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
                
                + "".join(workers_table_cells)
                + 
                </tbody>
                </table>
                
            )

            return (
                f"Welcome Back, **{horde.state.user.username}** !",
                workers_html,
            )

        user_update.click(fn=update_user_info, outputs=[user_welcome, workers]) """

    return user_ui


def get_settings_ui(status, running_type):
    with gr.Blocks() as settings_ui:
        with gr.Column():
            with gr.Box(scale=2):
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
                    config.enabled,
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

            toggle_running = gr.Button("Disable", elem_id=f"{tab_prefix}disable")

            def toggle_running_fn():
                if config.enabled:
                    config.enabled = False
                    status.update("Status: Stopped")
                    running_type.update("Running Type: Image Generation")
                    toggle_running.update(value="Enable")
                else:
                    config.enabled = True
                    status.update("Status: Running")
                    toggle_running.update(value="Disable")
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
