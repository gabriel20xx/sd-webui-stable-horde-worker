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

    return generator_ui


# Worker UI
def get_worker_ui(worker):
    with gr.Blocks() as worker_ui:
        # Worker functions
        horde_worker = API()
        worker_info = horde_worker.get_worker_info(session, config.apikey, worker)

        # Worker UI
        gr.Markdown("## Worker Details")
        with gr.Row():
            worker_update = gr.Button(
                "Update Worker Details", elem_id=f"{tab_prefix}worker-update"
            )
        with gr.Column():
            if "type" in worker_info:
                gr.Textbox(f"Type: {worker_info['type']}", interactive=False, lines=1)
            if "name" in worker_info:
                gr.Textbox(f"Name: {worker_info['name']}", interactive=False, lines=1)
            if "id" in worker_info:
                gr.Textbox(f"ID: {worker_info['id']}", interactive=False, lines=1)
            if "online" in worker_info:
                gr.Textbox(f"Online: {worker_info['online']}", interactive=False, lines=1)
            if "requests_fulfilled" in worker_info:
                gr.Textbox(
                    f"Requests Fulfilled: {worker_info['requests_fulfilled']}",
                    interactive=False,
                    lines=1,
                )
            if "kudos_rewards" in worker_info:
                gr.Textbox(
                    f"Kudos Rewards: {worker_info['kudos_rewards']}",
                    interactive=False,
                    lines=1,
                )
            if (
                "kudos_details" in worker_info
                and "generated" in worker_info["kudos_details"]
            ):
                gr.Textbox(
                    f"Kudos Generated: {worker_info['kudos_details']['generated']}",
                    interactive=False,
                    lines=1,
                )
            if "kudos_details" in worker_info and "uptime" in worker_info["kudos_details"]:
                gr.Textbox(
                    f"Kudos Uptime: {worker_info['kudos_details']['uptime']}",
                    interactive=False,
                    lines=1,
                )
            if "performance" in worker_info:
                gr.Textbox(
                    f"Performance: {worker_info['performance']}", interactive=False, lines=1
                )
            if "threads" in worker_info:
                gr.Textbox(f"Threads: {worker_info['threads']}", interactive=False, lines=1)
            if "uptime" in worker_info:
                gr.Textbox(f"Uptime: {worker_info['uptime']}", interactive=False, lines=1)
            if "maintenance_mode" in worker_info:
                gr.Textbox(
                    f"Maintenance Mode: {worker_info['maintenance_mode']}",
                    interactive=False,
                    lines=1,
                )
            if "paused" in worker_info:
                gr.Textbox(f"Paused: {worker_info['paused']}", interactive=False, lines=1)
            if "info" in worker_info:
                gr.Textbox(f"Info: {worker_info['info']}", interactive=False, lines=1)
            if "nsfw" in worker_info:
                gr.Textbox(f"NSFW: {worker_info['nsfw']}", interactive=False, lines=1)
            if "owner" in worker_info:
                gr.Textbox(f"Owner: {worker_info['owner']}", interactive=False, lines=1)
            if "ipaddr" in worker_info:
                gr.Textbox(
                    f"IP Address: {worker_info['ipaddr']}", interactive=False, lines=1
                )
            if "trusted" in worker_info:
                gr.Textbox(f"Trusted: {worker_info['trusted']}", interactive=False, lines=1)
            if "flagged" in worker_info:
                gr.Textbox(f"Flagged: {worker_info['flagged']}", interactive=False, lines=1)
            if "suspicious" in worker_info:
                gr.Textbox(
                    f"Suspicious: {worker_info['suspicious']}", interactive=False, lines=1
                )
            if "uncompleted_jobs" in worker_info:
                gr.Textbox(
                    f"Uncompleted Jobs: {worker_info['uncompleted_jobs']}",
                    interactive=False,
                    lines=1,
                )
            if "models" in worker_info:
                gr.Textbox(
                    f"Models: {', '.join(worker_info['models'])}",
                    interactive=False,
                    lines=1,
                )
            if "forms" in worker_info:
                gr.Textbox(
                    f"Forms: {', '.join(worker_info['forms'])}", interactive=False, lines=1
                )
            if "team" in worker_info and "name" in worker_info["team"]:
                gr.Textbox(
                    f"Team Name: {worker_info['team']['name']}", interactive=False, lines=1
                )
            if "team" in worker_info and "id" in worker_info["team"]:
                gr.Textbox(
                    f"Team ID: {worker_info['team']['id']}", interactive=False, lines=1
                )
            if "contact" in worker_info:
                gr.Textbox(f"Contact: {worker_info['contact']}", interactive=False, lines=1)
            if "bridge_agent" in worker_info:
                gr.Textbox(
                    f"Bridge Agent: {worker_info['bridge_agent']}",
                    interactive=False,
                    lines=1,
                )
            if "max_pixels" in worker_info:
                gr.Textbox(
                    f"Max Pixels: {worker_info['max_pixels']}", interactive=False, lines=1
                )
            if "megapixelsteps_generated" in worker_info:
                gr.Textbox(
                    f"Megapixelsteps Generated: {worker_info['megapixelsteps_generated']}",
                    interactive=False,
                    lines=1,
                )
            if "img2img" in worker_info:
                gr.Textbox(f"Img2Img: {worker_info['img2img']}", interactive=False, lines=1)
            if "painting" in worker_info:
                gr.Textbox(
                    f"Painting: {worker_info['painting']}", interactive=False, lines=1
                )
            if "post-processing" in worker_info:
                gr.Textbox(
                    f"Post-Processing: {worker_info['post-processing']}",
                    interactive=False,
                    lines=1,
                )
            if "lora" in worker_info:
                gr.Textbox(f"Lora: {worker_info['lora']}", interactive=False, lines=1)
            if "controlnet" in worker_info:
                gr.Textbox(
                    f"Controlnet: {worker_info['controlnet']}", interactive=False, lines=1
                )
            if "sdxl_controlnet" in worker_info:
                gr.Textbox(
                    f"SDXL Controlnet: {worker_info['sdxl_controlnet']}",
                    interactive=False,
                    lines=1,
                )
            if "max_length" in worker_info:
                gr.Textbox(
                    f"Max Length: {worker_info['max_length']}", interactive=False, lines=1
                )
            if "max_context_length" in worker_info:
                gr.Textbox(
                    f"Max Context Length: {worker_info['max_context_length']}",
                    interactive=False,
                    lines=1,
                )
            if "tokens_generated" in worker_info:
                gr.Textbox(
                    f"Tokens Generated: {worker_info['tokens_generated']}",
                    interactive=False,
                    lines=1,
                )

    return worker_ui


# User UI
def get_user_ui():
    with gr.Blocks() as user_ui:
        # User functions
        api = API()
        user_info = api.get_user_info(session, config.apikey)

        # User UI
        gr.Markdown("## User Details", elem_id="user_title")
        with gr.Row():
            user_update = gr.Button(
                "Update User Details", elem_id=f"{tab_prefix}user-update"
            )
        with gr.Column():
            if "username" in user_info:
                gr.Textbox(
                    f"Username: {user_info['username']}", interactive=False, lines=1
                )
            if "id" in user_info:
                gr.Textbox(f"ID: {user_info['id']}", interactive=False, lines=1)
            if "kudos" in user_info:
                gr.Textbox(f"Kudos: {user_info['kudos']}", interactive=False, lines=1)
            if "evaluating_kudos" in user_info:
                gr.Textbox(
                    f"Evaluating Kudos: {user_info['evaluating_kudos']}",
                    interactive=False,
                    lines=1,
                )
            if "concurrency" in user_info:
                gr.Textbox(
                    f"Concurrency: {user_info['concurrency']}",
                    interactive=False,
                    lines=1,
                )
            if "worker_invited" in user_info:
                gr.Textbox(
                    f"Worker Invited: {user_info['worker_invited']}",
                    interactive=False,
                    lines=1,
                )
            if "moderator" in user_info:
                gr.Textbox(
                    f"Moderator: {user_info['moderator']}", interactive=False, lines=1
                )
            if "kudos_details" in user_info:
                kudos_details = user_info["kudos_details"]
                if "accumulated" in kudos_details:
                    gr.Textbox(
                        f"Accumulated Kudos: {kudos_details['accumulated']}",
                        interactive=False,
                        lines=1,
                    )
                if "gifted" in kudos_details:
                    gr.Textbox(
                        f"Gifted Kudos: {kudos_details['gifted']}",
                        interactive=False,
                        lines=1,
                    )
                if "donated" in kudos_details:
                    gr.Textbox(
                        f"Donated Kudos: {kudos_details['donated']}",
                        interactive=False,
                        lines=1,
                    )
                if "admin" in kudos_details:
                    gr.Textbox(
                        f"Admin Kudos: {kudos_details['admin']}",
                        interactive=False,
                        lines=1,
                    )
                if "received" in kudos_details:
                    gr.Textbox(
                        f"Received Kudos: {kudos_details['received']}",
                        interactive=False,
                        lines=1,
                    )
                if "recurring" in kudos_details:
                    gr.Textbox(
                        f"Recurring Kudos: {kudos_details['recurring']}",
                        interactive=False,
                        lines=1,
                    )
                if "awarded" in kudos_details:
                    gr.Textbox(
                        f"Awarded Kudos: {kudos_details['awarded']}",
                        interactive=False,
                        lines=1,
                    )
            if "worker_count" in user_info:
                gr.Textbox(
                    f"Worker Count: {user_info['worker_count']}",
                    interactive=False,
                    lines=1,
                )
            if "worker_ids" in user_info:
                gr.Textbox(
                    f"Worker IDs: {', '.join(user_info['worker_ids'])}",
                    interactive=False,
                    lines=1,
                )
            if "sharedkey_ids" in user_info:
                gr.Textbox(
                    f"Shared Key IDs: {', '.join(user_info['sharedkey_ids'])}",
                    interactive=False,
                    lines=1,
                )
            if "monthly_kudos" in user_info:
                monthly_kudos = user_info["monthly_kudos"]
                if "amount" in monthly_kudos:
                    gr.Textbox(
                        f"Monthly Kudos Amount: {monthly_kudos['amount']}",
                        interactive=False,
                        lines=1,
                    )
                if "last_received" in monthly_kudos:
                    gr.Textbox(
                        f"Last Monthly Kudos Received: {monthly_kudos['last_received']}",
                        interactive=False,
                        lines=1,
                    )
            if "trusted" in user_info:
                gr.Textbox(
                    f"Trusted: {user_info['trusted']}", interactive=False, lines=1
                )
            if "flagged" in user_info:
                gr.Textbox(
                    f"Flagged: {user_info['flagged']}", interactive=False, lines=1
                )
            if "vpn" in user_info:
                gr.Textbox(f"VPN: {user_info['vpn']}", interactive=False, lines=1)
            if "service" in user_info:
                gr.Textbox(
                    f"Service: {user_info['service']}", interactive=False, lines=1
                )
            if "education" in user_info:
                gr.Textbox(
                    f"Education: {user_info['education']}", interactive=False, lines=1
                )
            if "customizer" in user_info:
                gr.Textbox(
                    f"Customizer: {user_info['customizer']}", interactive=False, lines=1
                )
            if "special" in user_info:
                gr.Textbox(
                    f"Special: {user_info['special']}", interactive=False, lines=1
                )
            if "suspicious" in user_info:
                gr.Textbox(
                    f"Suspicious: {user_info['suspicious']}", interactive=False, lines=1
                )
            if "pseudonymous" in user_info:
                gr.Textbox(
                    f"Pseudonymous: {user_info['pseudonymous']}",
                    interactive=False,
                    lines=1,
                )
            if "contact" in user_info:
                gr.Textbox(
                    f"Contact: {user_info['contact']}", interactive=False, lines=1
                )
            if "admin_comment" in user_info:
                gr.Textbox(
                    f"Admin Comment: {user_info['admin_comment']}",
                    interactive=False,
                    lines=1,
                )
            if "account_age" in user_info:
                gr.Textbox(
                    f"Account Age: {user_info['account_age']}",
                    interactive=False,
                    lines=1,
                )
            if "usage" in user_info:
                usage = user_info["usage"]
                if "megapixelsteps" in usage:
                    gr.Textbox(
                        f"Usage Megapixelsteps: {usage['megapixelsteps']}",
                        interactive=False,
                        lines=1,
                    )
                if "requests" in usage:
                    gr.Textbox(
                        f"Usage Requests: {usage['requests']}",
                        interactive=False,
                        lines=1,
                    )
            if "contributions" in user_info:
                contributions = user_info["contributions"]
                if "megapixelsteps" in contributions:
                    gr.Textbox(
                        f"Contribution Megapixelsteps: {contributions['megapixelsteps']}",
                        interactive=False,
                        lines=1,
                    )
                if "fulfillments" in contributions:
                    gr.Textbox(
                        f"Contribution Fulfillments: {contributions['fulfillments']}",
                        interactive=False,
                        lines=1,
                    )
            if "records" in user_info:
                records = user_info["records"]
                if "usage" in records:
                    usage_records = records["usage"]
                    if "megapixelsteps" in usage_records:
                        gr.Textbox(
                            f"Record Usage Megapixelsteps: {usage_records['megapixelsteps']}",
                            interactive=False,
                            lines=1,
                        )
                    if "tokens" in usage_records:
                        gr.Textbox(
                            f"Record Usage Tokens: {usage_records['tokens']}",
                            interactive=False,
                            lines=1,
                        )
                if "contribution" in records:
                    contribution_records = records["contribution"]
                    if "megapixelsteps" in contribution_records:
                        gr.Textbox(
                            f"Record Contribution Megapixelsteps: {contribution_records['megapixelsteps']}",
                            interactive=False,
                            lines=1,
                        )
                    if "tokens" in contribution_records:
                        gr.Textbox(
                            f"Record Contribution Tokens: {contribution_records['tokens']}",
                            interactive=False,
                            lines=1,
                        )
                if "fulfillment" in records:
                    fulfillment_records = records["fulfillment"]
                    if "image" in fulfillment_records:
                        gr.Textbox(
                            f"Fulfillment Image: {fulfillment_records['image']}",
                            interactive=False,
                            lines=1,
                        )
                    if "text" in fulfillment_records:
                        gr.Textbox(
                            f"Fulfillment Text: {fulfillment_records['text']}",
                            interactive=False,
                            lines=1,
                        )
                    if "interrogation" in fulfillment_records:
                        gr.Textbox(
                            f"Fulfillment Interrogation: {fulfillment_records['interrogation']}",
                            interactive=False,
                            lines=1,
                        )
                if "request" in records:
                    request_records = records["request"]
                    if "image" in request_records:
                        gr.Textbox(
                            f"Request Image: {request_records['image']}",
                            interactive=False,
                            lines=1,
                        )
                    if "text" in request_records:
                        gr.Textbox(
                            f"Request Text: {request_records['text']}",
                            interactive=False,
                            lines=1,
                        )
                    if "interrogation" in request_records:
                        gr.Textbox(
                            f"Request Interrogation: {request_records['interrogation']}",
                            interactive=False,
                            lines=1,
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

    return kudos_ui


# News UI
def get_news_ui():
    with gr.Blocks() as news_ui:
        # News functions
        api = API()
        news_info = api.get_horde_news(session)
        status_info = api.get_horde_status(session)

        # News UI
        gr.Markdown(
            "## News",
            elem_id="news_title",
        )
        with gr.Row():
            news_update = gr.Button("Update News", elem_id=f"{tab_prefix}news-update")

        with gr.Box(scale=2):
            with gr.Column():
                status_modes = {
                    "maintenance_mode": "Maintenance mode",
                    "invite_only_mode": "Invite only mode",
                    "raid_mode": "Raid mode",
                }

            for mode, label in status_modes.items():
                if mode in status_info:
                    gr.Textbox(
                        status_info[mode],
                        label=label,
                        elem_id=tab_prefix + f"status_{mode}",
                        visible=True,
                        interactive=False,
                    )
        with gr.Box(scale=2):
            with gr.Column():
                for news_item in news_info[:3]:
                    if "title" and "newspiece" and "date_published" in news_item:
                        gr.Textbox(
                            news_item["newspiece"],
                            label=news_item["date_published"]
                            + " - "
                            + news_item["title"],
                            elem_id=tab_prefix + "news_title",
                            visible=True,
                            interactive=False,
                        )

    return news_ui


def get_stats_ui(stats_info):
    with gr.Blocks() as stats_ui:
        # Stats functions
        api = API()
        stats_info = api.get_horde_stats(session)

        # Stats UI
        gr.Markdown(
            "## Stats",
            elem_id="stats_title",
        )
        with gr.Row():
            stats_update = gr.Button(
                "Update Stats", elem_id=f"{tab_prefix}stats-update"
            )
        stats_info = gr.JSON(
            value=stats_info,
            label="Stats",
            interactive=False,
            elem_id=f"{tab_prefix}stats-info",
            visible=False,
        )
        with gr.Box(scale=2):
            with gr.Column():
                if "minute" in stats_info:
                    minute_images = stats_info["minute"]["images"]
                    minute_ps = stats_info["minute"]["ps"]
                    gr.Textbox(
                        f"Minute Images: {minute_images}",
                        interactive=False,
                        lines=1,
                    )
                    gr.Textbox(
                        f"Minute Processing Speed: {minute_ps}",
                        interactive=False,
                        lines=1,
                    )

                # Displaying hour statistics
                if "hour" in stats_info:
                    hour_images = stats_info["hour"]["images"]
                    hour_ps = stats_info["hour"]["ps"]
                    gr.Textbox(
                        f"Hour Images: {hour_images}",
                        interactive=False,
                        lines=1,
                    )
                    gr.Textbox(
                        f"Hour Processing Speed: {hour_ps}",
                        interactive=False,
                        lines=1,
                    )

                # Displaying day statistics
                if "day" in stats_info:
                    day_images = stats_info["day"]["images"]
                    day_ps = stats_info["day"]["ps"]
                    gr.Textbox(
                        f"Day Images: {day_images}",
                        interactive=False,
                        lines=1,
                    )
                    gr.Textbox(
                        f"Day Processing Speed: {day_ps}",
                        interactive=False,
                        lines=1,
                    )

                # Displaying month statistics
                if "month" in stats_info:
                    month_images = stats_info["month"]["images"]
                    month_ps = stats_info["month"]["ps"]
                    gr.Textbox(
                        f"Month Images: {month_images}",
                        interactive=False,
                        lines=1,
                    )
                    gr.Textbox(
                        f"Month Processing Speed: {month_ps}",
                        interactive=False,
                        lines=1,
                    )

                # Displaying total statistics
                if "total" in stats_info:
                    total_images = stats_info["total"]["images"]
                    total_ps = stats_info["total"]["ps"]
                    gr.Textbox(
                        f"Total Images: {total_images}",
                        interactive=False,
                        lines=1,
                    )
                    gr.Textbox(
                        f"Total Processing Speed: {total_ps}",
                        interactive=False,
                        lines=1,
                    )

    return stats_ui


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
                    worker_info = api.get_worker_info(
                        session, config.apikey, worker
                    )
                    worker_name = worker_info["name"]
                    if worker_name == config.name:
                        print(f"Current Worker: {worker_name}")
                        break

            # General tabs
            with gr.Tab("Generation"):
                generator_ui = get_generator_ui()
            with gr.Tab("Worker"):
                worker_ui = get_worker_ui(worker)
            with gr.Tab("User"):
                user_ui = get_user_ui()
            with gr.Tab("Kudos"):
                kudos_ui = get_kudos_ui()
            with gr.Tab("News"):
                news_ui = get_news_ui()
            with gr.Tab("Stats"):
                stats_ui = get_stats_ui()
            with gr.Tab("Settings"):
                settings_ui = get_settings_ui(status, running_type)

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
        save_apikey.click(fn=save_apikey_fn(apikey))
        toggle_running.click(fn=toggle_running_fn)
        worker_ui.worker_update.click(
            fn=api.get_worker_info(session, config.apikey, worker),
            inputs=[apikey],
            outputs=[worker_info],
        )
        user_ui.user_update.click(
            fn=api.get_user_info(session, config.apikey),
            outputs=[user_info],
        )
        kudos_ui.transfer.click(
            fn=api.transfer_kudos(kudos_ui.username, kudos_ui.kudos_amount)
        )
        news_ui.news_update.click(
            fn=api.get_horde_news(session), outputs=[news_ui.news_info]
        )
        stats_ui.stats_update.click(
            fn=api.get_horde_stats(session), outputs=[stats_ui.stats_info]
        )
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
            output=[
                status, running_type
            ],
        )

    return ((ui_tabs, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
