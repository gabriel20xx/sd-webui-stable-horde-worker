import gradio as gr
from script import fetch_api_info


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
        models = gr.Textbox(
            value=worker_info.get("models").strip("[]").replace("'", ""),
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
