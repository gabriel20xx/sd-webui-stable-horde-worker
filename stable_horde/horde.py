import asyncio
import json
import requests
from os import path
from typing import Any, Dict, List, Optional, Tuple
from re import sub
import time

import aiohttp
import numpy as np
from PIL import Image
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from .job import HordeJob
from .config import StableHordeConfig
from .api import API
# from .save_image import save_image
from modules.images import save_image
from modules import (
    shared, call_queue, processing, sd_models, sd_samplers, scripts
)


stable_horde_supported_models_url = (
    "https://raw.githubusercontent.com/Haidra-Org/"
    "AI-Horde-image-model-reference/main/stable_diffusion.json"
)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor: Optional[AutoFeatureExtractor] = None
safety_checker: Optional[StableDiffusionSafetyChecker] = None

api = API()
basedir = scripts.basedir()
config = StableHordeConfig(basedir)
session = requests.Session()


class State:
    def __init__(self):
        self._status = ""
        self.id: Optional[str] = None
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.scale: Optional[float] = None
        self.steps: Optional[int] = None
        self.sampler: Optional[str] = None
        self.image: Optional[Image.Image] = None

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str):
        self._status = value
        if shared.cmd_opts.nowebui:
            print(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "scale": self.scale,
            "steps": self.steps,
            "sampler": self.sampler,
        }


class StableHorde:
    def __init__(self, basedir: str, config: StableHordeConfig):
        self.basedir = basedir
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.sfw_request_censor = Image.open(
            path.join(self.config.basedir, "assets", "nsfw_censor_sfw_request.png")
        )
        self.supported_models: List[Dict[str, Any]] = []
        self.current_models: Dict[str, str] = {}
        self.state = State()

    async def get_supported_models(self):
        for attempt in range(10, 0, -1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(stable_horde_supported_models_url) as resp:
                        if resp.status != 200:
                            raise aiohttp.ClientError()
                        self.supported_models = list(
                            json.loads(await resp.text()).values()
                        )
                        return
            except Exception as e:
                print(
                    f"Failed to get supported models, retrying in 1 second... \
                        ({attempt} attempts left) Error: {e}"
                )
                await asyncio.sleep(1)
        raise Exception("Failed to get supported models after 10 attempts")

    def detect_current_model(self) -> Optional[str]:
        model_checkpoint = shared.opts.sd_model_checkpoint
        checkpoint_info = sd_models.checkpoints_list.get(model_checkpoint)
        if checkpoint_info is None:
            return f"Model checkpoint {model_checkpoint} not found"

        for model in self.supported_models:
            remote_hash = model.get("config", {}).get("files", [{}])[0].get("sha256sum")
            print(
                f"Local hash: {shared.opts.sd_checkpoint_hash}, remote: {remote_hash}"
            )
            if shared.opts.sd_checkpoint_hash == remote_hash:
                self.current_models[model["name"]] = checkpoint_info.name

        # Print all matching models
        print("Matching Models:")
        for model_name in self.current_models.keys():
            print(model_name)

        if not self.current_models:
            return f"Current model {model_checkpoint} not found on StableHorde"
        return None

    def set_current_models(self, model_names: List[str]) -> Dict[str, str]:
        """Set the current models in horde and config"""
        remote_hashes = {
            model["config"]["files"][0]["sha256sum"].lower(): model["name"]
            # get the sha256 of all supported models
            for model in self.supported_models
            if "sha256sum" in model["config"]["files"][0]
        }
        # get the sha256 of all local models and compare it to the remote hashes
        # if the sha256 matches, add the model to the current models list
        for checkpoint in sd_models.checkpoints_list.values():
            checkpoint: sd_models.CheckpointInfo
            if checkpoint.name in model_names:
                # skip sha256 calculation if the model already has hash
                local_hash = checkpoint.sha256 or sd_models.hashes.sha256(
                    checkpoint.filename, f"checkpoint/{checkpoint.name}"
                )
                if local_hash in remote_hashes:
                    self.current_models[remote_hashes[local_hash]] = checkpoint.name
                    print(
                        f"sha256 for {checkpoint.name} is {local_hash} \
                            and it's supported by StableHorde"
                    )
                else:
                    print(
                        f"sha256 for {checkpoint.name} is {local_hash} \
                            but it's not supported by StableHorde"
                    )

        self.config.current_models = self.current_models
        self.config.save()
        return self.current_models

    async def run(self):
        await self.get_supported_models()
        self.current_models = self.config.current_models
        print("-" * 64)
        print("Stable Horde Worker")
        print(f"Available Models: {list(sorted(self.current_models.keys()))}")

        user_info = api.request(session, "User", config.apikey)
        username = user_info["username"]
        id = user_info["id"]
        worker_ids = user_info["worker_ids"]

        print(f"Username: {username}")
        print(f"User ID: {id}")

        for worker in worker_ids:
            worker_info = api.request(session, "Worker", config.apikey, worker)
            worker_name = worker_info["name"]
            worker_id = worker_info["id"]
            models = worker_info["models"]
            maintenance_mode = worker_info["maintenance_mode"]

            if worker_name == self.config.name:
                print(f"Worker ID: {worker_id}")
                print(f"Worker Name: {worker_name}")
                print(f"Worker Models: {models}")
                print("-" * 64)
                break

        start_time = time.time()
        while True:
            if not self.current_models:
                self.state.status = self.detect_current_model()
                if self.state.status:
                    # Wait 10 seconds before retrying to detect the current model
                    # if the current model is not listed in the Stable Horde supported
                    # models, we don't want to spam the server with requests
                    await asyncio.sleep(10)
                    continue

            if maintenance_mode:
                self.state.status = "Worker is in maintenance mode"
                await asyncio.sleep(10)
                continue

            await asyncio.sleep(self.config.interval)
            if self.config.enabled and not maintenance_mode:
                self.state.status = "Waiting for a request"
                try:
                    # Require a queue lock to prevent getting jobs when
                    # there are generation jobs from webui.
                    with call_queue.queue_lock:
                        self.type = None
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(
                            f"Looking for requests... ({elapsed_time:.0f}s)",
                            end="\r",
                        )
                        req = await HordeJob.get_request(
                            await self.get_session(),
                            self.config,
                            self.type,
                            list(self.current_models.keys()),
                        )
                    if req:
                        await self.handle_request(req)
                        start_time = time.time()
                except Exception:
                    import traceback

                    traceback.print_exc()

    def patch_sampler_names(self):
        """Add more samplers that the Stable Horde supports,
        but are not included in the default sd_samplers module.
        """
        from modules import sd_samplers

        try:
            # Old versions of webui put every samplers in `modules.sd_samplers`
            # But the newer version split them into several files
            # Happened in https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/4df63d2d197f26181758b5108f003f225fe84874 # noqa E501
            from modules.sd_samplers import KDiffusionSampler, SamplerData
        except ImportError:
            from modules.sd_samplers_common import SamplerData
            from modules.sd_samplers_kdiffusion import KDiffusionSampler

        if "euler a karras" in sd_samplers.samplers_map:
            # already patched
            return

        samplers = [
            SamplerData(
                name,
                lambda model, fn=func: KDiffusionSampler(fn, model),
                [alias],
                {"scheduler": "karras"},
            )
            for name, func, alias in [
                ("Euler a Karras", "sample_euler_ancestral", "k_euler_a_ka"),
                ("Euler Karras", "sample_euler", "k_euler_ka"),
                ("LMS Karras", "sample_lms", "k_lms_ka"),
                ("Heun Karras", "sample_heun", "k_heun_ka"),
                ("DPM2 Karras", "sample_dpm_2", "k_dpm_2_ka"),
                ("DPM2 a Karras", "sample_dpm_2_ancestral", "k_dpm_2_a_ka"),
                ("DPM++ 2S a Karras", "sample_dpmpp_2s_ancestral", "k_dpmpp_2s_a_ka"),
                ("DPM++ 2M Karras", "sample_dpmpp_2m", "k_dpmpp_2m_ka"),
                ("DPM++ SDE Karras", "sample_dpmpp_sde", "k_dpmpp_sde_ka"),
                ("DPM fast Karras", "sample_dpm_fast", "k_dpm_fast_ka"),
                ("DPM adaptive Karras", "sample_dpm_adaptive", "k_dpm_ad_ka"),
            ]
        ]

        sd_samplers.samplers.extend(samplers)
        sd_samplers.samplers_for_img2img.extend(samplers)
        sd_samplers.all_samplers_map.update({s.name: s for s in samplers})
        for sampler in samplers:
            sd_samplers.samplers_map[sampler.name.lower()] = sampler.name
            for alias in sampler.aliases:
                sd_samplers.samplers_map[alias.lower()] = sampler.name

    async def handle_request(self, job: HordeJob):
        """
        Handle a popped generation request
        """
        # Patch sampler names
        self.patch_sampler_names()

        self.state.status = f"Get popped generation request {job.id}, \
            model {job.model}, sampler {job.sampler}"
        sampler_name = job.sampler if job.sampler != "k_dpm_adaptive" else "k_dpm_ad"

        if job.karras:
            sampler_name += "_ka"

        # Map model name to model
        local_model = self.current_models.get(job.model, shared.sd_model)
        local_model_shorthash = self._get_model_shorthash(local_model)

        if local_model_shorthash is None:
            raise Exception(f"ERROR: Unknown model {local_model}")

        sampler = sd_samplers.samplers_map.get(sampler_name, None)
        if sampler is None:
            raise Exception(f"ERROR: Unknown sampler {sampler_name}")

        postprocessors = job.postprocessors

        # Create params
        params = self._create_params(job, local_model, sampler)

        if job.source_image:
            # Save source image
            p = processing.StableDiffusionProcessingImg2Img(
                init_images=[job.source_image], mask=job.source_mask, **params
            )
            if self.config.save_source_images:
                save_image(
                    job.source_image,
                    self.config.save_images_folder,
                    "",
                    job.seed,
                    job.prompt,
                    "png",
                    p=p,
                    suffix="-Source",
                )

        else:
            p = processing.StableDiffusionProcessingTxt2Img(**params)

        # Hijack clip skip if needed
        with call_queue.queue_lock:
            shared.state.begin()

            hijacked, old_clip_skip = self._hijack_clip_skip(job.clip_skip)

            if hijacked:
                shared.opts.CLIP_stop_at_last_layers = old_clip_skip
            shared.state.end()

        with call_queue.queue_lock:
            # Generating infotext
            processed = processing.process_images(p)
            infotext = self._generate_infotext(p, job)
            image = processed.images[0]

            # Saving image locally
            if self.config.save_images:
                save_image(
                    image,
                    self.config.save_images_folder,
                    "",
                    job.seed,
                    job.prompt,
                    "png",
                    info=infotext,
                    p=p,
                )

            # Checking safety
            has_nsfw = False
            if job.nsfw_censor:
                x_image = np.array(image)
                image, has_nsfw = self.check_safety(x_image)
                if has_nsfw:
                    job.censored = True

            # Apply postprocessors
            if not has_nsfw:
                image = self._apply_postprocessors(image, postprocessors)
        self._update_state(job, sampler_name, image)

        # Submit image
        res = await job.submit(image)
        if res:
            self.state.status = f"Submission accepted, reward {res} received."
            print(f"Submission accepted, reward {res} received.")

    def _get_model_shorthash(self, local_model: str) -> Optional[str]:
        # Short hash for info text
        local_model_shorthash = None
        for checkpoint in sd_models.checkpoints_list.values():
            checkpoint: sd_models.CheckpointInfo
            if checkpoint.name == local_model:
                if not checkpoint.shorthash:
                    checkpoint.calculate_shorthash()
                local_model_shorthash = checkpoint.shorthash
                return local_model_shorthash

        return None

    def _create_params(
        self, job: HordeJob, local_model: str, sampler: str
    ) -> Dict[str, Any]:
        params = {
            "sd_model": local_model,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "sampler_name": sampler,
            "cfg_scale": job.cfg_scale,
            "seed": job.seed,
            "denoising_strength": job.denoising_strength,
            "height": job.height,
            "width": job.width,
            "subseed": job.subseed,
            "steps": job.steps,
            "tiling": job.tiling,
            "n_iter": job.n_iter,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "override_settings": {"sd_model_checkpoint": local_model},
            "enable_hr": job.hires_fix,
            "hr_upscaler": self.config.hr_upscaler,
            "override_settings_restore_afterwards": self.config.restore_settings,
        }

        if job.hires_fix:
            ar = job.width / job.height
            params["firstphase_width"] = min(
                self.config.hires_firstphase_resolution,
                int(self.config.hires_firstphase_resolution * ar),
            )
            params["firstphase_height"] = min(
                self.config.hires_firstphase_resolution,
                int(self.config.hires_firstphase_resolution / ar),
            )

        return params

    def _hijack_clip_skip(self, clip_skip: int) -> Tuple[bool, Optional[int]]:
        # hijack clip skip
        hijacked = False
        old_clip_skip = shared.opts.CLIP_stop_at_last_layers
        if clip_skip >= 1 and clip_skip != shared.opts.CLIP_stop_at_last_layers:
            shared.opts.CLIP_stop_at_last_layers = clip_skip
            hijacked = True
        return hijacked, old_clip_skip

    def _generate_infotext(self, p: Any, job: HordeJob) -> Optional[str]:
        if shared.opts.enable_pnginfo:
            infotext = processing.create_infotext(
                p,
                p.all_prompts,
                p.all_seeds,
                p.all_subseeds,
                "Stable Horde",
                0,
                0,
            )
            # workaround for model name and hash since webui
            # uses shard.sd_model instead of local_model
            local_model = self.current_models.get(job.model, shared.sd_model)
            local_model_shorthash = self._get_model_shorthash(local_model)

            infotext = sub(
                "Model:(.*?),",
                "Model: " + local_model.split(".")[0] + ",",
                infotext,
            )
            infotext = sub(
                "Model hash:(.*?),",
                "Model hash: " + local_model_shorthash + ",",
                infotext,
            )

            return infotext

        return None

    def _apply_postprocessors(
        self, image: Image.Image, postprocessors: List[str]
    ) -> Image.Image:
        if "GFPGAN" in postprocessors or "CodeFormers" in postprocessors:
            model = "CodeFormer" if "CodeFormers" in postprocessors else "GFPGAN"
            face_restorators = [x for x in shared.face_restorers if x.name() == model]
            if face_restorators:
                with call_queue.queue_lock:
                    image = face_restorators[0].restore(np.array(image))
                image = Image.fromarray(image)
            else:
                print(f"ERROR: No face restorer for {model}")

        if "RealESRGAN_x4plus" in postprocessors:
            from modules.postprocessing import run_extras

            with call_queue.queue_lock:
                images, _info, _wtf = run_extras(
                    image=image,
                    extras_mode=0,
                    resize_mode=0,
                    show_extras_results=True,
                    upscaling_resize=2,
                    upscaling_resize_h=None,
                    upscaling_resize_w=None,
                    upscaling_crop=False,
                    upscale_first=False,
                    extras_upscaler_1="R-ESRGAN 4x+",  # 8 - RealESRGAN_x4plus
                    extras_upscaler_2=None,
                    extras_upscaler_2_visibility=0.0,
                    gfpgan_visibility=0.0,
                    codeformer_visibility=0.0,
                    codeformer_weight=0.0,
                    image_folder="",
                    input_dir="",
                    output_dir="",
                    save_output=False,
                )
            image = images[0]
        return image

    def _update_state(self, job: HordeJob, sampler_name: str, image: Image.Image):
        self.state.id = job.id
        self.state.prompt = job.prompt
        self.state.negative_prompt = job.negative_prompt
        self.state.scale = job.cfg_scale
        self.state.steps = job.steps
        self.state.sampler = sampler_name
        self.state.image = image

    # check and replace nsfw content
    def check_safety(self, x_image: np.ndarray) -> Tuple[Image.Image, bool]:
        global safety_feature_extractor, safety_checker

        if safety_feature_extractor is None:
            safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
                safety_model_id
            )
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                safety_model_id
            )

        safety_checker_input = safety_feature_extractor(x_image, return_tensors="pt")
        image, has_nsfw_concept = safety_checker(
            images=x_image, clip_input=safety_checker_input.pixel_values
        )

        if has_nsfw_concept and any(has_nsfw_concept):
            return self.sfw_request_censor, True
        return Image.fromarray(image), False

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            headers = {
                "apikey": self.config.apikey,
                "Content-Type": "application/json",
            }
            self.session = aiohttp.ClientSession(self.config.endpoint, headers=headers)
        # check if apikey has changed
        elif self.session.headers["apikey"] != self.config.apikey:
            await self.session.close()
            self.session = None
            self.session = await self.get_session()
        return self.session

    def handle_error(self, status: int, res: Dict[str, Any]):
        if status == 401:
            self.state.status = "ERROR: Invalid API Key"
        elif status == 403:
            self.state.status = f"ERROR: Access Denied. ({res.get('message', '')})"
        elif status == 404:
            self.state.status = "ERROR: Request Not Found"
        else:
            self.state.status = f"ERROR: Unknown Error {status}"
            print(f"ERROR: Unknown Error, {res}")
