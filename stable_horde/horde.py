import asyncio
import json
from os import path
from typing import Any, Dict, Optional
from re import sub

import aiohttp
import numpy as np
from PIL import Image
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)

from .job import HordeJob
from .config import StableHordeConfig
from modules.images import save_image
from modules import shared, call_queue, processing, sd_models, sd_samplers

stable_horde_supported_models_url = (
    "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/stable_diffusion.json"
)
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None


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
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        if shared.cmd_opts.nowebui:
            print(value)

    def to_dict(self):
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
        self.sfw_request_censor = Image.open(path.join(
            self.config.basedir, "assets", "nsfw_censor_sfw_request.png"
        ))
        self.supported_models = []
        self.current_models = {}
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
            except Exception:
                print(f"Failed to get supported models, retrying in 1 second... "
                      f"({attempt} attempts left)")
                await asyncio.sleep(1)
        raise Exception("Failed to get supported models after 10 attempts")

    def detect_current_model(self):
        model_checkpoint = shared.opts.sd_model_checkpoint
        checkpoint_info = sd_models.checkpoints_list.get(model_checkpoint)
        if checkpoint_info is None:
            return f"Model checkpoint {model_checkpoint} not found"

        for model in self.supported_models:
            remote_hash = model.get("config", {}).get("files", [{}])[0].get("sha256sum")
            if shared.opts.sd_checkpoint_hash == remote_hash:
                self.current_models[model["name"]] = checkpoint_info.name

        if not self.current_models:
            return f"Current model {model_checkpoint} not found on StableHorde"

    def set_current_models(self, model_names: list):
        remote_hashes = {
            model["config"]["files"][0]["sha256sum"].lower(): model["name"]
            for model in self.supported_models if "sha256sum" in (
                model["config"]["files"][0]
            )
        }
        for checkpoint in sd_models.checkpoints_list.values():
            if checkpoint.name in model_names:
                local_hash = checkpoint.sha256 or sd_models.hashes.sha256(
                    checkpoint.filename, f"checkpoint/{checkpoint.name}"
                )
                if local_hash in remote_hashes:
                    self.current_models[remote_hashes[local_hash]] = checkpoint.name
                    print(f"sha256 for {checkpoint.name} is {local_hash} and"
                          f"it's supported by StableHorde")
                else:
                    print(f"sha256 for {checkpoint.name} is {local_hash} but"
                          f"it's not supported by StableHorde")

        self.config.current_models = self.current_models
        self.config.save()
        return self.current_models

    async def run(self):
        await self.get_supported_models()
        self.current_models = self.config.current_models

        while True:
            if not self.current_models:
                self.state.status = self.detect_current_model()
                if self.state.status:
                    await asyncio.sleep(10)
                    continue

            await asyncio.sleep(self.config.interval)

            if self.config.enabled:
                try:
                    with call_queue.queue_lock:
                        req = await HordeJob.get(
                            await self.get_session(), self.config, list(
                                self.current_models.keys()
                            )
                        )
                    if req:
                        await self.handle_request(req)
                except Exception as e:
                    print(f"Error handling request: {e}")

    def patch_sampler_names(self):
        try:
            from modules.sd_samplers import KDiffusionSampler, SamplerData
        except ImportError:
            from modules.sd_samplers_kdiffusion import KDiffusionSampler
            from modules.sd_samplers_common import SamplerData

        if "euler a karras" in sd_samplers.samplers_map:
            return

        samplers = [
            SamplerData(name, lambda model, fn=func: KDiffusionSampler(
                fn, model
            ), [alias], {"scheduler": "karras"})
            for name, func, alias in [
                ("Euler a Karras", "sample_euler_ancestral", "k_euler_a_ka"),
                ("Euler Karras", "sample_euler", "k_euler_ka"),
                ("Heun Karras", "sample_heun", "k_heun_ka"),
                ("DPM adaptive Karras", "sample_dpm_adaptive", "k_dpm_ad_ka"),
                ("DPM fast Karras", "sample_dpm_fast", "k_dpm_fast_ka"),
                ("LMS Karras", "sample_lms", "k_lms_ka"),
                ("DPM++ SDE Karras", "sample_dpmpp_sde", "k_dpmpp_sde_ka"),
                ("DPM++ 2S a Karras", "sample_dpmpp_2s_ancestral", "k_dpmpp_2s_a_ka"),
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
        self.patch_sampler_names()
        self.state.status = f"Get popped generation request {job.id}, " + \
                            f"model {job.model}, sampler {job.sampler}"
        sampler_name = job.sampler if job.sampler != "k_dpm_adaptive" else "k_dpm_ad"
        if job.karras:
            sampler_name += "_ka"

        local_model = self.current_models.get(job.model, shared.sd_model)
        local_model_shorthash = None
        for checkpoint in sd_models.checkpoints_list.values():
            if checkpoint.name == local_model:
                local_model_shorthash = (
                    checkpoint.shorthash or checkpoint.calculate_shorthash()
                )
                break
        if not local_model_shorthash:
            raise Exception(f"ERROR: Unknown model {local_model}")

        sampler = sd_samplers.samplers_map.get(sampler_name)
        if not sampler:
            raise Exception(f"ERROR: Unknown sampler {sampler_name}")
        
        postprocessors = job.postprocessors

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
            params["firstphase_width"] = (
                min(self.config.hires_firstphase_resolution,
                    int(self.config.hires_firstphase_resolution * ar))
            )
            params["firstphase_height"] = (
                min(self.config.hires_firstphase_resolution, 
                    int(self.config.hires_firstphase_resolution / ar))
            )

        if job.source_image:
            p = processing.StableDiffusionProcessingImg2Img(
                init_images=[job.source_image], mask=job.source_mask, **params
            )
        else:
            p = processing.StableDiffusionProcessingTxt2Img(**params)

        with call_queue.queue_lock:
            shared.state.begin()
            # hijack clip skip
            hijacked = False
            old_clip_skip = shared.opts.CLIP_stop_at_last_layers
            if (
                job.clip_skip >= 1
                and job.clip_skip != shared.opts.CLIP_stop_at_last_layers
            ):
                shared.opts.CLIP_stop_at_last_layers = job.clip_skip
                hijacked = True
            processed = processing.process_images(p)

            if hijacked:
                shared.opts.CLIP_stop_at_last_layers = old_clip_skip
            shared.state.end()

        has_nsfw = False

        with call_queue.queue_lock:
            # Saving image locally
            infotext = (
                processing.create_infotext(
                    p,
                    p.all_prompts,
                    p.all_seeds,
                    p.all_subseeds,
                    "Stable Horde",
                    0,
                    0,
                )
                if shared.opts.enable_pnginfo
                else None
            )
            # workaround for model name and hash since webui
            # uses shard.sd_model instead of local_model
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

            if self.config.save_images:
                image = processed.images[0]
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

            if job.nsfw_censor:
                x_image = np.array(processed.images[0])
                image, has_nsfw = self.check_safety(x_image)
                if has_nsfw:
                    job.censored = True

            else:
                image = processed.images[0]

            if not has_nsfw and (
                "GFPGAN" in postprocessors or "CodeFormers" in postprocessors
            ):
                model = "CodeFormer" if "CodeFormers" in postprocessors else "GFPGAN"
                face_restorators = (
                    [x for x in shared.face_restorers if x.name() == model]
                )
                if len(face_restorators) == 0:
                    print(f"ERROR: No face restorer for {model}")

                else:
                    with call_queue.queue_lock:
                        image = face_restorators[0].restore(np.array(image))
                    image = Image.fromarray(image)

            if "RealESRGAN_x4plus" in postprocessors and not has_nsfw:
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

        self.state.id = job.id
        self.state.prompt = job.prompt
        self.state.negative_prompt = job.negative_prompt
        self.state.scale = job.cfg_scale
        self.state.steps = job.steps
        self.state.sampler = sampler_name
        self.state.image = image

        res = await job.submit(image)
        if res:
            self.state.status = f"Submission accepted, reward {res} received."

    # check and replace nsfw content
    def check_safety(self, x_image):
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
            return self.sfw_request_censor, has_nsfw_concept
        return Image.fromarray(image), has_nsfw_concept

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
