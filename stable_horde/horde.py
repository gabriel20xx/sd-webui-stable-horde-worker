import asyncio
import json
from os import path
from typing import Any, Dict, List, Optional, Tuple
from re import sub

import aiohttp
import numpy as np
from PIL import Image
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from .job import HordeJob
from .config import StableHordeConfig
from modules.images import save_image
from modules import shared, call_queue, processing, sd_models, sd_samplers


stable_horde_supported_models_url = (
    "https://raw.githubusercontent.com/Haidra-Org/"
    "AI-Horde-image-model-reference/main/stable_diffusion.json"
)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor: Optional[AutoFeatureExtractor] = None
safety_checker: Optional[StableDiffusionSafetyChecker] = None


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
        self.sfw_request_censor = Image.open(path.join(
            self.config.basedir, "assets", "nsfw_censor_sfw_request.png"
        ))
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
                        self.supported_models = list(json.loads(await resp.text()).values())
                        return
            except Exception as e:
                print(f"Failed to get supported models, retrying in 1 second... ({attempt} attempts left) Error: {e}")
                await asyncio.sleep(1)
        raise Exception("Failed to get supported models after 10 attempts")

    def detect_current_model(self) -> Optional[str]:
        model_checkpoint = shared.opts.sd_model_checkpoint
        checkpoint_info = sd_models.checkpoints_list.get(model_checkpoint)
        if checkpoint_info is None:
            print(f"Model checkpoint {model_checkpoint} not found")
            return f"Model checkpoint {model_checkpoint} not found"

        for model in self.supported_models:
            remote_hash = model.get("config", {}).get("files", [{}])[0].get("sha256sum")
            if shared.opts.sd_checkpoint_hash == remote_hash:
                self.current_models[model["name"]] = checkpoint_info.name

        if not self.current_models:
            print(f"Current model {model_checkpoint} not found on StableHorde")
            return f"Current model {model_checkpoint} not found on StableHorde"
        return None

    def set_current_models(self, model_names: List[str]) -> Dict[str, str]:
        remote_hashes = {
            model["config"]["files"][0]["sha256sum"].lower(): model["name"]
            for model in self.supported_models if "sha256sum" in model["config"]["files"][0]
        }
        for checkpoint in sd_models.checkpoints_list.values():
            checkpoint: sd_models.CheckpointInfo
            if checkpoint.name in model_names:
                local_hash = checkpoint.sha256 or sd_models.hashes.sha256(checkpoint.filename, f"checkpoint/{checkpoint.name}")
                if local_hash in remote_hashes:
                    self.current_models[remote_hashes[local_hash]] = checkpoint.name
                    print(f"sha256 for {checkpoint.name} is {local_hash} and it's supported by StableHorde")
                else:
                    print(f"sha256 for {checkpoint.name} is {local_hash} but it's not supported by StableHorde")

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
                        req = await HordeJob.get(await self.get_session(), self.config, list(self.current_models.keys()))
                    if req:
                        await self.handle_request(req)
                except Exception:
                    import traceback

                    traceback.print_exc()

    def patch_sampler_names(self):
        try:
            from modules.sd_samplers import KDiffusionSampler, SamplerData
        except ImportError:
            from modules.sd_samplers_kdiffusion import KDiffusionSampler
            from modules.sd_samplers_common import SamplerData

        if "euler a karras" in sd_samplers.samplers_map:
            return

        samplers = [
            SamplerData(name, lambda model, fn=func: KDiffusionSampler(fn, model), [alias], {"scheduler": "karras"})
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
        try:
            self.patch_sampler_names()
        except Exception as e:
            print(f"Error: patch_sampler_names {e}")
        self.state.status = f"Get popped generation request {job.id}, model {job.model}, sampler {job.sampler}"
        sampler_name = job.sampler if job.sampler != "k_dpm_adaptive" else "k_dpm_ad"
        if job.karras:
            sampler_name += "_ka"
        local_model = self.current_models.get(job.model, shared.sd_model)
        try:
            local_model_shorthash = self._get_model_shorthash(local_model)
            print(f"Local model shorthash 1: {local_model_shorthash}")
        except Exception as e:
            print(f"Error: _get_model_shorthash {e}")
        if local_model_shorthash is None:
            raise Exception(f"ERROR: Unknown model {local_model}")
        sampler = sd_samplers.samplers_map.get(sampler_name)
        if sampler is None:
            raise Exception(f"ERROR: Unknown sampler {sampler_name}")
        postprocessors = job.postprocessors
        try:
            params = self._create_params(job, local_model, sampler)
        except Exception as e:
            print(f"Error: _create_params {e}")
        if job.source_image:
            p = processing.StableDiffusionProcessingImg2Img(init_images=[job.source_image], mask=job.source_mask, **params)
        else:
            p = processing.StableDiffusionProcessingTxt2Img(**params)
        with call_queue.queue_lock:
            shared.state.begin()
            try:
                hijacked, old_clip_skip = self._hijack_clip_skip(job.clip_skip)
            except Exception as e:
                print(f"Error: _hijack_clip_skip {e}")
            processed = processing.process_images(p)
            if hijacked:
                shared.opts.CLIP_stop_at_last_layers = old_clip_skip
            shared.state.end()

        with call_queue.queue_lock:
            try:
                image = self._handle_postprocessing(processed, job, postprocessors)
            except Exception as e:
                print(f"Error: _handle_postprocessing {e}")
        try:
            self._update_state(job, sampler_name, image)
        except Exception as e:
            print(f"Error: _update_state {e}")
        res = await job.submit(image)
        if res:
            self.state.status = f"Submission accepted, reward {res} received."

    def _get_model_shorthash(self, local_model: str) -> Optional[str]:
        print("Step U")
        local_model_shorthash = None
        for checkpoint in sd_models.checkpoints_list.values():
            checkpoint: sd_models.CheckpointInfo
            if checkpoint.name == local_model:
                print("Step V")
                if not checkpoint.shorthash:
                    checkpoint.calculate_shorthash()
                local_model_shorthash = checkpoint.shorthash
                return local_model_shorthash
        print("Step Z")
        return None

    def _create_params(self, job: HordeJob, local_model: str, sampler: str) -> Dict[str, Any]:
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
            params["firstphase_width"] = min(self.config.hires_firstphase_resolution, int(self.config.hires_firstphase_resolution * ar))
            params["firstphase_height"] = min(self.config.hires_firstphase_resolution, int(self.config.hires_firstphase_resolution / ar))

        return params

    def _hijack_clip_skip(self, clip_skip: int) -> Tuple[bool, Optional[int]]:
        hijacked = False
        old_clip_skip = shared.opts.CLIP_stop_at_last_layers
        if clip_skip >= 1 and clip_skip != shared.opts.CLIP_stop_at_last_layers:
            shared.opts.CLIP_stop_at_last_layers = clip_skip
            hijacked = True
        return hijacked, old_clip_skip

    def _handle_postprocessing(self, processed: Any, job: HordeJob, postprocessors: List[str]) -> Image.Image:
        has_nsfw = False
        print("Step 1")
        try:
            infotext = self._generate_infotext(processed, job)
        except Exception as e:
            print(f"Error: _generate_infotext {e}")
        print("Step 2")
        if self.config.save_images:
            image = processed.images[0]
            print("Step 2.1")
            save_image(image, self.config.save_images_folder, "", job.seed, job.prompt, "png", info=infotext, p=processed)
        print("Step 3")
        if job.nsfw_censor:
            print("Step 3.1")
            x_image = np.array(processed.images[0])
            print("Step 3.2")
            try:
                image, has_nsfw = self.check_safety(x_image)
            except Exception as e:
                print(f"Error: check_safety {e}")
            print("Step 3.3")
            if has_nsfw:
                job.censored = True
        else:
            print("Step 3.B")
            image = processed.images[0]
        print("Step 4")
        if not has_nsfw:
            print("Step 4.1")
            try:
                image = self._apply_postprocessors(image, postprocessors)
                print("Step 4.2")
            except Exception as e:
                print(f"Error: _apply_postprocessors {e}")
        print("Step 5")
        return image

    def _generate_infotext(self, processed: Any, job: HordeJob) -> Optional[str]:
        print("Step A")
        if shared.opts.enable_pnginfo:
            try:
                print("Step B")
                # Debugging: Check attributes of the processed object
                print(f"Attributes of processed: {dir(processed)}")

                # Ensure required attributes are present before calling create_infotext
                required_attrs = ['all_prompts', 'all_seeds', 'all_subseeds', 'scheduler']
                for attr in required_attrs:
                    if not hasattr(processed, attr):
                        # Provide a default value for scheduler if it is missing
                        if attr == 'scheduler':
                            setattr(processed, attr, 'default_scheduler') # TODO: Replace with a real scheduler
                        else:
                            raise AttributeError(f"Processed object missing required attribute: {attr}")

                infotext = processing.create_infotext(
                    processed, processed.all_prompts, processed.all_seeds, processed.all_subseeds, "Stable Horde", 0, 0)
                print("Step C")

                local_model = self.current_models.get(job.model, shared.sd_model)
                print("Step D")
                try:
                    local_model_shorthash = self._get_model_shorthash(local_model)
                    print(f"Local model shorthash 2: {local_model_shorthash}")
                except Exception as e:
                    print(f"Error: _get_model_shorthash {e}")
                print("Step E")

                infotext = sub("Model:(.*?),", "Model: " + local_model.split(".")[0] + ",", infotext)
                print("Step F")
                infotext = sub("Model hash:(.*?),", "Model hash: " + local_model_shorthash + ",", infotext)
                print("Step G")
                return infotext
            except AttributeError as e:
                print(f"Error generating infotext: {e}")
                return None
        return None

    def _apply_postprocessors(self, image: Image.Image, postprocessors: List[str]) -> Image.Image:
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
                    extras_upscaler_1="R-ESRGAN 4x+",
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
            safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        safety_checker_input = safety_feature_extractor(x_image, return_tensors="pt")
        image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

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
