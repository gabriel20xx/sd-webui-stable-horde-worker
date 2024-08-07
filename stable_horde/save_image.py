import os
import re
import pytz
import string
import datetime
import functools
import hashlib
from modules import script_callbacks, shared, sd_samplers, errors
from modules.shared import opts


class FilenameGenerator:
    replacements = {
        "basename": lambda self: self.basename or "img",
        "seed": lambda self: self.seed if self.seed is not None else "",
        "seed_first": lambda self: (
            self.seed if self.p.batch_size == 1 else self.p.all_seeds[0]
        ),
        "seed_last": lambda self: (
            NOTHING_AND_SKIP_PREVIOUS_TEXT
            if self.p.batch_size == 1
            else self.p.all_seeds[-1]
        ),
        "steps": lambda self: self.p and self.p.steps,
        "cfg": lambda self: self.p and self.p.cfg_scale,
        "width": lambda self: self.image.width,
        "height": lambda self: self.image.height,
        "styles": lambda self: self.p
        and sanitize_filename_part(
            ", ".join([style for style in self.p.styles if not style == "None"])
            or "None",
            replace_spaces=False,
        ),
        "sampler": lambda self: self.p
        and sanitize_filename_part(self.p.sampler_name, replace_spaces=False),
        "sampler_scheduler": lambda self: self.p
        and get_sampler_scheduler(self.p, True),
        "scheduler": lambda self: self.p and get_sampler_scheduler(self.p, False),
        "model_hash": lambda self: getattr(
            self.p, "sd_model_hash", shared.sd_model.sd_model_hash
        ),
        "model_name": lambda self: sanitize_filename_part(
            shared.sd_model.sd_checkpoint_info.name_for_extra, replace_spaces=False
        ),
        "date": lambda self: datetime.datetime.now().strftime("%Y-%m-%d"),
        "datetime": lambda self, *args: self.datetime(
            *args
        ),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format>
        # <Time Zone>]
        "job_timestamp": lambda self: getattr(
            self.p, "job_timestamp", shared.state.job_timestamp
        ),
        "prompt_hash": lambda self, *args: self.string_hash(self.prompt, *args),
        "negative_prompt_hash": lambda self, *args: self.string_hash(
            self.p.negative_prompt, *args
        ),
        "full_prompt_hash": lambda self, *args: self.string_hash(
            f"{self.p.prompt} {self.p.negative_prompt}", *args
        ),  # a space in between to create a unique string
        "prompt": lambda self: sanitize_filename_part(self.prompt),
        "prompt_no_styles": lambda self: self.prompt_no_style(),
        "prompt_spaces": lambda self: sanitize_filename_part(
            self.prompt, replace_spaces=False
        ),
        "prompt_words": lambda self: self.prompt_words(),
        "batch_number": lambda self: (
            NOTHING_AND_SKIP_PREVIOUS_TEXT
            if self.p.batch_size == 1 or self.zip
            else self.p.batch_index + 1
        ),
        "batch_size": lambda self: self.p.batch_size,
        "generation_number": lambda self: (
            NOTHING_AND_SKIP_PREVIOUS_TEXT
            if (self.p.n_iter == 1 and self.p.batch_size == 1) or self.zip
            else self.p.iteration * self.p.batch_size + self.p.batch_index + 1
        ),
        "hasprompt": lambda self, *args: self.hasprompt(
            *args
        ),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        "clip_skip": lambda self: opts.data["CLIP_stop_at_last_layers"],
        "denoising": lambda self: (
            self.p.denoising_strength
            if self.p and self.p.denoising_strength
            else NOTHING_AND_SKIP_PREVIOUS_TEXT
        ),
        "user": lambda self: self.p.user,
        "vae_filename": lambda self: self.get_vae_filename(),
        "none": lambda self: "",  # Overrides the default, so you can get just the
        # sequence number
        "image_hash": lambda self, *args: self.image_hash(
            *args
        ),  # accepts formats: [image_hash<length>] default full hash
    }
    default_time_format = "%Y%m%d%H%M%S"

    def __init__(self, p, seed, prompt, image, zip=False, basename=""):
        self.p = p
        self.seed = seed
        self.prompt = prompt
        self.image = image
        self.zip = zip
        self.basename = basename

    def get_vae_filename(self):
        """Get the name of the VAE file."""

        import modules.sd_vae as sd_vae

        if sd_vae.loaded_vae_file is None:
            return "NoneType"

        file_name = os.path.basename(sd_vae.loaded_vae_file)
        split_file_name = file_name.split(".")
        if len(split_file_name) > 1 and split_file_name[0] == "":
            return split_file_name[1]  # if the first character of the filename is "."
        # then [1] is obtained.
        else:
            return split_file_name[0]

    def hasprompt(self, *args):
        lower = self.prompt.lower()
        if self.p is None or self.prompt is None:
            return None
        outres = ""
        for arg in args:
            if arg != "":
                division = arg.split("|")
                expected = division[0].lower()
                default = division[1] if len(division) > 1 else ""
                if lower.find(expected) >= 0:
                    outres = f"{outres}{expected}"
                else:
                    outres = outres if default == "" else f"{outres}{default}"
        return sanitize_filename_part(outres)

    def prompt_no_style(self):
        if self.p is None or self.prompt is None:
            return None

        prompt_no_style = self.prompt
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if style:
                for part in style.split("{prompt}"):
                    prompt_no_style = (
                        prompt_no_style.replace(part, "")
                        .replace(", ,", ",")
                        .strip()
                        .strip(",")
                    )

                prompt_no_style = (
                    prompt_no_style.replace(style, "").strip().strip(",").strip()
                )

        return sanitize_filename_part(prompt_no_style, replace_spaces=False)

    def prompt_words(self):
        words = [x for x in re_nonletters.split(self.prompt or "") if x]
        if len(words) == 0:
            words = ["empty"]
        return sanitize_filename_part(
            " ".join(words[0: opts.directories_max_prompt_words]), replace_spaces=False
        )

    def datetime(self, *args):
        time_datetime = datetime.datetime.now()

        time_format = args[0] if (args and args[0] != "") else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError:
            time_zone = None

        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError):
            formatted_time = time_zone_time.strftime(self.default_time_format)

        return sanitize_filename_part(formatted_time, replace_spaces=False)

    def image_hash(self, *args):
        length = int(args[0]) if (args and args[0] != "") else None
        return hashlib.sha256(self.image.tobytes()).hexdigest()[0:length]

    def string_hash(self, text, *args):
        length = int(args[0]) if (args and args[0] != "") else 8
        return hashlib.sha256(text.encode()).hexdigest()[0:length]

    def apply(self, x):
        res = ""

        for m in re_pattern.finditer(x):
            text, pattern = m.groups()

            if pattern is None:
                res += text
                continue

            pattern_args = []
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break

                pattern, arg = m.groups()
                pattern_args.insert(0, arg)

            fun = self.replacements.get(pattern.lower())
            if fun is not None:
                try:
                    replacement = fun(self, *pattern_args)
                except Exception:
                    replacement = None
                    errors.report(
                        f"Error adding [{pattern}] to filename", exc_info=True
                    )

                if replacement == NOTHING_AND_SKIP_PREVIOUS_TEXT:
                    continue
                elif replacement is not None:
                    res += text + str(replacement)
                    continue

            res += f"{text}[{pattern}]"

        return res


def save_image(
    image,
    path,
    basename,
    seed=None,
    prompt=None,
    extension="png",
    info=None,
    short_filename=False,
    no_prompt=False,
    grid=False,
    pnginfo_section_name="parameters",
    p=None,
    existing_info=None,
    forced_filename=None,
    suffix="",
    save_to_dirs=None,
):
    """Save an image.

    Args:
        image (`PIL.Image`):
            The image to be saved.
        path (`str`):
            The directory to save the image. Note, the option `save_to_dirs` will make
            the image to be saved into a sub directory.
        basename (`str`):
            The base filename which will be applied to `filename pattern`.
        seed, prompt, short_filename,
        extension (`str`):
            Image file extension, default is `png`.
        pngsectionname (`str`):
            Specify the name of the section which `info` will be saved in.
        info (`str` or `PngImagePlugin.iTXt`):
            PNG info chunks.
        existing_info (`dict`):
            Additional PNG info. `existing_info == {pngsectionname: info, ...}`
        no_prompt:
            TODO I don't know its meaning.
        p (`StableDiffusionProcessing`)
        forced_filename (`str`):
            If specified, `basename` and filename pattern will be ignored.
        save_to_dirs (bool):
            If true, the image will be saved into a subdirectory of `path`.

    Returns: (fullfn, txt_fullfn)
        fullfn (`str`):
            The full path of the saved imaged.
        txt_fullfn (`str` or None):
            If a text file is saved for this image, this will be its full path.
            Otherwise None.
    """
    namegen = FilenameGenerator(p, seed, prompt, image, basename=basename)

    # WebP and JPG formats have maximum dimension limits of 16383 and 65535
    # respectively. switch to PNG which has a much higher limit
    if (
        (image.height > 65535 or image.width > 65535)
        and extension.lower() in ("jpg", "jpeg")
        or (image.height > 16383 or image.width > 16383)
        and extension.lower() == "webp"
    ):
        print("Image dimensions too large; saving as PNG")
        extension = "png"

    if save_to_dirs is None:
        save_to_dirs = (grid and opts.grid_save_to_dirs) or (
            not grid and opts.save_to_dirs and not no_prompt
        )

    if save_to_dirs:
        dirname = (
            namegen.apply(opts.directories_filename_pattern or "[prompt_words]")
            .lstrip(" ")
            .rstrip("\\ /")
        )
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if forced_filename is None:
        if short_filename or seed is None:
            file_decoration = ""
        elif opts.save_to_dirs:
            file_decoration = opts.samples_filename_pattern or "[seed]"
        else:
            file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

        file_decoration = namegen.apply(file_decoration) + suffix

        add_number = opts.save_images_add_number or file_decoration == ""

        if file_decoration != "" and add_number:
            file_decoration = f"-{file_decoration}"

        if add_number:
            basecount = get_next_sequence_number(path, basename)
            fullfn = None
            for i in range(500):
                fn = (
                    f"{basecount + i:05}"
                    if basename == ""
                    else f"{basename}-{basecount + i:04}"
                )
                fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                if not os.path.exists(fullfn):
                    break
        else:
            fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    else:
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")

    pnginfo = existing_info or {}
    if info is not None:
        pnginfo[pnginfo_section_name] = info

    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    script_callbacks.before_image_saved_callback(params)

    image = params.image
    fullfn = params.filename
    info = params.pnginfo.get(pnginfo_section_name, None)


def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the
    specified directory.

    The sequence starts at 0.
    """
    result = -1
    if basename != "":
        basename = f"{basename}-"

    prefix_length = len(basename)
    for p in os.listdir(path):
        if p.startswith(basename):
            parts = os.path.splitext(p[prefix_length:])[0].split("-")  # splits the
            # filename (removing the basename first if one is defined, so the sequence
            # number is always the first element)
            try:
                result = max(int(parts[0]), result)
            except ValueError:
                pass

    return result + 1


def sanitize_filename_part(text, replace_spaces=True):
    if text is None:
        return None

    if replace_spaces:
        text = text.replace(" ", "_")

    text = text.translate({ord(x): "_" for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text


@functools.cache
def get_scheduler_str(sampler_name, scheduler_name):
    """Returns {Scheduler} if the scheduler is applicable to the sampler"""
    if scheduler_name == "Automatic":
        config = sd_samplers.find_sampler_config(sampler_name)
        scheduler_name = config.options.get("scheduler", "Automatic")
    return scheduler_name.capitalize()


@functools.cache
def get_sampler_scheduler_str(sampler_name, scheduler_name):
    """Returns the '{Sampler} {Scheduler}' if the scheduler is applicable to the
    sampler"""
    return f"{sampler_name} {get_scheduler_str(sampler_name, scheduler_name)}"


def get_sampler_scheduler(p, sampler):
    """Returns '{Sampler} {Scheduler}' / '{Scheduler}' /
    'NOTHING_AND_SKIP_PREVIOUS_TEXT'"""
    if hasattr(p, "scheduler") and hasattr(p, "sampler_name"):
        if sampler:
            sampler_scheduler = get_sampler_scheduler_str(p.sampler_name, p.scheduler)
        else:
            sampler_scheduler = get_scheduler_str(p.sampler_name, p.scheduler)
        return sanitize_filename_part(sampler_scheduler, replace_spaces=False)
    return NOTHING_AND_SKIP_PREVIOUS_TEXT


if not shared.cmd_opts.unix_filenames_sanitization:
    invalid_filename_chars = '#<>:"/\\|?*\n\r\t'
else:
    invalid_filename_chars = "/"
invalid_filename_prefix = " "
invalid_filename_postfix = " ."
re_nonletters = re.compile(r"[\s" + string.punctuation + "]+")
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
max_filename_part_length = shared.cmd_opts.filenames_max_length
NOTHING_AND_SKIP_PREVIOUS_TEXT = object()
