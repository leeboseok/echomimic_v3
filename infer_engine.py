import os
import math
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from moviepy import VideoFileClip, AudioFileClip
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler

from src.dist import set_multi_gpus_devices
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs, get_image_to_video_latent3, save_videos_grid
from src.face_detect import get_mask_coord

from infer_preview import Config, load_wav2vec_models, extract_audio_features, get_sample_size, get_ip_mask


class EchoMimicEngine:
    def __init__(self, anchor_image, prompt, config=None):
        self.config = config or Config()
        self.prompt = prompt
        self.anchor_image = anchor_image

        # GPU setup
        self.device = set_multi_gpus_devices(self.config.ulysses_degree, self.config.ring_degree)
        cfg = OmegaConf.load(self.config.config_path)

        # Load models
        transformer = WanTransformerAudioMask3DModel.from_pretrained(
            os.path.join(self.config.model_name, cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
            torch_dtype=self.config.weight_dtype,
        )
        state_dict = load_file(self.config.transformer_path)
        transformer.load_state_dict(state_dict, strict=False)

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
        ).to(self.config.weight_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(self.config.model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
            torch_dtype=self.config.weight_dtype,
        ).eval()

        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(self.config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        ).to(self.config.weight_dtype).eval()

        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs']))
        )

        self.pipeline = WanFunInpaintAudioPipeline(
            transformer=transformer, vae=vae, tokenizer=tokenizer,
            text_encoder=text_encoder, scheduler=scheduler, clip_image_encoder=clip_image_encoder,
        )
        self.vae = vae

        # Memory optimization
        try:
            from mmgp import offload, profile_type
            self.pipeline.to("cpu")
            offload.profile(self.pipeline, profile_type.LowRAM_LowVRAM, quantizeTransformer=True)
        except ImportError:
            self.pipeline.to(device=self.device)

        # Wav2Vec2
        self.wav2vec_processor, self.wav2vec_model = load_wav2vec_models(self.config.wav2vec_model_dir)
        self.wav2vec_model = self.wav2vec_model.to("cpu")

        # Precompute anchor image info
        self._prepare_anchor()

    def _prepare_anchor(self):
        self.ref_img = Image.open(self.anchor_image).convert("RGB")
        y1, y2, x1, x2, h_, w_ = get_mask_coord(self.anchor_image)

        self.sample_h, self.sample_w = get_sample_size(self.ref_img, self.config.sample_size)
        downratio = math.sqrt(self.sample_h * self.sample_w / h_ / w_)
        coords = (
            y1 * downratio // 16, y2 * downratio // 16,
            x1 * downratio // 16, x2 * downratio // 16,
            self.sample_h // 16, self.sample_w // 16,
        )
        self.ip_mask = get_ip_mask(coords).unsqueeze(0)
        self.ip_mask = torch.cat([self.ip_mask] * 3).to(device=self.device, dtype=self.config.weight_dtype)

    def generate(self, wav_path):
        """Generate video from a WAV file path. Returns the final MP4 file path."""
        tmp_dir = tempfile.mkdtemp(prefix="echomimic_")
        try:
            return self._generate_video(wav_path, tmp_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def generate_bytes(self, wav_path):
        """Generate video from a WAV file path. Returns MP4 bytes."""
        tmp_dir = tempfile.mkdtemp(prefix="echomimic_")
        try:
            final_path = self._generate_video(wav_path, tmp_dir)
            with open(final_path, "rb") as f:
                return f.read()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    def _generate_video(self, wav_path, tmp_dir):
        config = self.config

        # Audio features
        audio_features = extract_audio_features(wav_path, self.wav2vec_processor, self.wav2vec_model)
        audio_embeds = audio_features.unsqueeze(0).to(device=self.device, dtype=config.weight_dtype)

        # Video length
        audio_clip = AudioFileClip(wav_path)
        video_length = int(audio_clip.duration * config.fps)
        video_length = int((video_length - 1) // self.vae.config.temporal_compression_ratio * self.vae.config.temporal_compression_ratio) + 1

        partial_video_length = int((config.partial_video_length - 1) // self.vae.config.temporal_compression_ratio * self.vae.config.temporal_compression_ratio) + 1
        _, _, clip_image = get_image_to_video_latent3(self.ref_img, None, video_length=partial_video_length, sample_size=[self.sample_h, self.sample_w])

        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        mix_ratio = torch.linspace(0, 1, steps=config.overlap_video_length).view(1, 1, -1, 1, 1)

        init_frames = 0
        last_frames = partial_video_length
        new_sample = None
        current_ref = self.ref_img

        while init_frames < video_length:
            cur_len = partial_video_length
            if last_frames >= video_length:
                cur_len = video_length - init_frames
                cur_len = int((cur_len - 1) // self.vae.config.temporal_compression_ratio * self.vae.config.temporal_compression_ratio) + 1
            if cur_len <= 0:
                break

            input_video, input_video_mask, _ = get_image_to_video_latent3(
                current_ref, None, video_length=cur_len, sample_size=[self.sample_h, self.sample_w]
            )

            torch.cuda.empty_cache()
            sample = self.pipeline(
                self.prompt,
                num_frames=cur_len,
                negative_prompt=config.negative_prompt,
                audio_embeds=audio_embeds[:, init_frames * 2:(init_frames + cur_len) * 2],
                audio_scale=config.audio_scale,
                ip_mask=self.ip_mask, use_un_ip_mask=config.use_un_ip_mask,
                height=self.sample_h, width=self.sample_w, generator=generator,
                neg_scale=config.neg_scale, neg_steps=config.neg_steps,
                use_dynamic_cfg=config.use_dynamic_cfg, use_dynamic_acfg=config.use_dynamic_acfg,
                guidance_scale=config.guidance_scale,
                audio_guidance_scale=config.audio_guidance_scale,
                num_inference_steps=config.num_inference_steps,
                video=input_video, mask_video=input_video_mask, clip_image=clip_image,
                cfg_skip_ratio=config.cfg_skip_ratio, shift=config.shift,
                use_longvideo_cfg=config.use_longvideo_cfg,
                overlap_video_length=config.overlap_video_length,
                partial_video_length=cur_len,
            ).videos

            if init_frames != 0:
                new_sample[:, :, -config.overlap_video_length:] = (
                    new_sample[:, :, -config.overlap_video_length:] * (1 - mix_ratio) +
                    sample[:, :, :config.overlap_video_length] * mix_ratio
                )
                new_sample = torch.cat([new_sample, sample[:, :, config.overlap_video_length:]], dim=2)
            else:
                new_sample = sample

            if last_frames >= video_length:
                break

            current_ref = [
                Image.fromarray((sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8))
                for i in range(-config.overlap_video_length, 0)
            ]
            init_frames += cur_len - config.overlap_video_length
            last_frames = init_frames + partial_video_length

        # Save and merge audio
        video_path = os.path.join(tmp_dir, "output.mp4")
        final_path = os.path.join(tmp_dir, "final.mp4")
        save_videos_grid(new_sample[:, :, :video_length], video_path, fps=config.fps)

        vc = VideoFileClip(video_path)
        ac = audio_clip.subclipped(0, video_length / config.fps)
        vc = vc.with_audio(ac)
        vc.write_videofile(final_path, codec="libx264", audio_codec="aac", threads=2)

        return final_path
