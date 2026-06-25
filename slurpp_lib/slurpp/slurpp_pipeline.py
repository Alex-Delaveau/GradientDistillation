# this code is modified from Marigold: https://github.com/prs-eth/Marigold
from typing import Optional, Union
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class SlurppPipeline(DiffusionPipeline):

    rgb_latent_scale_factor = 0.18215
    pred_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
        self.skip_connection = False
        self.composite_latents = None
        self.pass_zero_conv = False
        self.vae_cld = None
    
    @torch.no_grad()
    def __call__(
        self,
        inputs,
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        multi_patch: bool = False,
        window_size: int = 96,
        stride: int = 48,
        is_dual: bool = False,
        return_latent: bool = False,
    ) :
        pred = self.single_infer(
            rgb_in=inputs,
            num_inference_steps=denoising_steps,
            show_pbar=show_progress_bar,
            generator=generator,
            is_dual=is_dual,
            return_latent=return_latent
        )

        return pred


    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        is_dual: bool = False,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        Perform an individual prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted image.
        """
        device = rgb_in[0].device
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        # rgb_in = rgb_in.to(device)
        batch_size = 1
        rgb_latents = [self.encode_rgb(rgb.to(device)) for rgb in rgb_in]  # [B, 4, h, w]
        rgb_latents = torch.cat(rgb_latents, dim=1)  # [B, 4*len(rgb_in), h, w]

        out_shape = list(rgb_latents[:,:4].shape)
        if not is_dual:
            out_shape[1] = self.unet.config["out_channels"]  # [B, 4, h, w]
        else:
            out_shape[1] = self.unet.unet1.config["out_channels"] + self.unet.unet2.config["out_channels"]  # [B, 8, h, w]
        if num_inference_steps == 1:
            pred_latent = torch.zeros(out_shape,
                                   device=device,
                                   dtype=self.dtype)  # [B, 4, h, w]
        else:
            pred_latent = torch.randn(out_shape,
                                   device=device,
                                   dtype=self.dtype,
                                   generator=generator,)  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (batch_size, 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for _, t in iterable:
            unet_input = torch.cat(
                [rgb_latents, pred_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            pred_latent = self.scheduler.step(
                noise_pred, t, pred_latent, generator=generator
            ).prev_sample


        if return_latent:
            return (self.decode_images(pred_latent), pred_latent)

        return self.decode_images(pred_latent)
    
    def encode_rgb(self, rgb_in: torch.Tensor, skip_connection: bool = None) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        if skip_connection is None:
            skip_connection = self.skip_connection
        if skip_connection and self.vae_cld is not None:
            out = self.vae_cld.encode(rgb_in, skip_connection=True)
            self.composite_latents = out.composite_latents
            self.pass_zero_conv = False
            latents = out.latent_dist.sample()
        else:
            out = self.vae.encode(rgb_in)
            latents = out.latent_dist.sample()
        # scale latent
        rgb_latent = latents * self.rgb_latent_scale_factor
        return rgb_latent
    
    def decode_images(self, img_latent: torch.Tensor) -> torch.Tensor:
            
        img_latent = img_latent / self.rgb_latent_scale_factor
        if self.skip_connection and self.vae_cld is not None:
            if not self.pass_zero_conv:
                self.composite_latents = self.vae_cld.pass_zero_conv(self.composite_latents)
                self.pass_zero_conv = True
            if img_latent.shape[1] == 4:
                image = self.vae_cld.decode(img_latent, composite_latents=self.composite_latents).sample
            else:
                img_splits = torch.split(img_latent, 4, dim=1)
                decoded_images = []

                # cld decode for the first split(clear)
                decoded_images.append(self.vae_cld.decode(img_splits[0], composite_latents=self.composite_latents).sample)

                # normal decode for the rest
                decoded_images.append(self.vae.decode(torch.cat(img_splits[1:], dim=0)).sample)

                image = torch.cat(decoded_images, dim=0)
            
        else:
            if img_latent.shape[1] == 4:
                image = self.vae.decode(img_latent).sample
            else:
                img_splits = img_latent.reshape(
                    img_latent.shape[0]* (img_latent.shape[1] // 4), 4, img_latent.shape[2], img_latent.shape[3]
                )
                image = self.vae.decode(img_splits).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        return image