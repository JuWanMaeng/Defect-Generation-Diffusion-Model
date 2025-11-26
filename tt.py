import torch
from diffusers import StableDiffusion3Pipeline

# 1. 파이프라인 로드 (학습이 목적이므로 gradient checkpointing 등 설정 필요)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")

# 학습할 대상인 Transformer만 그래디언트 계산 활성화
pipe.transformer.train() 
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False) # SD3는 3개의 텍스트 인코더가 있음 (text_encoder, 2, 3)
pipe.text_encoder_2.requires_grad_(False)
pipe.text_encoder_3.requires_grad_(False)

# --- Loss 계산을 위한 가상 데이터 준비 ---
# 실제 학습에선 데이터로더에서 가져온 이미지와 텍스트를 사용합니다.
dummy_image = torch.randn(1, 3, 1024, 1024).to("cuda").half() # (Batch, Channel, H, W)
dummy_prompt = ["A cat sitting on a bench"]

# 2. 이미지 인코딩 (Latent 변환)
# 이미지를 VAE를 통해 Latent Space로 압축합니다.
with torch.no_grad():
    latents = pipe.vae.encode(dummy_image).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

# 3. 텍스트 인코딩
# 복잡한 SD3 프롬프트 인코딩 과정을 파이프라인 내부 함수 활용
with torch.no_grad():
    # pipe._get_t5_embeddings, pipe._get_clip_embeddings 등의 내부 로직과 유사하게 처리해야 함
    # 편의상 파이프라인의 encode_prompt 함수를 사용
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=dummy_prompt,
        device="cuda",
        do_classifier_free_guidance=False
    )

# 4. 노이즈 및 타임스텝 생성 (Flow Matching 핵심)
noise = torch.randn_like(latents)
bsz = latents.shape[0]

# 0~1 사이의 랜덤한 타임스텝 t 생성 (Sigmoid 스케줄링 등이 적용될 수 있음)
# SD3 Scheduler에서 적절한 sigmas를 가져오거나, 간단히 0~1 uniform sampling 사용
u = torch.rand(bsz, device="cuda") 
timesteps = u * 1000 # Diffusers 스케줄러 포맷에 맞춤

# 5. Noisy Latent 만들기 (Rectified Flow: 직선 보간)
# t 시점의 이미지 = (1-t)*원본 + t*노이즈
# Diffusers의 scheduler.add_noise()를 써도 되지만, SD3 논문 수식은 아래와 같음
sigmas = pipe.scheduler.sigmas[0] # 예시값
# 정확한 add_noise 로직은 scheduler 종류(FlowMatchEulerDiscreteScheduler)에 따라 다름
noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

# 6. 모델 예측 (Transformer Forward)
# 파이프라인의 __call__ 대신 transformer를 직접 호출합니다.
model_pred = pipe.transformer(
    hidden_states=noisy_latents,
    timestep=timesteps,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    return_dict=False
)[0]

# 7. Loss 계산 (Flow Matching Loss)
# SD3의 목표(Target)는 노이즈(Noise) - 원본(Latent) 인 속도 벡터(Velocity)입니다.
# v_t = x_1 - x_0 (논문에 따라 방향이 다를 수 있으므로 scheduler 설정 확인 필요)
# FlowMatchEulerDiscreteScheduler 기준: target = noise - latents
target = noise - latents

loss = torch.nn.functional.mse_loss(model_pred, target)

print(f"Calculated Loss: {loss.item()}")

# 이후 loss.backward() 및 optimizer.step() 진행