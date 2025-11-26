from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64
import torch
from diffusers import StableDiffusionPipeline

# === 1. 설정 (연구원님 환경에 맞게 튜닝) ===
# 모델 ID: 처음엔 가벼운 v1.5 추천 (나중에 SDXL이나 파인튜닝 모델로 교체 가능)
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. FastAPI 앱 초기화 ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 3. 모델 로드 (서버 시작 시 1회 실행) ===
print(f"🚀 Loading model: {MODEL_ID} to {DEVICE}...")

try:
    # fp16: VRAM 절약 및 속도 향상 (연구용이라도 인퍼런스는 fp16이 국룰)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    
    # [선택 사항] VRAM이 8GB 미만일 경우 아래 주석 해제
    # pipe.enable_attention_slicing() 
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    pipe = None

# 데이터 모델
class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    print(f"🎨 Generating for: {req.prompt}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64
import torch
from diffusers import StableDiffusionPipeline

# === 1. 설정 (연구원님 환경에 맞게 튜닝) ===
# 모델 ID: 처음엔 가벼운 v1.5 추천 (나중에 SDXL이나 파인튜닝 모델로 교체 가능)
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. FastAPI 앱 초기화 ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 3. 모델 로드 (서버 시작 시 1회 실행) ===
print(f"🚀 Loading model: {MODEL_ID} to {DEVICE}...")

try:
    # fp16: VRAM 절약 및 속도 향상 (연구용이라도 인퍼런스는 fp16이 국룰)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    
    # [선택 사항] VRAM이 8GB 미만일 경우 아래 주석 해제
    # pipe.enable_attention_slicing() 
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    pipe = None

# 데이터 모델
class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    print(f"🎨 Generating for: {req.prompt}")

    try:
        # === 4. 추론 (Inference) ===
        # 여기서 GPU가 돌아갑니다.
        # num_inference_steps=30 : 퀄리티와 속도의 타협점
        image = pipe(req.prompt, num_inference_steps=30).images[0]

        # === 5. 이미지 후처리 (PIL -> Base64) ===
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image_url": f"data:image/png;base64,{img_str}"}

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Inpainting 추가 ===
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw

print(f"🚀 Loading inpainting model...")
try:
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    inpaint_pipe = inpaint_pipe.to(DEVICE)
    print("✅ Inpainting model loaded!")
except Exception as e:
    print(f"❌ Inpainting model load failed: {e}")
    inpaint_pipe = None

class InpaintRequest(BaseModel):
    image: str # Base64 encoded image
    prompt: str
    bbox: list # [x, y, w, h]

@app.post("/inpaint")
async def inpaint_image(req: InpaintRequest):
    if inpaint_pipe is None:
        raise HTTPException(status_code=500, detail="Inpainting model is not loaded")

    try:
        # 1. Base64 -> PIL Image
        image_data = base64.b64decode(req.image.split(",")[1])
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. Create Mask from BBox
        mask_image = Image.new("L", init_image.size, 0) # Black background
        draw = ImageDraw.Draw(mask_image)
        x, y, w, h = req.bbox
        draw.rectangle([x, y, x+w, y+h], fill=255) # White bbox
        
        # 3. Inpaint
        print(f"🎨 Inpainting for: {req.prompt}")
        output = inpaint_pipe(
            prompt=req.prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=30
        ).images[0]
        
        # 4. Response
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {"image_url": f"data:image/png;base64,{img_str}"}
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)