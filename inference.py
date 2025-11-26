import argparse, os, random, re, json, types, glob
import torch, numpy as np, cv2
from PIL import Image
from magic_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import \
    StableDiffusionInpaintPipeline_magic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-time mask-alignment augmentation.")
    # CAMA(Context-Aware Mask Alignment)를 위한 사전 계산된 좌표 DB
    parser.add_argument("--defect_json", required=True)
    parser.add_argument("--match_json",  required=True)
    # 
    # DreamBooth로 학습된 모델(체크포인트)들이 저장된 루트 폴더
    parser.add_argument("--model_ckpt_root", required=True)
    # MGNI용 커스텀 DDIM 스케줄러가 저장된 루트 폴더
    parser.add_argument("--ddim_scheduler_root", required=True)
    parser.add_argument("--categories", nargs='+', default=None)
    parser.add_argument("--blur_factor", type=int, default=0, help="마스크에 적용할 블러 강도")
    # GPP (Gaussian Prompt Perturbation)의 노이즈 강도
    parser.add_argument("--text_noise_scale", type=float, default=0.0)
    parser.add_argument("--output_name", default="./", help="결과물이 저장될 루트 폴더")
    # 
    # MGNI (Mask-Guided Noise Injection) 관련 파라미터 1: 노이즈 강도 최소값
    parser.add_argument("--anomaly_strength_min", type=float, default=0.0)
    # MGNI 관련 파라미터 2: 노이즈 강도 최대값 (이 사이에서 랜덤 선택)
    parser.add_argument("--anomaly_strength_max", type=float, default=0.0)
    # MGNI 관련 파라미터 3: 노이즈 주입을 멈추는 DDIM 스텝
    parser.add_argument("--anomaly_stop_step", type=int, default=999999)
    # 
    parser.add_argument("--normal_masks", default="./normal_masks", help="정상 객체의 '실루엣' 마스크가 저장된 폴더")
    parser.add_argument("--mask_dir",       default="./Aug_mask_3_shot", help="생성할 '결함 모양' 마스크가 저장된 폴더")
    parser.add_argument("--base_dir",       default="./mvtecad", help="MVTec 데이터셋 루트 폴더")
    # CAMA (Context-Aware Mask Alignment) 기능을 켤지 여부
    parser.add_argument("--CAMA",           action="store_true")
    parser.add_argument("--use_random_mask", action="store_true")
    parser.add_argument("--dataset_type", choices=["mvtec_3d", "mvtec"],
                        default="mvtec_3d",
                        help="mvtec_3d: MVTEC-3D Anomaly, mvtec: MVTEC-AD 2-D")
    return parser.parse_args()

args = parse_args()


def extract_number_from_filename(fname):
    m = re.search(r"\d+", fname)
    return int(m.group()) if m else float("inf")


def monkey_patch_encode_prompt(pipe):
    """
    '원숭이 패치' 기법을 사용해, 파이프라인의 `encode_prompt` 함수를 런타임에
    강제로 수정하여 GPP(텍스트 노이즈 주입) 기능을 추가합니다.
    """
    # 1. 원본(기존) `encode_prompt` 함수를 백업
    old_encode = pipe.encode_prompt

    # 2. 원본 함수의 기능을 대체할 '새로운' 함수 정의
    def new_encode_prompt(self, prompt, device, num_images_per_prompt,
                          do_classifier_free_guidance, negative_prompt=None,
                          prompt_embeds=None, negative_prompt_embeds=None,
                          lora_scale=None, clip_skip=None):
        
        # 3. 일단 원본 함수를 호출하여 기본 프롬프트 임베딩(prompt_embeds)을 생성
        prompt_embeds, neg_embeds = old_encode(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale, clip_skip=clip_skip)

        # 4. GPP 기능이 켜져 있으면 (text_noise_scale > 0)
        if getattr(self, "text_noise_scale", 0.0) > 0.0:
            s = self.text_noise_scale # 노이즈 강도 (e.g., 0.5)
            
            # CFG(Classifier-Free Guidance)를 사용하는 경우
            if do_classifier_free_guidance:
                # prompt_embeds는 [부정(uncond), 긍정(cond)] 2개가 합쳐진 형태
                half = prompt_embeds.shape[0] // 2
                uncond, cond = prompt_embeds[:half], prompt_embeds[half:]
                
                # ❗️ 핵심: 긍정 프롬프트('sks') 임베딩(cond)에만 가우시안 노이즈를 추가
                cond += torch.randn_like(cond) * s
                
                # 다시 합쳐서 반환
                prompt_embeds = torch.cat([uncond, cond], 0)
            else:
                # (CFG를 안 쓰는 경우, 드물지만) 그냥 전체에 노이즈 추가
                prompt_embeds += torch.randn_like(prompt_embeds) * s
                
        return prompt_embeds, neg_embeds

    # 5. 파이프라인(pipe)의 `encode_prompt` 함수를 우리가 방금 만든 '새로운' 함수로 교체
    pipe.encode_prompt = types.MethodType(new_encode_prompt, pipe)


def inpaint(pipe, image, prompt, mask=None, n_samples=4, device="cuda",
            blur_factor=0, anomaly_strength=0.0, anomaly_stop_step=999999):
    """
    StableDiffusion Inpainting 파이프라인을 실행하는 헬퍼 함수.
    MGNI를 위한 커스텀 인자(anomaly_strength, anomaly_stop_step)를 전달합니다.
    """
    from PIL import Image as PilImage
    # 이미지/마스크 로드 및 RGB 변환
    if isinstance(image, str):
        image_pil = PilImage.open(image).convert("RGB")
    else:
        image_pil = image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(mask, str):
        mask_pil = PilImage.open(mask).convert("RGB")
    else:
        mask_pil = mask.convert("RGB") if mask.mode != "RGB" else mask
    
    # 마스크에 블러 적용 (옵션)
    mask_pil = pipe.mask_processor.blur(mask_pil, blur_factor=blur_factor)
    
    # ❗️ 핵심: 커스텀 파이프라인(StableDiffusionInpaintPipeline_magic) 호출
    # MGNI 관련 인자(anomaly_strength 등)를 파이프라인 내부로 전달
    return pipe(
        prompt=[prompt]*n_samples, image=image_pil, mask_image=mask_pil,
        anomaly_strength=anomaly_strength, anomaly_stop_step=anomaly_stop_step,
        use_random_mask=args.use_random_mask).images


def get_random_image(img_dir):
    imgs = [f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        raise ValueError(f"No images in {img_dir}")
    return os.path.join(img_dir, random.choice(imgs))


def load_object_mask(category, normal_img_path, normal_masks_dir):
    """
    정상 이미지에 해당하는 '객체 실루엣' 마스크를 로드하는 함수.
    CAMA가 결함을 객체 밖으로 이동시키는 것을 방지하기 위해 사용됩니다.
    (e.g., 'screw/001.png' -> 'normal_masks/screw/train/masks/001_mask.png')
    """

    cat_dir = os.path.join(normal_masks_dir, category)
    base = os.path.splitext(os.path.basename(normal_img_path))[0]

    # Candidate directories to search first
    cand_dirs = [
        os.path.join(cat_dir, "train", "masks"),
        os.path.join(cat_dir, "masks"),
        cat_dir,
    ]
    candidates = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        candidates.append(os.path.join(d, f"{base}_mask.png"))
        candidates.extend(sorted(glob.glob(os.path.join(d, f"{base}_mask_*.png"))))
        candidates.append(os.path.join(d, f"{base}.png"))
        candidates.append(os.path.join(d, "mask.png"))

    # Remove duplicates and try only existing paths
    seen = set()
    ordered_paths = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            ordered_paths.append(p)

    for mask_path in ordered_paths:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (m > 127).astype(np.uint8)

    return None

def debug_save_masks(original_mask_bin, min_x, min_y, max_x, max_y,
                     shifted_mask_bin, debug_save_path):
    """CAMA 디버깅용: 원본 마스크(왼쪽)와 이동된 마스크(오른쪽)를 나란히 저장"""
    H, W = original_mask_bin.shape
    left = np.zeros((H, W, 3), np.uint8)
    left[original_mask_bin > 0] = (255, 255, 255)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(original_mask_bin, 8)
    for lbl in range(1, n_lbl):
        x, y, bw, bh, _ = stats[lbl]
        if bw and bh:
            cv2.rectangle(left, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 2)

    right = np.zeros((H, W, 3), np.uint8)
    right[shifted_mask_bin > 0] = (255, 255, 255)
    cv2.imwrite(debug_save_path, np.concatenate([left, right], axis=1))

###############################################################################
# 3) CAMA: Context-Aware Mask Alignment
###############################################################################
def CAMA_click(
    class_val,
    code_mask_bin,     # (A) 이동시킬 '결함 모양' 마스크 (numpy)
    obj_mask_np,       # (B) 결함이 그려질 '객체 실루엣' 마스크 (numpy)
    normal_image_path, # (C) 결함을 그릴 '정상 이미지' 경로
    category,
    defect_class,
    defect_data,       # (D) (사용 안 함)
    match_data,        # (E) (사용 안 함)
    debug_save_dir=None,
    debug_name=None,
):
    """
    [Modified] Interactive CAMA:
    사용자가 화면을 클릭하면 해당 좌표로 결함 마스크를 이동시킵니다.
    """
    H, W = code_mask_bin.shape
    base_normal = os.path.basename(normal_image_path)

    # ───────── ① 마스크 조각(Component) 분석 ─────────
    # 옮길 결함 덩어리가 몇 개인지 확인합니다.
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    comps = list(range(1, n_lbl))  # 0 is background
    
    if not comps:  # 마스크가 비어있으면 원본 그대로 리턴
        print("[Interactive] 마스크가 비어있습니다.")
        return code_mask_bin, -1, -1, False

    n_comp = len(comps)
    print(f"\n[Interactive] '{base_normal}' 이미지에 결함을 배치합니다.")
    print(f"👉 옮겨야 할 결함 조각 개수: {n_comp}개")
    print("👉 원하는 위치를 마우스로 클릭하세요 (조각 개수만큼 클릭).")

    # ───────── ② [핵심] 사용자 클릭 입력 받기 ─────────
    clicked_coords = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_coords.append((x, y))
            # 클릭한 위치에 시각적 피드백 (빨간 점)
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click Defect Position", param)
            print(f"   📍 클릭 {len(clicked_coords)}/{n_comp}: ({x}, {y})")

    # 정상 이미지를 로드해서 화면에 띄웁니다.
    display_img = cv2.imread(normal_image_path)
    if display_img is None: # 경로 문제 시 안전장치
        pil_img = Image.open(normal_image_path).convert("RGB")
        display_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    display_img = cv2.resize(display_img, (W, H))
    clone_img = display_img.copy()

    cv2.namedWindow("Click Defect Position")
    cv2.setMouseCallback("Click Defect Position", mouse_callback, param=clone_img)
    cv2.imshow("Click Defect Position", clone_img)

    # 사용자가 조각 개수만큼 클릭할 때까지 대기
    while len(clicked_coords) < n_comp:
        key = cv2.waitKey(10)
        if key == 27: # ESC 누르면 강제 종료 (혹은 건너뛰기)
            print("[Interactive] ESC 눌림. 원본 마스크 위치 사용.")
            cv2.destroyAllWindows()
            return code_mask_bin, -1, -1, False

    cv2.destroyAllWindows() # 창 닫기

    # ───────── ③ 이동 및 배치 (Translation) ─────────
    # 사용자가 찍은 좌표(clicked_coords)를 목표 좌표로 사용
    coords = clicked_coords 
    
    # 조각(comps)과 좌표(coords) 1:1 매칭
    target_pairs = list(zip(comps, coords))

    shifted = np.zeros_like(code_mask_bin, np.uint8) # 빈 마스크

    for lbl, (best_x, best_y) in target_pairs:
        # (이미지 상에서 클릭했으므로 480 스케일링 불필요. 바로 사용)
        
        # 해당 조각의 바운딩 박스 정보
        x, y, bw, bh, _ = stats[lbl]
        if bw == 0 or bh == 0: continue

        # 조각 잘라내기
        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8)

        # 목표 좌표가 '중심'이 되도록 왼쪽-위 좌표 계산
        tx = best_x - bw // 2
        ty = best_y - bh // 2

        # 픽셀 옮겨 그리기
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]:
                    yy = ty + r
                    xx = tx + c
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    # ───────── ④ 마무리 (Clipping & Debug) ─────────
    # class_val == 0 (객체 내부 결함)인 경우: 객체 실루엣 밖으로 나간 부분 제거
    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks(
            code_mask_bin,
            xs.min() if xs.size else 0, ys.min() if ys.size else 0,
            xs.max() if xs.size else 0, ys.max() if ys.size else 0,
            final,
            os.path.join(debug_save_dir, f"{debug_name}.jpg")
        )

    first_best = coords[0]
    return final, first_best[0], first_best[1], True

def CAMA_match(
    class_val,
    code_mask_bin,     # (A) 이동시킬 '결함 모양' 마스크 (numpy)
    obj_mask_np,       # (B) 결함이 그려질 '객체 실루엣' 마스크 (numpy)
    normal_image_path, # (C) 결함을 그릴 '정상 이미지' 경로
    category,
    defect_class,
    defect_data,       # (D) 'defect.json' 파일 내용
    match_data,        # (E) 'match.json' (사전 계산된 좌표 DB) 파일 내용
    debug_save_dir=None,
    debug_name=None,
):
    """
    Return (final_mask, first_best_x, first_best_y, is_shifted)
    """
    H, W = code_mask_bin.shape
    base_normal = os.path.basename(normal_image_path)

    # ───────── ① Collect coordinates per defect_img ─────────
    by_defect = {}
    for it in match_data.get(category, {}).get(defect_class, []):
        if it["normal_img"] != base_normal:
            continue
        by_defect.setdefault(it["defect_img"], []).append((it["best_x"], it["best_y"]))

    if not by_defect:
        fallback = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:
            debug_save_masks(
                code_mask_bin, 0, 0, 0, 0, fallback,
                os.path.join(debug_save_dir, f"{debug_name}_fallback.jpg")
            )
        return fallback, -1, -1, False
    
    # ⬇️⬇️⬇️ [디버깅 코드 추가 시작] ⬇️⬇️⬇️
    # matching 좌표들을 시각화하여 저장
    if debug_save_dir and debug_name:
        # 1. 원본 정상 이미지 로드
        debug_img = cv2.imread(normal_image_path)
        if debug_img is None:
            # 경로가 안 맞을 경우를 대비해 PIL로 읽고 변환 시도 (안전장치)
            pil_img = Image.open(normal_image_path).convert("RGB")
            debug_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 2. 이미지 크기에 맞게 리사이즈 (좌표가 480 기준일 수 있으므로 W, H로 맞춤)
        # match.json의 좌표는 보통 480x480 기준이므로, 현재 이미지 크기(W, H)로 변환해야 정확합니다.
        debug_img = cv2.resize(debug_img, (W, H))

        # 3. 모든 좌표 찍기
        for def_img, coords_list in by_defect.items():
            # 색상을 랜덤하게 생성 (결함 이미지별로 다른 색)
            color = np.random.randint(0, 255, (3,)).tolist()
            
            for (bx_480, by_480) in coords_list:
                # 좌표 스케일링 (480 -> 현재 W, H)
                bx = int(bx_480 * W / 480.0)
                by = int(by_480 * H / 480.0)
                
                # 점 찍기 (원)
                cv2.circle(debug_img, (bx, by), 5, color, -1) # 채워진 원
                # 십자 표시 (더 잘 보이게)
                cv2.drawMarker(debug_img, (bx, by), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # 4. 저장
        save_path = os.path.join(debug_save_dir, f"{debug_name}_points_debug.jpg")
        cv2.imwrite(save_path, debug_img)
        print(f"[DEBUG] Point visualization saved to: {save_path}")
    # ⬆️⬆️⬆️ [디버깅 코드 추가 끝] ⬆️⬆️⬆️

    # ───────── ② Randomly choose one defect_img ─────────
    chosen_defect, coords_all = random.choice(list(by_defect.items()))

    # ───────── ③ Extract components ─────────
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    comps = list(range(1, n_lbl))  # 0 is background
    if not comps:  # mask is empty
        return code_mask_bin, -1, -1, False

    n_comp = len(comps)
    n_coords = len(coords_all)

    # ───────── ④ Match coordinate list size to the number of components ─────────
    # 결함 조각(짐)의 개수에 맞춰서 좌표(목적지) 리스트의 길이를 강제로 동기화하는 과정
    # 마스크 조각이 한개면 신경쓸 필요가 없음
    def rand_point_inside(mask):
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            # If obj_mask is empty, sample from the whole image
            return random.randint(0, W - 1), random.randint(0, H - 1)
        idx = random.randrange(ys.size)
        # Return in (x, y) order consistently
        return int(xs[idx] * 480.0 / W), int(ys[idx] * 480.0 / H)

    if n_coords >= n_comp:
        coords = random.sample(coords_all, n_comp)
    else:
        coords = list(coords_all)
        # Fill the shortage with random coordinates
        for _ in range(n_comp - n_coords):
            rx, ry = rand_point_inside(obj_mask_np if class_val == 0 else np.ones_like(obj_mask_np))
            coords.append((rx, ry))

    # Now comps and coords have the same length (n_comp)
    # comps(결함 조각들)와 coords(목표 좌표들)가 1:1로 매칭된 상태(target_pairs)에서, 
    # 각 조각을 잘라내어 목표 위치로 이동(Translation)시키고, 
    # 마지막으로 삐져나온 부분을 다듬는(Clipping) 핵심 로직
    
    # 1. 짝짓기: (1번 조각, 좌표A), (2번 조각, 좌표B)...
    target_pairs = list(zip(comps, coords))

    # 2. 빈 트럭 준비: 이동된 결함들을 담을 새하얀 도화지(마스크) 생성
    shifted = np.zeros_like(code_mask_bin, np.uint8)

    # 3. 하나씩 이사 시작
    for lbl, (best_x_480, best_y_480) in target_pairs:
        # 좌표 스케일링: 480 기준 좌표를 현재 이미지 크기(W, H)에 맞게 변환
        best_x = int(best_x_480 * W / 480.0)
        best_y = int(best_y_480 * H / 480.0)

        # "이미 와 있는데요?"
        if 0 <= best_x < W and 0 <= best_y < H and code_mask_bin[best_y, best_x]:
            shifted |= (lbl_map == lbl).astype(np.uint8)
            continue

        x, y, bw, bh, _ = stats[lbl]  # 조각의 바운딩 박스(위치, 크기) 가져오기
        if bw == 0 or bh == 0:
            continue
        
        # 전체 라벨맵에서 '이 조각(lbl)'에 해당하는 부분만 딱 오려냄 (0과 1로 된 작은 네모)
        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8) 

        # 목표 좌표(best_x, best_y)가 이 조각의 '정중앙'에 오도록 새 왼쪽-위(Top-Left) 좌표 계산
        tx = best_x - bw // 2
        ty = best_y - bh // 2
        
        # 픽셀 한 땀 한 땀 옮겨 그리기
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]: # 조각의 흰색 부분만
                    yy = ty + r # 새 Y 좌표
                    xx = tx + c # 새 X 좌표
                    
                    # 화면 밖으로 나가는지 체크하고 그리기
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    # class_val == 0 (객체 내부 결함)인 경우: 객체 실루엣 밖으로 나간 부분 제거
    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks( # [왼쪽: 원본 마스크] vs [오른쪽: 최종 이동된 마스크]를 나란히 붙인 이미지를 생성해 저장
            code_mask_bin,
            xs.min() if xs.size else 0, ys.min() if ys.size else 0,
            xs.max() if xs.size else 0, ys.max() if ys.size else 0,
            final,
            os.path.join(debug_save_dir, f"{debug_name}.jpg")
        )

    first_best = coords[0]
    return final, first_best[0], first_best[1], True




def main():
    with open(args.defect_json, "r", encoding="utf-8") as f:
        defect_data = json.load(f)
    with open(args.match_json, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # ───────── Select default category list ─────────
    if args.dataset_type == "mvtec_3d":
        default_cats = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                        'foam', 'peach', 'potato', 'rope', 'tire']
    else:  # mvtec
        default_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    categories = args.categories or default_cats


    for category in categories:
        device = "cuda"

        # ───────── Path differences per dataset ─────────
        if args.dataset_type == "mvtec_3d":
            gt_path     = os.path.join(args.base_dir, category, "test")
            normal_root = os.path.join(args.base_dir, category,
                                       "train", "good", "rgb")
        else: 
            gt_path     = os.path.join(args.base_dir, category, "ground_truth") # 결함을 생성할 영역의 마스크
            normal_root = os.path.join(args.base_dir, category,                 # 정상 이미지
                                       "train", "good")

        if not os.path.exists(gt_path):
            print(f"[WARN] ground_truth path not found: {gt_path}")
            continue
        if not os.path.exists(normal_root):
            print(f"[WARN] normal image path not found: {normal_root}")
            continue

        defect_classes = [d for d in os.listdir(gt_path)
                          if os.path.isdir(os.path.join(gt_path, d))
                          and d != "good"]
        for defect_class in defect_classes:
            if defect_class not in defect_data.get(category, {}):
                print(f"[WARN] {defect_class} not in defect_json → skip")
                continue
            class_val = defect_data[category][defect_class]
            print(f"Category={category}, Defect={defect_class}, "
                  f"class_val={class_val}")

            mask_root = os.path.join(args.mask_dir, category, defect_class)
            if not os.path.exists(mask_root):
                print(f"[WARN] {mask_root} not found → skip")
                continue

            ckpt_root = os.path.join(args.model_ckpt_root,
                                     category, defect_class)
            
            # if not os.path.exists(ckpt_root):
            #     print(f"[WARN] checkpoint absent: {ckpt_root} → skip")
            #     continue
            # pipe = StableDiffusionInpaintPipeline_magic.from_pretrained(
            #     ckpt_root, torch_dtype=torch.float16)
            pipe = StableDiffusionInpaintPipeline_magic.from_pretrained(
                "alwold/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_pretrained(
                args.ddim_scheduler_root)
            pipe.text_noise_scale = args.text_noise_scale
            monkey_patch_encode_prompt(pipe)
            pipe.to(device)

            mask_imgs = sorted(
                (os.path.join(mask_root, f) for f in os.listdir(mask_root)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))),
                key=lambda x: extract_number_from_filename(
                    os.path.basename(x)))

            suffix = (f"noise_{args.text_noise_scale}_"
                      f"anomaly_{args.anomaly_strength_min}_"
                      f"{args.anomaly_strength_max}_"
                      f"dynamic_{args.anomaly_stop_step}_"
                      + ("align" if args.CAMA else "no_align"))
            save_root = os.path.join(args.output_name, suffix,
                                     category, defect_class)
            img_dir  = os.makedirs(os.path.join(save_root, "image"),
                                   exist_ok=True) or \
                       os.path.join(save_root, "image")
            norm_dir = os.makedirs(os.path.join(save_root, "normal"),
                                   exist_ok=True) or \
                       os.path.join(save_root, "normal")
            msk_dir  = os.makedirs(os.path.join(save_root, "masks"),
                                   exist_ok=True) or \
                       os.path.join(save_root, "masks")
            dbg_dir  = os.makedirs(os.path.join(save_root, "debug_mask"),
                                   exist_ok=True) or \
                       os.path.join(save_root, "debug_mask")

            for idx, mask_path in enumerate(mask_imgs):
                normal_img_path = get_random_image(normal_root)
                normal_img = Image.open(normal_img_path)

                # 결함 마스크
                mask_np = (np.array(Image.open(mask_path).convert("L")) > 127
                           ).astype(np.uint8)

                # 정상 이미지에 해당하는 '객체 실루엣' 마스크를 로드하는 함수.
                # CAMA가 결함을 객체 밖으로 이동시키는 것을 방지하기 위해 사용됩니다. path 체크 필요
                obj_mask = load_object_mask(category, normal_img_path,
                                            args.normal_masks)
                if obj_mask is None:
                    obj_mask = np.ones_like(mask_np, np.uint8)
                if obj_mask.shape != mask_np.shape:
                    obj_mask = cv2.resize(obj_mask, mask_np.shape[::-1],
                                          interpolation=cv2.INTER_NEAREST)

                if args.CAMA:
                    final_mask, *_ = CAMA_click(
                        class_val, mask_np, obj_mask, normal_img_path,
                        category, defect_class, defect_data, match_data,
                        debug_save_dir=dbg_dir, debug_name=f"{idx}",
                    )
                else:
                    final_mask = mask_np

                final_mask_pil = Image.fromarray(
                    (final_mask * 255).astype(np.uint8))

                a_strength = random.uniform(args.anomaly_strength_min,
                                            args.anomaly_strength_max)
                imgs = inpaint(pipe, normal_img,
                               prompt="a photo of a sks defect",
                               mask=final_mask_pil, n_samples=1, device=device,
                               blur_factor=args.blur_factor,
                               anomaly_strength=a_strength,
                               anomaly_stop_step=args.anomaly_stop_step)

                out = f"{idx}.jpg"
                imgs[0].save(os.path.join(img_dir,  out))
                normal_img.save(os.path.join(norm_dir, out))
                final_mask_pil.convert("RGB").save(os.path.join(msk_dir, out))
                print(f"Saved {out}")


if __name__ == "__main__":
    main()
