import os
import time
import sys
import shutil
import json
import mimetypes
import subprocess
import argparse
import base64
from pathlib import Path
from typing import Optional, List

from google import genai
from google.genai import types as gtypes

# Try to import PIL for the deterministic grid generation
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL (Pillow) not found. Install it for perfect board alignment: 'pip install pillow'")

# Try to import OpenAI for ChatGPT / Sora backend
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Backend selector: "gemini" (default) or "openai"
BACKEND = "gemini"

openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================================
# 0) BACKEND-LEVEL HELPERS
# ============================================================================

def openai_generate_image(
    prompt: str,
    input_paths: Optional[List[str]] = None,  # ignored for now
    out_dir: str = ".",
    n: int = 1,
    model: Optional[str] = None,
) -> List[str]:
    """
    OpenAI image generation using the image_generation tool on a chat model.
    Uses responses.create with tools=[{"type": "image_generation"}].
    """
    if openai_client is None:
        print("❌ OPENAI client not initialized. Set OPENAI_API_KEY and install 'openai'.")
        sys.exit(1)

    # Text/chat model that can call tools
    text_model = model or os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.1-chat-latest")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        print(f"   🔎 OpenAI responses.create(model={text_model!r}) with image_generation tool ...")
        response = openai_client.responses.create(
            model=text_model,
            input=prompt,
            tools=[{"type": "image_generation"}],
        )
    except Exception as e:
        print("❌ OpenAI image_generation tool call failed:")
        print(f"   type: {type(e)}")
        print(f"   repr: {repr(e)}")
        print(f"   str:  {e}")
        raise

    # Parse tool outputs -> base64 image(s)
    out_paths: List[str] = []
    ts = int(time.time() * 1000)

    outputs = getattr(response, "output", None)
    if not outputs:
        print("   ⚠️ response.output is empty, no images returned.")
        return out_paths

    for idx, output in enumerate(outputs):
        # Matches the pattern you pasted: type == "image_generation_call"
        if getattr(output, "type", None) != "image_generation_call":
            continue
        b64 = getattr(output, "result", None)
        if not b64:
            continue

        fname = f"openai_{ts}_{idx}.png"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(base64.b64decode(b64))
        out_paths.append(fpath)

        if len(out_paths) >= n:
            break

    if not out_paths:
        print("   ⚠️ No image files were written from OpenAI response (no image_generation_call outputs).")

    return out_paths

def openai_generate_video(
    prompt: str,
    image_path: Optional[str],
    out_dir: str = ".",
    model: Optional[str] = None,
) -> Optional[str]:
    """
    OpenAI Sora video generation using videos.create + polling + download_content.
    Currently uses prompt-only (image_path not used).
    """
    if openai_client is None:
        print("❌ OPENAI client not initialized. Set OPENAI_API_KEY and install 'openai'.")
        sys.exit(1)

    model = model or os.environ.get("OPENAI_VIDEO_MODEL", "sora-2")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("   ...sending to Sora (OpenAI Video API)...")
    try:
        video = openai_client.videos.create(
            model=model,
            prompt=prompt,
        )
    except Exception as e:
        if "rate_limit" in str(e) or "429" in str(e):
            print("\n🚨 SORA QUOTA / RATE LIMIT.")
            print("   The script is stopping safely.")
            sys.exit(0)
        print(f"❌ Sora Start Error: {e}")
        return None

    video_id = getattr(video, "id", None)
    status = getattr(video, "status", None)
    if not video_id:
        print("❌ No video id returned from OpenAI.")
        return None

    print(f"   Sora Job ID: {video_id}, initial status: {status}")

    # Poll for completion
    while status in ("queued", "in_progress", None):
        time.sleep(10)
        try:
            video = openai_client.videos.retrieve(video_id)
            status = getattr(video, "status", None)
            print(f"   ...status: {status}")
        except Exception as e:
            print(f"   ⚠️ Poll error: {e}")
            time.sleep(10)

    if status != "completed":
        print(f"❌ Video creation failed. Final status: {status}")
        return None

    out_path = os.path.join(out_dir, f"sora_{int(time.time()*1000)}.mp4")

    try:
        resp = openai_client.videos.download_content(video_id)
        if hasattr(resp, "iter_bytes"):
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        else:
            with open(out_path, "wb") as f:
                f.write(resp)  # type: ignore[arg-type]
    except Exception as e:
        print(f"❌ Error downloading Sora video: {e}")
        return None

    return out_path


def generate_image_backend(
    prompt: str,
    input_paths: Optional[List[str]],
    out_dir: str,
    n: int = 1,
) -> List[str]:
    """
    Unified image generation entrypoint:
    - BACKEND == 'gemini' -> Gemini/Banana
    - BACKEND == 'openai' -> OpenAI (dall-e-3)
    """
    if BACKEND == "openai":
        return openai_generate_image(prompt, input_paths=None, out_dir=out_dir, n=n)
    else:
        return banana_generate(prompt, input_paths=input_paths, out_dir=out_dir, n=n)


def generate_video_backend(
    prompt: str,
    image_path: Optional[str],
    out_dir: str,
) -> Optional[str]:
    """
    Unified video generation entrypoint:
    - BACKEND == 'gemini' -> Veo
    - BACKEND == 'openai' -> Sora
    """
    if BACKEND == "openai":
        return openai_generate_video(prompt, image_path=image_path, out_dir=out_dir)
    else:
        return veo_generate_video(prompt, image_path=image_path, out_dir=out_dir)

# ============================================================================
# 1) BANANA (Image Gen) - GEMINI BACKEND
# ============================================================================

def banana_generate(
    prompt: str,
    input_paths: Optional[List[str]] = None,
    out_dir: str = ".",
    n: int = 1,
    model: Optional[str] = None,
):
    model = model or os.environ.get("IMAGE_MODEL", "gemini-3-pro-image-preview")

    if not API_KEY:
        print("❌ GEMINI_API_KEY not found.")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    parts = [gtypes.Part.from_text(text=prompt)]
    input_paths = input_paths or []

    for p in input_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(p)
            parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt or "image/png"))

    contents = [gtypes.Content(role="user", parts=parts)]
    config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    out_paths = []
    try:
        stream = client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        idx = 0
        for chunk in stream:
            cand = getattr(chunk, "candidates", None)
            if not cand or not cand[0].content or not cand[0].content.parts:
                continue

            p = cand[0].content.parts[0]
            inline = getattr(p, "inline_data", None)

            if inline and inline.data:
                fname = f"banana_{int(time.time()*1000)}_{idx}.png"
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(inline.data)
                out_paths.append(fpath)
                idx += 1
                if len(out_paths) >= n:
                    break
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n🚨 BANANA QUOTA EXCEEDED (429).")
            print("   The script is stopping safely.")
            print("   ✅ Run this script again later/tomorrow to resume exactly here.")
            sys.exit(0)
        print(f"❌ Generation Error: {e}")

    return out_paths

# ============================================================================
# 2) VEO (Video Gen) - GEMINI BACKEND
# ============================================================================

def veo_generate_video(
    prompt: str,
    image_path: Optional[str],
    out_dir: str = ".",
    aspect_ratio="16:9",
    resolution="720p",
    model: Optional[str] = None,
):
    model = model or os.environ.get("VIDEO_MODEL", "veo-3.1-generate-preview")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=API_KEY)

    image_obj = None
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        mt, _ = mimetypes.guess_type(image_path)
        image_obj = gtypes.Image(image_bytes=data, mime_type=mt or "image/png")

    cfg = gtypes.GenerateVideosConfig(aspect_ratio=aspect_ratio, resolution=resolution)

    print(f"   ...sending to Veo...")
    try:
        op = client.models.generate_videos(
            model=model, prompt=prompt, image=image_obj, config=cfg
        )
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n🚨 VEO QUOTA EXCEEDED (429).")
            print("   The script is stopping safely.")
            print("   ✅ Run this script again later/tomorrow to resume exactly here.")
            sys.exit(0)
        print(f"❌ Veo Start Error: {e}")
        return None

    # Poll
    while not op.done:
        time.sleep(6)
        try:
            op = client.operations.get(op)
        except:
            pass

    if not op.result or not op.result.generated_videos:
        return None

    video = op.result.generated_videos[0].video
    out_path = os.path.join(out_dir, f"veo_{int(time.time()*1000)}.mp4")

    try:
        video_bytes = client.files.download(file=video)
        with open(out_path, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        print(f"❌ Save Error: {e}")
        return None

    return out_path

# ============================================================================
# 3) COMPRESSION (FFmpeg)
# ============================================================================

def compress_new_assets(video_dir):
    """
    Compresses all MP4s in the target directory using FFmpeg.
    """
    if shutil.which("ffmpeg") is None:
        print("\n⚠️  FFmpeg not found in system PATH. Skipping compression.")
        print("    Install FFmpeg to enable automatic file size reduction.")
        return

    print(f"\n🗜️  Compressing video assets in: {video_dir}...")

    files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    if not files:
        print("    No videos found to compress.")
        return

    count = 0
    total_saved = 0

    for file in files:
        full_path = os.path.join(video_dir, file)
        temp_path = full_path + ".temp.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            full_path,
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "slow",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            temp_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            original_size = os.path.getsize(full_path)
            new_size = os.path.getsize(temp_path)

            if new_size < original_size:
                os.replace(temp_path, full_path)
                saved = original_size - new_size
                total_saved += saved
                count += 1
            else:
                os.remove(temp_path)

        except Exception as e:
            print(f"    ❌ Error compressing {file}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    mb_saved = total_saved / (1024 * 1024)
    print(f"    ✅ Compression Complete. Optimized {count} files. Saved {mb_saved:.2f} MB total.")

# ============================================================================
# 4) PIPELINE STEPS
# ============================================================================

def create_guide_board(out_path, size=1024):
    """Creates a perfect black and white checkerboard to guide the AI."""
    if not PIL_AVAILABLE:
        return None

    print(f"   📐 Creating deterministic guide grid...")
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    square_size = size // 8

    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                x = c * square_size
                y = r * square_size
                draw.rectangle([x, y, x + square_size, y + square_size], fill="black")

    img.save(out_path)
    return out_path


def generate_board_texture(base_dir, board_desc, temp_dir):
    """Generates the single main board texture using a guide grid."""
    final_path = os.path.join(base_dir, "board_texture.png")
    if os.path.exists(final_path):
        print(f"   Skipping Board Texture (Exists): {final_path}")
        return final_path

    print(f"\n🎨 Generating Board Texture for '{board_desc}'...")

    guide_path = os.path.join(temp_dir, "guide_grid.png")
    created_guide = create_guide_board(guide_path)

    prompt = (
        f"Top-down view of a full chess board. Material: {board_desc}. "
        f"Strictly follow the checkerboard pattern of the reference image. "
        f"Perfect 8x8 grid, flat texture, no perspective distortion, orthographic view."
    )

    inputs = [created_guide] if created_guide else []
    paths = generate_image_backend(prompt, input_paths=inputs, out_dir=base_dir, n=1)

    if paths:
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(paths[0], final_path)
        print(f"   ✅ Saved {final_path}")
        return final_path

    print("   ❌ Failed to generate board.")
    return None


def generate_piece_sprites(base_dir, pieces_dir, temp_dir, piece_desc, board_desc, board_path):
    print(f"\n♟️  Generating Piece Sprites: '{piece_desc}'...")
    Path(pieces_dir).mkdir(parents=True, exist_ok=True)

    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = ["white", "black"]

    for color in colors:
        for piece in pieces:
            filename = f"{color}_{piece}.png"
            filepath = os.path.join(pieces_dir, filename)

            if os.path.exists(filepath):
                print(f"   Skipping {filename} (exists)")
                continue

            print(f"   Generating {color} {piece}...")

            if color == "white":
                bg_prompt = "Isolated on a pure solid BLACK background."
            else:
                bg_prompt = "Isolated on a pure solid WHITE background."

            prompt = (
                f"Studio product photography of a single {color} chess {piece} piece. "
                f"Style: {piece_desc}. "
                f"Use the provided board image (Image 1) as a material reference only. "
                f"{bg_prompt} "
                f"Full body shot, centered, symmetrical, front view. "
                f"NO smoke, NO dust, NO debris, NO magical effects in this shot. "
                f"Clean, sharp edges for cutout."
            )

            inputs = [board_path] if board_path else []
            paths = generate_image_backend(prompt, input_paths=inputs, out_dir=temp_dir, n=1)

            if paths:
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(paths[0], filepath)
                print(f"   ✅ Saved {filename}")
            else:
                print(f"   ❌ Failed {filename}")


def generate_animations(base_dir, video_dir, pieces_dir, temp_dir, board_path, anim_desc, piece_desc):
    """
    Generates ALL animations, including King Deaths.
    ENFORCED LOGIC: Attacker is always LEFT, Victim is always RIGHT.
    """
    print(f"\n⚔️  Generating Kill Animations: '{anim_desc}'...")

    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = [("white", "black"), ("black", "white")]

    for a_color, v_color in colors:
        for attacker in pieces:
            for victim in pieces:
                if attacker == "king" and victim == "king":
                    continue

                fname = f"{a_color}_{attacker}_takes_{v_color}_{victim}.mp4"
                out = os.path.join(video_dir, fname)

                if os.path.exists(out):
                    print(f"   Skipping {fname} (Exists)")
                    continue

                print(f"   Processing: {a_color} {attacker} vs {v_color} {victim}")

                attacker_path = os.path.join(pieces_dir, f"{a_color}_{attacker}.png")
                victim_path = os.path.join(pieces_dir, f"{v_color}_{victim}.png")

                inputs: List[str] = []
                if board_path:
                    inputs.append(board_path)
                if os.path.exists(attacker_path):
                    inputs.append(attacker_path)
                if os.path.exists(victim_path):
                    inputs.append(victim_path)

                setup_prompt = (
                    f"Cinematic side profile battle shot. "
                    f"On the LEFT side: The {a_color} {attacker} (Aggressor). "
                    f"On the RIGHT side: The {v_color} {victim} (Defender). "
                    f"They are facing each other on the chessboard. "
                    f"Style: {piece_desc}. "
                    f"The {a_color} piece on the left looks powerful and ready to strike. "
                    f"The {v_color} piece on the right looks vulnerable."
                )

                setup_png = generate_image_backend(
                    setup_prompt, input_paths=inputs, out_dir=temp_dir, n=1
                )

                if setup_png:
                    if victim == "king":
                        action = (
                            f"The {a_color} {attacker} (LEFT) delivers a fatal blow. "
                            f"The {v_color} King (RIGHT) falls and dies."
                        )
                    else:
                        action = (
                            f"The {a_color} {attacker} (LEFT) strikes and completely "
                            f"destroys the {v_color} {victim} (RIGHT)."
                        )

                    kill_prompt = (
                        f"{action} "
                        f"Action style: {anim_desc}. "
                        f"The piece on the LEFT MUST WIN. The piece on the RIGHT MUST LOSE. "
                        f"Start from the exact visual composition of the provided image. "
                        f"Violent, cinematic physics destruction of the victim only."
                    )

                    out_vid = generate_video_backend(
                        kill_prompt, image_path=setup_png[0], out_dir=temp_dir
                    )
                    if out_vid:
                        os.rename(out_vid, out)
                        print(f"   🎥 Saved {fname}")

                        print("   (Cooling down for 15s to respect API limits...)")
                        time.sleep(15)


def update_theme_manifest(theme_name):
    manifest_path = os.path.join("assets", "themes.json")
    themes = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                themes = json.load(f)
        except:
            themes = []
    if theme_name not in themes:
        themes.append(theme_name)
        with open(manifest_path, "w") as f:
            json.dump(themes, f)
        print(f"📝 Registered theme '{theme_name}' in assets/themes.json")

# ============================================================================
# 5) MAIN
# ============================================================================

def main():
    global BACKEND

    parser = argparse.ArgumentParser(description="Battle Chess Asset Generator")
    parser.add_argument(
        "--graphics-only",
        action="store_true",
        help="Generate only static graphics (board, pieces), skipping videos.",
    )
    args = parser.parse_args()

    print("=========================================")
    print("   BATTLE CHESS ASSET GENERATOR v4.0   ")
    print("   *** QUOTA-SAFE + AUTO-COMPRESS *** ")
    print("=========================================")

    # --- BACKEND SELECTION ---
    print("\n--- Backend Provider ---")
    print("1. Google / Gemini (Banana + Veo)")
    print("2. OpenAI / ChatGPT (DALL·E 3 + Sora)")
    backend_choice = input("Select backend [default: 1]: ").strip()

    if backend_choice == "2":
        BACKEND = "openai"
        print("   ✅ Using OpenAI / ChatGPT backend")
        if openai_client is None:
            print("   ⚠️ OPENAI client not initialized. Make sure OPENAI_API_KEY is set and 'openai' is installed.")
    else:
        BACKEND = "gemini"
        print("   ✅ Using Google / Gemini backend")

    # --- THEME SELECTION ---
    existing_themes = sorted(
        [d.name for d in Path("assets").iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    default_theme = "obsidian_gothic"

    print("\n--- Available Themes ---")
    for i, t in enumerate(existing_themes, 1):
        print(f"{i}. {t}")
    print("------------------------")

    user_input = input(
        f"Select number or enter new theme name [default: {default_theme}]: "
    ).strip()

    if user_input.isdigit() and 1 <= int(user_input) <= len(existing_themes):
        theme_title = existing_themes[int(user_input) - 1]
    else:
        theme_title = user_input or default_theme

    theme_title = "".join([c for c in theme_title if c.isalnum() or c in ("_", "-")]).lower()

    # --- MODEL SELECTION (VEO / SORA) ---
    current_video_model = os.environ.get("VIDEO_MODEL", "veo-3.1-generate-preview")
    current_image_model = os.environ.get("IMAGE_MODEL", "gemini-3-pro-image-preview")

    print(f"\n--- Model Configuration ---")
    if BACKEND == "gemini":
        print(f"   🖼️  Current Image Model: {current_image_model}")
        print(f"   🎥 Current Video Model: {current_video_model}")

        available_veo_models = [
            "veo-3.1-generate-preview",
            "veo-3.1-fast-generate-preview",
            "veo-3.0-generate-001",
            "veo-3.0-fast-generate-001",
            "veo-2.0-generate-001",
        ]

        print("\n--- Available Veo Models ---")
        for i, m in enumerate(available_veo_models, 1):
            print(f"{i}. {m}")

        model_input = input(
            f"Select number or enter model name [default: {current_video_model}]: "
        ).strip()

        if model_input.isdigit() and 1 <= int(model_input) <= len(available_veo_models):
            selected_video_model = available_veo_models[int(model_input) - 1]
        elif model_input:
            selected_video_model = model_input
        else:
            selected_video_model = current_video_model

        os.environ["VIDEO_MODEL"] = selected_video_model
        print(f"   ✅ Using Video Model: {selected_video_model}")
    else:
        current_openai_image_model = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
        current_openai_video_model = os.environ.get("OPENAI_VIDEO_MODEL", "sora-2")
        print(f"   🖼️  OpenAI Image Model: {current_openai_image_model}")
        print(f"   🎥 OpenAI Video Model: {current_openai_video_model}")
        print("   (Set OPENAI_IMAGE_MODEL / OPENAI_VIDEO_MODEL env vars to override.)")

    # 1. UPDATE MANIFEST IMMEDIATELY
    base_dir = os.path.join("assets", theme_title)
    pieces_dir = os.path.join(base_dir, "pieces")
    video_dir = os.path.join(base_dir, "videos")
    temp_dir = os.path.join(base_dir, "temp")

    for d in [base_dir, pieces_dir, video_dir, temp_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Asset Directory: {base_dir}")
    update_theme_manifest(theme_title)

    # --- PROMPT MANAGEMENT ---
    prompts_file = os.path.join(base_dir, "prompts.json")
    saved_prompts = {}
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, "r") as f:
                saved_prompts = json.load(f)
            print(f"   ℹ️ Loaded saved prompts from {prompts_file}")
        except:
            pass

    default_board = saved_prompts.get(
        "board_desc", "obsidian and white marble, ancient runes, glowing cracks"
    )
    board_desc = (
        input(f"Board Style [default: {default_board[:20]}...]: ").strip() or default_board
    )

    default_piece = saved_prompts.get(
        "piece_desc", "animated stone statues, glowing eyes, dark fantasy armor"
    )
    piece_desc = (
        input(f"Piece Style [default: {default_piece[:20]}...]: ").strip() or default_piece
    )

    default_anim = saved_prompts.get(
        "anim_desc", "violent shattering, magic explosions, debris flying, screen shake"
    )
    anim_desc = (
        input(f"Animation Style [default: {default_anim[:20]}...]: ").strip() or default_anim
    )

    try:
        with open(prompts_file, "w") as f:
            json.dump(
                {
                    "board_desc": board_desc,
                    "piece_desc": piece_desc,
                    "anim_desc": anim_desc,
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"   ⚠️ Could not save prompts: {e}")

    # 2. Execution
    board_png = generate_board_texture(base_dir, board_desc, temp_dir)

    if board_png:
        generate_piece_sprites(base_dir, pieces_dir, temp_dir, piece_desc, board_desc, board_path=board_png)
        if not args.graphics_only:
            generate_animations(
                base_dir,
                video_dir,
                pieces_dir,
                temp_dir,
                board_path=board_png,
                anim_desc=anim_desc,
                piece_desc=piece_desc,
            )
        else:
            print("\n🚫 Skipping animation generation (--graphics-only used)")
    else:
        print("❌ Board generation failed. Exiting.")

    # 3. Compression Step
    if os.path.exists(video_dir) and not args.graphics_only:
        compress_new_assets(video_dir)

    print("\n✅ GENERATION COMPLETE!")
    print("   Run 'python -m http.server' and open the updated game.html")


if __name__ == "__main__":
    main()
