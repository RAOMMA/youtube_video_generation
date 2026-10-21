@echo off
rem Downloads the LTX-2.3 lip-sync models. Safe to stop and re-run: every file resumes where it left off.
set M=%~dp0..\ComfyUI\models
if not exist "%M%\latent_upscale_models" mkdir "%M%\latent_upscale_models"

echo [1/4] LTX-2.3 model (16.5 GB)
curl.exe -L -C - --retry 20 --retry-delay 5 -o "%M%\unet\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf" "https://huggingface.co/QuantStack/LTX-2.3-GGUF/resolve/main/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"

echo [2/4] Text projection (2.2 GB)
curl.exe -L -C - --retry 20 --retry-delay 5 -o "%M%\text_encoders\ltx-2.3_text_projection_bf16.safetensors" "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

echo [3/4] Spatial upscaler (0.9 GB)
curl.exe -L -C - --retry 20 --retry-delay 5 -o "%M%\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

echo [4/4] Gemma 3 12B text encoder (12.3 GB)
curl.exe -L -C - --retry 20 --retry-delay 5 -o "%M%\text_encoders\gemma_3_12B_it_fp8_scaled.safetensors" "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"

echo.
echo All LTX downloads finished.
pause
