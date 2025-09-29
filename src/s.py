import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

# --- Config ---
glasses = {
    "url": "https://www.americasbest.com/archer--avery-wc-2020-2/p/286452",
    "sku": "352784"
}
face_dir = r'SCUT-FBP5500\Images'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get first face image
face_images = [f for f in os.listdir(face_dir) if os.path.isfile(os.path.join(face_dir, f))]
if not face_images:
    raise ValueError("No face images found in the 'faces' folder.")
face_img = face_images[0]
face_path = os.path.abspath(os.path.join(face_dir, face_img))

# --- Playwright Script ---
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Set to True for headless mode
    context = browser.new_context(
        permissions=[],  # Do not grant camera permissions
        ignore_https_errors=True,
        viewport={'width': 1280, 'height': 800}
    )
    page = context.new_page()

    try:
        # Go to product page
        page.goto(glasses["url"], timeout=30000)
        page.wait_for_timeout(5000)

        # Click the Try-On button
        tryon_btn = page.locator("button.fittingbox-trigger__trigger-btn")
        tryon_btn.click()
        page.wait_for_timeout(5000)

        # Switch to iframe
        iframe = page.frame_locator("#fitmixWidgetIframeContainer")

        # Upload the image
        if iframe.locator("input[nvi-selenium='no-camera-image-upload-input']").is_visible():
            iframe.locator("input[nvi-selenium='no-camera-image-upload-input']").set_input_files(face_path)
        elif iframe.locator("input[name='vtoAddImage']").is_visible():
            iframe.locator("input[name='vtoAddImage']").set_input_files(face_path)
        else:
            raise Exception("No upload input found in try-on iframe.")

        # Wait for the result to render
        page.wait_for_timeout(8000)

        # Screenshot
        screenshot_path = os.path.join(output_dir, f"{Path(face_img).stem}_{glasses['sku']}.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Saved result to {screenshot_path}")

    except Exception as e:
        print(f"Error processing {face_img} with {glasses['sku']}: {e}")

    browser.close()
