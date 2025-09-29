"""tryon.py
Run virtual try-on for a single product URL over a directory of face images.
This script prefers Playwright async approach; falls back to Selenium if playwright is not available.
"""
import argparse
import os
from pathlib import Path
import asyncio


async def run_playwright(url, sku, face_dir, output_dir):
    from playwright.async_api import async_playwright
    from pathlib import Path
    faces = [f for f in os.listdir(face_dir) if Path(face_dir, f).is_file()]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(permissions=[], ignore_https_errors=True)
        page = await context.new_page()
        for face in faces:
            face_path = str(Path(face_dir) / face)
            await page.goto(url, timeout=30000)
            await page.wait_for_timeout(3000)
            tryon_btn = page.locator("button.fittingbox-trigger__trigger-btn")
            await tryon_btn.click()
            await page.wait_for_timeout(2000)
            iframe = page.frame_locator("#fitmixWidgetIframeContainer")
            if await iframe.locator("input[nvi-selenium='no-camera-image-upload-input']").is_visible():
                await iframe.locator("input[nvi-selenium='no-camera-image-upload-input']").set_input_files(face_path)
            elif await iframe.locator("input[name='vtoAddImage']").is_visible():
                await iframe.locator("input[name='vtoAddImage']").set_input_files(face_path)
            else:
                print(f"No upload input found for {url}")
                continue
            await page.wait_for_timeout(6000)
            out = Path(output_dir) / f"{Path(face).stem}_{sku}.png"
            await page.screenshot(path=str(out), full_page=True)
            print(f"Saved {out}")
        await browser.close()


def main(url, sku, face_dir, output_dir):
    # Try playwright first
    try:
        asyncio.run(run_playwright(url, sku, face_dir, output_dir))
        return
    except Exception:
        print("Playwright run failed or not installed; fallback to Selenium")

    # Selenium fallback
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
    except Exception:
        raise RuntimeError("Neither playwright nor selenium available. Install one to use tryon.")

    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    os.makedirs(output_dir, exist_ok=True)
    faces = [f for f in os.listdir(face_dir) if Path(face_dir, f).is_file()]
    for face in faces:
        face_path = str(Path(face_dir) / face)
        try:
            driver.get(url)
            driver.implicitly_wait(3)
            tryon_btn = driver.find_element(By.CSS_SELECTOR, "button.fittingbox-trigger__trigger-btn")
            tryon_btn.click()
            driver.switch_to.frame(driver.find_element(By.ID, "fitmixWidgetIframeContainer"))
            try:
                upload_input = driver.find_element(By.CSS_SELECTOR, "input[nvi-selenium='no-camera-image-upload-input']")
            except Exception:
                upload_input = driver.find_element(By.CSS_SELECTOR, "input[name='vtoAddImage']")
            upload_input.send_keys(face_path)
            driver.switch_to.default_content()
            out = Path(output_dir) / f"{Path(face).stem}_{sku}.png"
            driver.save_screenshot(str(out))
            print(f"Saved {out}")
        except Exception as e:
            print(f"Error for {face}: {e}")
    driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--sku', required=True)
    parser.add_argument('--face-dir', default='faces')
    parser.add_argument('--out', default='output')
    args = parser.parse_args()
    main(args.url, args.sku, args.face_dir, args.out)
