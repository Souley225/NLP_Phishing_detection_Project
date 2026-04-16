"""
End-to-end Playwright tests for the deployed PhishGuard HF Space.
URL: https://sallsou-nlp-phishing-detection.hf.space
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "https://sallsou-nlp-phishing-detection.hf.space"
SCREENSHOTS = Path(__file__).parent / "screenshots"
SCREENSHOTS.mkdir(exist_ok=True)
TIMEOUT = 90_000


def _body(page) -> str:
    return page.inner_text("body").lower()


def _has_result_card(page) -> bool:
    """Markers that only appear inside a real scan result card."""
    b = _body(page)
    return "classification" in b and "url length" in b


def _wait_result(page, timeout_s=30) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _has_result_card(page):
            return True
        time.sleep(1.5)
    return False


def test_all():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # T1: Page loads
        print("\n[T1] Page loads...")
        page.goto(BASE_URL, timeout=TIMEOUT, wait_until="networkidle")
        page.screenshot(path=str(SCREENSHOTS / "t1_loaded.png"))
        t1 = "PhishGuard" in page.title() or "Streamlit" in page.title()
        results.append(("T1 Page loads", t1))
        print(f"     title={page.title()!r}  {'PASS' if t1 else 'FAIL'}")

        # T2: Model ONLINE (st.cache_resource loads on first render)
        print("\n[T2] Model loads and shows ONLINE...")
        online = False
        for attempt in range(10):
            time.sleep(3)
            page.reload(wait_until="networkidle")
            if "online" in _body(page):
                online = True
                break
            print(f"     attempt {attempt+1}/10 — not ONLINE yet")
        page.screenshot(path=str(SCREENSHOTS / "t2_status.png"))
        results.append(("T2 Model ONLINE", online))
        print(f"     {'PASS' if online else 'FAIL'}")

        # T3: Empty URL shows warning
        print("\n[T3] Empty URL submit shows warning...")
        try:
            page.locator("button").filter(has_text="SCAN").first.click(timeout=8000)
            time.sleep(2)
            page.screenshot(path=str(SCREENSHOTS / "t3_empty.png"))
            t3 = "please enter" in _body(page)
            results.append(("T3 Empty URL warning", t3))
            print(f"     {'PASS' if t3 else 'FAIL'}")
        except Exception as e:
            results.append(("T3 Empty URL warning", False))
            print(f"     FAIL ({e})")

        # T4: Scan legitimate URL
        print("\n[T4] Scan https://wikipedia.org...")
        try:
            inp = page.get_by_role("textbox").first
            inp.click()
            inp.fill("https://wikipedia.org")
            time.sleep(0.5)
            page.locator("button").filter(has_text="SCAN").first.click()
            got = _wait_result(page, timeout_s=30)
            page.screenshot(path=str(SCREENSHOTS / "t4_legit.png"), full_page=True)
            results.append(("T4 Legitimate URL scan", got))
            b = _body(page)
            verdict = "PHISHING" if "phishing detected" in b else "LEGITIMATE"
            print(f"     {'PASS' if got else 'FAIL'} — classified as {verdict}")
        except Exception as e:
            results.append(("T4 Legitimate URL scan", False))
            print(f"     FAIL ({e})")

        # T5: Scan phishing URL
        print("\n[T5] Scan http://paypal-secure.tk/login.php...")
        try:
            inp = page.get_by_role("textbox").first
            inp.click()
            inp.fill("http://paypal-secure.tk/login.php")
            time.sleep(0.5)
            page.locator("button").filter(has_text="SCAN").first.click()
            got = _wait_result(page, timeout_s=30)
            page.screenshot(path=str(SCREENSHOTS / "t5_phishing.png"), full_page=True)
            results.append(("T5 Phishing URL scan", got))
            b = _body(page)
            verdict = "PHISHING (correct)" if "phishing detected" in b else "LEGITIMATE (wrong)"
            print(f"     {'PASS' if got else 'FAIL'} — classified as {verdict}")
        except Exception as e:
            results.append(("T5 Phishing URL scan", False))
            print(f"     FAIL ({e})")

        # T6: History/stats updated
        print("\n[T6] History and session stats updated...")
        try:
            b = _body(page)
            t6 = "wikipedia" in b or "paypal" in b
            page.screenshot(path=str(SCREENSHOTS / "t6_history.png"), full_page=True)
            results.append(("T6 History updated", t6))
            print(f"     {'PASS' if t6 else 'FAIL'}")
        except Exception as e:
            results.append(("T6 History updated", False))
            print(f"     FAIL ({e})")

        # T7: CLEAR resets result card
        print("\n[T7] CLEAR button removes result card...")
        try:
            page.locator("button").filter(has_text="CLEAR").first.click(timeout=5000)
            time.sleep(2)
            t7 = not _has_result_card(page)
            page.screenshot(path=str(SCREENSHOTS / "t7_cleared.png"))
            results.append(("T7 CLEAR resets result", t7))
            print(f"     {'PASS' if t7 else 'FAIL'}")
        except Exception as e:
            results.append(("T7 CLEAR resets result", False))
            print(f"     FAIL ({e})")

        browser.close()

    print("\n" + "=" * 60)
    print("  PHISHGUARD E2E TEST RESULTS")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("=" * 60)
    print(f"  {passed}/{len(results)} tests passed")
    print(f"  Screenshots: {SCREENSHOTS}")
    print("=" * 60)
    return passed, len(results)


if __name__ == "__main__":
    passed, total = test_all()
    exit(0 if passed == total else 1)
