"""
Test end-to-end lewat BROWSER sungguhan (Playwright): mengetik di halaman demo seperti
manusia, lalu memeriksa bahwa PII diwarnai dan payload ke LLM bersih.

Stack yang dijalankan: NER Service asli (model asli) + server demo dengan
CS_AGENT_FAKE_LLM=1 — semua guardrail asli, hanya Gemini diganti LLM palsu supaya
gratis, cepat, konsisten, dan bisa jalan di CI tanpa API key.

    pytest tests/e2e                              # Windows: memakai Microsoft Edge terpasang
    E2E_BROWSER_CHANNEL= pytest tests/e2e         # Chromium bawaan Playwright (CI)
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
NIK = "3201234567890123"
CHANNEL = os.getenv("E2E_BROWSER_CHANNEL", "msedge" if sys.platform == "win32" else "") or None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ok(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{url} tidak siap dalam {timeout} detik")


def start(args, env, url):
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", *args], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ok(url)
    except Exception:
        proc.kill()
        raise
    return proc


@pytest.fixture(scope="module")
def stack():
    """NER Service asli + server demo (LLM palsu), plus satu server demo yang sengaja
    diarahkan ke NER yang mati untuk menguji fail-closed."""
    ner_port, ok_port, dead_port = free_port(), free_port(), free_port()
    env = {**os.environ, "CS_AGENT_FAKE_LLM": "1", "GUARDRAIL_FAIL_MODE": "closed"}
    procs = [start(["app:app", "--app-dir", "ner_service", "--port", str(ner_port)], env,
                   f"http://127.0.0.1:{ner_port}/health")]
    procs.append(start(["server:app", "--app-dir", "agent", "--port", str(ok_port)],
                       {**env, "NER_SERVICE_URL": f"http://127.0.0.1:{ner_port}"},
                       f"http://127.0.0.1:{ok_port}/api/status"))
    procs.append(start(["server:app", "--app-dir", "agent", "--port", str(dead_port)],
                       {**env, "NER_SERVICE_URL": f"http://127.0.0.1:{free_port()}"},
                       f"http://127.0.0.1:{dead_port}/api/status"))
    yield {"ok": f"http://127.0.0.1:{ok_port}", "ner_down": f"http://127.0.0.1:{dead_port}"}
    for p in procs:
        p.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        b = pw.chromium.launch(channel=CHANNEL)
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1500, "height": 950})
    p = ctx.new_page()
    yield p
    ctx.close()


def send(page, text):
    page.fill("#input", text)
    page.click("#send")
    # tunggu jawaban agent terakhir selesai (bukan indikator mengetik)
    page.wait_for_function(
        "() => { const b = [...document.querySelectorAll('.agent .bubble')].pop();"
        " return b && !b.classList.contains('typing'); }", timeout=30000)
    return page.locator(".agent .bubble").last.inner_text()


def test_pii_highlighted_and_payload_clean(stack, page):
    page.goto(stack["ok"])
    reply = send(page, f"Halo, saya Budi Santoso, NIK {NIK}, rumah di Jalan Sudirman No 10 Jakarta")

    # 1. pesan asli diwarnai per jenis
    assert page.locator(".user mark.NIK").count() == 1
    assert page.locator(".user mark.PERSON").count() >= 1
    assert page.locator(".user mark.ADDRESS").count() >= 1
    # 2. payload yang dikirim ke LLM: token ada, nilai asli TIDAK ada
    payload = page.locator("aside .payload").first.inner_text()
    assert "[REDACT_NIK]" in payload and "[REDACT_NAMA]" in payload
    assert NIK not in payload and "Budi" not in payload and "Sudirman" not in payload
    assert page.get_by_text("Sama persis dengan yang tersimpan di riwayat sesi").is_visible()
    # 3. LLM palsu membeo token -> after_model_callback harus merapikannya
    assert "[REDACT_" not in reply
    assert "yang Bapak/Ibu berikan" in reply


def test_message_without_pii_is_untouched(stack, page):
    page.goto(stack["ok"])
    send(page, "Tagihan saya Rp1508000 kok naik ya?")
    assert page.locator(".user mark").count() == 0
    assert page.get_by_text("Tidak ada data pribadi").is_visible()
    assert "Rp1508000" in page.locator("aside .payload").first.inner_text()


def test_sample_chip_sends_message(stack, page):
    page.goto(stack["ok"])
    page.locator(".samples button").first.click()
    page.wait_for_selector(".agent .bubble:not(.typing)", timeout=30000)
    assert page.locator(".user .bubble").count() == 1
    assert page.locator(".user mark").count() >= 1


def test_fail_closed_when_ner_down(stack, page):
    page.goto(stack["ner_down"])
    page.wait_for_function("() => document.querySelector('#nerVer').textContent === 'offline'")
    reply = send(page, f"Saya Budi, NIK {NIK}")
    assert "sistem pengaman data kami sedang tidak tersedia" in reply
    assert page.get_by_text("tidak dipanggil").is_visible()
    # tidak ada payload ke LLM sama sekali
    assert page.locator("aside .payload").count() == 0
