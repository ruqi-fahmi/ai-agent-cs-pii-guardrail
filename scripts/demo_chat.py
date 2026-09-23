"""
Demo end-to-end tanpa UI: kirim beberapa pesan ke agent dan tampilkan jawabannya.
Butuh NER Service hidup + GEMINI_API_KEY (atau GOOGLE_API_KEY).

    set GUARDRAIL_DEBUG=1          # opsional: lihat teks yang dikirim ke Gemini
    python scripts/demo_chat.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from google.adk.runners import InMemoryRunner   # noqa: E402
from google.genai import types                  # noqa: E402

from cs_agent.agent import app                  # noqa: E402

MESSAGES = [
    "Halo kak, nama saya Budi Santoso, NIK 3201234567890123. Internet di rumah saya "
    "Jalan Sudirman No. 10 Jakarta mati sejak kemarin.",
    "Kalau mau dihubungi teknisi, nomor saya 0812-3456-7890 atau email budi.s@gmail.com",
    "Oke, tadi alamat saya yang mana ya kak?",
]


async def main() -> None:
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name=app.name, user_id="demo")
    for msg in MESSAGES:
        print(f"\nUSER  : {msg}")
        async for event in runner.run_async(
                user_id="demo", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=msg)])):
            if event.is_final_response() and event.content and event.content.parts:
                print(f"AGENT : {''.join(p.text or '' for p in event.content.parts).strip()}")

    # Bukti plugin bekerja: yang tersimpan di riwayat sesi sudah tanpa PII.
    stored = await runner.session_service.get_session(
        app_name=app.name, user_id="demo", session_id=session.id)
    print("\n--- Pesan user yang TERSIMPAN di riwayat sesi ---")
    for ev in stored.events:
        if ev.author == "user" and ev.content and ev.content.parts:
            print(f"  {''.join(p.text or '' for p in ev.content.parts)}")


if __name__ == "__main__":
    asyncio.run(main())
