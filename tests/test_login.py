"""Gerbang login halaman demo.

Gerbang ini hanya menyala di deployment publik (DEMO_PASSWORD diisi). Yang diuji di
sini: gerbangnya benar-benar MATI saat variabel itu kosong, dan saat menyala tidak ada
jalan memutar ke endpoint yang memanggil Gemini.
"""
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client():
    with TestClient(app=server.app) as c:
        yield c


def _nyalakan(monkeypatch, password="rahasia-uji"):
    """Nyalakan gerbang seperti di VPS, termasuk token turunannya."""
    import hashlib
    monkeypatch.setattr(server, "DEMO_PASSWORD", password)
    monkeypatch.setattr(server, "_TOKEN",
                        hashlib.sha256(f"ai-guardrail-demo:{password}".encode()).hexdigest())
    # http polos di TestClient: cookie Secure tidak akan pernah dikirim balik.
    monkeypatch.setattr(server, "COOKIE_SECURE", False)
    return password


def test_gerbang_mati_saat_password_kosong(client, monkeypatch):
    """Default repo = tanpa login, supaya demo lokal dan docker compose tidak berubah."""
    monkeypatch.setattr(server, "DEMO_PASSWORD", "")
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200


def test_halaman_dialihkan_ke_login(client, monkeypatch):
    _nyalakan(monkeypatch)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_api_membalas_json_bukan_html(client, monkeypatch):
    """Halaman memanggil fetch(); balasan HTML akan bikin r.json() gagal parsing."""
    _nyalakan(monkeypatch)
    r = client.post("/api/session")
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")
    assert "detail" in r.json()


def test_form_login_selalu_terbuka(client, monkeypatch):
    """Kalau /login ikut dijaga, tidak ada cara masuk sama sekali."""
    _nyalakan(monkeypatch)
    r = client.get("/login")
    assert r.status_code == 200
    # Halaman sengaja tanpa teks, jadi yang dicek elemennya — bukan kata-katanya.
    assert 'id="pw"' in r.text and 'id="form"' in r.text


def test_password_salah_ditolak(client, monkeypatch):
    _nyalakan(monkeypatch)
    r = client.post("/api/login", json={"password": "tebakan-ngawur"})
    assert r.status_code == 401
    assert server.COOKIE_NAME not in r.cookies


def test_password_benar_membuka_semuanya(client, monkeypatch):
    pw = _nyalakan(monkeypatch)
    r = client.post("/api/login", json={"password": pw})
    assert r.status_code == 200
    # HttpOnly: cookie tidak boleh terbaca JavaScript kalau ada XSS di halaman.
    assert "httponly" in r.headers["set-cookie"].lower()
    # TestClient menyimpan cookie-nya, jadi permintaan berikutnya harus lolos.
    assert client.get("/", follow_redirects=False).status_code == 200
    assert client.get("/api/status").status_code == 200


def test_keluar_menutup_kembali(client, monkeypatch):
    pw = _nyalakan(monkeypatch)
    client.post("/api/login", json={"password": pw})
    client.post("/api/logout")
    assert client.get("/", follow_redirects=False).status_code == 302


def test_cookie_lama_mati_saat_password_diganti(client, monkeypatch):
    """Ganti password = semua sesi lama ikut batal, tanpa menyimpan daftar sesi."""
    pw = _nyalakan(monkeypatch)
    client.post("/api/login", json={"password": pw})
    assert client.get("/", follow_redirects=False).status_code == 200
    _nyalakan(monkeypatch, "password-yang-baru")
    assert client.get("/", follow_redirects=False).status_code == 302
