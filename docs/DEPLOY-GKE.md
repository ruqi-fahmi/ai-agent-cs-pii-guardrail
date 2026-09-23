# Deploy ke Google Kubernetes Engine (GKE)

Dua container → Artifact Registry → satu cluster GKE Autopilot.

```
            internet (tidak dibuka)            ┌─────────── namespace pii-guard ───────────┐
 laptop ── kubectl port-forward ──► Service cs-agent (ClusterIP) ──► Pod cs-agent           │
                                   │                                    │ HTTP              │
                                   │                                    ▼                   │
                                   │           Service ner-service (ClusterIP) ──► Pod ×2   │
                                   └────────────────────────────────────────────────────────┘
                                                        Pod cs-agent ──HTTPS──► Gemini API
```

## 0. Prasyarat

- Akun GCP dengan billing aktif (kredit trial cukup)
- `gcloud` CLI terpasang: `winget install Google.CloudSDK`
- `kubectl` (sudah ada lewat Docker Desktop) + plugin auth GKE:
  `gcloud components install gke-gcloud-auth-plugin`
- Docker lokal **tidak wajib** — image dibangun dengan Cloud Build (langkah 3)

## 1. Variabel

```powershell
$PROJECT = "<id-project-gcp>"
$REGION  = "asia-southeast2"          # Jakarta
$REPO    = "pii-guard"
$REGISTRY = "$REGION-docker.pkg.dev/$PROJECT/$REPO"
```

## 2. Aktifkan API & buat registry

```powershell
gcloud auth login
gcloud config set project $PROJECT
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

## 3. Build & push image — lewat Cloud Build (tanpa Docker di laptop)

Cloud Build membaca `Dockerfile` yang sama, membangun image di server Google (Linux amd64,
sama dengan node GKE), lalu langsung menyimpannya ke Artifact Registry. Laptop hanya
mengunggah kode sumber (beberapa MB) — tidak perlu Docker Desktop.

```powershell
gcloud services enable cloudbuild.googleapis.com
gcloud builds submit ner_service --tag "$REGISTRY/ner-service:1.3.0"
gcloud builds submit agent       --tag "$REGISTRY/cs-agent:1.1.0"
```

Alternatif bila Docker tersedia secara lokal:
```powershell
docker build --platform linux/amd64 -t "$REGISTRY/ner-service:1.3.0" ner_service
docker build --platform linux/amd64 -t "$REGISTRY/cs-agent:1.1.0" agent
docker push "$REGISTRY/ner-service:1.3.0"
docker push "$REGISTRY/cs-agent:1.1.0"
```

## 4. Buat cluster (Autopilot)

Autopilot: Google yang mengelola node; kita hanya membayar resource yang di-*request* pod.

```powershell
gcloud container clusters create-auto pii-guard-cluster --region $REGION
gcloud container clusters get-credentials pii-guard-cluster --region $REGION
```

## 5. Deploy

```powershell
kubectl apply -f k8s/namespace.yaml

# API key sebagai Secret — tidak pernah masuk image atau git
kubectl -n pii-guard create secret generic gemini-api-key --from-literal=GOOGLE_API_KEY="<api-key>"

# Ganti placeholder REGISTRY di manifest lalu apply
(Get-Content k8s/ner-service.yaml) -replace 'REGISTRY', $REGISTRY | kubectl apply -f -
(Get-Content k8s/cs-agent.yaml)    -replace 'REGISTRY', $REGISTRY | kubectl apply -f -
kubectl apply -f k8s/ner-podmonitoring.yaml   # metrik /metrics -> Cloud Monitoring

kubectl -n pii-guard get pods -w       # tunggu semua READY 1/1
```

## 6. Uji

```powershell
# NER Service dari dalam cluster
kubectl -n pii-guard run curl --rm -it --image=curlimages/curl --restart=Never -- `
  curl -s -X POST http://ner-service/ner -H "Content-Type: application/json" `
  -d '{\"text\":\"Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta\"}'

# Halaman demo agent
kubectl -n pii-guard port-forward svc/cs-agent 8000:80
# buka http://127.0.0.1:8000

# Resource aktual di cluster (bahan dokumentasi poin F)
kubectl -n pii-guard top pods
```

## 7. Bersih-bersih (supaya tagihan berhenti)

```powershell
gcloud container clusters delete pii-guard-cluster --region $REGION
gcloud artifacts repositories delete $REPO --location=$REGION
```

## Keputusan desain

| Keputusan | Alasan |
|---|---|
| NER = ClusterIP | Menerima teks ber-PII → tidak boleh dijangkau dari luar cluster |
| NER 2 replika + readiness probe | Guardrail fail-closed: kalau NER mati, agent menolak semua pesan → NER harus tahan satu pod mati / rolling update |
| Agent 1 replika | Sesi chat ADK disimpan di memori pod; >1 replika perlu session store bersama (mis. database) |
| Agent tidak dibuka ke internet | Halaman demo tanpa autentikasi dan memakai API key berbayar |
| API key di Secret | Tidak ikut ter-bake di image atau ter-commit ke git |
| Resource requests/limits | Dari pengukuran `docs/resource-performance.md`, bukan tebakan |
