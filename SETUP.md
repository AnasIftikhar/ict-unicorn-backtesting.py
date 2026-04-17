# ICT Unicorn — RunPod Serverless Optimizer

Run full parameter optimization on a 32-core RunPod CPU pod. No Docker needed locally.

---

## How It Works

```
Push code to GitHub
    → GitHub Actions builds Docker image
    → Pushes image to GitHub Container Registry (GHCR)
    → RunPod pulls image when you trigger a job
    → 32-core pod downloads data, runs optimization, uploads CSV to Google Drive
    → Results returned via API response
```

---

## One-Time Setup

### 1. Push This Repo to GitHub

1. Create a new GitHub repository (empty, no README)
2. Push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

### 2. GitHub Actions Auto-Builds the Docker Image

After your first push, go to:
`GitHub → Your Repo → Actions`

You will see the **Build and Push Docker Image** workflow running.
When it finishes, your image is live at:
```
ghcr.io/YOUR_USERNAME/YOUR_REPO:latest
```

### 3. Make the Image Public (Required for RunPod)

Go to:
`GitHub → Your Profile → Packages → your-repo → Package Settings → Change visibility → Public`

### 4. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select a **CPU pod** (e.g. 32 vCPU)
4. Set **Container Image** to: `ghcr.io/YOUR_USERNAME/YOUR_REPO:latest`
5. Set **Max Workers** as needed
6. Copy the **Endpoint ID** — you will need it

### 5. Set Up Google Drive (Optional but Recommended)

To have results uploaded directly to your Google Drive:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Drive API**
3. Create a **Service Account** → Download the JSON key
4. Share a Google Drive folder with the service account email
5. Base64-encode the JSON key:
   ```bash
   base64 -w 0 service_account.json
   ```
6. In RunPod Endpoint settings → **Environment Variables**:
   ```
   GOOGLE_SERVICE_ACCOUNT_JSON = <paste base64 string here>
   ```
7. Set `drive_folder_id` in `trigger_job.py` to your Drive folder ID
   (the long ID from the folder's URL)

---

## Running a Job

### Install trigger script dependencies (one-time)
```bash
pip install requests
```

### Set credentials
```bash
export RUNPOD_API_KEY=rp_your_api_key_here
export RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

### Run
```bash
python trigger_job.py
```

The script will:
1. Submit the job to RunPod
2. Poll every 15 seconds
3. Print results when done
4. Save CSV locally (if Drive not configured) or print the Drive URL

### Custom parameters
Edit `DEFAULT_PAYLOAD` in `trigger_job.py`, or create a JSON config:
```json
{
  "symbol": "BTCUSDT.P",
  "interval": "5m",
  "days_back": 180,
  "tpsl_methods": ["Unicorn", "Dynamic", "Fixed"],
  "metric_mode": "advanced"
}
```
Then run:
```bash
python trigger_job.py --config my_params.json
```

---

## Updating the Code

Just push to GitHub:
```bash
git add .
git commit -m "Update strategy"
git push
```

GitHub Actions rebuilds the Docker image automatically.
RunPod picks up the new image on the next job.

---

## Files

| File | Description |
|------|-------------|
| `unicorn.py` | ICT Unicorn strategy (unchanged) |
| `optimize.py` | Parameter optimizer (unchanged) |
| `Binance-Vision.py` | Data downloader (unchanged) |
| `handler.py` | RunPod serverless entry point |
| `trigger_job.py` | Local script to submit and poll jobs |
| `Dockerfile` | Container definition |
| `requirements.txt` | Python dependencies |
| `.github/workflows/build-push.yml` | Auto-build on push |
