# Nexus

Nexus is a full-stack personal finance app that helps users:
- upload Google Pay activity HTML
- parse and categorize transactions
- explore spending with dashboards and charts
- chat with an AI financial advisor

The project has two apps:
- `client/`: Next.js frontend (Clerk auth, dashboards, upload flow)
- `backend/`: FastAPI backend (parsing, categorization, advisor API)

## Quick Start

Run this from repo root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
cd client && npm install && cd ..
```

Create env files:
- copy `backend/.env.example` to `.env` (repo root)
- copy `client/.env.example` to `client/.env.local`

Start both apps in two terminals:

```bash
# terminal 1
python backend/run.py

# terminal 2
cd client
npm run dev
```

Open:
- frontend: `http://localhost:3000` (or next available port)
- backend: `http://127.0.0.1:8000`

## Tech Stack

### Frontend
- Next.js 15 (App Router + Turbopack)
- React 19
- TypeScript
- Clerk (`@clerk/nextjs`) for authentication
- Recharts for data visualizations

### Backend
- FastAPI + Uvicorn
- Python 3.10+
- LangChain + Hugging Face for advisor/categorization fallbacks
- BeautifulSoup for Google Pay HTML parsing
- Clerk backend SDK for token verification

## Project Structure

```text
Nexus/
  backend/
    app/
      api.py
      main.py
      config.py
      services/
        advisor.py
        categorization.py
        clerk_auth.py
        gpay_parser.py
    requirements.txt
    run.py
    .env.example
  client/
    app/
      upload-activity/
      dashboard/
      advisor/
    package.json
    .env.example
  README.md
```

## Local Development

### 1) Prerequisites
- Node.js 20+
- Python 3.10+
- npm
- Clerk account/app
- (Optional) Hugging Face token for better AI features

### 2) Clone and install

From repo root:

```bash
# backend deps
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# frontend deps
cd client
npm install
cd ..
```

### 3) Environment variables

### Backend (`.env` at repo root)

Copy values based on `backend/.env.example`:

```env
CLERK_SECRET_KEY=sk_test_or_sk_live_key
HF_API_TOKEN=hf_token_optional
HF_MODEL_ID=google/flan-t5-base
FRONTEND_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Frontend (`client/.env.local`)

Copy values based on `client/.env.example`:

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_or_pk_live_key
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Important: `CLERK_SECRET_KEY` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` must belong to the same Clerk application.

### 4) Run the apps

### Start backend (from repo root)

```bash
python backend/run.py
```

Backend default URL: `http://127.0.0.1:8000`

### Start frontend (new terminal)

```bash
cd client
npm run dev
```

Frontend default URL: `http://localhost:3000` (or next available port)

## Core API Endpoints

- `GET /api/health`
- `POST /api/process-transactions`
- `POST /api/financial-advisor`
- `GET /api/protected`

## Deployment

Recommended setup:
- Frontend: Vercel
- Backend: Render (or Railway/Fly)

## Backend deployment (Render example)

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `CLERK_SECRET_KEY`
  - `HF_API_TOKEN` (optional)
  - `HF_MODEL_ID` (optional)
  - `FRONTEND_ORIGINS=https://your-frontend-domain`

## Frontend deployment (Vercel example)

- Root directory: `client`
- Environment variables:
  - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
  - `NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain`
  - `CLERK_SECRET_KEY` (recommended for Clerk middleware/server features)

## Post-deploy checklist

- Verify backend health endpoint responds
- Verify Clerk sign-in works on deployed domain
- Test HTML upload end-to-end
- Confirm advisor chat can reach backend API
- Confirm CORS allows your frontend domain

## Security Notes

- Never commit `.env` or `.env.local`
- Rotate secrets immediately if exposed
- Keep Clerk test keys in dev and live keys in production

## Contribution Workflow

Typical flow when working with forks:

```bash
git checkout -b feature/your-change
git add .
git commit -m "Describe your change"
git push -u origin feature/your-change
```

Then open a Pull Request from your fork branch to `main`.

## Troubleshooting

### 401 Unauthorized from backend
- Usually Clerk key/session mismatch.
- Re-login after changing Clerk instance.
- Ensure frontend and backend use keys from the same Clerk app.

### CORS errors
- Set backend `FRONTEND_ORIGINS` to your actual frontend domain(s).

### Frontend still calls localhost after deploy
- Set `NEXT_PUBLIC_API_BASE_URL` in frontend deployment environment.
- Redeploy frontend after changing env vars.
