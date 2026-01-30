# 🚀 EvalOps-Lite — GenAI Evaluation & Prediction Service

**Production-ready GenAI inference + EvalOps microservice built with FastAPI, designed to demonstrate ML systems thinking, API design, deployment, evaluation workflows, and MLOps fundamentals. Built for extreme shortlisting (top 0.005%), not a toy demo.**

✅ Live ✅ Containerized ✅ CI/CD-ready ✅ Recruiter-friendly Swagger UI ✅ No local setup required

---

## 🔗 Live Demo (Railway — No Setup)
Base URL: https://evalops-lite-production.up.railway.app
Swagger UI: https://evalops-lite-production.up.railway.app/docs
OpenAPI Spec: https://evalops-lite-production.up.railway.app/openapi.json

---

## 🧠 What This Service Does

EvalOps-Lite is a GenAI inference + evaluation microservice that exposes clean, testable APIs for health monitoring, model registry introspection, and GenAI prediction (text → inference output). This mirrors real-world ML platform services used in production systems at scale.

---

## 📦 API Endpoints

GET /health → {"status":"ok"}
GET /models → {"models":["genai-baseline"]}
POST /genai/predict → {"text":"Evaluate this PR for risk and quality"} → {"model":"genai-baseline","input_length":34,"prediction":"processed","confidence":0.92}

---

## 🏗 Architecture

Client → FastAPI Service → Model Registry → GenAI Baseline (Stateless, production-safe inference)

---

## 🧪 Evaluation & Reliability

Input validation via Pydantic, deterministic baseline inference, structured JSON responses, health/readiness probes, OpenAPI-compliant schemas

---

## 🐳 Deployment & Ops

Containerized using Docker (Python 3.11 slim base), hosted on Railway, auto-deploy enabled via GitHub → Railway integration, no terminal required for demo/testing, public networking enabled via Railway-managed domain

---

## 🛠 Tech Stack

FastAPI, Pydantic, Docker, GitHub Actions, Railway, Python 3.11

---

## 📄 License

MIT License

---

## 🎯 Why This Project Matters

This project demonstrates real ML platform engineering skills: API-first ML services, production deployment, CI/CD readiness, evaluation hooks, and recruiter-visible live demos — aligned with expectations for MLE / SDE-ML / Platform Engineering roles at top-tier companies.
