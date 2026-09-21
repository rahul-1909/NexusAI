# 🤖 Autonomous Multi-Agent RAG Engine

[![Render Deployment](https://img.shields.io/badge/Live_Demo-Render_Cloud-brightgreen?style=for-the-badge&logo=render)](https://multi-agent-autonomous-rag-engine.onrender.com/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF4F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **An enterprise-grade, Corrective Multi-Agent RAG (CRAG) and Self-RAG system built with LangGraph, Qdrant Vector Search, FastEmbed (CPU), and Groq LPU inference. Implements query intent routing, dynamic document relevance grading, self-correcting query reformulation loops, and strict anti-hallucination verification.**

🌐 **Live Application**: [https://multi-agent-autonomous-rag-engine.onrender.com](https://multi-agent-autonomous-rag-engine.onrender.com)  
📖 **Interactive Swagger Docs**: [https://multi-agent-autonomous-rag-engine.onrender.com/docs](https://multi-agent-autonomous-rag-engine.onrender.com/docs)

---

## 📑 Table of Contents
- [Architectural Overview](#-architectural-overview)
- [Key Engineering Differentiators](#-key-engineering-differentiators)
- [Multi-Agent State Machine Specification](#-multi-agent-state-machine-specification)
- [Empirical Evaluation & Benchmarks](#-empirical-evaluation--benchmarks)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [REST API Reference](#-rest-api-reference)
- [Local Quickstart & Installation](#-local-quickstart--installation)
- [Docker Deployment](#-docker-deployment)
- [Author & Profile](#-author--profile)

---

## 🏛️ Architectural Overview

Standard Naive RAG architectures blindly feed top-$k$ retrieved chunks into an LLM context window, resulting in context dilution, irrelevant generation, and hallucinations. 

This engine implements **Corrective RAG (CRAG)** and **Self-RAG** patterns through a stateful directed acyclic graph (DAG) managed by **LangGraph**:

```mermaid
flowchart TD
    Start([User Query]) --> Router{Router Agent}

    Router -->|"Smalltalk / Direct Conversational"| DirectLLM[Direct LLM Generator]
    Router -->|"Domain Knowledge Required"| Retriever[Qdrant Dense Retriever]

    Retriever --> GradeDocs{Document Relevance Grader}

    GradeDocs -->|"Chunks Irrelevant & Retries < 2"| Rewriter[Query Rewriter Agent]
    Rewriter -->|"Reformulated Query"| Retriever

    GradeDocs -->|"Relevant Context Found"| Generator[Grounded Answer Generator]

    Generator --> HallucinationChecker{Hallucination & Fidelity Grader}

    HallucinationChecker -->|"Fails Groundedness Check & Retries < 2"| Generator
    HallucinationChecker -->|"Grounded & Verified"| Response([Validated Output with Citations])
    DirectLLM --> Response