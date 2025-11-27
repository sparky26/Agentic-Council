# 🧠 Council of Experts — Multi-Agent Debate System for India-Focused Topics

The **Council of Experts** is a custom-built multi-agent debate engine designed to simulate **high-level expert discussions** on topics related to Indian history, society, religion, anthropology, governance, and civilizational analysis.

It uses:

- **Ollama** with the `gpt-oss:latest` model for local, streaming responses
- A set of **specialized agents**  
- A structured **debate → rebuttal → consensus** protocol  
- A **Streamlit UI** that streams expert messages live  
- Persistent storage of debates  

---

## 🚀 Features Overview

### ✔️ Five Specialist Agents ("The Council")
| Role | Expertise |
|------|-----------|
| Indian Historian | Empirical, archival, region-specific history |
| Civilizational Historian | Comparative historical + civilizational analysis |
| Religion Expert | Theology, sects, ritual, interpretation |
| Anthropology Expert | Fieldwork, culture, caste, kinship |
| Policymaker | Governance, trade-offs, mechanisms, reforms |

Each expert has:
- Role-specific system prompt  
- Assigned LLM model  
- Unique image  
- Live streaming output panel  

---

## 🔁 Debate Pipeline

### **1. Opening Statements**
Each expert independently presents:
- Their domain analysis  
- Sources and evidence  
- Hypotheses and uncertainties  

No previous messages are shown to ensure independence.

---

### **2. Rebuttal Rounds** (0–3)
Each expert sees:
- Topic prompt  
- Rebuttal instructions  
- A **sliding window** of the last N messages (token-safe)

They react to:
- Arguments of other experts  
- Contradictions  
- Missing evidence  
- Methodological issues  

This creates a realistic back-and-forth.

---

### **3. Consensus Stage**
The Policymaker agent receives:
- Topic prompt  
- A compressed transcript summary  
- Role instructions

Produces:
- Final council-wide assessment  
- Trade-offs  
- Evidence-based recommendations  

---

## 🧱 Architecture

```
src/
 └── council/
      ├── agents/               # Expert agent classes
      ├── config/               # Settings, prompts
      ├── debate/               # Debate protocol + consensus logic
      ├── io/                   # Streamlit UI + persistence
      ├── llm/                  # Ollama client wrapper
      └── utils/                # Helpers
```

---

## 🖥️ Streamlit UI

The interface shows:
- Expert image  
- Role/title  
- Scrollable real-time output box  
- Saved debates list  
- Consensus section  

---

## ⚙️ Configuration

Environment variables via `.env`:

```
OLLAMA_HOST=http://localhost:11434
COUNCIL_DEFAULT_MODEL_ALIAS=gpt_oss_latest
```

---

## ▶️ How to Run

```
pip install -r requirements.txt
streamlit run src/council/io/streamlit_app.py
```

---

## 💾 Persistence

Saved debates appear in:

```
debates/<id>.json
```

including:
- Metadata  
- Full transcript  
- Consensus  

---

## 🧩 Token Safety Mechanisms

To avoid overrunning context limits:
- Sliding-window transcripts
- Truncated consensus transcript
- Lower `max_completion_tokens`
- Model-specific token budgeting

---

## 📚 Purpose

The system is designed for:
- Research  
- Policy simulations  
- Educational analysis  
- Multi-perspective reasoning  

It encourages **non-politically-correct but evidence-grounded** expert debate within safe boundaries.

---

## 📄 License

Private use only unless you specify otherwise.
