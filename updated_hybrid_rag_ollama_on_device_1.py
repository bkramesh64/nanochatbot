#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 07:58:51 2025

@author: rameshbk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved offline RAG backend for Nano Diagnostics

This module is imported by local_api_server.py and must expose:

    - load_data_from_files()
    - run_on_device_rag(query: str) -> dict

Key improvements:
- Uses a hybrid KG + vector search (similar to Claude backend)
- Builds a rich SYSTEM prompt with context (triples + manual chunks)
- Calls Ollama's /api/chat endpoint with system + user messages
- Returns HTML-structured answers for better readability
"""

import os
import json
from typing import List, Dict, Tuple

import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# GLOBALS & CONFIG
# ---------------------------------------------------------------------------

# Choose your local Ollama model (override with env if needed)
#OLLAMA_MODEL = os.environ.get("OFFLINE_LLM_MODEL", "llama3.1:8b")
OLLAMA_MODEL = os.environ.get("OFFLINE_LLM_MODEL", "gemma2:9b")

# Sentence-transformer for embeddings (cached locally after first download)
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------ KNOWLEDGE GRAPH (same idea as online backend) ----------

KNOWLEDGE_GRAPH: Dict[str, Dict] = {
    "P0117": {
        "type": "DTC",
        "fault_cause": "Short Circuit to Ground",
        "blink_code": "19",
        "symptoms": ["Continuous Fan", "Sluggish Performance", "Cold Start Problem"],
        "affects": ["Radiator Fan", "Coolant Sensor", "Fuel Consumption"],
        "ecu_pins": ["30", "44"],
        "repair_steps": [
            "Check coolant level between MIN-MAX marks",
            "Inspect connector pins for corrosion or damage",
            "Test continuity Pin 44↔1, Pin 30↔2",
            "Verify no short to ground on Pin 44",
            "Measure voltage 3.3V ± 0.2V",
            "Test resistance 1.954-2.160 K Ohm at 25°C",
        ],
    },
    "Coolant Sensor": {
        "type": "Component",
        "sensor_type": "NTC Thermistor",
        "location": "Thermostat Housing",
        "voltage": "3.3V",
        "resistance": "1.954-2.160 K Ohm at 25°C",
        "connects_to": ["ECU Pin 44", "ECU Pin 30"],
        "related_dtcs": ["P0117", "P0118"],
    },
    "Radiator Fan": {
        "type": "Component",
        "controlled_by": "ECU",
        "fuse": "30A",
        "on_temp": "95-98°C",
        "off_temp": "92-95°C",
        "rotation": "Anticlockwise",
        "rpm": "2200-2300",
        "related_dtcs": ["P0117", "P0118", "P0691"],
    },
    "Window Motor": {
        "type": "Component",
        "fuses": ["WW RH 30A", "WW LH 30A", "WW MOTOR 10A"],
    },
    "Continuous Fan": {
        "type": "Symptom",
        "indicates": ["P0117", "P0118"],
        "description": "Fan runs continuously in limp-home mode",
    },
    "Sluggish Performance": {
        "type": "Symptom",
        "caused_by": ["P0117"],
        "description": "Increased engine load due to continuous fan",
    },
    "Cold Start Problem": {
        "type": "Symptom",
        "caused_by": ["P0117"],
        "description": "Engine struggles to start when cold",
    },
}

# ------------------ MANUAL CHUNKS (small demo KB; extend as needed) ---------

MANUAL_CHUNKS: List[Dict] = [
    {
        "id": "chunk_1",
        "text": (
            "DTC P0117 indicates Engine Coolant Temperature Circuit Low. "
            "This fault occurs when the ECU detects a short circuit to ground "
            "in the ECT sensor circuit, interpreting it as an extremely high "
            "temperature reading above 137.3°C."
        ),
        "dtc": "P0117",
        "component": "Coolant Sensor",
        "page": 165,
        "section": "Fault Description",
    },
    {
        "id": "chunk_2",
        "text": (
            "When P0117 is active, the radiator fan runs continuously as a "
            "protective limp-home mode to prevent overheating. This causes "
            "increased fuel consumption and sluggish vehicle performance due "
            "to increased engine load."
        ),
        "dtc": "P0117",
        "component": "Radiator Fan",
        "page": 166,
        "section": "Impact on Vehicle",
    },
    {
        "id": "chunk_3",
        "text": (
            "The coolant temperature sensor is an NTC (Negative Temperature "
            "Coefficient) thermistor located in the thermostat housing. "
            "Supply voltage is 3.3V ± 0.2V. Normal resistance at 25°C is "
            "1.954 to 2.160 K Ohm. It connects to ECU Pin 44 (signal) and "
            "Pin 30 (ground)."
        ),
        "component": "Coolant Sensor",
        "page": 45,
        "section": "Component Specifications",
    },
    {
        "id": "chunk_4",
        "text": (
            "P0117 Repair Procedure: Step 1 - Check coolant level between MIN "
            "and MAX marks. Step 2 - Inspect connector pins for back-out, "
            "corrosion, or damage. Step 3 - Test continuity from sensor to ECU "
            "(Pin 44↔Pin 1, Pin 30↔Pin 2). Step 4 - Verify no short to ground "
            "on Pin 44. Step 5 - Measure 3.3V ± 0.2V at sensor. Step 6 - Test "
            "sensor resistance at room temperature."
        ),
        "dtc": "P0117",
        "page": 167,
        "section": "Repair Procedure",
    },
    {
        "id": "chunk_5",
        "text": (
            "The radiator fan is controlled by the ECU and turns ON at coolant "
            "temperature 95-98°C and OFF at 92-95°C. Fan rotation direction is "
            "anticlockwise when viewed from front. Normal operating RPM is "
            "2200-2300. The fan fuse rating is 30A."
        ),
        "component": "Radiator Fan",
        "page": 52,
        "section": "Radiator Fan Specifications",
    },
    {
        "id": "chunk_6",
        "text": (
            "Window motor fuses: WW RH (Window Winding Right Hand) is 30A, "
            "WW LH (Window Winding Left Hand) is 30A, and WW MOTOR (Window "
            "Motor Control) is 10A. Located in main fuse box."
        ),
        "component": "Window Motor",
        "page": 28,
        "section": "Fuse Specifications",
    },
    {
        "id": "chunk_7",
        "text": (
            "Cold start problems with P0117 occur because the ECU incorrectly "
            "believes the engine is hot due to the sensor fault. This affects "
            "the fuel mixture calculations and can prevent proper engine "
            "starting when the engine is actually cold."
        ),
        "dtc": "P0117",
        "page": 166,
        "section": "Cold Start Issues",
    },
    {
        "id": "chunk_8",
        "text": (
            "Most common cause of P0117 is a faulty coolant temperature sensor. "
            "Second most common is wiring harness damage causing short to ground. "
            "Check sensor connector first before replacing sensor. If wiring is "
            "damaged, repair or replace harness."
        ),
        "dtc": "P0117",
        "page": 168,
        "section": "Common Causes",
    },
]

# ---------------------------------------------------------------------------
# 1. ENTITY EXTRACTION (lightweight, domain-specific)
# ---------------------------------------------------------------------------

def extract_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract DTC codes, components, symptoms and detect query type.

    query_type:
      - 'explanation'  : wants meaning/details
      - 'repair'       : how to fix
      - 'image_request': wants location / picture (not used heavily offline)
      - 'general'      : default
    """
    q = query.lower()
    entities = {
        "dtc_codes": [],
        "components": [],
        "symptoms": [],
        "wants_image": False,
        "query_type": "general",
    }

    # Image vs explanation/repair keywords
    image_keywords = [
        "show", "display", "picture", "image", "photo", "diagram",
        "where is", "location of",
    ]
    detail_keywords = [
        "details", "detail", "description", "describe", "what is",
        "what does", "tell me", "explain", "mean", "meaning",
    ]
    repair_keywords = [
        "repair", "fix", "steps", "procedure", "how to", "how do i fix",
    ]

    has_image = any(w in q for w in image_keywords)
    has_detail = any(w in q for w in detail_keywords)
    has_repair = any(w in q for w in repair_keywords)

    if has_repair:
        entities["query_type"] = "repair"
    elif has_detail:
        entities["query_type"] = "explanation"
    elif has_image:
        entities["query_type"] = "image_request"
        entities["wants_image"] = True
    else:
        entities["query_type"] = "general"

    # DTC codes
    dtc_codes = ["p0117", "p0118", "p0300", "p0691"]
    for dtc in dtc_codes:
        if dtc in q:
            entities["dtc_codes"].append(dtc.upper())

    # Components
    component_map = {
        "coolant sensor": "Coolant Sensor",
        "temperature sensor": "Coolant Sensor",
        "ect sensor": "Coolant Sensor",
        "radiator fan": "Radiator Fan",
        "fan": "Radiator Fan",
        "window motor": "Window Motor",
        "window": "Window Motor",
        "thermostat": "Thermostat",
        "ecu": "ECU",
    }
    for kw, comp in component_map.items():
        if kw in q:
            entities["components"].append(comp)

    # Symptoms
    symptom_map = {
        "continuous fan": "Continuous Fan",
        "always on": "Continuous Fan",
        "always running": "Continuous Fan",
        "won't turn off": "Continuous Fan",
        "fan running": "Continuous Fan",
        "fan runs": "Continuous Fan",
        "sluggish": "Sluggish Performance",
        "slow": "Sluggish Performance",
        "cold start": "Cold Start Problem",
        "won't start": "Cold Start Problem",
    }
    for kw, sym in symptom_map.items():
        if kw in q:
            entities["symptoms"].append(sym)

    return entities


# ---------------------------------------------------------------------------
# 2. KG RETRIEVAL
# ---------------------------------------------------------------------------

def kg_query(entities: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    """Get a small set of triples from the knowledge graph."""
    triples: List[Tuple[str, str, str]] = []

    # By DTC
    for dtc in entities.get("dtc_codes", []):
        if dtc in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[dtc]
            if "fault_cause" in node:
                triples.append((dtc, "FAULT_CAUSE", node["fault_cause"]))
            if "blink_code" in node:
                triples.append((dtc, "BLINK_CODE", node["blink_code"]))
            for sym in node.get("symptoms", []):
                triples.append((dtc, "SYMPTOM", sym))
            for comp in node.get("affects", []):
                triples.append((dtc, "AFFECTS", comp))
            if entities.get("query_type") == "repair":
                for step in node.get("repair_steps", [])[:4]:
                    triples.append((dtc, "REPAIR_STEP", step))

    # By component
    for comp in entities.get("components", []):
        if comp in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[comp]
            if "location" in node:
                triples.append((comp, "LOCATION", node["location"]))
            if "voltage" in node:
                triples.append((comp, "VOLTAGE", node["voltage"]))
            if "resistance" in node:
                triples.append((comp, "RESISTANCE", node["resistance"]))
            for dtc in node.get("related_dtcs", []):
                triples.append((comp, "RELATED_TO", dtc))
            for fuse in node.get("fuses", []):
                triples.append((comp, "FUSE", fuse))

    # By symptom
    for sym in entities.get("symptoms", []):
        if sym in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[sym]
            for dtc in node.get("indicates", []):
                triples.append((sym, "INDICATES", dtc))
            for cause in node.get("caused_by", []):
                triples.append((sym, "CAUSED_BY", cause))

    # De-duplicate and limit
    return list({t for t in triples})[:10]


# ---------------------------------------------------------------------------
# 3. VECTOR RETRIEVAL
# ---------------------------------------------------------------------------

def _ensure_embeddings():
    """Compute embeddings once (also called by load_data_from_files)."""
    for chunk in MANUAL_CHUNKS:
        if "embedding" not in chunk:
            chunk["embedding"] = EMBEDDER.encode(chunk["text"])


def vector_search(query: str,
                  entities: Dict[str, List[str]],
                  top_k: int = 5) -> List[Dict]:
    """Retrieve semantically relevant chunks from MANUAL_CHUNKS."""
    _ensure_embeddings()

    query_emb = EMBEDDER.encode(query)
    filtered = list(MANUAL_CHUNKS)

    if entities.get("dtc_codes"):
        filtered = [
            c for c in filtered
            if c.get("dtc") in entities["dtc_codes"] or not c.get("dtc")
        ]

    if entities.get("components"):
        filtered = [
            c for c in filtered
            if c.get("component") in entities["components"] or not c.get("component")
        ]

    results: List[Dict] = []
    for chunk in filtered:
        sim = cosine_similarity(
            query_emb.reshape(1, -1),
            chunk["embedding"].reshape(1, -1)
        )[0][0]
        results.append(
            {
                "text": chunk["text"],
                "score": float(sim),
                "page": chunk.get("page", "N/A"),
                "section": chunk.get("section", "N/A"),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# 4. PROMPT BUILDING & OLLAMA CALL
# ---------------------------------------------------------------------------

def build_system_prompt(triples: List[Tuple[str, str, str]],
                        chunks: List[Dict],
                        query_type: str) -> str:
    """Create a rich system prompt similar to Claude backend, but for Ollama."""
    if triples:
        triples_text = "\n".join(
            f"- {subj} --[{pred}]--> {obj}" for subj, pred, obj in triples[:5]
        )
    else:
        triples_text = "No specific graph relationships found."

    if chunks:
        chunks_text = "\n\n".join(
            f"[Page {c['page']}, {c['section']}]\n{c['text']}"
            for c in chunks[:3]
        )
    else:
        chunks_text = "No relevant manual sections found."

    # Tailor sections based on query type
    if query_type == "explanation":
        response_sections = """
2. For explanation/detail queries:
   - Use headings:
     <h3>📋 [DTC/Topic]: [Short description]</h3>
     <h4>🔍 What This Means</h4>
     <h4>⚠️ Symptoms You Might Notice</h4>
     <h4>🔍 Most Likely Causes</h4>
   - Do NOT include detailed repair steps or pin-level wiring unless explicitly asked.
"""
    elif query_type == "repair":
        response_sections = """
2. For repair/fix queries:
   - Use headings:
     <h3>📋 [DTC/Component]: [Short description]</h3>
     <h4>🔍 What This Means</h4>
     <h4>🔧 Repair Procedure</h4> (use <ol><li>...</li></ol> for steps)
     <h4>📍 Component Location</h4>
     <h4>⚠️ Important Notes</h4>
"""
    elif query_type == "image_request":
        response_sections = """
2. For location/image-type queries:
   - Use headings:
     <h3>📋 [Component]: [Short description]</h3>
     <h4>📍 Component Location</h4>
     <h4>🔌 Connection Details</h4>
     <h4>🔧 Quick Visual Checks</h4>
   - Describe location and checks clearly so a mechanic can find it physically.
"""
    else:
        response_sections = """
2. For general queries:
   - Choose appropriate sections depending on context (explanation, checks, or repair),
     but keep structure clear with <h3>, <h4>, <ul>, <ol>, and <p>.
"""

    system_prompt = f"""
You are an expert TATA Nano diagnostic technician assistant.

You MUST answer using ONLY the information in the context below.
If the context does not contain the answer, say you don't know and suggest checking the service manual.

KNOWLEDGE GRAPH RELATIONSHIPS:
{triples_text}

RELEVANT MANUAL SECTIONS:
{chunks_text}

CRITICAL FORMATTING INSTRUCTIONS:
1. Structure your answer with clear HTML tags:
   - <h3> for main headings
   - <h4> for subheadings
   - <strong> for emphasis
   - <ul><li>...</li></ul> for bullet lists
   - <ol><li>...</li></ol> for numbered steps
   - <p> for normal paragraphs

{response_sections}

3. Always cite page numbers when you reference specific data (e.g., 'See Page 165').
4. End with: <p><em>Source: TATA Nano EMS Service Manual v5.0</em></p>
5. Use simple, clear language suitable for mechanics with basic technical knowledge.
6. Do NOT invent voltages, resistances, or pin numbers beyond the provided context.
"""
    return system_prompt.strip()


def call_ollama_chat(model: str,
                     system_prompt: str,
                     user_query: str,
                     temperature: float = 0.3,
                     num_predict: int = 768) -> str:
    """
    Call local Ollama /api/chat endpoint with system + user messages.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # Typical Ollama chat response: {"message": {"role": "assistant", "content": "..."}}
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "").strip()

        # Fallback: try to read as plain "content"
        return str(data)
    except Exception as e:
        return (
            f"<p>⚠️ Error calling local LLM model '{model}': {e}</p>"
            "<p>Please check that Ollama is running and the model is installed.</p>"
        )


# ---------------------------------------------------------------------------
# 5. PUBLIC API FOR local_api_server.py
# ---------------------------------------------------------------------------

def load_data_from_files():
    """
    Called once on server startup from local_api_server.py.

    In this improved version we:
      - Precompute embeddings for manual chunks (for faster queries).
      - (You can extend this to load extra chunks from PDF/JSON in the future.)
    """
    _ensure_embeddings()
    print("✅ Offline RAG KB initialized (manual chunks + KG).")
    print(f"   - KG nodes: {len(KNOWLEDGE_GRAPH)}")
    print(f"   - Manual chunks: {len(MANUAL_CHUNKS)}")
    print(f"   - Ollama model: {OLLAMA_MODEL}")


def run_on_device_rag(query: str) -> Dict:
    """
    Main entry point used by /api/chat and /api/speech in local_api_server.py.

    Returns:
        {
          "answer": "<html-formatted answer>",
          "vdb_chunks": [ {text, score, page, section}, ... ],
          "kg_triples": [ (subj, pred, obj), ... ],
          "locked_specs": {... any extra metadata ...}
        }
    """
    # Step 1: entity extraction
    entities = extract_entities(query)

    # Step 2: KG + vector retrieval
    triples = kg_query(entities)
    chunks = vector_search(query, entities, top_k=5)

    # Step 3: build system prompt
    query_type = entities.get("query_type", "general")
    system_prompt = build_system_prompt(triples, chunks, query_type)

    # Step 4: call local LLM via Ollama
    answer_html = call_ollama_chat(
        model=OLLAMA_MODEL,
        system_prompt=system_prompt,
        user_query=query,
    )

    # Step 5: pack results for local_api_server.py
    locked_specs = {
        "pages_used": sorted({c["page"] for c in chunks}),
        "dtc_codes": entities.get("dtc_codes", []),
        "components": entities.get("components", []),
        "query_type": query_type,
    }

    return {
        "answer": answer_html,
        "vdb_chunks": chunks,
        "kg_triples": triples,
        "locked_specs": locked_specs,
    }
