"""
TATA Nano Diagnostics Chatbot - RAG POC with Smart Image Display
KG + VectorDB Hybrid (Manual KB, In-Memory)
"""

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import anthropic
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import sys

app = Flask(__name__)
CORS(app)

# Check for API key BEFORE initializing
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_API_KEY:
    print("=" * 60)
    print("❌ ERROR: ANTHROPIC_API_KEY not set!")
    print("=" * 60)
    print("\nSet environment variable:")
    print("export ANTHROPIC_API_KEY='sk-ant-api03-your-key-here'")
    print("=" * 60)
    sys.exit(1)

print(f"✅ API Key loaded: {ANTHROPIC_API_KEY[:20]}...")

# Initialize models
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# HTML Template
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TATA Nano RAG Diagnostics</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            height: 85vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 20px 20px 0 0;
        }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-top: 8px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin: 15px 0;
            padding: 12px 18px;
            border-radius: 15px;
            max-width: 85%;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
        }
        .bot {
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .bot h3 { color: #667eea; margin: 10px 0; font-size: 18px; }
        .bot h4 { color: #764ba2; margin: 8px 0; font-size: 16px; }
        .bot strong { color: #667eea; }
        .bot ul { margin: 10px 0; padding-left: 20px; }
        .bot ol { margin: 10px 0; padding-left: 20px; }
        .bot li { margin: 5px 0; }
        .bot p { margin: 8px 0; line-height: 1.6; }
        .bot em { color: #666; font-size: 13px; }
        .component-image {
            margin: 15px 0;
            text-align: center;
        }
        .component-image img {
            max-width: 100%;
            height: auto;
            border: 2px solid #667eea;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .image-caption {
            margin-top: 8px;
            font-size: 13px;
            color: #666;
            font-style: italic;
        }
        .sources {
            margin-top: 10px;
            padding: 8px;
            background: #f0f0f0;
            border-radius: 5px;
            font-size: 12px;
            color: #666;
        }
        .input-area {
            padding: 15px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 20px;
            font-size: 14px;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        .quick-btns {
            padding: 10px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            background: white;
            border-bottom: 1px solid #ddd;
        }
        .quick-btn {
            padding: 6px 12px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
        }
        .quick-btn:hover {
            background: #667eea;
            color: white;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 TATA Nano RAG Diagnostics</h1>
            <p>KG + Vector Hybrid Retrieval with Claude AI</p>
            <span class="badge">🧠 Knowledge Graph</span>
            <span class="badge">📊 Vector Search</span>
            <span class="badge">🤖 Claude AI</span>
        </div>
        <div class="quick-btns">
            <button class="quick-btn" onclick="ask('Why is my radiator fan always running?')">Fan Issue</button>
            <button class="quick-btn" onclick="ask('P0117 is showing, what does this mean?')">P0117 Details</button>
            <button class="quick-btn" onclick="ask('Show me picture of coolant sensor location')">Sensor Image</button>
            <button class="quick-btn" onclick="ask('Window fuse rating?')">Fuses</button>
            <button class="quick-btn" onclick="ask('Car won\\'t start when cold')">Cold Start</button>
        </div>
        <div class="messages" id="messages">
            <div class="message bot">
                <strong>👋 Welcome to TATA Nano RAG Diagnostics!</strong><br><br>
                This system uses:<br>
                • <strong>Knowledge Graph</strong> - Understanding relationships between DTCs, components & symptoms<br>
                • <strong>Vector Search</strong> - Finding relevant manual sections semantically<br>
                • <strong>Claude AI</strong> - Generating natural, accurate answers<br><br>
                Try asking:<br>
                • "P0117 is showing, what does this mean?" (explanation only)<br>
                • "Show me picture of coolant sensor" (with images)
            </div>
        </div>
        <div class="loading" id="loading">🔄 Thinking...</div>
        <div class="input-area">
            <input id="input" placeholder="Ask about DTCs, symptoms, or components..." onkeypress="if(event.key==='Enter')send()">
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        function add(msg, isUser) {
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            div.innerHTML = msg;
            document.getElementById('messages').appendChild(div);
            document.getElementById('messages').scrollTop = 999999;
        }

        async function send() {
            const input = document.getElementById('input');
            const msg = input.value.trim();
            if (!msg) return;
            
            add(msg, true);
            input.value = '';
            input.disabled = true;
            document.getElementById('loading').style.display = 'block';
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await res.json();
                
                let response = data.answer;
                if (data.sources) {
                    response += `<div class="sources">📚 Sources: ${data.sources.kg_triples} KG triples, ${data.sources.vector_chunks} manual chunks</div>`;
                }
                
                add(response, false);
            } catch(e) {
                add('❌ Error: ' + e.message, false);
            }
            
            document.getElementById('loading').style.display = 'none';
            input.disabled = false;
            input.focus();
        }

        function ask(q) {
            document.getElementById('input').value = q;
            send();
        }
    </script>
</body>
</html>'''

# ==================== COMPONENT IMAGES (SVG) ====================

COOLANT_SENSOR_IMG = '''<div class="component-image">
<img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect fill='%23f0f0f0' width='400' height='300'/%3E%3Ctext x='200' y='100' font-family='Arial' font-size='20' fill='%23667eea' text-anchor='middle' font-weight='bold'%3ECoolant Temperature Sensor%3C/text%3E%3Ccircle cx='200' cy='180' r='50' fill='%23667eea' opacity='0.3'/%3E%3Cline x1='200' y1='130' x2='200' y2='230' stroke='%23667eea' stroke-width='3'/%3E%3Cline x1='150' y1='180' x2='250' y2='180' stroke='%23667eea' stroke-width='3'/%3E%3Ctext x='200' y='260' font-family='Arial' font-size='14' fill='%23666' text-anchor='middle'%3ELocation: Thermostat Housing%3C/text%3E%3Ctext x='80' y='180' font-family='Arial' font-size='12' fill='%23666'%3EPin 1 (ECU 44)%3C/text%3E%3Ctext x='260' y='180' font-family='Arial' font-size='12' fill='%23666'%3EPin 2 (ECU 30)%3C/text%3E%3C/svg%3E">
<div class="image-caption">📍 Coolant Temperature Sensor - Thermostat Housing</div>
</div>'''

ECU_PINS_IMG = '''<div class="component-image">
<img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='250' viewBox='0 0 400 250'%3E%3Crect fill='%23f0f0f0' width='400' height='250'/%3E%3Ctext x='200' y='30' font-family='Arial' font-size='18' fill='%23667eea' text-anchor='middle' font-weight='bold'%3EECU Pin Configuration%3C/text%3E%3Crect x='80' y='60' width='240' height='120' fill='%23667eea' opacity='0.2' rx='10'/%3E%3Ccircle cx='140' cy='100' r='18' fill='%23667eea'/%3E%3Ctext x='140' y='107' font-family='Arial' font-size='14' fill='white' text-anchor='middle' font-weight='bold'%3E30%3C/text%3E%3Ctext x='140' y='135' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3ESensor Ground%3C/text%3E%3Ccircle cx='260' cy='100' r='18' fill='%23764ba2'/%3E%3Ctext x='260' y='107' font-family='Arial' font-size='14' fill='white' text-anchor='middle' font-weight='bold'%3E44%3C/text%3E%3Ctext x='260' y='135' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3ESensor Input%3C/text%3E%3Ctext x='200' y='165' font-family='Arial' font-size='12' fill='%23666' text-anchor='middle'%3EConnector: Black | 3.3V%3C/text%3E%3C/svg%3E">
<div class="image-caption">🔌 ECU Pins 30 & 44</div>
</div>'''

FUSE_BOX_IMG = '''<div class="component-image">
<img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='280' viewBox='0 0 400 280'%3E%3Crect fill='%23f0f0f0' width='400' height='280'/%3E%3Ctext x='200' y='30' font-family='Arial' font-size='20' fill='%23667eea' text-anchor='middle' font-weight='bold'%3EFuse Box Layout%3C/text%3E%3Crect x='50' y='60' width='100' height='60' fill='%23667eea' opacity='0.7'/%3E%3Ctext x='100' y='85' font-family='Arial' font-size='14' fill='white' text-anchor='middle' font-weight='bold'%3EWW RH%3C/text%3E%3Ctext x='100' y='105' font-family='Arial' font-size='16' fill='white' text-anchor='middle' font-weight='bold'%3E30A%3C/text%3E%3Crect x='160' y='60' width='100' height='60' fill='%23667eea' opacity='0.7'/%3E%3Ctext x='210' y='85' font-family='Arial' font-size='14' fill='white' text-anchor='middle' font-weight='bold'%3EWW LH%3C/text%3E%3Ctext x='210' y='105' font-family='Arial' font-size='16' fill='white' text-anchor='middle' font-weight='bold'%3E30A%3C/text%3E%3Crect x='270' y='60' width='80' height='60' fill='%23764ba2' opacity='0.7'/%3E%3Ctext x='310' y='85' font-family='Arial' font-size='12' fill='white' text-anchor='middle' font-weight='bold'%3EWW MOTOR%3C/text%3E%3Ctext x='310' y='105' font-family='Arial' font-size='16' fill='white' text-anchor='middle' font-weight='bold'%3E10A%3C/text%3E%3Ctext x='50' y='160' font-family='Arial' font-size='12' fill='%23666'%3E• WW RH: Right Window (30A)%3C/text%3E%3Ctext x='50' y='180' font-family='Arial' font-size='12' fill='%23666'%3E• WW LH: Left Window (30A)%3C/text%3E%3Ctext x='50' y='200' font-family='Arial' font-size='12' fill='%23666'%3E• WW MOTOR: Control (10A)%3C/text%3E%3C/svg%3E">
<div class="image-caption">⚡ Window Fuse Locations</div>
</div>'''

# ==================== KNOWLEDGE GRAPH (Manual) ====================

KNOWLEDGE_GRAPH = {
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
            "Test resistance 1.954-2.160 K Ohm at 25°C"
        ]
    },
    
    "Coolant Sensor": {
        "type": "Component",
        "sensor_type": "NTC Thermistor",
        "location": "Thermostat Housing",
        "voltage": "3.3V",
        "resistance": "1.954-2.160 K Ohm at 25°C",
        "connects_to": ["ECU Pin 44", "ECU Pin 30"],
        "related_dtcs": ["P0117", "P0118"]
    },
    
    "Radiator Fan": {
        "type": "Component",
        "controlled_by": "ECU",
        "fuse": "30A",
        "on_temp": "95-98°C",
        "off_temp": "92-95°C",
        "rotation": "Anticlockwise",
        "rpm": "2200-2300",
        "related_dtcs": ["P0117", "P0118", "P0691"]
    },
    
    "Window Motor": {
        "type": "Component",
        "fuses": ["WW RH 30A", "WW LH 30A", "WW MOTOR 10A"]
    },
    
    "Continuous Fan": {
        "type": "Symptom",
        "indicates": ["P0117", "P0118"],
        "description": "Fan runs continuously in limp-home mode"
    },
    
    "Sluggish Performance": {
        "type": "Symptom",
        "caused_by": ["P0117"],
        "description": "Increased engine load due to continuous fan"
    },
    
    "Cold Start Problem": {
        "type": "Symptom",
        "caused_by": ["P0117"],
        "description": "Engine struggles to start when cold"
    }
}

# ==================== VECTOR DATABASE (Manual Chunks) ====================

MANUAL_CHUNKS = [
    {
        "id": "chunk_1",
        "text": "DTC P0117 indicates Engine Coolant Temperature Circuit Low. This fault occurs when the ECU detects a short circuit to ground in the ECT sensor circuit, interpreting it as an extremely high temperature reading above 137.3°C.",
        "dtc": "P0117",
        "component": "Coolant Sensor",
        "page": 165,
        "section": "Fault Description"
    },
    {
        "id": "chunk_2",
        "text": "When P0117 is active, the radiator fan runs continuously as a protective limp-home mode to prevent overheating. This causes increased fuel consumption and sluggish vehicle performance due to increased engine load.",
        "dtc": "P0117",
        "component": "Radiator Fan",
        "page": 166,
        "section": "Impact on Vehicle"
    },
    {
        "id": "chunk_3",
        "text": "The coolant temperature sensor is an NTC (Negative Temperature Coefficient) thermistor located in the thermostat housing. Supply voltage is 3.3V ± 0.2V. Normal resistance at 25°C is 1.954 to 2.160 K Ohm. It connects to ECU Pin 44 (signal) and Pin 30 (ground).",
        "component": "Coolant Sensor",
        "page": 45,
        "section": "Component Specifications"
    },
    {
        "id": "chunk_4",
        "text": "P0117 Repair Procedure: Step 1 - Check coolant level between MIN and MAX marks. Step 2 - Inspect connector pins for back-out, corrosion, or damage. Step 3 - Test continuity from sensor to ECU (Pin 44↔Pin 1, Pin 30↔Pin 2). Step 4 - Verify no short to ground on Pin 44. Step 5 - Measure 3.3V ± 0.2V at sensor. Step 6 - Test sensor resistance at room temperature.",
        "dtc": "P0117",
        "page": 167,
        "section": "Repair Procedure"
    },
    {
        "id": "chunk_5",
        "text": "The radiator fan is controlled by the ECU and turns ON at coolant temperature 95-98°C and OFF at 92-95°C. Fan rotation direction is anticlockwise when viewed from front. Normal operating RPM is 2200-2300. The fan fuse rating is 30A.",
        "component": "Radiator Fan",
        "page": 52,
        "section": "Radiator Fan Specifications"
    },
    {
        "id": "chunk_6",
        "text": "Window motor fuses: WW RH (Window Winding Right Hand) is 30A, WW LH (Window Winding Left Hand) is 30A, and WW MOTOR (Window Motor Control) is 10A. Located in main fuse box.",
        "component": "Window Motor",
        "page": 28,
        "section": "Fuse Specifications"
    },
    {
        "id": "chunk_7",
        "text": "Cold start problems with P0117 occur because the ECU incorrectly believes the engine is hot due to the sensor fault. This affects the fuel mixture calculations and can prevent proper engine starting when the engine is actually cold.",
        "dtc": "P0117",
        "page": 166,
        "section": "Cold Start Issues"
    },
    {
        "id": "chunk_8",
        "text": "Most common cause of P0117 is a faulty coolant temperature sensor. Second most common is wiring harness damage causing short to ground. Check sensor connector first before replacing sensor. If wiring is damaged, repair or replace harness.",
        "dtc": "P0117",
        "page": 168,
        "section": "Common Causes"
    }
]

# Lazy loading - compute embeddings on first use
print("✅ Embedder initialized (embeddings computed on first query)")

# ==================== ENTITY EXTRACTION ====================

def extract_entities(query: str) -> Dict[str, List[str]]:
    """Enhanced entity extraction with strict image detection"""
    query_lower = query.lower()
    
    entities = {
        "dtc_codes": [],
        "components": [],
        "symptoms": [],
        "wants_image": False,
        "query_type": "general"
    }
    
    # VERY STRICT image request detection
    # Must have explicit image words WITHOUT detail/description words
    image_keywords = ['show', 'display', 'picture', 'image', 'photo', 'diagram', 'where is', 'location of']
    detail_keywords = ['details', 'detail', 'description', 'describe', 'what is', 'what does', 'tell me', 'explain', 'mean', 'meaning']
    
    has_image_keyword = any(word in query_lower for word in image_keywords)
    has_detail_keyword = any(word in query_lower for word in detail_keywords)
    
    # CRITICAL: If query has detail/description words, it's NOT an image request
    # Even if it has "show" in "show us details"
    if has_detail_keyword:
        entities["wants_image"] = False
        entities["query_type"] = "explanation"
    elif has_image_keyword:
        entities["wants_image"] = True
        entities["query_type"] = "image_request"
    elif any(word in query_lower for word in ['repair', 'fix', 'steps', 'procedure', 'how to']):
        entities["query_type"] = "repair"
    else:
        entities["query_type"] = "general"
    
    # Extract DTC codes
    dtc_codes = ["p0117", "p0118", "p0300", "p0691"]
    for dtc in dtc_codes:
        if dtc in query_lower:
            entities["dtc_codes"].append(dtc.upper())
    
    # Extract components
    components = {
        "coolant sensor": "Coolant Sensor",
        "temperature sensor": "Coolant Sensor",
        "ect sensor": "Coolant Sensor",
        "radiator fan": "Radiator Fan",
        "fan": "Radiator Fan",
        "window motor": "Window Motor",
        "window": "Window Motor",
        "thermostat": "Thermostat",
        "ecu": "ECU",
        "faulty part": "Coolant Sensor",  # When asking for "faulty part" with P0117
        "part": "Coolant Sensor"
    }
    for keyword, component in components.items():
        if keyword in query_lower:
            if keyword in ["faulty part", "part"] and entities["dtc_codes"]:
                entities["components"].append(component)
            elif keyword not in ["faulty part", "part"]:
                entities["components"].append(component)
    
    # Extract symptoms
    symptoms = {
        "continuous fan": "Continuous Fan",
        "always on": "Continuous Fan",
        "always running": "Continuous Fan",
        "won't turn off": "Continuous Fan",
        "fan running": "Continuous Fan",
        "fan runs": "Continuous Fan",
        "sluggish": "Sluggish Performance",
        "slow": "Sluggish Performance",
        "cold start": "Cold Start Problem",
        "won't start": "Cold Start Problem"
    }
    for keyword, symptom in symptoms.items():
        if keyword in query_lower:
            entities["symptoms"].append(symptom)
    
    return entities

# ==================== KG RETRIEVAL ====================

def kg_query(entities: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    """Retrieve triples from knowledge graph"""
    triples = []
    
    # Query by DTC codes
    for dtc in entities.get("dtc_codes", []):
        if dtc in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[dtc]
            triples.append((dtc, "FAULT_CAUSE", node.get("fault_cause", "")))
            triples.append((dtc, "BLINK_CODE", node.get("blink_code", "")))
            for symptom in node.get("symptoms", []):
                triples.append((dtc, "SYMPTOM", symptom))
            for component in node.get("affects", []):
                triples.append((dtc, "AFFECTS", component))
            
            # Only include repair steps if query type is repair or image request
            if entities.get("query_type") in ["repair", "image_request"]:
                for step in node.get("repair_steps", [])[:3]:
                    triples.append((dtc, "REPAIR_STEP", step))
    
    # Query by components
    for component in entities.get("components", []):
        if component in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[component]
            if "location" in node and entities.get("query_type") in ["image_request", "repair"]:
                triples.append((component, "LOCATION", node["location"]))
            if "voltage" in node:
                triples.append((component, "VOLTAGE", node["voltage"]))
            if "resistance" in node:
                triples.append((component, "RESISTANCE", node["resistance"]))
            if "related_dtcs" in node:
                for dtc in node["related_dtcs"]:
                    triples.append((component, "RELATED_TO", dtc))
            if "fuses" in node:
                for fuse in node["fuses"]:
                    triples.append((component, "FUSE", fuse))
    
    # Query by symptoms
    for symptom in entities.get("symptoms", []):
        if symptom in KNOWLEDGE_GRAPH:
            node = KNOWLEDGE_GRAPH[symptom]
            for dtc in node.get("indicates", []):
                triples.append((symptom, "INDICATES", dtc))
            for cause in node.get("caused_by", []):
                triples.append((symptom, "CAUSED_BY", cause))
    
    return list(set(triples))[:10]

# ==================== VECTOR RETRIEVAL ====================

def vector_search(query: str, entities: Dict[str, List[str]], top_k: int = 5) -> List[Dict]:
    """Retrieve relevant chunks using vector similarity"""
    # Encode query
    query_embedding = embedder.encode(query)
    
    # Filter chunks
    filtered_chunks = MANUAL_CHUNKS.copy()
    
    if entities.get("dtc_codes"):
        filtered_chunks = [c for c in filtered_chunks if c.get("dtc") in entities["dtc_codes"] or not c.get("dtc")]
    
    if entities.get("components"):
        filtered_chunks = [c for c in filtered_chunks if c.get("component") in entities["components"] or not c.get("component")]
    
    # Calculate similarity
    results = []
    for chunk in filtered_chunks:
        # Compute embedding if not cached
        if "embedding" not in chunk:
            chunk["embedding"] = embedder.encode(chunk["text"])
        
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk["embedding"].reshape(1, -1)
        )[0][0]
        
        results.append({
            "text": chunk["text"],
            "score": float(similarity),
            "page": chunk.get("page", "N/A"),
            "section": chunk.get("section", "N/A")
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ==================== HYBRID FUSION ====================

def hybrid_fusion(triples: List[Tuple], chunks: List[Dict]) -> Dict:
    """Combine KG and Vector results"""
    K = 60
    kg_score = 0.4 / (K + 1) if triples else 0
    vector_scores = [chunk["score"] for chunk in chunks]
    avg_vector_score = sum(vector_scores) / len(vector_scores) if vector_scores else 0
    
    return {
        "kg_weight": 0.4,
        "vector_weight": 0.6,
        "kg_score": kg_score,
        "vector_score": avg_vector_score,
        "combined_score": (0.4 * kg_score) + (0.6 * avg_vector_score)
    }

# ==================== CLAUDE GENERATION ====================

def generate_answer(query: str, triples: List[Tuple], chunks: List[Dict], query_type: str = "general") -> str:
    """Generate answer using Claude with context-aware formatting"""
    
    # Format triples
    triples_text = "\n".join([
        f"- {subj} --[{pred}]--> {obj}"
        for subj, pred, obj in triples[:5]
    ]) if triples else "No specific graph relationships found."
    
    # Format chunks
    chunks_text = "\n\n".join([
        f"[Page {chunk['page']}, {chunk['section']}]\n{chunk['text']}"
        for chunk in chunks[:3]
    ]) if chunks else "No relevant manual sections found."
    
    # Adjust instructions based on query type
    if query_type == "explanation":
        response_sections = """
2. For DTC explanation queries (details/description WITHOUT image request), include ONLY:
   - <h3>📋 [DTC Code]: [Description]</h3>
   - <h4>🔍 What This Means</h4>
   - <h4>⚠️ Symptoms You Might Notice</h4>
   - <h4>🔍 Most Likely Causes</h4>
   
   STRICTLY EXCLUDE:
   - Repair steps or procedures
   - Detailed location information
   - Component specifications (voltage, resistance, pin details)
   - Installation or removal instructions
   
   Keep response concise - focused ONLY on understanding the problem."""
    
    elif query_type == "image_request":
        response_sections = """
2. For image/location requests (show/picture/diagram), include:
   - <h3>📋 [Component Name]: [Brief Description]</h3>
   - <h4>📍 Component Location</h4> (detailed physical location)
   - <h4>🔌 Connection Details</h4> (pin numbers, wire colors, connector type)
   - <h4>🔧 Quick Visual Check</h4> (what to look for visually)
   
   Provide detailed location and connection info since images will be displayed."""
    
    elif query_type == "repair":
        response_sections = """
2. For repair queries, include:
   - <h3>📋 [DTC Code]: [Description]</h3>
   - <h4>🔍 What This Means</h4>
   - <h4>🔧 Repair Procedure</h4> (numbered steps)
   - <h4>📍 Component Location</h4>
   - <h4>⚠️ Important Notes</h4>"""
    
    else:
        response_sections = """
2. For general queries, provide appropriate sections based on context."""
    
    # Enhanced system prompt
    system_prompt = f"""You are an expert TATA Nano diagnostic technician assistant.

KNOWLEDGE GRAPH RELATIONSHIPS:
{triples_text}

RELEVANT MANUAL SECTIONS:
{chunks_text}

CRITICAL FORMATTING INSTRUCTIONS:
1. Structure your response with clear HTML formatting:
   - Use <h3> for main headings
   - Use <h4> for subheadings
   - Use <strong> for emphasis
   - Use <ul> and <li> for bullet lists
   - Use <ol> and <li> for numbered lists
   - Use <p> for paragraphs

{response_sections}

3. Always cite page numbers when available
4. End with: <p><em>Source: TATA Nano EMS Service Manual v5.0</em></p>

ANSWER STYLE:
- Professional but accessible
- Use automotive terminology correctly
- Match detail level EXACTLY to query type
- Provide ONLY what's requested

CRITICAL RULES FOR EXPLANATION QUERIES:
- If user asks for "details" or "description" WITHOUT mentioning images/pictures/show:
  * Give ONLY: explanation, symptoms, causes
  * DO NOT include: repair steps, location details, pin numbers, connector specs, voltages, resistance values
  * Keep it diagnostic understanding ONLY
  
- If user asks to "show" or wants "picture/image/location":
  * Include detailed location, connector info, pin numbers
  * Images will be added automatically

NEVER make up information not in the provided context."""

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# ==================== IMAGE HANDLING ====================

def add_images_to_response(response: str, entities: Dict, triples: List[Tuple], query: str) -> str:
    """Add relevant component images ONLY when explicitly requested"""
    
    # Only add images if user explicitly wants them
    wants_image = entities.get("wants_image", False)
    query_type = entities.get("query_type", "general")
    
    # CRITICAL: Never add images for explanation queries
    if query_type == "explanation" or not wants_image:
        return response
    
    images_to_add = []
    query_lower = query.lower()
    
    # Determine which images based on what user is asking about
    if (any(entity in ['P0117', 'P0118'] for entity in entities.get('dtc_codes', [])) or 
        any('Coolant Sensor' in str(component) for component in entities.get('components', [])) or
        'coolant' in query_lower or 'temperature sensor' in query_lower or 'ect' in query_lower or
        'faulty part' in query_lower or ('part' in query_lower and entities.get('dtc_codes'))):
        images_to_add.append(('coolant', COOLANT_SENSOR_IMG))
        images_to_add.append(('ecu', ECU_PINS_IMG))
    
    elif (any('Window Motor' in str(component) for component in entities.get('components', [])) or
          'window' in query_lower or 'fuse' in query_lower):
        images_to_add.append(('fuse', FUSE_BOX_IMG))
    
    # Add images to response
    if images_to_add:
        image_html = "\n\n<h4>📷 Component Images & Location</h4>\n"
        for img_name, img_html in images_to_add:
            image_html += img_html + "\n"
        
        # Insert before source citation
        if "Source: TATA Nano" in response:
            response = response.replace(
                "<p><em>Source: TATA Nano",
                image_html + "\n<p><em>Source: TATA Nano"
            )
        else:
            response += image_html
    
    return response

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.json.get('message', '')
    
    # Step 1: Entity extraction
    entities = extract_entities(query)
    
    # Step 2: KG retrieval
    triples = kg_query(entities)
    
    # Step 3: Vector retrieval
    chunks = vector_search(query, entities, top_k=5)
    
    # Step 4: Hybrid fusion
    fusion_scores = hybrid_fusion(triples, chunks)
    
    # Step 5: Generate answer with Claude (context-aware)
    answer = generate_answer(query, triples, chunks, query_type=entities.get("query_type", "general"))
    
    # Step 6: Add images ONLY if explicitly requested (not for explanation queries)
    answer = add_images_to_response(answer, entities, triples, query)
    
    # Step 7: Format response
    response = {
        "answer": answer,
        "sources": {
            "kg_triples": len(triples),
            "vector_chunks": len(chunks),
            "scores": f"KG:{fusion_scores['kg_score']:.2f}, Vec:{fusion_scores['vector_score']:.2f}"
        },
        "entities": entities
    }
    
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "mode": "POC - Manual KB",
        "kg_nodes": len(KNOWLEDGE_GRAPH),
        "vector_chunks": len(MANUAL_CHUNKS),
        "claude_api": "configured" if ANTHROPIC_API_KEY else "missing"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    
    print("=" * 60)
    print("🚀 Starting TATA Nano RAG POC on port", port)
    print("📚 Knowledge Graph:", len(KNOWLEDGE_GRAPH), "nodes")
    print("📄 Vector Chunks:", len(MANUAL_CHUNKS), "documents")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)