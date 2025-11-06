"""
TATA Nano Diagnostics Chatbot - SMART CONTEXT VERSION
Remembers context from symptoms, not just DTC codes
"""

from flask import Flask, render_template_string, request, jsonify, session
from flask_cors import CORS
import re
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app, supports_credentials=True)

# HTML Template (same as before)
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TATA Nano Diagnostics</title>
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
        .context-bar {
            padding: 8px 15px;
            background: #e3f2fd;
            border-bottom: 1px solid #90caf9;
            font-size: 13px;
            color: #1976d2;
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
        .bot h3 { color: #667eea; margin: 10px 0; }
        .bot h4 { color: #764ba2; margin: 8px 0; }
        .bot table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .bot th {
            background: #667eea;
            color: white;
            padding: 8px;
            text-align: left;
        }
        .bot td {
            padding: 8px;
            border: 1px solid #ddd;
            vertical-align: top;
        }
        .bot tr:nth-child(even) { background: #f9f9f9; }
        .bot ul, .bot ol { margin: 10px 0; padding-left: 20px; }
        .bot li { margin: 5px 0; }
        .bot strong { color: #667eea; }
        .bot code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            color: #e91e63;
            font-weight: bold;
        }
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 TATA Nano Diagnostics Assistant</h1>
            <p>Smart context from symptoms and DTCs</p>
        </div>
        <div class="context-bar" id="contextBar">
            💡 Ask about any symptom or DTC
        </div>
        <div class="quick-btns">
            <button class="quick-btn" onclick="ask('Fan runs continuously')">Fan Issue</button>
            <button class="quick-btn" onclick="ask('Show images')">Images</button>
            <button class="quick-btn" onclick="ask('How to fix')">Fix</button>
            <button class="quick-btn" onclick="ask('Window fuse')">Fuse</button>
        </div>
        <div class="messages" id="messages">
            <div class="message bot">
                <strong>👋 Welcome!</strong><br>
                I remember context from both <strong>symptoms</strong> and <strong>DTCs</strong>!<br><br>
                <strong>Try:</strong><br>
                • "Fan runs continuously" → Then ask "show images" or "how to fix"<br>
                • "P0117 showing" → Then ask for details<br>
                • Any diagnostic question!
            </div>
        </div>
        <div class="input-area">
            <input id="input" placeholder="Describe symptoms or ask about DTCs..." onkeypress="if(event.key==='Enter')send()">
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        function updateContextBar(context) {
            const bar = document.getElementById('contextBar');
            if (context) {
                bar.innerHTML = `🔧 Context: <strong>${context}</strong> | Ask: "show images", "how to fix", "show table"`;
                bar.style.background = '#c8e6c9';
                bar.style.color = '#2e7d32';
            } else {
                bar.innerHTML = '💡 Ask about any symptom or DTC';
                bar.style.background = '#e3f2fd';
                bar.style.color = '#1976d2';
            }
        }

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
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg}),
                    credentials: 'include'
                });
                const data = await res.json();
                add(data.response, false);
                if (data.context !== undefined) {
                    updateContextBar(data.context);
                }
            } catch(e) {
                add('Error: ' + e.message, false);
            }
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

# Component Images
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

# P0117 Data
P0117_DESCRIPTION = '''<h3>📋 P0117: Engine Coolant Temperature Circuit Low</h3>

<h4>🔍 Problem Description</h4>
<p>DTC <code>P0117</code> indicates a <strong>Short Circuit to Ground</strong> in the Engine Coolant Temperature (ECT) sensor circuit. This fault occurs when the ECU detects an implausibly low voltage signal from the coolant temperature sensor, which it interprets as an extremely high temperature reading (above 137.3°C).</p>

<h4>⚠️ Detecting Condition</h4>
<p>The fault sets when:</p>
<ul>
<li>Ignition is ON for some time OR engine is running</li>
<li>Coolant temperature measured is <strong>above 137.3°C</strong></li>
<li>If Intake Air Temperature sensor error is also present, fault detection takes <strong>60 seconds</strong></li>
</ul>

<h4>💥 Impact on Vehicle</h4>
<table>
<tr><th>Category</th><th>Effect</th></tr>
<tr><td><strong>ECU Response</strong></td><td>Replacement values taken from sensor model</td></tr>
<tr><td><strong>Radiator Fan</strong></td><td>Runs continuously (limp-home mode)</td></tr>
<tr><td><strong>Fuel Consumption</strong></td><td>Increased due to continuous fan</td></tr>
<tr><td><strong>Drivability</strong></td><td>Sluggish performance</td></tr>
<tr><td><strong>Cold Start</strong></td><td>Severely affected</td></tr>
<tr><td><strong>Engine Risk</strong></td><td>May overheat and seize</td></tr>
</table>

<h4>⚙️ Probable Trouble Areas (8 Components)</h4>
<ol>
<li>Coolant level</li>
<li>Coolant Temperature Sensor Circuit</li>
<li>Coolant Temperature Sensor</li>
<li>Radiator fan control circuit/relay</li>
<li>Radiator fan fuse</li>
<li>Radiator fan assembly</li>
<li>Thermostat valve</li>
<li>ECU</li>
</ol>

<h4>🔧 Technical Specifications</h4>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>DTC Code</td><td><code>P0117</code></td></tr>
<tr><td>Fault Cause</td><td>Short Circuit to Ground</td></tr>
<tr><td>Blink Code</td><td>19</td></tr>
<tr><td>ECU Pins</td><td>30 (Ground), 44 (Signal)</td></tr>
<tr><td>Sensor Type</td><td>NTC Thermistor</td></tr>
<tr><td>Supply Voltage</td><td>3.3 V</td></tr>
<tr><td>Resistance at 25°C</td><td>1.954 - 2.160 K Ohm</td></tr>
</table>

<p><em>Source: Form No. 9, Pages 165-168, TATA Nano EMS Service Manual v5.0</em></p>'''

P0117_IMAGES = COOLANT_SENSOR_IMG + ECU_PINS_IMG

P0117_REPAIR = '''<h3>🛠️ How to Rectify P0117 Problem</h3>

<table>
<tr><th>Step</th><th>Procedure</th><th>Specification</th></tr>
<tr><td>1</td><td>Check coolant level</td><td>Between MIN-MAX marks</td></tr>
<tr><td>2</td><td>Inspect connector pins</td><td>No corrosion/damage</td></tr>
<tr><td>3</td><td>Test continuity</td><td>Pin 44↔1, Pin 30↔2</td></tr>
<tr><td>4</td><td>Check for short to ground</td><td>Pin 44: OPEN to ground</td></tr>
<tr><td>5</td><td>Measure voltage</td><td><strong>3.3V ± 0.2V</strong></td></tr>
<tr><td>6</td><td>Test resistance</td><td><strong>1.954-2.160 K Ohm at 25°C</strong></td></tr>
<tr><td>7</td><td>Check fan circuit</td><td>ON at 95-98°C</td></tr>
<tr><td>8</td><td>Verify relay/fuse</td><td>Clicks & has continuity</td></tr>
<tr><td>9</td><td>Check fan rotation</td><td>Anticlockwise, 2200-2300 rpm</td></tr>
<tr><td>10</td><td>Test thermostat</td><td>Opens at 82-87°C</td></tr>
<tr><td>11</td><td>Replace faulty parts</td><td>Sensor/wiring/thermostat</td></tr>
<tr><td>12</td><td>Clear DTC & confirm</td><td>3 fault-free cycles</td></tr>
</table>

<p><em>Source: Form No. 9, Repair Procedure</em></p>'''

P0117_TABLE = '''<h3>📊 P0117: P Code, Fault Cause & Repair Steps</h3>

<table>
<tr style="background: #667eea;">
<th colspan="3" style="color: white; text-align: center; font-size: 16px; padding: 10px;">P0117 COMPLETE DIAGNOSTIC TABLE</th>
</tr>

<tr style="background: #e3f2fd;">
<th style="width: 25%;">P Code</th>
<td colspan="2"><strong><code>P0117</code></strong> - Engine Coolant Temperature Circuit Low</td>
</tr>

<tr style="background: #fff3e0;">
<th>Fault Cause</th>
<td colspan="2"><strong>Short Circuit to Ground</strong></td>
</tr>

<tr style="background: #e8f5e9;">
<th>Related Info</th>
<td colspan="2">Blink Code: 19 | ECU Pins: 30, 44 | Sensor: NTC Thermistor | Voltage: 3.3V | Resistance: 1.954-2.160 K Ohm at 25°C</td>
</tr>

<tr style="background: #667eea;">
<th colspan="3" style="color: white; text-align: center; padding: 10px;">REPAIR STEPS</th>
</tr>

<tr style="background: #f5f5f5;">
<th style="width: 8%; text-align: center;">Step</th>
<th style="width: 42%;">Procedure</th>
<th style="width: 50%;">Specification / Expected Result</th>
</tr>

<tr>
<td style="text-align: center;"><strong>1</strong></td>
<td>Check coolant level</td>
<td>Between MIN-MAX marks</td>
</tr>

<tr>
<td style="text-align: center;"><strong>2</strong></td>
<td>Inspect connector pins</td>
<td>No corrosion/damage/back-out</td>
</tr>

<tr>
<td style="text-align: center;"><strong>3</strong></td>
<td>Test continuity sensor to ECU</td>
<td>Pin 44↔Pin 1: Continuity | Pin 30↔Pin 2: Continuity</td>
</tr>

<tr>
<td style="text-align: center;"><strong>4</strong></td>
<td>Check for short to ground</td>
<td>Pin 44 to ground: <strong>OPEN circuit</strong></td>
</tr>

<tr>
<td style="text-align: center;"><strong>5</strong></td>
<td>Measure supply voltage</td>
<td><strong>3.3V ± 0.2V</strong> between pins 1 & 2</td>
</tr>

<tr>
<td style="text-align: center;"><strong>6</strong></td>
<td>Test thermistor resistance</td>
<td><strong>1.954-2.160 K Ohm at 25°C</strong></td>
</tr>

<tr>
<td style="text-align: center;"><strong>7</strong></td>
<td>Check radiator fan circuit</td>
<td>Fan ON at 95-98°C, OFF at 92-95°C</td>
</tr>

<tr>
<td style="text-align: center;"><strong>8</strong></td>
<td>Verify fan relay and fuse</td>
<td>Relay clicks | Fuse has continuity</td>
</tr>

<tr>
<td style="text-align: center;"><strong>9</strong></td>
<td>Check fan rotation & RPM</td>
<td>Direction: <strong>Anticlockwise</strong> | RPM: <strong>2200-2300</strong></td>
</tr>

<tr>
<td style="text-align: center;"><strong>10</strong></td>
<td>Test thermostat valve</td>
<td>Opens at <strong>82-87°C</strong></td>
</tr>

<tr>
<td style="text-align: center;"><strong>11</strong></td>
<td>Replace faulty components</td>
<td>Most common: Sensor, wiring harness, thermostat</td>
</tr>

<tr>
<td style="text-align: center;"><strong>12</strong></td>
<td>Clear DTC and confirm repair</td>
<td>Drive <strong>3 fault-free driving cycles</strong> | CKNL remains OFF</td>
</tr>

<tr style="background: #fff3e0;">
<td colspan="3" style="padding: 8px; font-size: 12px;"><em><strong>Source:</strong> Form No. 9, Pages 165-168, TATA Nano EMS Service Manual v5.0</em></td>
</tr>

</table>'''
# Other responses
FAN_CONTINUOUS = '''<h3>🌀 Radiator Fan Running Continuously</h3>
<p>Continuous fan = coolant sensor fault. ECU in <strong>limp-home mode</strong>.</p>

<h4>🔴 Related DTCs</h4>
<table>
<tr><th>DTC</th><th>Cause</th></tr>
<tr><td><code>P0117</code></td><td>ECT Circuit Low (Short to Ground)</td></tr>
<tr><td><code>P0118</code></td><td>ECT Circuit High</td></tr>
<tr><td><code>P0691</code></td><td>Fan Control Low</td></tr>
</table>

<h4>🛠️ Quick Fix Steps</h4>
<ol>
<li>Read DTCs with diagnostic tool</li>
<li>Check coolant level</li>
<li>Test ECT sensor (1.954-2.160 K Ohm at 25°C)</li>
<li>Check fan relay/fuse</li>
<li>Inspect wiring for shorts</li>
<li>Replace faulty parts</li>
<li>Clear DTCs</li>
</ol>

<p><em>Most common cause: Faulty coolant temperature sensor causing P0117</em></p>'''

WINDOW_FUSE = '''<h3>⚡ Window Motor Fuse Rating</h3>
''' + FUSE_BOX_IMG + '''

<table>
<tr><th>Fuse</th><th>Rating</th><th>Circuit</th></tr>
<tr><td><strong>WW RH</strong></td><td><code>30A</code></td><td>Right Window</td></tr>
<tr><td><strong>WW LH</strong></td><td><code>30A</code></td><td>Left Window</td></tr>
<tr><td><strong>WW MOTOR</strong></td><td><code>10A</code></td><td>Control Circuit</td></tr>
</table>

<p><em>Source: Section 2.4</em></p>'''

DTC_INFO = '''<h3>📚 What is DTC?</h3>
<p><strong>Diagnostic Trouble Code</strong> - Standard fault identifier (SAE J 2012v003)</p>

<table>
<tr><th>Code</th><th>Type</th></tr>
<tr><td><code>P</code></td><td>Powertrain</td></tr>
<tr><td><code>B</code></td><td>Body</td></tr>
<tr><td><code>C</code></td><td>Chassis</td></tr>
<tr><td><code>U</code></td><td>Network</td></tr>
</table>

<p><strong>CKNL:</strong> Lights after 3 cycles, clears after 3 fault-free cycles</p>'''

OUT_OF_SCOPE = '''<h3>⚠️ Out of Scope</h3>
<p>That's not automotive diagnostics. Try: DTCs, components, symptoms.</p>'''

def find_match(query, context):
    """Smart matching - remembers context from symptoms AND DTCs"""
    q = query.lower()
    q = re.sub(r'\b(the|a|an|is|are|was|were|in|on|at|to|for|of|with|from)\b', ' ', q)
    q = ' '.join(q.split())
    
    # Out of scope
    if any(kw in q for kw in ['program', '.c', 'python', 'java', 'adding', 'write program']):
        return OUT_OF_SCOPE, None
    if any(kw in q for kw in ['dwg', 'autocad', 'diameter', 'cad', '2d']):
        return OUT_OF_SCOPE, None
    
    # === DETECT NEW CONTEXT (from DTC or Symptom) ===
    new_context = None
    
    # Direct DTC mention
    if 'p0117' in q or 'p 0117' in q or ('tml' in q and 'diagnostics' in q):
        new_context = 'P0117'
    
    # Symptom-based detection (FAN CONTINUOUS = P0117 context)
    fan_kw = ['fan', 'radiator', 'cooling']
    continuous_kw = ['continuous', 'always', 'runs', 'running', 'wont stop', 'constantly', 'on always']
    if any(f in q for f in fan_kw) and any(c in q for c in continuous_kw):
        new_context = 'P0117'  # Fan continuous is symptom of P0117
    
    # === CONTEXT-SPECIFIC RESPONSES ===
    if context == 'P0117' or new_context == 'P0117':
        # Just asking about symptom/DTC - give full description
        if new_context == 'P0117' and not any(x in q for x in ['picture', 'image', 'rectify', 'fix', 'table', 'repair']):
            if 'fan' in q and 'continuous' in q:
                return FAN_CONTINUOUS, 'P0117'
            elif 'description' in q or 'detail' in q:
                return P0117_DESCRIPTION, 'P0117'
            else:
                return P0117_DESCRIPTION, 'P0117'
        
        # PRIORITY 1: Show table (check this FIRST)
        if 'table' in q or ('list' in q and ('p code' in q or 'fault' in q or 'repair' in q or 'cause' in q or 'steps' in q)):
            return P0117_TABLE, 'P0117'
        
        # PRIORITY 2: Show images
        if ('picture' in q or 'image' in q or 'photo' in q or 'show') and ('faulty' in q or 'part' in q or 'component' in q or 'sensor' in q):
            return P0117_IMAGES, 'P0117'
        
        # Just "show images" or "show picture" without context clues
        if ('show' in q or 'display' in q) and ('image' in q or 'picture' in q or 'photo' in q):
            return P0117_IMAGES, 'P0117'
        
        # PRIORITY 3: Show repair steps (only if NOT table)
        if ('rectify' in q or 'fix' in q or 'repair' in q or 'solve' in q or 'how to') and ('problem' in q or 'issue' in q or 'fault' in q or 'it' in q):
            return P0117_REPAIR, 'P0117'
        
        # Just "how to fix" without specific mention
        if ('how' in q and 'fix' in q) or ('how' in q and 'repair' in q) or 'rectify' in q:
            return P0117_REPAIR, 'P0117'
    
    # === OTHER QUESTIONS (No context needed) ===
    
    # Window fuse
    window_kw = ['window', 'ww', 'winding']
    fuse_kw = ['fuse', 'rating', 'ampere', 'amp']
    if any(w in q for w in window_kw) and any(f in q for f in fuse_kw):
        return WINDOW_FUSE, None
    
    # DTC info
    if 'what is dtc' in q or 'dtc meaning' in q or 'define dtc' in q:
        return DTC_INFO, None
    
    # === SMART SUGGESTIONS ===
    if context:
        return f'''<h3>💡 Context: {context}</h3>
<p>Ask:</p>
<ul>
<li>"Show images" or "Show me picture of faulty part"</li>
<li>"How to fix" or "Show me how to rectify"</li>
<li>"Show table" or "List P Code, Fault cause"</li>
</ul>''', context
    
    return '''<h3>❓ Try These</h3>
<ul>
<li>"Radiator fan runs continuously" (sets P0117 context)</li>
<li>"P0117 showing" (sets P0117 context)</li>
<li>"Window fuse rating"</li>
<li>"What is DTC?"</li>
</ul>''', None

@app.route('/')
def home():
    session.clear()
    return render_template_string(HTML)

@app.route('/api/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '')
    current_context = session.get('fault_context', None)
    
    response, new_context = find_match(msg, current_context)
    
    if new_context:
        session['fault_context'] = new_context
    
    return jsonify({
        'response': response,
        'context': session.get('fault_context', None)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 TATA Nano Diagnostics - SMART CONTEXT!")
    print("=" * 60)
    print("✅ Remembers context from symptoms too!")
    print("✅ Open: http://localhost:5001")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)