#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:54:42 2025

@author: rameshbk
"""

# local_api_server.py
# Flask server to expose the RAG logic to the HTML client.
# Now includes:
#   - /api/chat   : text → RAG (run_on_device_rag)
#   - /api/speech : WAV audio → Vosk STT → RAG

from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import os
import sys
import uuid
import wave
import audioop
import json

# Offline STT (Vosk)
from vosk import Model, KaldiRecognizer

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})

# IMPORTANT: Ensure your RAG script file is accessible and importable
try:
    # Import the main RAG function from your script
    from updated_hybrid_rag_ollama_on_device_1 import (
        run_on_device_rag,
        load_data_from_files,
    )
except ImportError:
    print(
        "FATAL ERROR: Could not import 'run_on_device_rag' from "
        "updated_hybrid_rag_ollama_on_device.py"
    )
    print("Please ensure the RAG script is in the same directory.")
    sys.exit(1)

# --- Load the persistent knowledge base once on startup ---
load_data_from_files()

# --- Load Vosk model for offline speech recognition ---
# Set VOSK_MODEL_PATH env var or keep the default path below
VOSK_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH", "./models/vosk-model-small-en-us-0.15"
)

vosk_model = None
if os.path.isdir(VOSK_MODEL_PATH):
    print(f"✅ Loading Vosk model from: {VOSK_MODEL_PATH}")
    vosk_model = Model(VOSK_MODEL_PATH)
else:
    print(
        f"⚠️ Vosk model not found at {VOSK_MODEL_PATH}. "
        f"Set VOSK_MODEL_PATH or download the model there."
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def serve_root():
    """Serve the main Nano chatbot UI."""
    return send_from_directory(BASE_DIR, "nano_3.html")

@app.route("/nano_3.html")
def serve_nano3():
    """Also serve the UI if someone explicitly asks for /nano_3.html."""
    return send_from_directory(BASE_DIR, "nano_3.html")


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Text query endpoint used by the Nano HTML UI.
    Expects JSON: { "message": "your question" }
    """
    try:
        # Get the query from the HTML client's JSON payload
        data = request.get_json()
        query = data.get("message", "")

        if not query:
            return jsonify({"error": "No query message provided."}), 400

        # Execute the core RAG logic (calling Ollama locally)
        output = run_on_device_rag(query)

        # Prepare response for the HTML client
        response_data = {
            "answer": output["answer"],
            "sources": {
                "vector_chunks": [
                    {
                        "text": c["text"],
                        "score": c["score"],
                        "page": c["page"],
                    }
                    for c in output["vdb_chunks"]
                ],
                "scores": f"KG Triples: {len(output['kg_triples'])}",
                "locked_specs": output["locked_specs"],
            },
        }

        return jsonify(response_data)

    except Exception as e:
        # Return a generic error to the client
        return jsonify(
            {"error": f"Internal Server Error during RAG process: {str(e)}"}
        ), 500


@app.route("/api/speech", methods=["POST"])
def speech_endpoint():
    """
    Offline speech endpoint.

    Frontend sends:
        FormData with field 'audio' = WAV blob from browser (mono/16-bit/any rate).
    Steps:
        1. Save WAV to temp file
        2. Use Vosk to transcribe (convert to mono + 16 kHz internally)
        3. Call run_on_device_rag(transcript)
        4. Return same structure as /api/chat plus 'transcript'
    """
    if vosk_model is None:
        return jsonify({"error": "Vosk model not available on server"}), 500

    if "audio" not in request.files:
        return jsonify(
            {"error": "No audio file uploaded. Use field name 'audio'."}
        ), 400

    audio_file = request.files["audio"]

    # Save to a temporary path
    tmp_path = os.path.join("/tmp", f"nano_upload_{uuid.uuid4().hex}.wav")
    audio_file.save(tmp_path)

    try:
        wf = wave.open(tmp_path, "rb")
    except Exception as e:
        return jsonify({"error": f"Cannot read WAV file: {str(e)}"}), 400

    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()

    if sampwidth != 2:
        wf.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return jsonify({"error": "Audio must be 16-bit PCM WAV."}), 400

    TARGET_RATE = 16000
    recognizer = KaldiRecognizer(vosk_model, TARGET_RATE)

    transcript_parts = []

    try:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            # Convert stereo → mono if needed
            if n_channels > 1:
                data_mono = audioop.tomono(data, sampwidth, 0.5, 0.5)
            else:
                data_mono = data

            # Resample to 16 kHz if needed
            if framerate != TARGET_RATE:
                data_mono, _ = audioop.ratecv(
                    data_mono, sampwidth, 1, framerate, TARGET_RATE, None
                )

            if recognizer.AcceptWaveform(data_mono):
                res = json.loads(recognizer.Result())
                if "text" in res:
                    transcript_parts.append(res["text"])

        final_res = json.loads(recognizer.FinalResult())
        if "text" in final_res:
            transcript_parts.append(final_res["text"])
    finally:
        wf.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    transcript = " ".join(t.strip() for t in transcript_parts if t.strip())
    transcript = transcript.strip()

    if not transcript:
        return jsonify(
            {"error": "Could not recognize any speech from audio."}
        ), 400

    # ✅ Now reuse the same RAG pipeline used for text
    try:
        output = run_on_device_rag(transcript)
    except Exception as e:
        return jsonify(
            {
                "error": f"Recognized speech, but RAG failed: {e}",
                "transcript": transcript,
            }
        ), 500

    response_data = {
        "answer": output["answer"],
        "sources": {
            "vector_chunks": [
                {
                    "text": c["text"],
                    "score": c["score"],
                    "page": c["page"],
                }
                for c in output["vdb_chunks"]
            ],
            "scores": f"KG Triples: {len(output['kg_triples'])}",
            "locked_specs": output["locked_specs"],
        },
        "transcript": transcript,
    }

    return jsonify(response_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("-" * 50)
    print(f"🚀 Starting RAG API Server on http://localhost:{port}")
    print("Make sure Ollama is running!")
    print(f"Vosk model path: {VOSK_MODEL_PATH} (loaded={vosk_model is not None})")
    print("-" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
