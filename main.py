from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class
import torchaudio
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Ganti daftar origin sesuai kebutuhan deployment
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model selection
MODEL_NAME = "Jzuluaga/accent-id-commonaccent_ecapa"  

# Switch-case untuk inisialisasi classifier
if MODEL_NAME == "Jzuluaga/accent-id-commonaccent_ecapa":
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_NAME,
        savedir="pretrained_models/accent-id-commonaccent_ecapa"
    )
elif MODEL_NAME == "Jzuluaga/accent-id-commonaccent_xlsr-en-english":
    classifier = foreign_class(
        source=MODEL_NAME,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
else:
    raise ValueError(f"Unknown model name: {MODEL_NAME}")

@app.post("/classify-us-accent")
async def classify_us_accent(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Classify the accent
        out_prob, score, index, text_lab = classifier.classify_file(tmp_path)
        # Find the index for 'us' accent
        us_index = None
        for i, label in enumerate(classifier.hparams.label_encoder.lab2ind.keys()):
            if label == 'us':
                us_index = i
                break
        if us_index is None:
            raise HTTPException(status_code=500, detail="US accent label not found in model.")
        us_confidence = float(out_prob[0][us_index]) * 100
        return {"us_confidence": round(us_confidence, 1)}
    finally:
        os.remove(tmp_path)