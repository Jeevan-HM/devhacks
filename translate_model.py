import io
import json
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn

# Device and transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=80).to(device)


# Model class
class HumToText(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, output_dim=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.mean(dim=1)
        return self.classifier(out)


# Feature extraction from in-memory audio (bytes)
def extract_features_from_buffer(wav_bytes):
    waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
    waveform = waveform.to(device)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    mel_spec = mel_transform(waveform)
    return mel_spec.squeeze(0).transpose(0, 1)


# Prediction function
def predict_from_buffer(model, wav_bytes, phrase_list):
    model.eval()
    features = extract_features_from_buffer(wav_bytes).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[1]]).to(device)

    with torch.no_grad():
        output = model(features, lengths)
        predicted_index = output.argmax(dim=1).item()
        return phrase_list[predicted_index]


# === USAGE EXAMPLE ===
def run_prediction(audio_buffer):
    # Load phrase labels
    with open("outputs/labels.json") as f:
        label_map = json.load(f)
    phrases = sorted(set(entry["phrase"] for entry in label_map.values()))

    # Load model and weights
    model = HumToText(output_dim=len(phrases)).to(device)
    model.load_state_dict(torch.load("hum_to_text.pt", map_location=device))

    # Predict
    predicted_phrase = predict_from_buffer(model, audio_buffer, phrases)
    return predicted_phrase
