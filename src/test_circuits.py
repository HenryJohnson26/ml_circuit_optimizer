import sys
import json
import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    from src.cnn import CNNPredictor
    checkpoint = torch.load(path, map_location=DEVICE)
    model = CNNPredictor(
        gate_vocab=checkpoint['vocab_size'],
        num_qubits=checkpoint['num_qubits'],
        embed_dim=checkpoint['embed_dim'],
        q_embed=checkpoint['q_embed'],
        hidden_dim=checkpoint['hidden_dim']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def optimize_circuit(model, circuit_ints, window, no_q_idx, max_steps=100):
    optimized = []
    for i in range(len(circuit_ints)):
        if i < window:
            optimized.append(circuit_ints[i])
        else:
            win = optimized[-window:]
            win_mapped = [[g, q1, q2 if q2 >= 0 else no_q_idx] for g, q1, q2 in win]
            x = torch.tensor([win_mapped], dtype=torch.long).to(DEVICE)

            with torch.no_grad():
                g_logits, q1_logits, q2_logits = model(x)
                pred_gate = int(torch.argmax(g_logits).item())
                pred_q1 = int(torch.argmax(q1_logits).item())
                pred_q2 = int(torch.argmax(q2_logits).item())

            pred_q2 = pred_q2 if pred_q2 != no_q_idx else -1
            optimized.append([pred_gate, pred_q1, pred_q2])

            if len(optimized) >= max_steps:
                break
    return optimized

def ints_to_tokens(ints, gate_vocab):
    tokens = []
    for g_id, q1, q2 in ints:
        gate_name = gate_vocab[g_id]
        if q2 == -1:
            tokens.append([gate_name, [q1], None])
        else:
            tokens.append([gate_name, [q1, q2], None])
    return tokens

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ml_opt_circuit <model_path> [input_circuit.json]")
        sys.exit(1)

    model_path = sys.argv[1]
    model, meta = load_model(model_path)
    print(f"Loaded model: {model_path}")

    if len(sys.argv) >= 3:
        input_file = sys.argv[2]
        with open(input_file) as f:
            data = json.load(f)
        circuits = [data["raw_ints"]]
    else:
        data_dir = Path("data/tokenized")
        circuits = []
        for f in list(data_dir.glob("*.json"))[:5]:
            with open(f) as fp:
                circuits.append(json.load(fp)["raw_ints"])

    output_dir = Path("output_circuits")
    output_dir.mkdir(exist_ok=True)

    for idx, circuit in enumerate(circuits):
        optimized_ints = optimize_circuit(model, circuit, meta['window'], meta['no_q_idx'])
        optimized_tokens = ints_to_tokens(optimized_ints, meta['gate_vocab'])

        output = {
            "input_tokens": ints_to_tokens(circuit, meta['gate_vocab']),
            "input_ints": circuit,
            "optimized_tokens": optimized_tokens,
            "optimized_ints": optimized_ints,
            "n_qubits": meta['num_qubits']
        }

        output_file = output_dir / f"optimized_circuit_{idx}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Circuit {idx}: {len(circuit)} gates -> {len(optimized_ints)} gates")
        print(f"  Saved to {output_file}")
