from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
inference_model = InferenceModel()
smile = "CCCCCCC"
smiles_input = []
smiles_input.append(preprocess_smiles(smile))
emb = inference_model.seq_to_emb(smiles_input)
print(emb)
