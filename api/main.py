from flask import Flask, request, jsonify
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv
import dgl
import dgl.data
import pandas as pd

# Define the GraphSAGE class and other functions/constants from your original code

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Initialize Flask
app = Flask(__name__)

# Load the saved model and other data
model_path = "../model/dgl_model.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Load author data and prepare necessary data structures
# ...
author_ids = pd.read_csv('../data/author_id.csv')
author_id_to_number = {author_id: idx for idx, author_id in enumerate(author_ids['Author'])}
author_num_to_id = {v: k for k, v in author_id_to_number.items()}

def get_top_coauthors(author_id, model, g, author_id_to_number, author_num_to_id, num_top_coauthors=5):
    if author_id not in author_id_to_number:
        print("Invalid author ID. Please provide a valid author ID.")
        return [], []
    
    author_number = author_id_to_number[author_id]
    
    with torch.no_grad():
        
        h = model(g, g.ndata['feat'].type(torch.float32))
        author_embedding = h[author_number]

        similarity_scores = torch.cosine_similarity(author_embedding, h, dim=1)
        top_indices = np.argsort(similarity_scores.numpy())[-num_top_coauthors:][::-1]
        top_coauthors = [author_num_to_id[i] for i in top_indices]
        likeliness_scores = [float(similarity_scores[i]) for i in top_indices]

        return top_coauthors, likeliness_scores


@app.route('/get_possible_coauthors', methods=['GET'])
def predict():
    input_author_id = request.args.get('author_id')
    
    if input_author_id is None:
        return jsonify({"error": "Missing author_id parameter"}), 400
    dataset = dgl.data.CSVDataset('../data/author_data')
    g = dataset[0]
    g = dgl.add_self_loop(g)
    top_coauthors, likeliness_scores = get_top_coauthors(input_author_id, model, g, author_id_to_number, author_num_to_id)

    response = []
    for author_id, likeliness in zip(top_coauthors, likeliness_scores):
        response.append({"author_id": author_id, "likeliness": likeliness})

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
