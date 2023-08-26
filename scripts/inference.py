import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv
import dgl
import dgl.data
import pandas as pd

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
    
# Load the saved model
model_path = "../model/dgl_model.pt"  # Replace with the actual path to your saved model
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

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
    
def main():
    dataset = dgl.data.CSVDataset('../data/author_data')
    g = dataset[0]
    g = dgl.add_self_loop(g)
    input_author_id = input("Enter an author ID: ")  # Input in the format: authorID_5ef6f_df325_13aa7_cd11f_72bec

    top_coauthors, likeliness_scores = get_top_coauthors(input_author_id, model, g, author_id_to_number, author_num_to_id)

    print("Input Author ID:", input_author_id)
    print("Top Coauthors:", top_coauthors)
    print("Likeliness Scores:", likeliness_scores)

if __name__ == "__main__":
    main()
