import argparse

import torch

from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data, load_ki_graph_data
from utils.constants import *


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def predict(gat, node_features, edge_index, pred_indices, slide_name):
    node_dim = 0  # node axis

    # train_labels = node_labels.index_select(node_dim, train_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it

    def main_loop():
        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        gat.eval()

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, pred_indices)

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)

        print(f'exporting csv: {slide_name}_gat_prediction.csv')
        class_predictions.cpu().numpy().tofile(f'{slide_name}_gat_prediction.csv', sep=',')
        print('done!')

    return main_loop  # return the decorated function
ki_prediction_paths = [
    # train
    ['P01_1_1', 'P01_1_1_delaunay_forGAT_pred_edges.csv', 'P01_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['N10_1_1', 'N10_1_1_delaunay_forGAT_pred_edges.csv', 'N10_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['N10_1_2', 'N10_1_2_delaunay_forGAT_pred_edges.csv', 'N10_1_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_2_1', 'N10_2_1_delaunay_forGAT_pred_edges.csv', 'N10_2_1_delaunay_forGAT_pred_nodes.csv'],
    ['N10_2_2', 'N10_2_2_delaunay_forGAT_pred_edges.csv', 'N10_2_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_3_1', 'N10_3_1_delaunay_forGAT_pred_edges.csv', 'N10_3_1_delaunay_forGAT_pred_nodes.csv'],
    ['N10_3_2', 'N10_3_2_delaunay_forGAT_pred_edges.csv', 'N10_3_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_4_1', 'N10_4_1_delaunay_forGAT_pred_edges.csv', 'N10_4_1_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_1_1', 'P7_HE_Default_Extended_1_1_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_3_2', 'P7_HE_Default_Extended_3_2_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_3_2_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_4_2', 'P7_HE_Default_Extended_4_2_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_4_2_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_5_2', 'P7_HE_Default_Extended_5_2_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_5_2_delaunay_forGAT_pred_nodes.csv'],
    ['P11_1_1', 'P11_1_1_delaunay_forGAT_pred_edges.csv', 'P11_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['P9_1_1', 'P9_1_1_delaunay_forGAT_pred_edges.csv', 'P9_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['P9_3_1', 'P9_3_1_delaunay_forGAT_pred_edges.csv', 'P9_3_1_delaunay_forGAT_pred_nodes.csv'],
    ['P20_6_1', 'P20_6_1_delaunay_forGAT_pred_edges.csv',
     'P20_6_1_delaunay_forGAT_pred_nodes.csv'],
    ['P19_1_1', 'P19_1_1_delaunay_forGAT_pred_edges.csv', 'P19_1_1_delaunay_forGAT_pred_nodes.csv'],
    ['P19_3_2', 'P19_3_2_delaunay_forGAT_pred_edges.csv',
     'P19_3_2_delaunay_forGAT_pred_nodes.csv'],

    # val
    ['N10_4_2', 'N10_4_2_delaunay_forGAT_pred_edges.csv', 'N10_4_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_5_2', 'N10_5_2_delaunay_forGAT_pred_edges.csv', 'N10_5_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_6_2', 'N10_6_2_delaunay_forGAT_pred_edges.csv', 'N10_6_2_delaunay_forGAT_pred_nodes.csv'],
    ['P19_2_1', 'P19_2_1_delaunay_forGAT_pred_edges.csv', 'P19_2_1_delaunay_forGAT_pred_nodes.csv'],
    ['P9_4_1', 'P9_4_1_delaunay_forGAT_pred_edges.csv', 'P9_4_1_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_2_1', 'P7_HE_Default_Extended_2_1_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_2_1_delaunay_forGAT_pred_nodes.csv'],

    # test
    ['P9_2_1', 'P9_2_1_delaunay_forGAT_pred_edges.csv', 'P9_2_1_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_2_2', 'P7_HE_Default_Extended_2_2_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_2_2_delaunay_forGAT_pred_nodes.csv'],
    ['P7_HE_Default_Extended_3_1', 'P7_HE_Default_Extended_3_1_delaunay_forGAT_pred_edges.csv',
     'P7_HE_Default_Extended_3_1_delaunay_forGAT_pred_nodes.csv'],
    ['P20_5_1', 'P20_5_1_delaunay_forGAT_pred_edges.csv', 'P20_5_1_delaunay_forGAT_pred_nodes.csv'],
    ['P19_3_1', 'P19_3_1_delaunay_forGAT_pred_edges.csv', 'P19_3_1_delaunay_forGAT_pred_nodes.csv'],
    ['P13_1_1', 'P13_1_1_delaunay_forGAT_pred_edges.csv', 'P13_1_1_delaunay_forGAT_pred_nodes.csv'],
    # ['P7_HE_Default_Extended_4_1', 'P7_HE_Default_Extended_4_1_delaunay_forGAT_pred_edges.csv',
    #  'P7_HE_Default_Extended_4_1_delaunay_forGAT_pred_nodes.csv'],
    ['P13_2_2', 'P13_2_2_delaunay_forGAT_pred_edges.csv', 'P13_2_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_7_2', 'N10_7_2_delaunay_forGAT_pred_edges.csv', 'N10_7_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_7_3', 'N10_7_3_delaunay_forGAT_pred_edges.csv', 'N10_7_3_delaunay_forGAT_pred_nodes.csv'],
    ['N10_8_2', 'N10_8_2_delaunay_forGAT_pred_edges.csv', 'N10_8_2_delaunay_forGAT_pred_nodes.csv'],
    ['N10_8_3', 'N10_8_3_delaunay_forGAT_pred_edges.csv', 'N10_8_3_delaunay_forGAT_pred_nodes.csv'],
    ['P11_1_2', 'P11_1_2_delaunay_forGAT_pred_edges.csv', 'P11_1_2_delaunay_forGAT_pred_nodes.csv'],
    ['P11_2_2', 'P11_2_2_delaunay_forGAT_pred_edges.csv', 'P11_2_2_delaunay_forGAT_pred_nodes.csv'],

]




def predict_gat_ki(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 2: prepare the model
    gat = GAT(num_of_layers=config['num_of_layers'],
              num_heads_per_layer=config['num_heads_per_layer'],
              num_features_per_layer=config['num_features_per_layer'],
              add_skip_connection=config['add_skip_connection'],
              bias=config['bias'],
              dropout=config['dropout'],
              layer_type=config['layer_type'],
              log_attention_weights=False).to(device)  # TODO

    gat.load_state_dict(torch.load('./models/binaries/gat_KI_000087.pth')['state_dict'])  # 0.83 acc

    # Step 1: load the graph data
    for paths in ki_prediction_paths:

        print(f'loading {paths[0]}')
        node_features, node_labels, edge_index, pred_indices = \
            load_ki_graph_data(device, paths, '')
        print(f'loaded!')
        slide_name = paths[0]  # TODO

        main_loop = predict(gat, node_features, edge_index, pred_indices, slide_name)

        # Prediction
        try:
            main_loop()
        except Exception as e:  # "patience has run out" exception :O
            print(str(e))


def get_prediction_args():
    parser = argparse.ArgumentParser()

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.KI.name)
    # Logging/debugging/checkpoint related (helps a lot with experimentation)

    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        # "num_heads_per_layer": [8, 1],
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [KI_NUM_INPUT_FEATURES, 8, KI_NUM_CLASSES],
        "add_skip_connection": True,  # hurts perf on Cora
        "bias": False,  # result is not so sensitive to bias
        # "dropout": 0.6,  # result is sensitive to dropout
        "dropout": 0,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    prediction_config = dict()
    for arg in vars(args):
        prediction_config[arg] = getattr(args, arg)

    # Add additional config information
    prediction_config.update(gat_config)

    return prediction_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    predict_gat_ki(get_prediction_args())
