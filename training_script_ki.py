import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from sklearn import metrics

from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data, load_ki_graph_data
from utils.constants import *
import utils.utils as utils


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, cross_entropy_loss, optimizer,
                  train_node_features, train_node_labels, train_edge_index, train_indices,
                  patience_period, time_start, val_node_features=None, val_node_labels=None, val_edge_index=None,
                  val_indices=None,
                  test_node_features=None, test_node_labels=None, test_edge_index=None, test_indices=None):
    node_dim = 0  # node axis

    # train_labels = node_labels.index_select(node_dim, train_indices)
    # val_labels = node_labels.index_select(node_dim, val_indices)
    # test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    train_graph_data = (train_node_features, train_edge_index)
    # I pack data into tuples because GAT uses nn.Sequential which requires it
    val_graph_data = (val_node_features, val_edge_index)
    test_graph_data = (test_node_features, test_edge_index)

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        elif phase == LoopPhase.TEST:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            return train_node_labels
        elif phase == LoopPhase.VAL:
            return val_node_labels
        elif phase == LoopPhase.TEST:
            return test_node_labels

    def get_graph_data(phase):
        if phase == LoopPhase.TRAIN:
            return train_graph_data
        elif phase == LoopPhase.VAL:
            return val_graph_data
        elif phase == LoopPhase.TEST:
            return test_graph_data

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices = get_node_indices(phase)
        gt_node_labels = get_node_labels(phase)  # gt stands for ground truth
        graph_data = get_graph_data(phase)
        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

        # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
        # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
        # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
        # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
        # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
        # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
        # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        if phase == LoopPhase.TEST:
            print('exporting csv...')
            class_predictions.cpu().numpy().tofile('gat_prediction.csv', sep=',')
            print('done!')

            print(metrics.classification_report(gt_node_labels.cpu().numpy(), class_predictions.cpu().numpy(), digits=4))

        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

            if config['final'] and config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(
                    f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | train acc={accuracy}')

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(
                    f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                print(f'last epoch: {epoch}')
                print(metrics.classification_report(gt_node_labels.cpu().numpy(), class_predictions.cpu().numpy(),
                                                    digits=4))
                raise Exception('Stopping the training, the universe has no more patience for this training.')

            # {0: "green", 1: "yellow", 2: "blue", 3: "red"}

        elif phase == LoopPhase.TEST:
            return accuracy  # recalls, precisions, f1_scores
            # in the case of test phase we just report back the test accuracy
        else:
            return

    return main_loop  # return the decorated function


ki_prediction_paths = [
    # ['train', 'train_edges_k6.csv', 'train_nodes.csv'],
    ['train', 'train_edges_delaunay.csv', 'train_nodes.csv'],
    # ['val', 'val_edges_k6.csv', 'val_nodes.csv'],
    ['val', 'val_edges_delaunay.csv', 'val_nodes.csv'],
    # ['test', 'test_edges_k6.csv', 'test_nodes.csv'],
    ['test', 'test_edges_delaunay.csv', 'test_nodes.csv'],
]

# ki_train_paths = [
#     ["HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_edges.csv", "HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_edges.csv",
#     #  "HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_edges.csv",
#     #  "HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_edges.csv", "HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_edges.csv",
#     #  "HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_edges.csv",
#     #  "HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["N10_1_1_gen0_delaunay_forGAT_train_edges.csv", "N10_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["N10_1_1_gen0_k10_forGAT_train_edges.csv", "N10_1_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["N10_1_1_gen0_k6_forGAT_train_edges.csv", "N10_1_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["N10_1_3_gen0_delaunay_forGAT_train_edges.csv", "N10_1_3_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["N10_1_3_gen0_k10_forGAT_train_edges.csv", "N10_1_3_gen0_k10_forGAT_train_nodes.csv"],
#     # ["N10_1_3_gen0_k6_forGAT_train_edges.csv", "N10_1_3_gen0_k6_forGAT_train_nodes.csv"],
#     ["P11_1_1_gen0_delaunay_forGAT_train_edges.csv", "P11_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P11_1_1_gen0_k10_forGAT_train_edges.csv", "P11_1_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P11_1_1_gen0_k6_forGAT_train_edges.csv", "P11_1_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P11_2_2_gen0_delaunay_forGAT_train_edges.csv", "P11_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P11_2_2_gen0_k10_forGAT_train_edges.csv", "P11_2_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P11_2_2_gen0_k6_forGAT_train_edges.csv", "P11_2_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P13_1_1_gen0_delaunay_forGAT_train_edges.csv", "P13_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P13_1_1_gen0_k10_forGAT_train_edges.csv", "P13_1_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P13_1_1_gen0_k6_forGAT_train_edges.csv", "P13_1_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P13_1_2_gen0_delaunay_forGAT_train_edges.csv", "P13_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P13_1_2_gen0_k10_forGAT_train_edges.csv", "P13_1_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P13_1_2_gen0_k6_forGAT_train_edges.csv", "P13_1_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P19_1_1_gen1_delaunay_forGAT_train_edges.csv", "P19_1_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_1_1_gen1_k10_forGAT_train_edges.csv", "P19_1_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_1_1_gen1_k6_forGAT_train_edges.csv", "P19_1_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P19_1_2_gen1_delaunay_forGAT_train_edges.csv", "P19_1_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_1_2_gen1_k10_forGAT_train_edges.csv", "P19_1_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_1_2_gen1_k6_forGAT_train_edges.csv", "P19_1_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P19_2_1_gen1_delaunay_forGAT_train_edges.csv", "P19_2_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_2_1_gen1_k10_forGAT_train_edges.csv", "P19_2_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_2_1_gen1_k6_forGAT_train_edges.csv", "P19_2_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_1_3_gen1_delaunay_forGAT_train_edges.csv", "P20_1_3_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_1_3_gen1_k10_forGAT_train_edges.csv", "P20_1_3_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_1_3_gen1_k6_forGAT_train_edges.csv", "P20_1_3_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_2_2_gen0_delaunay_forGAT_train_edges.csv", "P20_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_2_2_gen0_k10_forGAT_train_edges.csv", "P20_2_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_2_2_gen0_k6_forGAT_train_edges.csv", "P20_2_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_2_3_gen1_delaunay_forGAT_train_edges.csv", "P20_2_3_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_2_3_gen1_k10_forGAT_train_edges.csv", "P20_2_3_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_2_3_gen1_k6_forGAT_train_edges.csv", "P20_2_3_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_2_4_gen0_delaunay_forGAT_train_edges.csv", "P20_2_4_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_2_4_gen0_k10_forGAT_train_edges.csv", "P20_2_4_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_2_4_gen0_k6_forGAT_train_edges.csv", "P20_2_4_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_3_1_gen0_delaunay_forGAT_train_edges.csv", "P20_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_3_1_gen0_k10_forGAT_train_edges.csv", "P20_3_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_3_1_gen0_k6_forGAT_train_edges.csv", "P20_3_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_3_2_gen1_delaunay_forGAT_train_edges.csv", "P20_3_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_3_2_gen1_k10_forGAT_train_edges.csv", "P20_3_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_3_2_gen1_k6_forGAT_train_edges.csv", "P20_3_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_3_3_gen1_delaunay_forGAT_train_edges.csv", "P20_3_3_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_3_3_gen1_k10_forGAT_train_edges.csv", "P20_3_3_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_3_3_gen1_k6_forGAT_train_edges.csv", "P20_3_3_gen1_k6_forGAT_train_nodes.csv"],
#     # ["P25_2_1_gen0_delaunay_forGAT_train_edges.csv", "P25_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_2_1_gen0_k10_forGAT_train_edges.csv", "P25_2_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_2_1_gen0_k6_forGAT_train_edges.csv", "P25_2_1_gen0_k6_forGAT_train_nodes.csv"],
#     # ["P25_3_1_gen0_delaunay_forGAT_train_edges.csv", "P25_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_3_1_gen0_k10_forGAT_train_edges.csv", "P25_3_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_3_1_gen0_k6_forGAT_train_edges.csv", "P25_3_1_gen0_k6_forGAT_train_nodes.csv"],
#     # ["P25_3_2_gen0_delaunay_forGAT_train_edges.csv", "P25_3_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_3_2_gen0_k10_forGAT_train_edges.csv", "P25_3_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_3_2_gen0_k6_forGAT_train_edges.csv", "P25_3_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P28_7_5_gen0_delaunay_forGAT_train_edges.csv", "P28_7_5_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P28_7_5_gen0_k10_forGAT_train_edges.csv", "P28_7_5_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P28_7_5_gen0_k6_forGAT_train_edges.csv", "P28_7_5_gen0_k6_forGAT_train_nodes.csv"],
#     ["P28_8_5_gen0_delaunay_forGAT_train_edges.csv", "P28_8_5_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P28_8_5_gen0_k10_forGAT_train_edges.csv", "P28_8_5_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P28_8_5_gen0_k6_forGAT_train_edges.csv", "P28_8_5_gen0_k6_forGAT_train_nodes.csv"],
#     ["P28_10_5_gen0_delaunay_forGAT_train_edges.csv", "P28_10_5_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P28_10_5_gen0_k6_forGAT_train_edges.csv", "P28_10_5_gen0_k6_forGAT_train_nodes.csv"],
#     ["P9_1_1_gen1_delaunay_forGAT_train_edges.csv", "P9_1_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_1_1_gen1_k6_forGAT_train_edges.csv", "P9_1_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P9_2_1_gen1_delaunay_forGAT_train_edges.csv", "P9_2_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_2_1_gen1_k10_forGAT_train_edges.csv", "P9_2_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_2_1_gen1_k6_forGAT_train_edges.csv", "P9_2_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P9_2_2_gen1_delaunay_forGAT_train_edges.csv", "P9_2_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_2_2_gen1_k10_forGAT_train_edges.csv", "P9_2_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_2_2_gen1_k6_forGAT_train_edges.csv", "P9_2_2_gen1_k6_forGAT_train_nodes.csv"],
# ]
# 
# ki_val_paths = [
#     ["N10_2_2_gen0_delaunay_forGAT_train_edges.csv", "N10_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["N10_2_2_gen0_k10_forGAT_train_edges.csv", "N10_2_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["N10_2_2_gen0_k6_forGAT_train_edges.csv", "N10_2_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P19_3_1_gen1_delaunay_forGAT_train_edges.csv", "P19_3_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_3_1_gen1_k10_forGAT_train_edges.csv", "P19_3_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_3_1_gen1_k6_forGAT_train_edges.csv", "P19_3_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_7_1_gen0_delaunay_forGAT_train_edges.csv", "P20_7_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_7_1_gen0_k10_forGAT_train_edges.csv", "P20_7_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_7_1_gen0_k6_forGAT_train_edges.csv", "P20_7_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_7_2_gen0_delaunay_forGAT_train_edges.csv", "P20_7_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_7_2_gen0_k10_forGAT_train_edges.csv", "P20_7_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_7_2_gen0_k6_forGAT_train_edges.csv", "P20_7_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_8_1_gen0_delaunay_forGAT_train_edges.csv", "P20_8_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_8_1_gen0_k10_forGAT_train_edges.csv", "P20_8_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_8_1_gen0_k6_forGAT_train_edges.csv", "P20_8_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_9_1_gen0_delaunay_forGAT_train_edges.csv", "P20_9_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_9_1_gen0_k10_forGAT_train_edges.csv", "P20_9_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_9_1_gen0_k6_forGAT_train_edges.csv", "P20_9_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_9_2_gen0_delaunay_forGAT_train_edges.csv", "P20_9_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_9_2_gen0_k10_forGAT_train_edges.csv", "P20_9_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_9_2_gen0_k6_forGAT_train_edges.csv", "P20_9_2_gen0_k6_forGAT_train_nodes.csv"],
#     # ["P25_8_2_gen0_delaunay_forGAT_train_edges.csv", "P25_8_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_8_2_gen0_k10_forGAT_train_edges.csv", "P25_8_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_8_2_gen0_k6_forGAT_train_edges.csv", "P25_8_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P28_10_4_gen0_delaunay_forGAT_train_edges.csv", "P28_10_4_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P28_10_4_gen0_k10_forGAT_train_edges.csv", "P28_10_4_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P28_10_4_gen0_k6_forGAT_train_edges.csv", "P28_10_4_gen0_k6_forGAT_train_nodes.csv"],
#     ["P9_4_2_gen1_delaunay_forGAT_train_edges.csv", "P9_4_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_4_2_gen1_k10_forGAT_train_edges.csv", "P9_4_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_4_2_gen1_k6_forGAT_train_edges.csv", "P9_4_2_gen1_k6_forGAT_train_nodes.csv"],
# ]
# 
# ki_test_paths = [
#     ["N10_1_2_gen0_delaunay_forGAT_train_edges.csv", "N10_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["N10_1_2_gen0_k10_forGAT_train_edges.csv", "N10_1_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["N10_1_2_gen0_k6_forGAT_train_edges.csv", "N10_1_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P11_1_2_gen0_delaunay_forGAT_train_edges.csv", "P11_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P11_1_2_gen0_k10_forGAT_train_edges.csv", "P11_1_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P11_1_2_gen0_k6_forGAT_train_edges.csv", "P11_1_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P11_2_1_gen0_delaunay_forGAT_train_edges.csv", "P11_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P11_2_1_gen0_k10_forGAT_train_edges.csv", "P11_2_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P11_2_1_gen0_k6_forGAT_train_edges.csv", "P11_2_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P13_2_1_gen0_delaunay_forGAT_train_edges.csv", "P13_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P13_2_1_gen0_k10_forGAT_train_edges.csv", "P13_2_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P13_2_1_gen0_k6_forGAT_train_edges.csv", "P13_2_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P13_2_2_gen0_delaunay_forGAT_train_edges.csv", "P13_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P13_2_2_gen0_k10_forGAT_train_edges.csv", "P13_2_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P13_2_2_gen0_k6_forGAT_train_edges.csv", "P13_2_2_gen0_k6_forGAT_train_nodes.csv"],
#     ["P19_2_2_gen1_delaunay_forGAT_train_edges.csv", "P19_2_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_2_2_gen1_k10_forGAT_train_edges.csv", "P19_2_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_2_2_gen1_k6_forGAT_train_edges.csv", "P19_2_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P19_3_2_gen1_delaunay_forGAT_train_edges.csv", "P19_3_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P19_3_2_gen1_k10_forGAT_train_edges.csv", "P19_3_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P19_3_2_gen1_k6_forGAT_train_edges.csv", "P19_3_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_4_1_gen1_delaunay_forGAT_train_edges.csv", "P20_4_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_4_1_gen1_k10_forGAT_train_edges.csv", "P20_4_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_4_1_gen1_k6_forGAT_train_edges.csv", "P20_4_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_4_2_gen1_delaunay_forGAT_train_edges.csv", "P20_4_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_4_2_gen1_k10_forGAT_train_edges.csv", "P20_4_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_4_2_gen1_k6_forGAT_train_edges.csv", "P20_4_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_4_3_gen0_delaunay_forGAT_train_edges.csv", "P20_4_3_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_4_3_gen0_k10_forGAT_train_edges.csv", "P20_4_3_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P20_4_3_gen0_k6_forGAT_train_edges.csv", "P20_4_3_gen0_k6_forGAT_train_nodes.csv"],
#     ["P20_5_1_gen1_delaunay_forGAT_train_edges.csv", "P20_5_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_5_1_gen1_k10_forGAT_train_edges.csv", "P20_5_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_5_1_gen1_k6_forGAT_train_edges.csv", "P20_5_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_5_2_gen1_delaunay_forGAT_train_edges.csv", "P20_5_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_5_2_gen1_k10_forGAT_train_edges.csv", "P20_5_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_5_2_gen1_k6_forGAT_train_edges.csv", "P20_5_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P20_6_1_gen1_delaunay_forGAT_train_edges.csv", "P20_6_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_6_1_gen1_k10_forGAT_train_edges.csv", "P20_6_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_6_1_gen1_k6_forGAT_train_edges.csv", "P20_6_1_gen1_k6_forGAT_train_nodes.csv"],
#     # ["P20_6_2_gen1_delaunay_forGAT_train_edges.csv", "P20_6_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P20_6_2_gen1_k10_forGAT_train_edges.csv", "P20_6_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P20_6_2_gen1_k6_forGAT_train_edges.csv", "P20_6_2_gen1_k6_forGAT_train_nodes.csv"],
#     # ["P25_4_2_gen0_delaunay_forGAT_train_edges.csv", "P25_4_2_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_4_2_gen0_k10_forGAT_train_edges.csv", "P25_4_2_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_4_2_gen0_k6_forGAT_train_edges.csv", "P25_4_2_gen0_k6_forGAT_train_nodes.csv"],
#     # ["P25_5_1_gen0_delaunay_forGAT_train_edges.csv", "P25_5_1_gen0_delaunay_forGAT_train_nodes.csv"],
#     # ["P25_5_1_gen0_k10_forGAT_train_edges.csv", "P25_5_1_gen0_k10_forGAT_train_nodes.csv"],
#     # ["P25_5_1_gen0_k6_forGAT_train_edges.csv", "P25_5_1_gen0_k6_forGAT_train_nodes.csv"],
#     ["P9_3_1_gen1_delaunay_forGAT_train_edges.csv", "P9_3_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_3_1_gen1_k10_forGAT_train_edges.csv", "P9_3_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_3_1_gen1_k6_forGAT_train_edges.csv", "P9_3_1_gen1_k6_forGAT_train_nodes.csv"],
#     ["P9_3_2_gen1_delaunay_forGAT_train_edges.csv", "P9_3_2_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_3_2_gen1_k10_forGAT_train_edges.csv", "P9_3_2_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_3_2_gen1_k6_forGAT_train_edges.csv", "P9_3_2_gen1_k6_forGAT_train_nodes.csv"],
#     ["P9_4_1_gen1_delaunay_forGAT_train_edges.csv", "P9_4_1_gen1_delaunay_forGAT_train_nodes.csv"],
#     # ["P9_4_1_gen1_k10_forGAT_train_edges.csv", "P9_4_1_gen1_k10_forGAT_train_nodes.csv"],
#     # ["P9_4_1_gen1_k6_forGAT_train_edges.csv", "P9_4_1_gen1_k6_forGAT_train_nodes.csv"],
# 
# ]

ki_train_paths = [

    ["N10_1_3_gen0_delaunay_forGAT_train_edges.csv", "N10_1_3_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["N10_1_3_gen0_k10_forGAT_train_edges.csv", "N10_1_3_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_3_gen0_k6_forGAT_train_edges.csv", "N10_1_3_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_delaunay_forGAT_train_edges.csv", "P11_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_k10_forGAT_train_edges.csv", "P11_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_k6_forGAT_train_edges.csv", "P11_1_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_delaunay_forGAT_train_edges.csv", "P11_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_k10_forGAT_train_edges.csv", "P11_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_k6_forGAT_train_edges.csv", "P11_2_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_1_1_gen0_delaunay_forGAT_train_edges.csv", "P13_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_1_1_gen0_k10_forGAT_train_edges.csv", "P13_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_1_1_gen0_k6_forGAT_train_edges.csv", "P13_1_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_1_2_gen0_delaunay_forGAT_train_edges.csv", "P13_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_1_2_gen0_k10_forGAT_train_edges.csv", "P13_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_1_2_gen0_k6_forGAT_train_edges.csv", "P13_1_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P19_1_1_gen1_delaunay_forGAT_train_edges.csv", "P19_1_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_1_1_gen1_k10_forGAT_train_edges.csv", "P19_1_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_1_1_gen1_k6_forGAT_train_edges.csv", "P19_1_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_1_2_gen1_delaunay_forGAT_train_edges.csv", "P19_1_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_1_2_gen1_k10_forGAT_train_edges.csv", "P19_1_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_1_2_gen1_k6_forGAT_train_edges.csv", "P19_1_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_1_3_gen1_delaunay_forGAT_train_edges.csv", "P20_1_3_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_1_3_gen1_k10_forGAT_train_edges.csv", "P20_1_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_1_3_gen1_k6_forGAT_train_edges.csv", "P20_1_3_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_2_2_gen0_delaunay_forGAT_train_edges.csv", "P20_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_2_2_gen0_k10_forGAT_train_edges.csv", "P20_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_2_2_gen0_k6_forGAT_train_edges.csv", "P20_2_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_2_3_gen1_delaunay_forGAT_train_edges.csv", "P20_2_3_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_2_3_gen1_k10_forGAT_train_edges.csv", "P20_2_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_2_3_gen1_k6_forGAT_train_edges.csv", "P20_2_3_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_2_4_gen0_delaunay_forGAT_train_edges.csv", "P20_2_4_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_2_4_gen0_k10_forGAT_train_edges.csv", "P20_2_4_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_2_4_gen0_k6_forGAT_train_edges.csv", "P20_2_4_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_3_1_gen0_delaunay_forGAT_train_edges.csv", "P20_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_3_1_gen0_k10_forGAT_train_edges.csv", "P20_3_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_3_1_gen0_k6_forGAT_train_edges.csv", "P20_3_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_delaunay_forGAT_train_edges.csv", "P25_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_k10_forGAT_train_edges.csv", "P25_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_k6_forGAT_train_edges.csv", "P25_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_delaunay_forGAT_train_edges.csv", "P25_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_k10_forGAT_train_edges.csv", "P25_3_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_k6_forGAT_train_edges.csv", "P25_3_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_delaunay_forGAT_train_edges.csv", "P25_3_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_k10_forGAT_train_edges.csv", "P25_3_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_k6_forGAT_train_edges.csv", "P25_3_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_7_5_gen0_delaunay_forGAT_train_edges.csv", "P28_7_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_7_5_gen0_k10_forGAT_train_edges.csv", "P28_7_5_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_7_5_gen0_k6_forGAT_train_edges.csv", "P28_7_5_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_8_5_gen0_delaunay_forGAT_train_edges.csv", "P28_8_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_8_5_gen0_k10_forGAT_train_edges.csv", "P28_8_5_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_8_5_gen0_k6_forGAT_train_edges.csv", "P28_8_5_gen0_k6_forGAT_train_nodes.csv"],
    # # ["P28_10_5_gen0_delaunay_forGAT_train_edges.csv", "P28_10_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_10_5_gen0_k6_forGAT_train_edges.csv", "P28_10_5_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_1_1_gen1_delaunay_forGAT_train_edges.csv", "P9_1_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_1_1_gen1_k6_forGAT_train_edges.csv", "P9_1_1_gen1_k6_forGAT_train_nodes.csv"],
    # ["P9_1_1_gen1_k10_forGAT_train_edges.csv", "P9_1_1_gen1_k10_forGAT_train_nodes.csv"],
    ["P9_2_2_gen1_delaunay_forGAT_train_edges.csv", "P9_2_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_2_2_gen1_k10_forGAT_train_edges.csv", "P9_2_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_2_2_gen1_k6_forGAT_train_edges.csv", "P9_2_2_gen1_k6_forGAT_train_nodes.csv"],

    ["P19_3_2_gen1_delaunay_forGAT_train_edges.csv", "P19_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_3_2_gen1_k10_forGAT_train_edges.csv", "P19_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_3_2_gen1_k6_forGAT_train_edges.csv", "P19_3_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_5_2_gen1_delaunay_forGAT_train_edges.csv", "P20_5_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_5_2_gen1_k10_forGAT_train_edges.csv", "P20_5_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_5_2_gen1_k6_forGAT_train_edges.csv", "P20_5_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_6_1_gen1_delaunay_forGAT_train_edges.csv", "P20_6_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_6_1_gen1_k10_forGAT_train_edges.csv", "P20_6_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_6_1_gen1_k6_forGAT_train_edges.csv", "P20_6_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_6_2_gen1_delaunay_forGAT_train_edges.csv", "P20_6_2_gen1_delaunay_forGAT_train_nodes.csv"],
    # ["P20_6_2_gen1_k10_forGAT_train_edges.csv", "P20_6_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_6_2_gen1_k6_forGAT_train_edges.csv", "P20_6_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_3_1_gen1_delaunay_forGAT_train_edges.csv", "P9_3_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_3_1_gen1_k10_forGAT_train_edges.csv", "P9_3_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_3_1_gen1_k6_forGAT_train_edges.csv", "P9_3_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_3_2_gen1_delaunay_forGAT_train_edges.csv", "P9_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_3_2_gen1_k10_forGAT_train_edges.csv", "P9_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_3_2_gen1_k6_forGAT_train_edges.csv", "P9_3_2_gen1_k6_forGAT_train_nodes.csv"],
]

ki_val_paths = [
    ["P19_2_1_gen1_delaunay_forGAT_train_edges.csv", "P19_2_1_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P19_2_1_gen1_k10_forGAT_train_edges.csv", "P19_2_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_2_1_gen1_k6_forGAT_train_edges.csv", "P19_2_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_3_2_gen1_delaunay_forGAT_train_edges.csv", "P20_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P20_3_2_gen1_k10_forGAT_train_edges.csv", "P20_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_3_2_gen1_k6_forGAT_train_edges.csv", "P20_3_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_3_3_gen1_delaunay_forGAT_train_edges.csv", "P20_3_3_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P20_3_3_gen1_k10_forGAT_train_edges.csv", "P20_3_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_3_3_gen1_k6_forGAT_train_edges.csv", "P20_3_3_gen1_k6_forGAT_train_nodes.csv"],
    ["N10_2_2_gen0_delaunay_forGAT_train_edges.csv", "N10_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["N10_2_2_gen0_k10_forGAT_train_edges.csv", "N10_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_2_2_gen0_k6_forGAT_train_edges.csv", "N10_2_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_7_1_gen0_delaunay_forGAT_train_edges.csv", "P20_7_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_7_1_gen0_k10_forGAT_train_edges.csv", "P20_7_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_7_1_gen0_k6_forGAT_train_edges.csv", "P20_7_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_7_2_gen0_delaunay_forGAT_train_edges.csv", "P20_7_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_7_2_gen0_k10_forGAT_train_edges.csv", "P20_7_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_7_2_gen0_k6_forGAT_train_edges.csv", "P20_7_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_8_1_gen0_delaunay_forGAT_train_edges.csv", "P20_8_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_8_1_gen0_k10_forGAT_train_edges.csv", "P20_8_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_8_1_gen0_k6_forGAT_train_edges.csv", "P20_8_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_9_1_gen0_delaunay_forGAT_train_edges.csv", "P20_9_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_9_1_gen0_k10_forGAT_train_edges.csv", "P20_9_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_9_1_gen0_k6_forGAT_train_edges.csv", "P20_9_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_9_2_gen0_delaunay_forGAT_train_edges.csv", "P20_9_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_9_2_gen0_k10_forGAT_train_edges.csv", "P20_9_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_9_2_gen0_k6_forGAT_train_edges.csv", "P20_9_2_gen0_k6_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_delaunay_forGAT_train_edges.csv", "P25_8_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_k10_forGAT_train_edges.csv", "P25_8_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_k6_forGAT_train_edges.csv", "P25_8_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_10_4_gen0_delaunay_forGAT_train_edges.csv", "P28_10_4_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_10_4_gen0_k10_forGAT_train_edges.csv", "P28_10_4_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_10_4_gen0_k6_forGAT_train_edges.csv", "P28_10_4_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_4_2_gen1_delaunay_forGAT_train_edges.csv", "P9_4_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P9_4_2_gen1_k10_forGAT_train_edges.csv", "P9_4_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_4_2_gen1_k6_forGAT_train_edges.csv", "P9_4_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_2_2_gen1_delaunay_forGAT_train_edges.csv", "P19_2_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P19_2_2_gen1_k10_forGAT_train_edges.csv", "P19_2_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_2_2_gen1_k6_forGAT_train_edges.csv", "P19_2_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_4_1_gen1_delaunay_forGAT_train_edges.csv", "P9_4_1_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P9_4_1_gen1_k10_forGAT_train_edges.csv", "P9_4_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_4_1_gen1_k6_forGAT_train_edges.csv", "P9_4_1_gen1_k6_forGAT_train_nodes.csv"],
]

ki_test_paths = [
    ["N10_1_1_gen0_delaunay_forGAT_train_edges.csv", "N10_1_1_gen0_delaunay_forGAT_train_nodes.csv"],  # test
    # ["N10_1_1_gen0_k10_forGAT_train_edges.csv", "N10_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_1_gen0_k6_forGAT_train_edges.csv", "N10_1_1_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_2_1_gen1_delaunay_forGAT_train_edges.csv", "P9_2_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P9_2_1_gen1_k10_forGAT_train_edges.csv", "P9_2_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_2_1_gen1_k6_forGAT_train_edges.csv", "P9_2_1_gen1_k6_forGAT_train_nodes.csv"],
    ["N10_1_2_gen0_delaunay_forGAT_train_edges.csv", "N10_1_2_gen0_delaunay_forGAT_train_nodes.csv"],  # test
    # ["N10_1_2_gen0_k10_forGAT_train_edges.csv", "N10_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_2_gen0_k6_forGAT_train_edges.csv", "N10_1_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_delaunay_forGAT_train_edges.csv", "P11_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_k10_forGAT_train_edges.csv", "P11_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_k6_forGAT_train_edges.csv", "P11_1_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_delaunay_forGAT_train_edges.csv", "P11_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_k10_forGAT_train_edges.csv", "P11_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_k6_forGAT_train_edges.csv", "P11_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_2_1_gen0_delaunay_forGAT_train_edges.csv", "P13_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_2_1_gen0_k10_forGAT_train_edges.csv", "P13_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_2_1_gen0_k6_forGAT_train_edges.csv", "P13_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_2_2_gen0_delaunay_forGAT_train_edges.csv", "P13_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_2_2_gen0_k10_forGAT_train_edges.csv", "P13_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_2_2_gen0_k6_forGAT_train_edges.csv", "P13_2_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_4_1_gen1_delaunay_forGAT_train_edges.csv", "P20_4_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_4_1_gen1_k10_forGAT_train_edges.csv", "P20_4_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_4_1_gen1_k6_forGAT_train_edges.csv", "P20_4_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_4_2_gen1_delaunay_forGAT_train_edges.csv", "P20_4_2_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_4_2_gen1_k10_forGAT_train_edges.csv", "P20_4_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_4_2_gen1_k6_forGAT_train_edges.csv", "P20_4_2_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_4_3_gen0_delaunay_forGAT_train_edges.csv", "P20_4_3_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_4_3_gen0_k10_forGAT_train_edges.csv", "P20_4_3_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_4_3_gen0_k6_forGAT_train_edges.csv", "P20_4_3_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_5_1_gen1_delaunay_forGAT_train_edges.csv", "P20_5_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_5_1_gen1_k10_forGAT_train_edges.csv", "P20_5_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_5_1_gen1_k6_forGAT_train_edges.csv", "P20_5_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_3_1_gen1_delaunay_forGAT_train_edges.csv", "P19_3_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P19_3_1_gen1_k10_forGAT_train_edges.csv", "P19_3_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_3_1_gen1_k6_forGAT_train_edges.csv", "P19_3_1_gen1_k6_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_delaunay_forGAT_train_edges.csv", "P25_4_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_k10_forGAT_train_edges.csv", "P25_4_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_k6_forGAT_train_edges.csv", "P25_4_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_delaunay_forGAT_train_edges.csv", "P25_5_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_k10_forGAT_train_edges.csv", "P25_5_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_k6_forGAT_train_edges.csv", "P25_5_1_gen0_k6_forGAT_train_nodes.csv"],
    ["HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_edges.csv",
     "HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_edges.csv", "HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_nodes.csv"],
    ["HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_edges.csv",
     "HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_nodes.csv"],

]

ki_full_train_paths = [

    ["N10_1_3_gen0_delaunay_forGAT_train_edges.csv", "N10_1_3_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["N10_1_3_gen0_k10_forGAT_train_edges.csv", "N10_1_3_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_3_gen0_k6_forGAT_train_edges.csv", "N10_1_3_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_delaunay_forGAT_train_edges.csv", "P11_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_k10_forGAT_train_edges.csv", "P11_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_1_1_gen0_k6_forGAT_train_edges.csv", "P11_1_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_delaunay_forGAT_train_edges.csv", "P11_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_k10_forGAT_train_edges.csv", "P11_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_2_2_gen0_k6_forGAT_train_edges.csv", "P11_2_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_1_1_gen0_delaunay_forGAT_train_edges.csv", "P13_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_1_1_gen0_k10_forGAT_train_edges.csv", "P13_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_1_1_gen0_k6_forGAT_train_edges.csv", "P13_1_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_1_2_gen0_delaunay_forGAT_train_edges.csv", "P13_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_1_2_gen0_k10_forGAT_train_edges.csv", "P13_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_1_2_gen0_k6_forGAT_train_edges.csv", "P13_1_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P19_1_1_gen1_delaunay_forGAT_train_edges.csv", "P19_1_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_1_1_gen1_k10_forGAT_train_edges.csv", "P19_1_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_1_1_gen1_k6_forGAT_train_edges.csv", "P19_1_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_1_2_gen1_delaunay_forGAT_train_edges.csv", "P19_1_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_1_2_gen1_k10_forGAT_train_edges.csv", "P19_1_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_1_2_gen1_k6_forGAT_train_edges.csv", "P19_1_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_1_3_gen1_delaunay_forGAT_train_edges.csv", "P20_1_3_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_1_3_gen1_k10_forGAT_train_edges.csv", "P20_1_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_1_3_gen1_k6_forGAT_train_edges.csv", "P20_1_3_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_2_2_gen0_delaunay_forGAT_train_edges.csv", "P20_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_2_2_gen0_k10_forGAT_train_edges.csv", "P20_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_2_2_gen0_k6_forGAT_train_edges.csv", "P20_2_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_2_3_gen1_delaunay_forGAT_train_edges.csv", "P20_2_3_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_2_3_gen1_k10_forGAT_train_edges.csv", "P20_2_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_2_3_gen1_k6_forGAT_train_edges.csv", "P20_2_3_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_2_4_gen0_delaunay_forGAT_train_edges.csv", "P20_2_4_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_2_4_gen0_k10_forGAT_train_edges.csv", "P20_2_4_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_2_4_gen0_k6_forGAT_train_edges.csv", "P20_2_4_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_3_1_gen0_delaunay_forGAT_train_edges.csv", "P20_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_3_1_gen0_k10_forGAT_train_edges.csv", "P20_3_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_3_1_gen0_k6_forGAT_train_edges.csv", "P20_3_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_delaunay_forGAT_train_edges.csv", "P25_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_k10_forGAT_train_edges.csv", "P25_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_2_1_gen0_k6_forGAT_train_edges.csv", "P25_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_delaunay_forGAT_train_edges.csv", "P25_3_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_k10_forGAT_train_edges.csv", "P25_3_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_3_1_gen0_k6_forGAT_train_edges.csv", "P25_3_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_delaunay_forGAT_train_edges.csv", "P25_3_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_k10_forGAT_train_edges.csv", "P25_3_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_3_2_gen0_k6_forGAT_train_edges.csv", "P25_3_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_7_5_gen0_delaunay_forGAT_train_edges.csv", "P28_7_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_7_5_gen0_k10_forGAT_train_edges.csv", "P28_7_5_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_7_5_gen0_k6_forGAT_train_edges.csv", "P28_7_5_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_8_5_gen0_delaunay_forGAT_train_edges.csv", "P28_8_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_8_5_gen0_k10_forGAT_train_edges.csv", "P28_8_5_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_8_5_gen0_k6_forGAT_train_edges.csv", "P28_8_5_gen0_k6_forGAT_train_nodes.csv"],
    # # ["P28_10_5_gen0_delaunay_forGAT_train_edges.csv", "P28_10_5_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_10_5_gen0_k6_forGAT_train_edges.csv", "P28_10_5_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_1_1_gen1_delaunay_forGAT_train_edges.csv", "P9_1_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_1_1_gen1_k6_forGAT_train_edges.csv", "P9_1_1_gen1_k6_forGAT_train_nodes.csv"],
    # ["P9_1_1_gen1_k10_forGAT_train_edges.csv", "P9_1_1_gen1_k10_forGAT_train_nodes.csv"],
    ["P9_2_2_gen1_delaunay_forGAT_train_edges.csv", "P9_2_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_2_2_gen1_k10_forGAT_train_edges.csv", "P9_2_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_2_2_gen1_k6_forGAT_train_edges.csv", "P9_2_2_gen1_k6_forGAT_train_nodes.csv"],

    ["P19_3_2_gen1_delaunay_forGAT_train_edges.csv", "P19_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P19_3_2_gen1_k10_forGAT_train_edges.csv", "P19_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_3_2_gen1_k6_forGAT_train_edges.csv", "P19_3_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_5_2_gen1_delaunay_forGAT_train_edges.csv", "P20_5_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_5_2_gen1_k10_forGAT_train_edges.csv", "P20_5_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_5_2_gen1_k6_forGAT_train_edges.csv", "P20_5_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_6_1_gen1_delaunay_forGAT_train_edges.csv", "P20_6_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P20_6_1_gen1_k10_forGAT_train_edges.csv", "P20_6_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_6_1_gen1_k6_forGAT_train_edges.csv", "P20_6_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_6_2_gen1_delaunay_forGAT_train_edges.csv", "P20_6_2_gen1_delaunay_forGAT_train_nodes.csv"],
    # ["P20_6_2_gen1_k10_forGAT_train_edges.csv", "P20_6_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_6_2_gen1_k6_forGAT_train_edges.csv", "P20_6_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_3_1_gen1_delaunay_forGAT_train_edges.csv", "P9_3_1_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_3_1_gen1_k10_forGAT_train_edges.csv", "P9_3_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_3_1_gen1_k6_forGAT_train_edges.csv", "P9_3_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_3_2_gen1_delaunay_forGAT_train_edges.csv", "P9_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # train
    # ["P9_3_2_gen1_k10_forGAT_train_edges.csv", "P9_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_3_2_gen1_k6_forGAT_train_edges.csv", "P9_3_2_gen1_k6_forGAT_train_nodes.csv"],

    # ki_val_paths
    ["P19_2_1_gen1_delaunay_forGAT_train_edges.csv", "P19_2_1_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P19_2_1_gen1_k10_forGAT_train_edges.csv", "P19_2_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_2_1_gen1_k6_forGAT_train_edges.csv", "P19_2_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_3_2_gen1_delaunay_forGAT_train_edges.csv", "P20_3_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P20_3_2_gen1_k10_forGAT_train_edges.csv", "P20_3_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_3_2_gen1_k6_forGAT_train_edges.csv", "P20_3_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_3_3_gen1_delaunay_forGAT_train_edges.csv", "P20_3_3_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P20_3_3_gen1_k10_forGAT_train_edges.csv", "P20_3_3_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_3_3_gen1_k6_forGAT_train_edges.csv", "P20_3_3_gen1_k6_forGAT_train_nodes.csv"],
    ["N10_2_2_gen0_delaunay_forGAT_train_edges.csv", "N10_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["N10_2_2_gen0_k10_forGAT_train_edges.csv", "N10_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_2_2_gen0_k6_forGAT_train_edges.csv", "N10_2_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_7_1_gen0_delaunay_forGAT_train_edges.csv", "P20_7_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_7_1_gen0_k10_forGAT_train_edges.csv", "P20_7_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_7_1_gen0_k6_forGAT_train_edges.csv", "P20_7_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_7_2_gen0_delaunay_forGAT_train_edges.csv", "P20_7_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_7_2_gen0_k10_forGAT_train_edges.csv", "P20_7_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_7_2_gen0_k6_forGAT_train_edges.csv", "P20_7_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_8_1_gen0_delaunay_forGAT_train_edges.csv", "P20_8_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_8_1_gen0_k10_forGAT_train_edges.csv", "P20_8_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_8_1_gen0_k6_forGAT_train_edges.csv", "P20_8_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_9_1_gen0_delaunay_forGAT_train_edges.csv", "P20_9_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_9_1_gen0_k10_forGAT_train_edges.csv", "P20_9_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_9_1_gen0_k6_forGAT_train_edges.csv", "P20_9_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P20_9_2_gen0_delaunay_forGAT_train_edges.csv", "P20_9_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_9_2_gen0_k10_forGAT_train_edges.csv", "P20_9_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_9_2_gen0_k6_forGAT_train_edges.csv", "P20_9_2_gen0_k6_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_delaunay_forGAT_train_edges.csv", "P25_8_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_k10_forGAT_train_edges.csv", "P25_8_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P25_8_2_gen0_k6_forGAT_train_edges.csv", "P25_8_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P28_10_4_gen0_delaunay_forGAT_train_edges.csv", "P28_10_4_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P28_10_4_gen0_k10_forGAT_train_edges.csv", "P28_10_4_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P28_10_4_gen0_k6_forGAT_train_edges.csv", "P28_10_4_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_4_2_gen1_delaunay_forGAT_train_edges.csv", "P9_4_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P9_4_2_gen1_k10_forGAT_train_edges.csv", "P9_4_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_4_2_gen1_k6_forGAT_train_edges.csv", "P9_4_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_2_2_gen1_delaunay_forGAT_train_edges.csv", "P19_2_2_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P19_2_2_gen1_k10_forGAT_train_edges.csv", "P19_2_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_2_2_gen1_k6_forGAT_train_edges.csv", "P19_2_2_gen1_k6_forGAT_train_nodes.csv"],
    ["P9_4_1_gen1_delaunay_forGAT_train_edges.csv", "P9_4_1_gen1_delaunay_forGAT_train_nodes.csv"],  # val
    # ["P9_4_1_gen1_k10_forGAT_train_edges.csv", "P9_4_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_4_1_gen1_k6_forGAT_train_edges.csv", "P9_4_1_gen1_k6_forGAT_train_nodes.csv"],

    # ki_test_paths
    ["N10_1_1_gen0_delaunay_forGAT_train_edges.csv", "N10_1_1_gen0_delaunay_forGAT_train_nodes.csv"],  # test
    # ["N10_1_1_gen0_k10_forGAT_train_edges.csv", "N10_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_1_gen0_k6_forGAT_train_edges.csv", "N10_1_1_gen0_k6_forGAT_train_nodes.csv"],
    ["P9_2_1_gen1_delaunay_forGAT_train_edges.csv", "P9_2_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P9_2_1_gen1_k10_forGAT_train_edges.csv", "P9_2_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P9_2_1_gen1_k6_forGAT_train_edges.csv", "P9_2_1_gen1_k6_forGAT_train_nodes.csv"],
    ["N10_1_2_gen0_delaunay_forGAT_train_edges.csv", "N10_1_2_gen0_delaunay_forGAT_train_nodes.csv"],  # test
    # ["N10_1_2_gen0_k10_forGAT_train_edges.csv", "N10_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["N10_1_2_gen0_k6_forGAT_train_edges.csv", "N10_1_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_delaunay_forGAT_train_edges.csv", "P11_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_k10_forGAT_train_edges.csv", "P11_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_1_2_gen0_k6_forGAT_train_edges.csv", "P11_1_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_delaunay_forGAT_train_edges.csv", "P11_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_k10_forGAT_train_edges.csv", "P11_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P11_2_1_gen0_k6_forGAT_train_edges.csv", "P11_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_2_1_gen0_delaunay_forGAT_train_edges.csv", "P13_2_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_2_1_gen0_k10_forGAT_train_edges.csv", "P13_2_1_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_2_1_gen0_k6_forGAT_train_edges.csv", "P13_2_1_gen0_k6_forGAT_train_nodes.csv"],
    # ["P13_2_2_gen0_delaunay_forGAT_train_edges.csv", "P13_2_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P13_2_2_gen0_k10_forGAT_train_edges.csv", "P13_2_2_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P13_2_2_gen0_k6_forGAT_train_edges.csv", "P13_2_2_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_4_1_gen1_delaunay_forGAT_train_edges.csv", "P20_4_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_4_1_gen1_k10_forGAT_train_edges.csv", "P20_4_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_4_1_gen1_k6_forGAT_train_edges.csv", "P20_4_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P20_4_2_gen1_delaunay_forGAT_train_edges.csv", "P20_4_2_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_4_2_gen1_k10_forGAT_train_edges.csv", "P20_4_2_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_4_2_gen1_k6_forGAT_train_edges.csv", "P20_4_2_gen1_k6_forGAT_train_nodes.csv"],
    # ["P20_4_3_gen0_delaunay_forGAT_train_edges.csv", "P20_4_3_gen0_delaunay_forGAT_train_nodes.csv"],
    # # ["P20_4_3_gen0_k10_forGAT_train_edges.csv", "P20_4_3_gen0_k10_forGAT_train_nodes.csv"],
    # # ["P20_4_3_gen0_k6_forGAT_train_edges.csv", "P20_4_3_gen0_k6_forGAT_train_nodes.csv"],
    ["P20_5_1_gen1_delaunay_forGAT_train_edges.csv", "P20_5_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P20_5_1_gen1_k10_forGAT_train_edges.csv", "P20_5_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P20_5_1_gen1_k6_forGAT_train_edges.csv", "P20_5_1_gen1_k6_forGAT_train_nodes.csv"],
    ["P19_3_1_gen1_delaunay_forGAT_train_edges.csv", "P19_3_1_gen1_delaunay_forGAT_train_nodes.csv"],  # test
    # ["P19_3_1_gen1_k10_forGAT_train_edges.csv", "P19_3_1_gen1_k10_forGAT_train_nodes.csv"],
    # ["P19_3_1_gen1_k6_forGAT_train_edges.csv", "P19_3_1_gen1_k6_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_delaunay_forGAT_train_edges.csv", "P25_4_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_k10_forGAT_train_edges.csv", "P25_4_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_4_2_gen0_k6_forGAT_train_edges.csv", "P25_4_2_gen0_k6_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_delaunay_forGAT_train_edges.csv", "P25_5_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_k10_forGAT_train_edges.csv", "P25_5_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["P25_5_1_gen0_k6_forGAT_train_edges.csv", "P25_5_1_gen0_k6_forGAT_train_nodes.csv"],
    ["HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_edges.csv",
     "HE_T12193_90_Default_Extended_1_2_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_2_gen0_k10_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_edges.csv", "HE_T12193_90_Default_Extended_1_2_gen0_k6_forGAT_train_nodes.csv"],
    ["HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_edges.csv",
     "HE_T12193_90_Default_Extended_1_1_gen0_delaunay_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_1_gen0_k10_forGAT_train_nodes.csv"],
    # ["HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_edges.csv",
    #  "HE_T12193_90_Default_Extended_1_1_gen0_k6_forGAT_train_nodes.csv"],

]

# hover-net experiment
ki_train_paths_hn = [
    ["train/N10_1_3_delaunay_orig_edges.csv", "train/N10_1_3_delaunay_orig_nodes.csv"],
    # ["train/P19_1_1_delaunay_orig_edges.csv","train/P19_1_1_delaunay_orig_nodes.csv"], # train Pred not gen yet
    ["train/P19_1_2_delaunay_orig_edges.csv", "train/P19_1_2_delaunay_orig_nodes.csv"],  # train
    ["train/P20_1_3_delaunay_orig_edges.csv", "train/P20_1_3_delaunay_orig_nodes.csv"],  # train
    ["train/P20_2_3_delaunay_orig_edges.csv", "train/P20_2_3_delaunay_orig_nodes.csv"],  # train
    ["train/P9_1_1_delaunay_orig_edges.csv", "train/P9_1_1_delaunay_orig_nodes.csv"],  # train
    ["train/P9_2_2_delaunay_orig_edges.csv", "train/P9_2_2_delaunay_orig_nodes.csv"],  # train
    ["train/P19_3_2_delaunay_orig_edges.csv", "train/P19_3_2_delaunay_orig_nodes.csv"],  # train
    ["train/P20_5_2_delaunay_orig_edges.csv", "train/P20_5_2_delaunay_orig_nodes.csv"],  # train
    ["train/P20_6_1_delaunay_orig_edges.csv", "train/P20_6_1_delaunay_orig_nodes.csv"],  # train
    ["train/P20_6_2_delaunay_orig_edges.csv", "train/P20_6_2_delaunay_orig_nodes.csv"],
    ["train/P9_3_1_delaunay_orig_edges.csv", "train/P9_3_1_delaunay_orig_nodes.csv"],  # train
    ["train/P9_3_2_delaunay_orig_edges.csv", "train/P9_3_2_delaunay_orig_nodes.csv"],  # train
]
ki_val_paths_hn = [
    ["val/P19_2_1_delaunay_orig_edges.csv", "val/P19_2_1_delaunay_orig_nodes.csv"],  # val
    ["val/P20_3_2_delaunay_orig_edges.csv", "val/P20_3_2_delaunay_orig_nodes.csv"],  # val
    ["val/P20_3_3_delaunay_orig_edges.csv", "val/P20_3_3_delaunay_orig_nodes.csv"],  # val
    ["val/N10_2_2_delaunay_orig_edges.csv", "val/N10_2_2_delaunay_orig_nodes.csv"],
    ["val/P9_4_2_delaunay_orig_edges.csv", "val/P9_4_2_delaunay_orig_nodes.csv"],  # val
    ["val/P19_2_2_delaunay_orig_edges.csv", "val/P19_2_2_delaunay_orig_nodes.csv"],  # val
    ["val/P9_4_1_delaunay_orig_edges.csv", "val/P9_4_1_delaunay_orig_nodes.csv"],  # val
]
ki_test_paths_hn = [
    ["test/N10_1_1_delaunay_orig_edges.csv", "test/N10_1_1_delaunay_orig_nodes.csv"],  # test
    ["test/P9_2_1_delaunay_orig_edges.csv", "test/P9_2_1_delaunay_orig_nodes.csv"],  # test
    ["test/N10_1_2_delaunay_orig_edges.csv", "test/N10_1_2_delaunay_orig_nodes.csv"],  # test
    ["test/P20_4_1_delaunay_orig_edges.csv", "test/P20_4_1_delaunay_orig_nodes.csv"],  # test
    ["test/P20_4_2_delaunay_orig_edges.csv", "test/P20_4_2_delaunay_orig_nodes.csv"],  # test
    ["test/P20_5_1_delaunay_orig_edges.csv", "test/P20_5_1_delaunay_orig_nodes.csv"],  # test
    ["test/P19_3_1_delaunay_orig_edges.csv", "test/P19_3_1_delaunay_orig_nodes.csv"],  # test
    ["test/HE_T12193_90_Default_Extended_1_2_delaunay_orig_edges.csv",
     "test/HE_T12193_90_Default_Extended_1_2_delaunay_orig_nodes.csv"],
    ["test/HE_T12193_90_Default_Extended_1_1_delaunay_orig_edges.csv",
     "test/HE_T12193_90_Default_Extended_1_1_delaunay_orig_nodes.csv"],
]


def train_gat_ki(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    # node_features, node_labels, edge_index, train_indices, val_indices, test_indices, final_indices = \
    #     load_graph_data(config, device)

    # tnf, tnl, tei = load_ki_graph_data(device, ki_prediction_paths[0], multiple=True)

    val_node_features, val_node_labels, val_edge_index, val_indices = None, None, None, None
    test_node_features, test_node_labels, test_edge_index, test_indices = None, None, None, None

    if not config['final']:
        train_node_features, train_node_labels, train_edge_index, train_indices = \
            load_ki_graph_data(device, ki_train_paths_hn, multiple=True, dataset="train")

        val_node_features, val_node_labels, val_edge_index, val_indices = \
            load_ki_graph_data(device, ki_val_paths_hn, multiple=True, dataset="val")

        test_node_features, test_node_labels, test_edge_index, test_indices = \
            load_ki_graph_data(device, ki_test_paths_hn, multiple=True, dataset="test")
    else:
        train_node_features, train_node_labels, train_edge_index, train_indices = \
            load_ki_graph_data(device, ki_full_train_paths, multiple=True, dataset="train")
    # if config['final']:
    #     train_indices = final_indices

    # Step 2: prepare the model

    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops

    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        train_node_features,
        train_node_labels,
        train_edge_index,
        train_indices,
        config['patience_period'],
        time.time(),
        val_node_features=val_node_features,
        val_node_labels=val_node_labels,
        val_edge_index=val_edge_index,
        val_indices=val_indices,
        test_node_features=test_node_features,
        test_node_labels=test_node_labels,
        test_edge_index=test_edge_index,
        test_indices=test_indices,
    )

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        if not config['final']:
            with torch.no_grad():
                try:
                    main_loop(phase=LoopPhase.VAL, epoch=epoch)
                except Exception as e:  # "patience has run out" exception :O
                    print(str(e))
                    break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if not config['final']:
        if config['should_test']:
            test_acc = main_loop(phase=LoopPhase.TEST)
            # recall, precision, f1_score = \

            config['test_perf'] = test_acc
            # print(f'    accuracy                           {test_acc}')

        else:
            config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        utils.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    )
    print('finished saving model')


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int,
                        help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)  # 0.005
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)  # 0.0005
    parser.add_argument("--should_test", action='store_true',
                        help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--final", action='store_true',
                        help='train model with full dataset, final model (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.KI.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)",
                        default=100)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
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
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_gat_ki(get_training_args())
