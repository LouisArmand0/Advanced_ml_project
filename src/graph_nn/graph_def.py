import networkx as nx
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def compute_adj_matrix_based_on_sector(sector_df: pd.DataFrame, plot: bool) -> nx.Graph:
        adj_matrix = np.array([
            [sector_df.loc[stock1, 'sector'] == sector_df.loc[stock2, 'sector'] * (stock1 != stock2) for stock1 in sector_df.index]
            for stock2 in sector_df.index
        ]).astype(int)
        graph = nx.from_numpy_array(adj_matrix)
        graph = nx.relabel_nodes(graph, dict(enumerate(sector_df.index)))

        if plot:
            plt.rcParams['font.family'] = 'serif'
            plt.figure(figsize=(12, 8))
            nx.draw(graph, with_labels=True, node_size=400, node_color='lightgreen', font_size=8,
                    font_weight='bold', font_color='black', pos=nx.spring_layout(graph, k=.5))
            plt.title('Stocks Graph by Sector')
            plt.show()
        return graph

def compute_adj_matrix_based_on_correlation(ret_df: pd.DataFrame, threshold: float, plot: bool) -> nx.Graph:
    corr_df = ret_df.corr(method="spearman")
    # remove self-correlation
    np.fill_diagonal(corr_df.values, 0)

    # thresholding: keep only weights above threshold
    adj_matrix = corr_df.where(abs(corr_df) >= threshold, other=0)
    graph = nx.from_numpy_array(adj_matrix.to_numpy())
    graph = nx.relabel_nodes(graph, dict(enumerate(corr_df.columns)))

    # remove edges with weight 0
    graph.remove_edges_from(
        [(u, v) for u, v, w in graph.edges(data="weight") if w == 0]
    )

    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        nx.draw(graph, with_labels=True, node_size=400, node_color='lightgreen', font_size=8,
                font_weight='bold', font_color='black', pos=nx.spring_layout(graph, k=.5))
        ax.set_title(r'Stocks Graph ($|\rho| \geq {}$)'.format(threshold), fontsize=16)
        plt.show()
    return graph



if __name__ == '__main__':
    sector_df = pd.read_csv('src/data/raw/trading_universe.csv').set_index('symbol')
    graph_sector = compute_adj_matrix_based_on_sector(sector_df, True)

    ret_df = pd.read_csv('src/data/raw/historical_prices.csv').set_index(['Date'])
    ret_df = ret_df.filter(like="_close")
    graph_corr = compute_adj_matrix_based_on_correlation(ret_df, .85, True)

