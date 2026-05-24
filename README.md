# AMSFM
Official code implementation of CIKM paper "ZoomLens Stock Forecasting: Adaptive Multi-Scale Modeling for Market-Aware Investment". This work proposes AMSFM, an adaptive multi-scale signal selection and fusion model for interpretable stock forecasting. It consists of three core components: an adaptive temporal scale selector that dynamically aligns modeling horizons with real-time market regimes, an interpretable attribution and interaction module based on RWKV for fine-grained feature-level contributions, and a market-aware forecasting strategy with dual-branch architecture to capture cross-stock spillover effects without relying on prior knowledge.
# Environment
Python 3.8

torch~=2.0.0

numpy~=1.24.0

pandas, tqdm, matplotlib
# Dataset and Preprocessing

NASDAQ/NYSE: Temporal Relational Ranking for Stock Prediction (Feng et al., 2019).
We use the processed opening price, closing price, highest price, lowest price, and trading volume as five features
S&P500: Efficient Integration of Multi-Order Dynamics and Internal Dynamics in Stock Movement Prediction (Huynh et al., 2022).
We collect and process data from Yahoo Finance.

# code
Currently, the core code of the model has been made public for submission purposes, and all the code will be released after the paper is accepted.
