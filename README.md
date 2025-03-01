# vocaloid_modeling

# How to start
See `vocaloid_pyneuralfx/Pyneuralfx.ipynb`, run the blocks, you can run the grid_search first to find the best configurations (you have to modify the search space in `vocaloid_pyneuralfx/Pyneuralfx/frame_work/grid_search.py`), then manually modify the snapshot model configuration in `vocaloid_pyneuralfx/Pyneuralfx/configs/cnn/tcn/snapshot_tcn.yml` to the best configuration. Then run `vocaloid_pyneuralfx/Pyneuralfx/frame_work/main_snapshot.py` for best model training. After you complete training and inferenceing, the inferenced file will be stored in `vocaloid_pyneuralfx/Pyneuralfx/frame_work/exp/vocaloid/valid_gen`, or you can just listen to the validation file listed in the last block.

# Data
Training data is in `data/audio/`, the audio `sv_teto_X_X_X_X_MixDown.wav` are the audio file for training, the parameters are `{loudness}_{?}_{?}_{gender}`.