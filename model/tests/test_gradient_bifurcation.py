def test_both_branches_receive_gradients():
    """After loss.backward(), every parameter in SpatialCNN and TemporalLSTM
    has a non-None, non-zero grad (the fusion head really uses both)."""
    # TODO
    pass
