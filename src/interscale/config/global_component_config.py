from yacs.config import CfgNode as CN


def get_global_component_cfg(cfg, global_component_name):
    """
    Defines global component configuration.
    """
    cfg.model.global_component.parameters = CN()

    if global_component_name == "self-attn-transformer":
        cfg.model.global_component.parameters.n_heads = 4
        cfg.model.global_component.parameters.dim_feedforward = 256
        cfg.model.global_component.parameters.dropout_global = 0.1
        cfg.model.global_component.parameters.activation_func = "relu"
        cfg.model.global_component.parameters.num_layers = 2
        cfg.model.global_component.parameters.max_seq_len = (
            2000  # optionally adjust to maximum number of cells, ideally shouldnt be larger than 4000
        )
        # If True, blocks attention inside the local component's receptive field. There is no
        # radius to set: the mask always covers exactly what the local module mixed (its
        # `receptive_field_hops`), so the two components stay disjoint by construction.
        cfg.model.global_component.parameters.long_range_attention = False
        cfg.model.global_component.parameters.type_gex_embedding = None
        cfg.model.global_component.latent_obsm_key = None  # Use the obms key where precomputed embeddings are stored, only if type_gex_embedding is "Precomputed"
    return cfg
