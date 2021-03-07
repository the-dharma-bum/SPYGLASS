from ..model import LightningModel

def init_model(cfg):
    return LightningModel(
                window=cfg.window,
                stride=cfg.stride,
                early_fusion_dim=cfg.early_fusion_dim,
                fusion_mode=cfg.fusion_mode,
                encoder_base_net=cfg.encoder_base_net,
                pretrained=cfg.pretrained,
                encoder_hidden_dim_1=cfg.encoder_hidden_dim_1,
                encoder_hidden_dim_2=cfg.encoder_hidden_dim_2,
                encoder_dropout_rate=cfg.encoder_dropout_rate,
                embed_dim=cfg.embed_dim,
                decoder=cfg.decoder,
                decoder_dropout_rate=cfg.decoder_dropout_rate,
                fc_hidden_dim=cfg.fc_hidden_dim,
                h_RNN_layers=cfg.h_RNN_layers,
                h_RNN=cfg.h_RNN,
                num_classes=cfg.num_classes,
                aggregation_mode=cfg.aggregation_mode,
                use_label_smoothing=cfg.use_label_smoothing,
                smoothing=cfg.smoothing,
                criteria=cfg.criteria,
    )