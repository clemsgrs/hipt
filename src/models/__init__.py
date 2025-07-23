from omegaconf import DictConfig

from src.models.hipt import GlobalHIPT, LocalHIPT
from src.models.base import LinearLayer, MultiLayerPerceptron


class ModelFactory:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        options: DictConfig | None = None,
    ):
        if model_name == "lp":
            assert options.level == "slide" or options.level == "case"
            self.model = LinearLayer(
                options.embed_dim_slide,
                output_dim=num_classes,
            )
        elif model_name == "mlp":
            assert options.level == "slide" or options.level == "case"
            self.model = MultiLayerPerceptron(
                options.embed_dim_slide,
                hidden_dim=256,
                output_dim=num_classes,
                num_layers=options.num_layers,
            )
        elif model_name == "hipt":
            if options.level == "global":
                self.model = GlobalHIPT(
                    num_classes=num_classes,
                    embed_dim_region=options.embed_dim_region,
                    d_model=options.embed_dim_slide,
                    dropout=options.dropout,
                )
            elif options.level == "local":
                self.model = LocalHIPT(
                    num_classes=num_classes,
                    region_size=options.region_size,
                    patch_size=options.patch_size,
                    embed_dim_patch=options.embed_dim_patch,
                    embed_dim_region=options.embed_dim_region,
                    embed_dim_slide=options.embed_dim_slide,
                    dropout=options.dropout,
                    mask_attn_region=options.mask_attn_region,
                    num_register_tokens_region=options.num_register_tokens_region,
                    pretrain_vit_region=options.pretrain_weights,
                    img_size_pretrained=options.img_size_pretrained,
                )

    def get_model(self):
        return self.model
