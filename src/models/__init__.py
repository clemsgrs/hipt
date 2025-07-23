from omegaconf import DictConfig

from src.models.hipt import GlobalHIPT, LocalHIPT
from src.models.base import LinearLayer, MultiLayerPerceptron


class ModelFactory:
    def __init__(
        self,
        name: str,
        num_classes: int,
        options: DictConfig | None = None,
    ):
        if name == "lp":
            assert options.level == "slide" or options.level == "case"
            self.model = LinearLayer(
                options.embed_dim_slide,
                output_dim=num_classes,
            )
        elif name == "mlp":
            assert options.level == "slide" or options.level == "case"
            self.model = MultiLayerPerceptron(
                options.embed_dim_slide,
                hidden_dim=256,
                output_dim=num_classes,
                num_layers=options.num_layers,
            )
        elif name == "hipt":
            if options.level == "global":
                self.model = GlobalHIPT(
                    num_classes=num_classes,
                    embed_dim_region=options.embed_dim_region,
                    embed_dim_slide=options.embed_dim_slide,
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
                    mask_attn=options.mask_attn,
                    num_register_tokens=options.num_register_tokens,
                    pretrained_weights=options.pretrained_weights,
                    img_size_pretrained=options.img_size_pretrained,
                )

    def get_model(self):
        return self.model
