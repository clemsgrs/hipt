from omegaconf import DictConfig

from src.models.hipt import GlobalHIPT, LocalHIPT


class ModelFactory:
    def __init__(
        self,
        level: str,
        num_classes: int,
        options: DictConfig | None = None,
    ):
        if level == "global":
            self.model = GlobalHIPT(
                num_classes=num_classes,
                embed_dim_region=options.embed_dim_region,
                embed_dim_slide=options.embed_dim_slide,
                dropout=options.dropout,
            )
        elif level == "local":
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
                num_heads=options.num_heads,
                pretrained_weights=options.pretrained_weights,
                img_size_pretrained=options.img_size_pretrained,
            )

    def get_model(self):
        return self.model
