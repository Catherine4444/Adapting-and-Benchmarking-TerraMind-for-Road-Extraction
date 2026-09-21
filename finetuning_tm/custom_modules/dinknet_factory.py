from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.registry import MODEL_FACTORY_REGISTRY
from .dinknet import MultiBandDinkNet34


class DinkNetWrapper(Model):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model  # MultiBandDinkNet34

    def forward(self, x):
        out = self.model(x)  # DinkNet34 returns raw logits [B, num_classes, H, W]
        return ModelOutput(output=out)

    def freeze_encoder(self):
        inner = self.model.model  # the underlying DinkNet34
        for m in [inner.firstconv, inner.firstbn,
                  inner.encoder1, inner.encoder2,
                  inner.encoder3, inner.encoder4]:
            for p in m.parameters():
                p.requires_grad_(False)

    def freeze_decoder(self):
        inner = self.model.model
        for m in [inner.dblock, inner.decoder4, inner.decoder3,
                  inner.decoder2, inner.decoder1]:
            for p in m.parameters():
                p.requires_grad_(False)


@MODEL_FACTORY_REGISTRY.register
class DinkNetModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        bands: list,
        num_classes: int = 1,
        **kwargs,
    ) -> Model:
        if task != "segmentation":
            raise ValueError(f"DinkNet34 only supports segmentation, got task={task}")

        num_channels = len(bands)
        net = MultiBandDinkNet34(num_channels=num_channels, num_classes=num_classes)
        return DinkNetWrapper(net)