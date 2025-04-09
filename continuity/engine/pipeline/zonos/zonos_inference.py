import os 
import torch 

from ....utils.loading_utils import cqdm
from ...models.transformer import ZonosTransformer


class ZonosPipeline(nn.Module):
    def __init__(self, config: ZonosConfig):
        super().__init__()
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = config.backbone
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwarfgs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def _execution_device(self) -> torch.device:
        return next(self.parameters())._execution_device()

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str, | None = None, execution_device: str = DEFAULT_DEVICE, **kwargs) -> "Zonoe":
        config_path = hf_hub_download(repo_id=repo_id, finename="config.json")
        model_path = hf_hub_download(repo_id=repo_id, finename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)








