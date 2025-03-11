import numpy as np
from torch import nn
from torch_geometric.loader import DataLoader


class SaveInputOutput:
    """Object to save input and output of a layer during forward pass"""

    def __init__(self, mask=None, vq=False):
        self.io = {}
        self.mask = mask
        self.vq = vq

    def __call__(self, module, module_in, module_out):
        if self.vq:
            input_array1, input_array = module_in
            if input_array is None:
                input_array = input_array1
            input_array = input_array.cpu().detach().numpy()
            _, _, output_array = module_out
            output_array = output_array.cpu().detach().numpy()
        else:
            input_array = module_in[0].detach().numpy()
            output_array = module_out[0].detach().numpy()
        if self.mask is not None:
            input_array = input_array[self.mask]
            output_array = output_array[self.mask]
        layer_name = module.__name__
        if layer_name not in self.io:
            self.io[layer_name] = {"inputs": input_array, "outputs": output_array}
        else:
            self.io[layer_name]["inputs"] = np.concatenate(
                (self.io[layer_name]["inputs"], input_array), axis=0
            )
            self.io[layer_name]["outputs"] = np.concatenate(
                (self.io[layer_name]["outputs"], output_array), axis=0
            )

    def clear(self):
        self.io = {}

    def size(self):
        for layer_name, values in self.io.items():
            print(
                f"{layer_name}: {np.shape(values['inputs'])} {np.shape(values['outputs'])}"
            )


class FeatureExtractor(nn.Module):
    """Model wrapper that saves input and output of specified layers"""

    def __init__(self, model, layer_names, mask, vq=False):
        super().__init__()
        self.model = model
        self.saver = SaveInputOutput(mask, vq=vq)
        self.handlers = []
        some_dict = dict([*self.model.named_modules()])
        # print()
        # print("Keys")
        # print(some_dict.keys())
        # print()
        for layer_id in layer_names:
            layer = some_dict[layer_id]
            layer.__name__ = layer_id
            self.handlers.append(layer.register_forward_hook(self.saver))

    def deregister_hooks(self):
        for handler in self.handlers:
            handler.remove()

    def forward(self, data):
        self.model.eval()
        return self.model(
            data.x, data.edge_index, data.batch if "batch" in data else None
        )


def extract_features(
    model,
    layer_names,
    dataset,
    device,
    loader=None,
    mask=None,
    vq=False,
    batch_size=128,
):
    """Extract features from specified layers of a model.
    Returns a dict, where keys are layer names and values are dicts with keys 'inputs' and 'outputs',
    whose values are a tensor containing all feature maps from all batches.
    """
    extractor_net = FeatureExtractor(model, layer_names, mask=mask, vq=vq)
    model.eval()
    if mask is not None:    
        data = dataset.to(device)
        _ = extractor_net(data)
        return extractor_net.saver.io

    if loader is None:
        data_loader = DataLoader(dataset, batch_size=batch_size)
    else:
        data_loader = loader

    for data in data_loader:
        data = data.to(device)
        _ = extractor_net(data)

    extractor_net.deregister_hooks()
    return extractor_net.saver.io
