from collections import OrderedDict


def layer_split(network):
    # Tuning the learning rates
    # Split into three groups of layers, to have three different learning rates
    layer_1 = OrderedDict()
    layer_2 = OrderedDict()
    layer_3 = OrderedDict()
    layer_1_to_include = [
        "blueBlock",
        "resBlock1",
        "resBlock2"
    ]
    layer_2_to_include = [
        "resBlock3",
        "resBlock4"
    ]

    for name, param in network.named_parameters():
        if any(substring in name for substring in layer_1_to_include):
            layer_1[name] = param
        elif any(substring in name for substring in layer_2_to_include):
            layer_1[name] = param
        else:
            layer_3[name] = param

    return layer_1, layer_2, layer_3
