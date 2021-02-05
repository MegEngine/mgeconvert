import argparse

import megengine.core.tensor.megbrain_graph as G
import megengine.utils.comp_graph_tools as cgtools
from megengine.core._imperative_rt import make_h2d


def change_batch_and_dump(inp_file, oup_file):
    cg, _, outputs = G.load_graph(inp_file)
    inputs = cgtools.get_dep_vars(outputs[0], "Host2DeviceCopy")
    replace_dict = {}
    for var in inputs:
        n_shape = list(var.shape)
        n_shape[0] = 1
        new_input = make_h2d(cg, "xpux", var.dtype, n_shape, var.name)
        replace_dict[var] = new_input

    new_outputs = cgtools.replace_vars(outputs, replace_dict)
    dump_content, _ = G.dump_graph(map(G.VarNode, new_outputs), keep_var_name=2)
    with open(oup_file, "wb") as file:
        file.write(dump_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input megengine dump model file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output batch1 model file"
    )
    args = parser.parse_args()

    change_batch_and_dump(args.input, args.output)


if __name__ == "__main__":
    main()
