from ..backend.ir_to_trt import TensorRTConverter
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.mge_to_ir import MGE_FrontEnd


def mge_to_trt(
    mge_fpath,
    output="out.trt",
    *,
    graph_name="graph",
):
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath).resolve()

    transformer_options = [
        TransformerRule.FUSE_SOFTMAX,
        TransformerRule.FUSE_CONV_BN,
        TransformerRule.REMOVE_AXISADDREMOVE,
        # TransformerRule.REMOVE_GETVARSHAPE,
    ]
    
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    converter = TensorRTConverter(transformed_irgraph, graph_name)
    input_names,output_names = converter.convert()

    assert isinstance(output, str), "trt_fpath must be string"
    converter.save_model(output)

    return input_names,output_names