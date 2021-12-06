
from logging import error
import warnings
from typing import Sequence
import tensorrt as trt
from tqdm import tqdm

from mgeconvert.converter_ir.ir_tensor import IRTensor
from ...converter_ir.ir_graph import IRGraph
from .trt_op import (
    MGE2TRT,
    BatchNormalizationOpr
)
from .utils import default_irtensor_names, mge_device_to_trt, mge_dtype_to_trt, get_dynamic_dims


class TensorRTConverter:
    def __init__(self,
                net,
                graph_name="graph",
                max_batch_size = 64,
                explicit_batch_dimension = False,
                explicit_precision = False,
                fp16_mode=False,
                int8_mode=False,
                force_fp32_output=False,
                strict_type_constraints=False,
                algorithm_selector=None,
                timing_cache=None,
                ):
        assert isinstance(net, IRGraph), "net must be instance of IRGraph"
       
        if int8_mode and not self.builder.platform_has_fast_int8:
            warnings.warn("Current platform doesn't support fast native int8!")

        if fp16_mode and not self.builder.platform_has_fast_fp16:
            warnings.warn("Current platform doesn't support fast native fp16!")

        self.net = net
        self.graph_name = graph_name
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.builder = trt.Builder(self.logger)
        self.builder_config = self.builder.create_builder_config()
        self.builder_config.max_workspace_size = 1<<25
        self.builder.int8_mode = int8_mode
        self.builder.max_batch_size = max_batch_size
        self.builder.strict_type_constraints = True

        if fp16_mode:
            self.builder_config.set_flag(trt.BuilderFlag.FP16)

        if int8_mode:
            self.builder_config.set_flag(trt.BuilderFlag.INT8)

        if strict_type_constraints:
            self.builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # if self.optimization_profiles:
        #     for optimization_profile in self.optimization_profiles:
        #         self.builder_config.add_optimization_profile(optimization_profile)

        if algorithm_selector:
            self.builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            self.builder_config.algorithm_selector = algorithm_selector

        flag = 0
        if explicit_batch_dimension:
            EXPLICIT_BATCH = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
            flag |= EXPLICIT_BATCH

        if explicit_precision:
            EXPLICIT_PRECISION = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION
            )
            flag |= EXPLICIT_PRECISION

        self.network = self.builder.create_network(flag)

        self._var2tensor = dict()

    def preprocess(self, irtensor_list):
        _irtensor_names = default_irtensor_names(irtensor_list)
        self.input_names = _irtensor_names

        for i, irtensor in enumerate(irtensor_list):
            if irtensor in self._var2tensor:
                continue
            trt_tensor = self.network.add_input(
                name=_irtensor_names[i],
                shape=tuple(irtensor.shape)[1:],
                # shape=tuple(irtensor.shape),
                dtype=mge_dtype_to_trt(irtensor.dtype),
            )
            # trt_tensor.location = mge_device_to_trt(repr(irtensor_list[i].device)[1:4])
            self._var2tensor[irtensor] = trt_tensor

    def postprocess(self, irtensor_list):
        _irtensor_names = default_irtensor_names(irtensor_list)
        self.output_names = _irtensor_names
        for i, output in enumerate(irtensor_list):
            trt_tensor = self._var2tensor[output]
            trt_tensor.name = _irtensor_names[i]
            # trt_tensor.location = mge_device_to_trt(repr(irtensor_list[i].device)[1:4])
            trt_tensor.dtype = mge_dtype_to_trt(output.dtype)
            self.network.mark_output(trt_tensor)

    def convert(self):
        self.preprocess(self.net.graph_inputs)

        def isconsant(input):
            return input.np_data is not None

        for mge_opr in tqdm(self.net.all_oprs):

            inp_tensors = mge_opr.inp_tensors

            # for tensor in inp_tensors:
            #     if tensor not in self._var2tensor and not isconsant(tensor):
            #         raise AssertionError("invalid input {} for network".format(tensor.name))   
            
            trt_outpus=MGE2TRT[type(mge_opr)](mge_opr, self.network, self._var2tensor)
            irtensors=mge_opr.out_tensors

            trt_outpus = (trt_outpus,) if not isinstance(trt_outpus, Sequence) else trt_outpus

            if isinstance(mge_opr, BatchNormalizationOpr):
                irtensor = irtensors[-1]
                assert irtensor not in self._var2tensor, "The irtensor is already existed."
                self._var2tensor[irtensor] = trt_outpus[0]
                continue
            for i, trt_tensor in enumerate(trt_outpus):
                irtensor = irtensors[i]
                assert irtensor not in self._var2tensor, "The irtensor is already existed."
                self._var2tensor[irtensor] = trt_tensor

        self.postprocess(self.net.graph_outputs)

        return self.input_names,self.output_names

    # def validate_input_specs(self):
    #     for input_tensor in self.net.graph_inputs:
    #         shape = input_tensor.shape
    #         if not self.network.has_implicit_batch_dimension:
    #             assert (
    #                 has_batch_dim
    #             ), "It's required to specify batch dimension when it's explicit in TensorRT network."

    #         dynamic_dims = get_dynamic_dims(shape)
    #         if len(dynamic_dims):
    #             assert not self.network.has_implicit_batch_dimension, (
    #                 "Can't have dynamic dim when "
    #                 f"batch dim is implicit, got {shape}."
    #             )
    #             assert len(
    #                 shape_ranges
    #             ), "shape_ranges must be provided when shape has dynamic dim."

    #             if self.optimization_profiles:
    #                 assert len(shape_ranges) == len(self.optimization_profiles), (
    #                     "Number of optimization "
    #                     f"profiles {len(self.optimization_profiles)} doesn't match with the number of shape_range"
    #                     f" {len(shape_ranges)} provided."
    #                 )
    #             else:
    #                 self.optimization_profiles = [
    #                     self.builder.create_optimization_profile()
    #                     for _ in range(len(shape_ranges))
    #                 ]

    #             for shape_range in shape_ranges:
    #                 assert (
    #                     len(shape_range) == 3
    #                 ), f"Expect three elements in shape_range, got {len(shape_range)}"
    #                 assert all(len(s) == len(shape) for s in shape_range), (
    #                     "Expect elements in shape_range"
    #                     f" {shape_range} have the same number of dimension as the provided shape {len(shape)}"
    #                 )

    #                 for i in range(len(shape)):
    #                     if i in dynamic_dims:
    #                         assert all(
    #                             shape_range[j][i] <= shape_range[j + 1][i]
    #                             for j in range(2)
    #                         ), (
    #                             "Expect dynamic dim"
    #                             f" {i} to have incremental value for shapes in shape_range {shape_range}."
    #                         )
    #                     else:
    #                         assert all(s[i] == shape[i] for s in shape_range), (
    #                             f"Expect non dynamic dim {i} to be the same"
    #                             f" for all shapes in shape_range {shape_range}."
    #                         )
    #         else:
    #             assert (
    #                 len(shape_ranges) == 0
    #             ), "shape_ranges are provided for input that doesn't have dynamic dim."

    def save_model(self, path=None):
        engine = self.builder.build_engine(self.network, self.builder_config)
        assert(engine)
        with open(path, "wb") as f:
            f.write(engine.serialize())