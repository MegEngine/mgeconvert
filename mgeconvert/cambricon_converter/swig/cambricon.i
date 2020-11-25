/*
 * $File: cambricon.i
 *
 * $Copyright: Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

%module cambriconLib

%{
#define SWIG_FILE_WITH_INIT
#include "cnrt.h"
#include "cnml.h"
%}

// ignore functions those are not implemented
%ignore cnmlSetBaseOpJobType;
%ignore cnmlGetBitWidth;
%ignore cnmlSetFusionOpCoreVersionChangable;
%ignore cnmlGetOffset;
%ignore cnmlGetScale;
%ignore cnmlComputeNdCycleNEqualOpForward;
%ignore cnmlGetPosition;
%ignore cnmlSetFusionOperationComputingLayout;
%ignore cnmlCreateNdCycleNEqualOp;
%ignore cnmlGetMovingPosition;
%ignore cnmlComputeNdCycleGreaterOpForward;
%ignore cnmlCreateCastOpForward;
%ignore cnmlCreateBroadcastMultOpForward;
%ignore cnmlGetInterval;
%ignore cnmlComputeSquaredDiffOpForward_V3;
%ignore cnmlSetBasicDivHighPrecision;
%ignore cnmlCreateBroadcastOpForward;
%ignore cnmlCreateArgminOp;

// use numpy swig for python-c type convertions, mainly (python list/numpy array <==> c array)
%include "numpy.i"
%init %{
    import_array();
%}

%apply (int DIM1, int *IN_ARRAY1) { (int dim_nums, int dim_values[]) }; 
%apply (int DIM1, float *IN_ARRAY1) { (int data_count, float *data_cpu) };
%apply (int DIM1, int8_t *IN_ARRAY1) { (int data_count, int8_t *data_cpu) };
%apply (int DIM1, uint8_t *IN_ARRAY1) { (int data_count, uint8_t* data_cpu) };
%apply (int DIM1, int16_t *IN_ARRAY1) { (int data_count, int16_t* data_cpu) };
%apply (int DIM1, float *INPLACE_ARRAY1) { (int data_out_count, float *data_out_cpu) };
%apply (int DIM1, int16_t *INPLACE_ARRAY1) { (int data_out_count, int16_t *data_out_cpu) };
%apply (int DIM1, int *INPLACE_ARRAY1) { (int dim, int *shape) };

// cnmlStatus_t cnmlSetTensorDimMutable(cnmlTensor_t tensor, bool *dim_mutable, int dim_num);
%typemap(in) (bool *dim_mutable, int dim_num) {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_ValueError, "Expecting a list");
        return NULL;
    }
    $2 = PyList_Size($input);
    $1 = (bool *) malloc(($2) * sizeof(bool));
    int i;
    for (i = 0; i < $2; ++i) {
        PyObject *s = PyList_GetItem($input, i);
        if (!PyBool_Check(s)) {
            free($1);
            PyErr_SetString(PyExc_ValueError, "List items must be bools");
            return NULL;
        }
        // $1[i] = PyBool_AsBool(s);
        // $1[i] = PyNumber_AsSsize_t(s);
        $1[i] = (s == Py_True);
    }
}
%typemap(freearg) (bool *dim_mutable, int dim_num) {
    if ($1) free($1);
}

// void cnGetShape(cnmlTensor_t tensor, int tensor_shape[4]) 
// will return a list with 4 elements in Python
%typemap(in, numinputs=0) int tensor_shape[4] (int temp[4]) {
    $1 = temp;
}

%typemap(argout) int tensor_shape[4] {
    int i;
    $result = PyList_New(4);
    for (i = 0; i < 4; i++) {
        PyObject *o = PyInt_FromLong((int) $1[i]);
        PyList_SetItem($result, i, o);
    }
}

%include "std_vector.i"

%include "cnrt.h"
%include "cnml.h"

// handle convertion between array of class and python list
namespace std {
    %template(VectorcnmlTensor) vector<cnmlTensor_t>;
    %template(Vectorvoid) vector<void *>;
};

%inline{
    cnmlTensor_t cnTensor_V2(cnmlTensorType_t tensor_type) {
        cnmlTensor_t ptensor;
        cnmlCreateTensor_V2(&ptensor, tensor_type);
        return ptensor;
    }

    cnmlTensor_t cnTensor_V3() {
        cnmlTensor_t ptensor;
        cnmlCreateTensor_V3(&ptensor);
        return ptensor;
    }

    void cnDestroyTensor(cnmlTensor_t ptensor) {
        cnmlDestroyTensor(&ptensor);
    }

    cnmlModel_t cnModel(char *name) {
        cnmlModel_t p_model;
        cnmlCreateModel(&p_model, name);
        return p_model;
    }

    cnrtQueue_t cnQueue() {
        cnrtQueue_t pqueue;
        cnrtCreateQueue(&pqueue);
        return pqueue;
    }

    void *cnMalloc(size_t sz) {
        void *pvar;
        cnrtRet_t ret = cnrtMalloc(&pvar, sz);
        return pvar;
    }

    void cnH2d(void *data_mlu, int data_count, float *data_cpu) {
        void *void_data_cpu = reinterpret_cast<void *>(data_cpu);
        cnrtMemcpy(data_mlu, void_data_cpu, data_count * sizeof(float), 
                                    CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    void cnH2dUint8(void *data_mlu, int data_count, uint8_t* data_cpu) {
        void* void_data_cpu = reinterpret_cast<void*>(data_cpu);
        cnrtMemcpy(data_mlu, void_data_cpu, data_count * sizeof(uint8_t),
                                    CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    void cnH2dFloat16(void* data_mlu, int data_count, float* data_cpu) {
        // input h2d: cpu fp32 -> cpu int16 -> dev fp16
        // declare cpu int16
        int16_t* int16_data_cpu = (int16_t*)malloc(data_count * sizeof(int16_t));
        // cpu fp32 -> cpu int16
        cnrtCastDataType(data_cpu, CNRT_FLOAT32, int16_data_cpu, CNRT_FLOAT16, data_count, NULL);
        // cpu int16 -> dev fp16
        cnrtMemcpy(data_mlu, int16_data_cpu, data_count * sizeof(int16_t),
                                        CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    void cnD2h(int data_out_count, float *data_out_cpu, void *data_mlu) {
        void *tmp_data_cpu = reinterpret_cast<void *>(data_out_cpu);
        cnrtMemcpy(tmp_data_cpu, data_mlu, data_out_count * sizeof(float),
                                    CNRT_MEM_TRANS_DIR_DEV2HOST);
        data_out_cpu = reinterpret_cast<float *>(tmp_data_cpu);
    }

    void cnD2hFloat16(int data_out_count, float* data_out_cpu, void* data_mlu) {
        // output d2h: dev fp16 -> cpu int16 -> cpu fp32
        // declare cpu int16
        int16_t* int16_data_cpu = (int16_t*)malloc(data_out_count * sizeof(int16_t));
        // dev fp16 -> cpu int16
        cnrtMemcpy(int16_data_cpu, data_mlu, data_out_count * sizeof(int16_t),
                                        CNRT_MEM_TRANS_DIR_DEV2HOST);
        // cpu int16 -> cpu fp32
        cnrtCastDataType(int16_data_cpu, CNRT_FLOAT16, data_out_cpu, CNRT_FLOAT32, data_out_count, NULL);

    }

    void cnH2dConstInt8(cnmlTensor_t tensor, int data_count, int8_t *data_cpu, bool free_aftercompile) {
        void *void_data_cpu = reinterpret_cast<void *>(data_cpu);
        cnmlBindConstData_V2(tensor, void_data_cpu, free_aftercompile);
    }

    void cnH2dConstFloat16(cnmlTensor_t tensor, int data_count, float *data_cpu, bool free_aftercompile) {
        // bind const fp16: float32 -> float16 -> bind
        int16_t* int16_data_cpu = (int16_t*)malloc(data_count * sizeof(int16_t));
        cnrtCastDataType(data_cpu, CNRT_FLOAT32, int16_data_cpu, CNRT_FLOAT16, data_count, NULL);
        cnmlBindConstData_V2(tensor, int16_data_cpu, free_aftercompile);
    }

    void cnH2dConst(cnmlTensor_t tensor, int data_count, float *data_cpu, bool free_aftercompile) {
        void *void_data_cpu = reinterpret_cast<void *>(data_cpu);
        cnmlBindConstData_V2(tensor, void_data_cpu, free_aftercompile);
    }

    cnmlBaseOp_t cnCastOp(cnmlTensor_t inp, cnmlTensor_t oup, cnmlCastType_t type) {
        cnmlBaseOp_t op;
        cnmlCreateCastOp(&op, type, inp, oup);
        return op;
    }

    cnmlFusionOp_t cnFusionOp() {
        cnmlFusionOp_t fuse_op;
        cnmlCreateFusionOp(&fuse_op);
        return fuse_op;
    }

    void cnGetShape(cnmlTensor_t tensor, int tensor_shape[4]) {
        cnmlGetTensorShape(tensor, tensor_shape);
    }

    cnmlBaseOp_t cnDevMemcpyOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t cpy_op;
        cnmlCreateDeviceMemcpyOp(&cpy_op, inp, oup);
        return cpy_op;
    }

#define REG_ELEMWISE_1(type) \
    cnmlBaseOp_t cn##type##Op(cnmlTensor_t inp, cnmlTensor_t oup)   \
    {                                                               \
        cnmlBaseOp_t elemwise_op;                                   \
        cnmlCreate##type##Op(&elemwise_op, inp, oup);               \
        return elemwise_op;                                         \
    }
REG_ELEMWISE_1(Abs)
REG_ELEMWISE_1(Exp)
REG_ELEMWISE_1(Log)
#undef REG_ELEMWISE

    cnmlBaseOp_t cnBasicDivOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t basic_div_op;
        cnmlCreateBasicDivOp(&basic_div_op, inp, oup);
        return basic_div_op;
    }

    cnmlBaseOp_t cnSqrtOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t sqrt_op;
        cnmlCreateSqrtOp(&sqrt_op, inp, oup);
        return sqrt_op;
    }

    cnmlBaseOp_t cnRsqrtOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t rsqrt_op;
        cnmlCreateRsqrtOp(&rsqrt_op, inp, oup);
        return rsqrt_op;
    }

    cnmlBaseOp_t cnSquareOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t square_op;
        cnmlCreateSquareOp(&square_op, inp, oup);
        return square_op;
    }

    cnmlBaseOp_t cnPowOp(cnmlTensor_t inp, cnmlTensor_t oup, float power_c) {
        cnmlBaseOp_t pow_op;
        cnmlCreatePowerOp(&pow_op, inp, oup, power_c);
        return pow_op;
    }

    cnmlBaseOp_t cnArgmaxOp(cnmlDimension_t dim_axis, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t argmax_op;
        cnmlCreateArgmaxOp(&argmax_op, dim_axis, inp, oup);
        return argmax_op;
    }

    cnmlBaseOp_t cnFloorOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t floor_op;
        cnmlCreateFloorOp(&floor_op, inp, oup);
        return floor_op;
    }

    cnmlBaseOp_t cnActiveOp(cnmlActiveFunction_t active_func, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t active_op;
        cnmlCreateActiveOp(&active_op, active_func, inp, oup);
        return active_op;
    }

#define REG_ELEMWISE_2(type) \
    cnmlBaseOp_t cn##type##Op(cnmlTensor_t inp1, cnmlTensor_t inp2, cnmlTensor_t oup) \
    {                                                                   \
        cnmlBaseOp_t elemwise_op;                                       \
        cnmlCreateBroadcast##type##Op(&elemwise_op, inp1, inp2, oup);   \
        return elemwise_op;                                             \
    }
REG_ELEMWISE_2(Add)
REG_ELEMWISE_2(Sub)
REG_ELEMWISE_2(Mult)
#undef REG_ELEMWISE

    cnmlBaseOp_t cnDivOp(cnmlTensor_t inp1, cnmlTensor_t inp2, cnmlTensor_t oup) {
        cnmlBaseOp_t div_op;
        cnmlCreateRealDivOp(&div_op, inp1, inp2, oup);
        return div_op;
    }

#define REG_CYCLE(type) \
    cnmlBaseOp_t cnCycle##type##Op(cnmlTensor_t inp1, cnmlTensor_t inp2, cnmlTensor_t oup) { \
        cnmlBaseOp_t op; \
        cnmlCreateCycle##type##Op(&op, inp1, inp2, oup); \
        return op; \
    }
REG_CYCLE(Add)
REG_CYCLE(Mult)
#undef REG_CYCLE

#define REG_REDUCE(type) \
    cnmlBaseOp_t cnReduce##type##Op(cnmlDimension_t dim, cnmlTensor_t inp, cnmlTensor_t oup) \
    {                                                           \
        cnmlBaseOp_t reduce_op;                                 \
        cnmlCreateReduce##type##Op(&reduce_op, dim, inp, oup);  \
        return reduce_op;                                       \
    }
REG_REDUCE(Max)
REG_REDUCE(Mean)
REG_REDUCE(Sum)
REG_REDUCE(Product)
#undef REG_REDUCE

    cnmlBaseOp_t cnReduceAndOp(cnmlReduce_andDim_t dim, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t reduce_op;
        cnmlCreateReduceAndOp(&reduce_op, dim, inp, oup);
        return reduce_op;
    }

    cnmlBaseOp_t cnReduceOrOp(cnmlReduce_orDim_t dim, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t reduce_op;
        cnmlCreateReduceOrOp(&reduce_op, dim, inp, oup);
        return reduce_op;
    }

    cnmlBaseOp_t cnBatchNormOp(cnmlTensor_t inp, cnmlTensor_t oup, cnmlTensor_t mean, cnmlTensor_t var) {
        cnmlBaseOp_t bn_op;
        cnmlCreateBatchNormOpForward(&bn_op, inp, oup, mean, var);
        return bn_op;
    }

    cnmlBaseOp_t cnMatMulOp(cnmlTensor_t inp, cnmlTensor_t oup, cnmlTensor_t filter, cnmlTensor_t bias) {
        cnmlBaseOp_t matmul_op;
        cnmlCreateMlpOp(&matmul_op, inp, oup, filter, bias);
        return matmul_op;
    }

    cnmlBaseOp_t cnBatchDotOp(cnmlTensor_t lef, cnmlTensor_t rht, cnmlTensor_t oup, bool trans_a, bool trans_b) {
        cnmlBaseOp_t batchdot_op;
        cnmlCreateBatchDotOp(&batchdot_op, lef, rht, oup, trans_a, trans_b);
        return batchdot_op;
    }

    cnmlConvOpParam_t cnConvOpParam(int stride_h, int stride_w, 
                                        int dilation_h, int dilation_w, 
                                        int pad_h, int pad_w) {
        cnmlConvOpParam_t param;
        cnmlCreateConvOpParam(&param, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w);
        return param;
    }

    void cnDestroyConvOpParam(cnmlConvOpParam_t param) {
        cnmlDestroyConvOpParam(&param);
    }

    cnmlDeconvOpParam_t cnDeconvOpParam(int stride_h, int stride_w,
                                        int dilation_h, int dilation_w,
                                        int pad_h, int pad_w) {
        cnmlDeconvOpParam_t param;
        cnmlCreateDeconvOpParam_V3(&param, stride_h, stride_w, 0, 0, 0, 0, pad_w, pad_h, dilation_h, dilation_w);
        return param;
    }

    void cnDestoryDeconvOpParam(cnmlDeconvOpParam_t param) {
        cnmlDestroyDeconvOpParam(&param);
    }

    cnmlQuantizedParam_t cnQuantizedParam(int pos, float scale = 1., float offset = 0.) {
        cnmlQuantizedParam_t input_quant_param;
        cnmlCreateQuantizedParam(&input_quant_param, pos, scale, offset);
        return input_quant_param;
    }

    void cnDestroyQuantizedParam(cnmlQuantizedParam_t qparam) {
        cnmlDestroyQuantizedParam(&qparam);
    }

    cnmlBaseOp_t cnConvOp(cnmlConvOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup, 
                            cnmlTensor_t filter, cnmlTensor_t bias, int groups) {
        cnmlBaseOp_t conv_op;
        if (groups == 1) cnmlCreateConvOpForward(&conv_op, param, inp, oup, filter, bias);
        else cnmlCreateConvGroupOpForward(&conv_op, param, inp, oup, filter, bias, groups);
        return conv_op;
    }

    cnmlBaseOp_t cnDeconvOp(cnmlDeconvOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup,
                            cnmlTensor_t filter, cnmlTensor_t bias) {
        cnmlBaseOp_t deconv_op;
        cnmlCreateDeconvOpForward(&deconv_op, param, inp, oup, filter, bias);
        return deconv_op;
    }

    cnmlConvFirstOpParam_t cnConvFirstOpParam(int sh, int sw, int dh, int dw, int ph, int pw) {
        cnmlConvFirstOpParam_t param;
        cnmlCreateConvFirstOpParam_V2(&param, sh, sw, dh, dw, pw, pw, ph, ph);
        return param;
    }

    cnmlBaseOp_t cnConvFirstOp(cnmlConvFirstOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup,
                               cnmlTensor_t filter, cnmlTensor_t bias,
                               cnmlTensor_t mean, cnmlTensor_t std) {
        cnmlBaseOp_t op;
        cnmlCreateConvFirstOpForward(&op, param, inp, mean, oup, filter, bias, std);
        return op;
    }

    void cnDestroyConvFirstOpParam(cnmlConvFirstOpParam_t param) {
        cnmlDestroyConvFirstOpParam(&param);
    }

    cnmlReshapeOpParam_t cnReshapeOpParam(int no, int co, int ho, int wo,
                                            cnmlDataOrder_t data_order = CNML_NHWC) {
        cnmlReshapeOpParam_t param;
        cnmlCreateReshapeOpParam(&param, no, co, ho, wo, data_order);
        return param;
    }

    cnmlBaseOp_t cnReshapeOp(cnmlReshapeOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t reshape_op;
        cnmlCreateReshapeOp(&reshape_op, param, inp, oup);
        return reshape_op;
    }

    cnmlAddPadOpParam_t cnAddPadOpParam(int pt, int pb, int pl, int pr, float pad_value) {
        cnmlAddPadOpParam_t param;
        cnmlCreateAddPadOpParam_V2(&param, pt, pb, pl, pr, pad_value);
        return param;
    }

    cnmlBaseOp_t cnAddPadOp(cnmlAddPadOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t op;
        cnmlCreateAddPadOp(&op, param, inp, oup);
        return op;
    }

    cnmlStridedSliceOpParam_t cnSliceOpParam(int nb, int cb, int hb, int wb, int ne, int ce, int he, int we, int ns, int cs, int hs, int ws) {
        cnmlStridedSliceOpParam_t param;
        cnmlCreateStridedSliceOpParam(&param, nb, cb, hb, wb, ne, ce, he, we, ns, cs, hs, ws);
        return param;
    }

    cnmlBaseOp_t cnSliceOp(cnmlStridedSliceOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t slice_op;
        cnmlCreateStridedSliceOp(&slice_op, param, inp, oup);
        return slice_op;
    }

    cnmlTransposeOpParam_t cnTransposeOpParam(int d0, int d1, int d2, int d3,
                                                cnmlDataOrder_t data_order = CNML_NCHW) {
        cnmlTransposeOpParam_t param;
        cnmlCreateTransposeOpParam(&param, data_order, d0, d1, d2, d3);
        return param;
    }
    
    cnmlBaseOp_t cnTransposeProOp(cnmlTransposeOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t transpose_op;
        cnmlCreateTransposeProOp(&transpose_op, inp, oup, param);
        return transpose_op;
    }

    cnmlBaseOp_t cnBroadcastOp(cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t broadcast_op;
        cnmlCreateBroadcastOp(&broadcast_op, inp, oup);
        return broadcast_op;
    }

    cnmlPoolOpParam_t cnPoolOpParam(int window_h, int window_w, int stride_h, 
                                    int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, 
                                    cnmlPoolMode_t pool_mode = CNML_POOL_MAX, 
                                    cnmlPoolStrategyMode_t strategy_mode = CNML_POOL_KVALID, 
                                    bool real = false) {
        cnmlPoolOpParam_t param;
        cnmlCreatePoolOpParam(&param, window_h, window_w, stride_h, stride_w, 
                                pad_h, pad_w, dilation_h, dilation_w, pool_mode, strategy_mode, real);
        return param;
    }

    void cnDestroyPoolOpParam(cnmlPoolOpParam_t param) {
        cnmlDestroyPoolOpParam(&param);
    }

    cnmlBaseOp_t cnPoolOp(cnmlPoolOpParam_t param, cnmlTensor_t inp, cnmlTensor_t oup) {
        cnmlBaseOp_t pool_op;
        cnmlCreatePoolOp(&pool_op, param, inp, oup);
        return pool_op;
    }

    cnmlConcatOpParam_t cnConcatOpParam(int input_num, int output_num, cnmlDimension_t concat_mode) {
        cnmlConcatOpParam_t param;
        cnmlCreateConcatOpParam(&param, input_num, output_num, concat_mode);
        return param;
    }

    cnmlBaseOp_t cnConcatOp(cnmlConcatOpParam_t param, 
                            std::vector<cnmlTensor_t> input_tensors, 
                            std::vector<cnmlTensor_t> output_tensors) {
        cnmlBaseOp_t concat_op;
        cnmlCreateConcatOp(&concat_op, param, input_tensors.data(), input_tensors.size(), output_tensors.data(), output_tensors.size());
        return concat_op;
    }

    void cnComputeConcatOp(cnmlBaseOp_t concatop, cnmlTensor_t *inp_tensors, std::vector<void *> inputs,
                            cnmlTensor_t *oup_tensors, std::vector<void *> outputs, cnrtQueue_t cnq, void *extra) {
        cnmlComputeConcatOpForward_V4(concatop, inp_tensors, inputs.data(), inputs.size(), oup_tensors, outputs.data(), outputs.size(), cnq, extra);
    }

    void cnFusionIO(cnmlFusionOp_t op, std::vector<cnmlTensor_t> input_tensors, std::vector<cnmlTensor_t> output_tensors) {
        cnmlSetFusionIO(op, input_tensors.data(), input_tensors.size(), output_tensors.data(), output_tensors.size());
    }

    // void cnComputeFusionOp(cnmlFusionOp_t op, cnmlTensor_t input_tensors[], std::vector<void *> inputs, 
    //                         cnmlTensor_t output_tensors[], std::vector<void *> outputs, cnrtQueue_t cnq, void *extra) {
    void cnComputeFusionOp(cnmlFusionOp_t op, std::vector<cnmlTensor_t> input_tensors, std::vector<void*> inputs,
                            std::vector<cnmlTensor_t> output_tensors, std::vector<void*> outputs, cnrtQueue_t cnq, void* extra) {
        cnmlComputeFusionOpForward_V4(op, input_tensors.data(), inputs.data(), inputs.size(), output_tensors.data(), outputs.data(), outputs.size(), cnq, extra);
    }

    void cnDestroyBaseOp(cnmlBaseOp_t op) {
        cnmlDestroyBaseOp(&op);
    }

    void cnDestroyFusionOp(cnmlFusionOp_t op) {
        cnmlDestroyFusionOp(&op);
    }
}
