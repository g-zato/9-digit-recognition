á¤
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8Ò¶

sequential_64/dense_210/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name sequential_64/dense_210/kernel

2sequential_64/dense_210/kernel/Read/ReadVariableOpReadVariableOpsequential_64/dense_210/kernel* 
_output_shapes
:
*
dtype0

sequential_64/dense_210/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namesequential_64/dense_210/bias

0sequential_64/dense_210/bias/Read/ReadVariableOpReadVariableOpsequential_64/dense_210/bias*
_output_shapes	
:*
dtype0

sequential_64/dense_211/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*/
shared_name sequential_64/dense_211/kernel

2sequential_64/dense_211/kernel/Read/ReadVariableOpReadVariableOpsequential_64/dense_211/kernel*
_output_shapes
:	
*
dtype0

sequential_64/dense_211/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namesequential_64/dense_211/bias

0sequential_64/dense_211/bias/Read/ReadVariableOpReadVariableOpsequential_64/dense_211/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
¨
%Adam/sequential_64/dense_210/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/sequential_64/dense_210/kernel/m
¡
9Adam/sequential_64/dense_210/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/sequential_64/dense_210/kernel/m* 
_output_shapes
:
*
dtype0

#Adam/sequential_64/dense_210/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_64/dense_210/bias/m

7Adam/sequential_64/dense_210/bias/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_64/dense_210/bias/m*
_output_shapes	
:*
dtype0
§
%Adam/sequential_64/dense_211/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*6
shared_name'%Adam/sequential_64/dense_211/kernel/m
 
9Adam/sequential_64/dense_211/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/sequential_64/dense_211/kernel/m*
_output_shapes
:	
*
dtype0

#Adam/sequential_64/dense_211/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/sequential_64/dense_211/bias/m

7Adam/sequential_64/dense_211/bias/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_64/dense_211/bias/m*
_output_shapes
:
*
dtype0
¨
%Adam/sequential_64/dense_210/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/sequential_64/dense_210/kernel/v
¡
9Adam/sequential_64/dense_210/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/sequential_64/dense_210/kernel/v* 
_output_shapes
:
*
dtype0

#Adam/sequential_64/dense_210/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_64/dense_210/bias/v

7Adam/sequential_64/dense_210/bias/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_64/dense_210/bias/v*
_output_shapes	
:*
dtype0
§
%Adam/sequential_64/dense_211/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*6
shared_name'%Adam/sequential_64/dense_211/kernel/v
 
9Adam/sequential_64/dense_211/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/sequential_64/dense_211/kernel/v*
_output_shapes
:	
*
dtype0

#Adam/sequential_64/dense_211/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/sequential_64/dense_211/bias/v

7Adam/sequential_64/dense_211/bias/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_64/dense_211/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
¹
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ô
valueêBç Bà

layer-0
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9
 

	0

1
2
3

	0

1
2
3

regularization_losses
trainable_variables

layers
layer_regularization_losses
metrics
non_trainable_variables
	variables
 
][
VARIABLE_VALUEsequential_64/dense_210/kernel)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_64/dense_210/bias'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1

regularization_losses
trainable_variables

layers
layer_regularization_losses
 metrics
!non_trainable_variables
	variables
][
VARIABLE_VALUEsequential_64/dense_211/kernel)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_64/dense_211/bias'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses
trainable_variables

"layers
#layer_regularization_losses
$metrics
%non_trainable_variables
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

&0
 
 
 
 
 
 
 
 
 
x
	'total
	(count
)
_fn_kwargs
*regularization_losses
+trainable_variables
,	variables
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

'0
(1

*regularization_losses
+trainable_variables

.layers
/layer_regularization_losses
0metrics
1non_trainable_variables
,	variables
 
 
 

'0
(1
~
VARIABLE_VALUE%Adam/sequential_64/dense_210/kernel/mElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_64/dense_210/bias/mClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE%Adam/sequential_64/dense_211/kernel/mElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_64/dense_211/bias/mClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE%Adam/sequential_64/dense_210/kernel/vElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_64/dense_210/bias/vClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE%Adam/sequential_64/dense_211/kernel/vElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_64/dense_211/bias/vClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential_64/dense_210/kernelsequential_64/dense_210/biassequential_64/dense_211/kernelsequential_64/dense_211/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_4219130
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2sequential_64/dense_210/kernel/Read/ReadVariableOp0sequential_64/dense_210/bias/Read/ReadVariableOp2sequential_64/dense_211/kernel/Read/ReadVariableOp0sequential_64/dense_211/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9Adam/sequential_64/dense_210/kernel/m/Read/ReadVariableOp7Adam/sequential_64/dense_210/bias/m/Read/ReadVariableOp9Adam/sequential_64/dense_211/kernel/m/Read/ReadVariableOp7Adam/sequential_64/dense_211/bias/m/Read/ReadVariableOp9Adam/sequential_64/dense_210/kernel/v/Read/ReadVariableOp7Adam/sequential_64/dense_210/bias/v/Read/ReadVariableOp9Adam/sequential_64/dense_211/kernel/v/Read/ReadVariableOp7Adam/sequential_64/dense_211/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_save_4219301

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_64/dense_210/kernelsequential_64/dense_210/biassequential_64/dense_211/kernelsequential_64/dense_211/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount%Adam/sequential_64/dense_210/kernel/m#Adam/sequential_64/dense_210/bias/m%Adam/sequential_64/dense_211/kernel/m#Adam/sequential_64/dense_211/bias/m%Adam/sequential_64/dense_210/kernel/v#Adam/sequential_64/dense_210/bias/v%Adam/sequential_64/dense_211/kernel/v#Adam/sequential_64/dense_211/bias/v*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_restore_4219370Þ

­
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219166

inputs,
(dense_210_matmul_readvariableop_resource-
)dense_210_biasadd_readvariableop_resource,
(dense_211_matmul_readvariableop_resource-
)dense_211_biasadd_readvariableop_resource
identity¢ dense_210/BiasAdd/ReadVariableOp¢dense_210/MatMul/ReadVariableOp¢ dense_211/BiasAdd/ReadVariableOp¢dense_211/MatMul/ReadVariableOp­
dense_210/MatMul/ReadVariableOpReadVariableOp(dense_210_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_210/MatMul/ReadVariableOp
dense_210/MatMulMatMulinputs'dense_210/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/MatMul«
 dense_210/BiasAdd/ReadVariableOpReadVariableOp)dense_210_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_210/BiasAdd/ReadVariableOpª
dense_210/BiasAddBiasAdddense_210/MatMul:product:0(dense_210/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/BiasAddw
dense_210/ReluReludense_210/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/Relu¬
dense_211/MatMul/ReadVariableOpReadVariableOp(dense_211_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02!
dense_211/MatMul/ReadVariableOp§
dense_211/MatMulMatMuldense_210/Relu:activations:0'dense_211/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/MatMulª
 dense_211/BiasAdd/ReadVariableOpReadVariableOp)dense_211_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_211/BiasAdd/ReadVariableOp©
dense_211/BiasAddBiasAdddense_211/MatMul:product:0(dense_211/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/BiasAdd
dense_211/SoftmaxSoftmaxdense_211/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/Softmaxù
IdentityIdentitydense_211/Softmax:softmax:0!^dense_210/BiasAdd/ReadVariableOp ^dense_210/MatMul/ReadVariableOp!^dense_211/BiasAdd/ReadVariableOp ^dense_211/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2D
 dense_210/BiasAdd/ReadVariableOp dense_210/BiasAdd/ReadVariableOp2B
dense_210/MatMul/ReadVariableOpdense_210/MatMul/ReadVariableOp2D
 dense_211/BiasAdd/ReadVariableOp dense_211/BiasAdd/ReadVariableOp2B
dense_211/MatMul/ReadVariableOpdense_211/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ó
ê
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219073
input_1,
(dense_210_statefulpartitionedcall_args_1,
(dense_210_statefulpartitionedcall_args_2,
(dense_211_statefulpartitionedcall_args_1,
(dense_211_statefulpartitionedcall_args_2
identity¢!dense_210/StatefulPartitionedCall¢!dense_211/StatefulPartitionedCall²
!dense_210/StatefulPartitionedCallStatefulPartitionedCallinput_1(dense_210_statefulpartitionedcall_args_1(dense_210_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_210_layer_call_and_return_conditional_losses_42190272#
!dense_210/StatefulPartitionedCallÔ
!dense_211/StatefulPartitionedCallStatefulPartitionedCall*dense_210/StatefulPartitionedCall:output:0(dense_211_statefulpartitionedcall_args_1(dense_211_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_211_layer_call_and_return_conditional_losses_42190502#
!dense_211/StatefulPartitionedCallÆ
IdentityIdentity*dense_211/StatefulPartitionedCall:output:0"^dense_210/StatefulPartitionedCall"^dense_211/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2F
!dense_210/StatefulPartitionedCall!dense_210/StatefulPartitionedCall2F
!dense_211/StatefulPartitionedCall!dense_211/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ï	
ß
F__inference_dense_210_layer_call_and_return_conditional_losses_4219195

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
¦1
Ï	
 __inference__traced_save_4219301
file_prefix=
9savev2_sequential_64_dense_210_kernel_read_readvariableop;
7savev2_sequential_64_dense_210_bias_read_readvariableop=
9savev2_sequential_64_dense_211_kernel_read_readvariableop;
7savev2_sequential_64_dense_211_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_adam_sequential_64_dense_210_kernel_m_read_readvariableopB
>savev2_adam_sequential_64_dense_210_bias_m_read_readvariableopD
@savev2_adam_sequential_64_dense_211_kernel_m_read_readvariableopB
>savev2_adam_sequential_64_dense_211_bias_m_read_readvariableopD
@savev2_adam_sequential_64_dense_210_kernel_v_read_readvariableopB
>savev2_adam_sequential_64_dense_210_bias_v_read_readvariableopD
@savev2_adam_sequential_64_dense_211_kernel_v_read_readvariableopB
>savev2_adam_sequential_64_dense_211_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2fc44ce26be94f9fb5a4756eb4b87929/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¬	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¾
value´B±B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¾	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_sequential_64_dense_210_kernel_read_readvariableop7savev2_sequential_64_dense_210_bias_read_readvariableop9savev2_sequential_64_dense_211_kernel_read_readvariableop7savev2_sequential_64_dense_211_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_adam_sequential_64_dense_210_kernel_m_read_readvariableop>savev2_adam_sequential_64_dense_210_bias_m_read_readvariableop@savev2_adam_sequential_64_dense_211_kernel_m_read_readvariableop>savev2_adam_sequential_64_dense_211_bias_m_read_readvariableop@savev2_adam_sequential_64_dense_210_kernel_v_read_readvariableop>savev2_adam_sequential_64_dense_210_bias_v_read_readvariableop@savev2_adam_sequential_64_dense_211_kernel_v_read_readvariableop>savev2_adam_sequential_64_dense_211_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
~: :
::	
:
: : : : : : : :
::	
:
:
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ù
¬
+__inference_dense_210_layer_call_fn_4219202

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_210_layer_call_and_return_conditional_losses_42190272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ð
é
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219105

inputs,
(dense_210_statefulpartitionedcall_args_1,
(dense_210_statefulpartitionedcall_args_2,
(dense_211_statefulpartitionedcall_args_1,
(dense_211_statefulpartitionedcall_args_2
identity¢!dense_210/StatefulPartitionedCall¢!dense_211/StatefulPartitionedCall±
!dense_210/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_210_statefulpartitionedcall_args_1(dense_210_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_210_layer_call_and_return_conditional_losses_42190272#
!dense_210/StatefulPartitionedCallÔ
!dense_211/StatefulPartitionedCallStatefulPartitionedCall*dense_210/StatefulPartitionedCall:output:0(dense_211_statefulpartitionedcall_args_1(dense_211_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_211_layer_call_and_return_conditional_losses_42190502#
!dense_211/StatefulPartitionedCallÆ
IdentityIdentity*dense_211/StatefulPartitionedCall:output:0"^dense_210/StatefulPartitionedCall"^dense_211/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2F
!dense_210/StatefulPartitionedCall!dense_210/StatefulPartitionedCall2F
!dense_211/StatefulPartitionedCall!dense_211/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ð
é
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219086

inputs,
(dense_210_statefulpartitionedcall_args_1,
(dense_210_statefulpartitionedcall_args_2,
(dense_211_statefulpartitionedcall_args_1,
(dense_211_statefulpartitionedcall_args_2
identity¢!dense_210/StatefulPartitionedCall¢!dense_211/StatefulPartitionedCall±
!dense_210/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_210_statefulpartitionedcall_args_1(dense_210_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_210_layer_call_and_return_conditional_losses_42190272#
!dense_210/StatefulPartitionedCallÔ
!dense_211/StatefulPartitionedCallStatefulPartitionedCall*dense_210/StatefulPartitionedCall:output:0(dense_211_statefulpartitionedcall_args_1(dense_211_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_211_layer_call_and_return_conditional_losses_42190502#
!dense_211/StatefulPartitionedCallÆ
IdentityIdentity*dense_211/StatefulPartitionedCall:output:0"^dense_210/StatefulPartitionedCall"^dense_211/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2F
!dense_210/StatefulPartitionedCall!dense_210/StatefulPartitionedCall2F
!dense_211/StatefulPartitionedCall!dense_211/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ï	
ß
F__inference_dense_210_layer_call_and_return_conditional_losses_4219027

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
÷
¬
+__inference_dense_211_layer_call_fn_4219220

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_211_layer_call_and_return_conditional_losses_42190502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ñ	
ß
F__inference_dense_211_layer_call_and_return_conditional_losses_4219213

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ý
ö
"__inference__wrapped_model_4219012
input_1:
6sequential_64_dense_210_matmul_readvariableop_resource;
7sequential_64_dense_210_biasadd_readvariableop_resource:
6sequential_64_dense_211_matmul_readvariableop_resource;
7sequential_64_dense_211_biasadd_readvariableop_resource
identity¢.sequential_64/dense_210/BiasAdd/ReadVariableOp¢-sequential_64/dense_210/MatMul/ReadVariableOp¢.sequential_64/dense_211/BiasAdd/ReadVariableOp¢-sequential_64/dense_211/MatMul/ReadVariableOp×
-sequential_64/dense_210/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_210_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_64/dense_210/MatMul/ReadVariableOp½
sequential_64/dense_210/MatMulMatMulinput_15sequential_64/dense_210/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_64/dense_210/MatMulÕ
.sequential_64/dense_210/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_210_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_64/dense_210/BiasAdd/ReadVariableOpâ
sequential_64/dense_210/BiasAddBiasAdd(sequential_64/dense_210/MatMul:product:06sequential_64/dense_210/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_64/dense_210/BiasAdd¡
sequential_64/dense_210/ReluRelu(sequential_64/dense_210/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_64/dense_210/ReluÖ
-sequential_64/dense_211/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_211_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02/
-sequential_64/dense_211/MatMul/ReadVariableOpß
sequential_64/dense_211/MatMulMatMul*sequential_64/dense_210/Relu:activations:05sequential_64/dense_211/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential_64/dense_211/MatMulÔ
.sequential_64/dense_211/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_211_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.sequential_64/dense_211/BiasAdd/ReadVariableOpá
sequential_64/dense_211/BiasAddBiasAdd(sequential_64/dense_211/MatMul:product:06sequential_64/dense_211/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2!
sequential_64/dense_211/BiasAdd©
sequential_64/dense_211/SoftmaxSoftmax(sequential_64/dense_211/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2!
sequential_64/dense_211/Softmax¿
IdentityIdentity)sequential_64/dense_211/Softmax:softmax:0/^sequential_64/dense_210/BiasAdd/ReadVariableOp.^sequential_64/dense_210/MatMul/ReadVariableOp/^sequential_64/dense_211/BiasAdd/ReadVariableOp.^sequential_64/dense_211/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2`
.sequential_64/dense_210/BiasAdd/ReadVariableOp.sequential_64/dense_210/BiasAdd/ReadVariableOp2^
-sequential_64/dense_210/MatMul/ReadVariableOp-sequential_64/dense_210/MatMul/ReadVariableOp2`
.sequential_64/dense_211/BiasAdd/ReadVariableOp.sequential_64/dense_211/BiasAdd/ReadVariableOp2^
-sequential_64/dense_211/MatMul/ReadVariableOp-sequential_64/dense_211/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1

ø
/__inference_sequential_64_layer_call_fn_4219184

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_42191052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ó
ê
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219063
input_1,
(dense_210_statefulpartitionedcall_args_1,
(dense_210_statefulpartitionedcall_args_2,
(dense_211_statefulpartitionedcall_args_1,
(dense_211_statefulpartitionedcall_args_2
identity¢!dense_210/StatefulPartitionedCall¢!dense_211/StatefulPartitionedCall²
!dense_210/StatefulPartitionedCallStatefulPartitionedCallinput_1(dense_210_statefulpartitionedcall_args_1(dense_210_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_210_layer_call_and_return_conditional_losses_42190272#
!dense_210/StatefulPartitionedCallÔ
!dense_211/StatefulPartitionedCallStatefulPartitionedCall*dense_210/StatefulPartitionedCall:output:0(dense_211_statefulpartitionedcall_args_1(dense_211_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_211_layer_call_and_return_conditional_losses_42190502#
!dense_211/StatefulPartitionedCallÆ
IdentityIdentity*dense_211/StatefulPartitionedCall:output:0"^dense_210/StatefulPartitionedCall"^dense_211/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2F
!dense_210/StatefulPartitionedCall!dense_210/StatefulPartitionedCall2F
!dense_211/StatefulPartitionedCall!dense_211/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
T
µ
#__inference__traced_restore_4219370
file_prefix3
/assignvariableop_sequential_64_dense_210_kernel3
/assignvariableop_1_sequential_64_dense_210_bias5
1assignvariableop_2_sequential_64_dense_211_kernel3
/assignvariableop_3_sequential_64_dense_211_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count=
9assignvariableop_11_adam_sequential_64_dense_210_kernel_m;
7assignvariableop_12_adam_sequential_64_dense_210_bias_m=
9assignvariableop_13_adam_sequential_64_dense_211_kernel_m;
7assignvariableop_14_adam_sequential_64_dense_211_bias_m=
9assignvariableop_15_adam_sequential_64_dense_210_kernel_v;
7assignvariableop_16_adam_sequential_64_dense_210_bias_v=
9assignvariableop_17_adam_sequential_64_dense_211_kernel_v;
7assignvariableop_18_adam_sequential_64_dense_211_bias_v
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1²	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¾
value´B±B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp/assignvariableop_sequential_64_dense_210_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp/assignvariableop_1_sequential_64_dense_210_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp1assignvariableop_2_sequential_64_dense_211_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp/assignvariableop_3_sequential_64_dense_211_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11²
AssignVariableOp_11AssignVariableOp9assignvariableop_11_adam_sequential_64_dense_210_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOp7assignvariableop_12_adam_sequential_64_dense_210_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13²
AssignVariableOp_13AssignVariableOp9assignvariableop_13_adam_sequential_64_dense_211_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOp7assignvariableop_14_adam_sequential_64_dense_211_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp9assignvariableop_15_adam_sequential_64_dense_210_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp7assignvariableop_16_adam_sequential_64_dense_210_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17²
AssignVariableOp_17AssignVariableOp9assignvariableop_17_adam_sequential_64_dense_211_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp7assignvariableop_18_adam_sequential_64_dense_211_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix

ù
/__inference_sequential_64_layer_call_fn_4219093
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_42190862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1

­
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219148

inputs,
(dense_210_matmul_readvariableop_resource-
)dense_210_biasadd_readvariableop_resource,
(dense_211_matmul_readvariableop_resource-
)dense_211_biasadd_readvariableop_resource
identity¢ dense_210/BiasAdd/ReadVariableOp¢dense_210/MatMul/ReadVariableOp¢ dense_211/BiasAdd/ReadVariableOp¢dense_211/MatMul/ReadVariableOp­
dense_210/MatMul/ReadVariableOpReadVariableOp(dense_210_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_210/MatMul/ReadVariableOp
dense_210/MatMulMatMulinputs'dense_210/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/MatMul«
 dense_210/BiasAdd/ReadVariableOpReadVariableOp)dense_210_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_210/BiasAdd/ReadVariableOpª
dense_210/BiasAddBiasAdddense_210/MatMul:product:0(dense_210/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/BiasAddw
dense_210/ReluReludense_210/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_210/Relu¬
dense_211/MatMul/ReadVariableOpReadVariableOp(dense_211_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02!
dense_211/MatMul/ReadVariableOp§
dense_211/MatMulMatMuldense_210/Relu:activations:0'dense_211/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/MatMulª
 dense_211/BiasAdd/ReadVariableOpReadVariableOp)dense_211_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_211/BiasAdd/ReadVariableOp©
dense_211/BiasAddBiasAdddense_211/MatMul:product:0(dense_211/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/BiasAdd
dense_211/SoftmaxSoftmaxdense_211/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_211/Softmaxù
IdentityIdentitydense_211/Softmax:softmax:0!^dense_210/BiasAdd/ReadVariableOp ^dense_210/MatMul/ReadVariableOp!^dense_211/BiasAdd/ReadVariableOp ^dense_211/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2D
 dense_210/BiasAdd/ReadVariableOp dense_210/BiasAdd/ReadVariableOp2B
dense_210/MatMul/ReadVariableOpdense_210/MatMul/ReadVariableOp2D
 dense_211/BiasAdd/ReadVariableOp dense_211/BiasAdd/ReadVariableOp2B
dense_211/MatMul/ReadVariableOpdense_211/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

ø
/__inference_sequential_64_layer_call_fn_4219175

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_42190862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ñ	
ß
F__inference_dense_211_layer_call_and_return_conditional_losses_4219050

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

ù
/__inference_sequential_64_layer_call_fn_4219112
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_42191052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
â
ï
%__inference_signature_wrapper_4219130
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_42190122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:g

layer-0
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
:_default_save_signature
;__call__
*<&call_and_return_all_conditional_losses"§
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_64", "layers": [{"class_name": "Dense", "config": {"name": "dense_210", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_211", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 784]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_64", "layers": [{"class_name": "Dense", "config": {"name": "dense_210", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_211", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 784]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ø

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_210", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_210", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
ú

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "dense_211", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_211", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}

iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9"
	optimizer
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
·
regularization_losses
trainable_variables

layers
layer_regularization_losses
metrics
non_trainable_variables
	variables
;__call__
:_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
2:0
2sequential_64/dense_210/kernel
+:)2sequential_64/dense_210/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper

regularization_losses
trainable_variables

layers
layer_regularization_losses
 metrics
!non_trainable_variables
	variables
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
1:/	
2sequential_64/dense_211/kernel
*:(
2sequential_64/dense_211/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses
trainable_variables

"layers
#layer_regularization_losses
$metrics
%non_trainable_variables
	variables
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	'total
	(count
)
_fn_kwargs
*regularization_losses
+trainable_variables
,	variables
-	keras_api
B__call__
*C&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper

*regularization_losses
+trainable_variables

.layers
/layer_regularization_losses
0metrics
1non_trainable_variables
,	variables
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
7:5
2%Adam/sequential_64/dense_210/kernel/m
0:.2#Adam/sequential_64/dense_210/bias/m
6:4	
2%Adam/sequential_64/dense_211/kernel/m
/:-
2#Adam/sequential_64/dense_211/bias/m
7:5
2%Adam/sequential_64/dense_210/kernel/v
0:.2#Adam/sequential_64/dense_210/bias/v
6:4	
2%Adam/sequential_64/dense_211/kernel/v
/:-
2#Adam/sequential_64/dense_211/bias/v
á2Þ
"__inference__wrapped_model_4219012·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
2
/__inference_sequential_64_layer_call_fn_4219112
/__inference_sequential_64_layer_call_fn_4219175
/__inference_sequential_64_layer_call_fn_4219093
/__inference_sequential_64_layer_call_fn_4219184À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219148
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219063
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219166
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219073À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_dense_210_layer_call_fn_4219202¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_210_layer_call_and_return_conditional_losses_4219195¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_211_layer_call_fn_4219220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_211_layer_call_and_return_conditional_losses_4219213¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
4B2
%__inference_signature_wrapper_4219130input_1
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
"__inference__wrapped_model_4219012n	
1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
¨
F__inference_dense_210_layer_call_and_return_conditional_losses_4219195^	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_210_layer_call_fn_4219202Q	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_dense_211_layer_call_and_return_conditional_losses_4219213]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_dense_211_layer_call_fn_4219220P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¶
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219063h	
9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¶
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219073h	
9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219148g	
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
J__inference_sequential_64_layer_call_and_return_conditional_losses_4219166g	
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
/__inference_sequential_64_layer_call_fn_4219093[	
9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

/__inference_sequential_64_layer_call_fn_4219112[	
9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

/__inference_sequential_64_layer_call_fn_4219175Z	
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

/__inference_sequential_64_layer_call_fn_4219184Z	
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¢
%__inference_signature_wrapper_4219130y	
<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
