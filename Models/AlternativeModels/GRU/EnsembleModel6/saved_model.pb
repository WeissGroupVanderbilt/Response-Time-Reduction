��/
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softplus
features"T
activations"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.9.12v2.9.0-18-gd8ce9f9c3018��-
�
Adam/gru_17/gru_cell_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_17/gru_cell_32/bias/v
�
2Adam/gru_17/gru_cell_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_17/gru_cell_32/bias/v*
_output_shapes

:*
dtype0
�
*Adam/gru_17/gru_cell_32/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_17/gru_cell_32/recurrent_kernel/v
�
>Adam/gru_17/gru_cell_32/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_17/gru_cell_32/recurrent_kernel/v*
_output_shapes

:*
dtype0
�
 Adam/gru_17/gru_cell_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_17/gru_cell_32/kernel/v
�
4Adam/gru_17/gru_cell_32/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_17/gru_cell_32/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/gru_16/gru_cell_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_16/gru_cell_31/bias/v
�
2Adam/gru_16/gru_cell_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_16/gru_cell_31/bias/v*
_output_shapes
:	�*
dtype0
�
*Adam/gru_16/gru_cell_31/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*;
shared_name,*Adam/gru_16/gru_cell_31/recurrent_kernel/v
�
>Adam/gru_16/gru_cell_31/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_16/gru_cell_31/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
 Adam/gru_16/gru_cell_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" Adam/gru_16/gru_cell_31/kernel/v
�
4Adam/gru_16/gru_cell_31/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_16/gru_cell_31/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/gru_15/gru_cell_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_15/gru_cell_30/bias/v
�
2Adam/gru_15/gru_cell_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_15/gru_cell_30/bias/v*
_output_shapes
:	�*
dtype0
�
*Adam/gru_15/gru_cell_30/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/gru_15/gru_cell_30/recurrent_kernel/v
�
>Adam/gru_15/gru_cell_30/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_15/gru_cell_30/recurrent_kernel/v* 
_output_shapes
:
��*
dtype0
�
 Adam/gru_15/gru_cell_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_15/gru_cell_30/kernel/v
�
4Adam/gru_15/gru_cell_30/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_15/gru_cell_30/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/gru_17/gru_cell_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_17/gru_cell_32/bias/m
�
2Adam/gru_17/gru_cell_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_17/gru_cell_32/bias/m*
_output_shapes

:*
dtype0
�
*Adam/gru_17/gru_cell_32/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_17/gru_cell_32/recurrent_kernel/m
�
>Adam/gru_17/gru_cell_32/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_17/gru_cell_32/recurrent_kernel/m*
_output_shapes

:*
dtype0
�
 Adam/gru_17/gru_cell_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_17/gru_cell_32/kernel/m
�
4Adam/gru_17/gru_cell_32/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_17/gru_cell_32/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/gru_16/gru_cell_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_16/gru_cell_31/bias/m
�
2Adam/gru_16/gru_cell_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_16/gru_cell_31/bias/m*
_output_shapes
:	�*
dtype0
�
*Adam/gru_16/gru_cell_31/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*;
shared_name,*Adam/gru_16/gru_cell_31/recurrent_kernel/m
�
>Adam/gru_16/gru_cell_31/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_16/gru_cell_31/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
 Adam/gru_16/gru_cell_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" Adam/gru_16/gru_cell_31/kernel/m
�
4Adam/gru_16/gru_cell_31/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_16/gru_cell_31/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/gru_15/gru_cell_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_15/gru_cell_30/bias/m
�
2Adam/gru_15/gru_cell_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_15/gru_cell_30/bias/m*
_output_shapes
:	�*
dtype0
�
*Adam/gru_15/gru_cell_30/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/gru_15/gru_cell_30/recurrent_kernel/m
�
>Adam/gru_15/gru_cell_30/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_15/gru_cell_30/recurrent_kernel/m* 
_output_shapes
:
��*
dtype0
�
 Adam/gru_15/gru_cell_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_15/gru_cell_30/kernel/m
�
4Adam/gru_15/gru_cell_30/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_15/gru_cell_30/kernel/m*
_output_shapes
:	�*
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
�
gru_17/gru_cell_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namegru_17/gru_cell_32/bias
�
+gru_17/gru_cell_32/bias/Read/ReadVariableOpReadVariableOpgru_17/gru_cell_32/bias*
_output_shapes

:*
dtype0
�
#gru_17/gru_cell_32/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#gru_17/gru_cell_32/recurrent_kernel
�
7gru_17/gru_cell_32/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_17/gru_cell_32/recurrent_kernel*
_output_shapes

:*
dtype0
�
gru_17/gru_cell_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_namegru_17/gru_cell_32/kernel
�
-gru_17/gru_cell_32/kernel/Read/ReadVariableOpReadVariableOpgru_17/gru_cell_32/kernel*
_output_shapes

:d*
dtype0
�
gru_16/gru_cell_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_16/gru_cell_31/bias
�
+gru_16/gru_cell_31/bias/Read/ReadVariableOpReadVariableOpgru_16/gru_cell_31/bias*
_output_shapes
:	�*
dtype0
�
#gru_16/gru_cell_31/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*4
shared_name%#gru_16/gru_cell_31/recurrent_kernel
�
7gru_16/gru_cell_31/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_16/gru_cell_31/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
gru_16/gru_cell_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_namegru_16/gru_cell_31/kernel
�
-gru_16/gru_cell_31/kernel/Read/ReadVariableOpReadVariableOpgru_16/gru_cell_31/kernel* 
_output_shapes
:
��*
dtype0
�
gru_15/gru_cell_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_15/gru_cell_30/bias
�
+gru_15/gru_cell_30/bias/Read/ReadVariableOpReadVariableOpgru_15/gru_cell_30/bias*
_output_shapes
:	�*
dtype0
�
#gru_15/gru_cell_30/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#gru_15/gru_cell_30/recurrent_kernel
�
7gru_15/gru_cell_30/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_15/gru_cell_30/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
gru_15/gru_cell_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_namegru_15/gru_cell_30/kernel
�
-gru_15/gru_cell_30/kernel/Read/ReadVariableOpReadVariableOpgru_15/gru_cell_30/kernel*
_output_shapes
:	�*
dtype0

NoOpNoOp
�I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_random_generator
&cell
'
state_spec*
C
(0
)1
*2
+3
,4
-5
.6
/7
08*
C
(0
)1
*2
+3
,4
-5
.6
/7
08*
* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
6trace_0
7trace_1
8trace_2
9trace_3* 
6
:trace_0
;trace_1
<trace_2
=trace_3* 
* 
�
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate(m�)m�*m�+m�,m�-m�.m�/m�0m�(v�)v�*v�+v�,v�-v�.v�/v�0v�*

Cserving_default* 

(0
)1
*2*

(0
)1
*2*
* 
�

Dstates
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
6
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_3* 
* 
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
X_random_generator

(kernel
)recurrent_kernel
*bias*
* 

+0
,1
-2*

+0
,1
-2*
* 
�

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
_trace_0
`trace_1
atrace_2
btrace_3* 
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
* 
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator

+kernel
,recurrent_kernel
-bias*
* 

.0
/1
02*

.0
/1
02*
* 
�

nstates
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 
6
xtrace_0
ytrace_1
ztrace_2
{trace_3* 
* 
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

.kernel
/recurrent_kernel
0bias*
* 
YS
VARIABLE_VALUEgru_15/gru_cell_30/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_15/gru_cell_30/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_15/gru_cell_30/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_16/gru_cell_31/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_16/gru_cell_31/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_16/gru_cell_31/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_17/gru_cell_32/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_17/gru_cell_32/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_17/gru_cell_32/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1
*2*

(0
)1
*2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1
-2*

+0
,1
-2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 

&0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1
02*

.0
/1
02*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_15/gru_cell_30/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_15/gru_cell_30/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_15/gru_cell_30/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_16/gru_cell_31/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_16/gru_cell_31/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_16/gru_cell_31/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_17/gru_cell_32/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_17/gru_cell_32/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_17/gru_cell_32/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_15/gru_cell_30/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_15/gru_cell_30/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_15/gru_cell_30/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_16/gru_cell_31/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_16/gru_cell_31/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_16/gru_cell_31/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_17/gru_cell_32/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_17/gru_cell_32/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_17/gru_cell_32/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_gru_15_inputPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_15_inputgru_15/gru_cell_30/biasgru_15/gru_cell_30/kernel#gru_15/gru_cell_30/recurrent_kernelgru_16/gru_cell_31/biasgru_16/gru_cell_31/kernel#gru_16/gru_cell_31/recurrent_kernelgru_17/gru_cell_32/biasgru_17/gru_cell_32/kernel#gru_17/gru_cell_32/recurrent_kernel*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_2905470
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-gru_15/gru_cell_30/kernel/Read/ReadVariableOp7gru_15/gru_cell_30/recurrent_kernel/Read/ReadVariableOp+gru_15/gru_cell_30/bias/Read/ReadVariableOp-gru_16/gru_cell_31/kernel/Read/ReadVariableOp7gru_16/gru_cell_31/recurrent_kernel/Read/ReadVariableOp+gru_16/gru_cell_31/bias/Read/ReadVariableOp-gru_17/gru_cell_32/kernel/Read/ReadVariableOp7gru_17/gru_cell_32/recurrent_kernel/Read/ReadVariableOp+gru_17/gru_cell_32/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/gru_15/gru_cell_30/kernel/m/Read/ReadVariableOp>Adam/gru_15/gru_cell_30/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_15/gru_cell_30/bias/m/Read/ReadVariableOp4Adam/gru_16/gru_cell_31/kernel/m/Read/ReadVariableOp>Adam/gru_16/gru_cell_31/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_16/gru_cell_31/bias/m/Read/ReadVariableOp4Adam/gru_17/gru_cell_32/kernel/m/Read/ReadVariableOp>Adam/gru_17/gru_cell_32/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_17/gru_cell_32/bias/m/Read/ReadVariableOp4Adam/gru_15/gru_cell_30/kernel/v/Read/ReadVariableOp>Adam/gru_15/gru_cell_30/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_15/gru_cell_30/bias/v/Read/ReadVariableOp4Adam/gru_16/gru_cell_31/kernel/v/Read/ReadVariableOp>Adam/gru_16/gru_cell_31/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_16/gru_cell_31/bias/v/Read/ReadVariableOp4Adam/gru_17/gru_cell_32/kernel/v/Read/ReadVariableOp>Adam/gru_17/gru_cell_32/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_17/gru_cell_32/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_2908829
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegru_15/gru_cell_30/kernel#gru_15/gru_cell_30/recurrent_kernelgru_15/gru_cell_30/biasgru_16/gru_cell_31/kernel#gru_16/gru_cell_31/recurrent_kernelgru_16/gru_cell_31/biasgru_17/gru_cell_32/kernel#gru_17/gru_cell_32/recurrent_kernelgru_17/gru_cell_32/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/gru_15/gru_cell_30/kernel/m*Adam/gru_15/gru_cell_30/recurrent_kernel/mAdam/gru_15/gru_cell_30/bias/m Adam/gru_16/gru_cell_31/kernel/m*Adam/gru_16/gru_cell_31/recurrent_kernel/mAdam/gru_16/gru_cell_31/bias/m Adam/gru_17/gru_cell_32/kernel/m*Adam/gru_17/gru_cell_32/recurrent_kernel/mAdam/gru_17/gru_cell_32/bias/m Adam/gru_15/gru_cell_30/kernel/v*Adam/gru_15/gru_cell_30/recurrent_kernel/vAdam/gru_15/gru_cell_30/bias/v Adam/gru_16/gru_cell_31/kernel/v*Adam/gru_16/gru_cell_31/recurrent_kernel/vAdam/gru_16/gru_cell_31/bias/v Adam/gru_17/gru_cell_32/kernel/v*Adam/gru_17/gru_cell_32/recurrent_kernel/vAdam/gru_17/gru_cell_32/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_2908941��+
�
�
while_cond_2905021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2905021___redundant_placeholder05
1while_while_cond_2905021___redundant_placeholder15
1while_while_cond_2905021___redundant_placeholder25
1while_while_cond_2905021___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2907487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907487___redundant_placeholder05
1while_while_cond_2907487___redundant_placeholder15
1while_while_cond_2907487___redundant_placeholder25
1while_while_cond_2907487___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�V
�
&sequential_5_gru_17_while_body_2903158D
@sequential_5_gru_17_while_sequential_5_gru_17_while_loop_counterJ
Fsequential_5_gru_17_while_sequential_5_gru_17_while_maximum_iterations)
%sequential_5_gru_17_while_placeholder+
'sequential_5_gru_17_while_placeholder_1+
'sequential_5_gru_17_while_placeholder_2C
?sequential_5_gru_17_while_sequential_5_gru_17_strided_slice_1_0
{sequential_5_gru_17_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_17_tensorarrayunstack_tensorlistfromtensor_0Q
?sequential_5_gru_17_while_gru_cell_32_readvariableop_resource_0:X
Fsequential_5_gru_17_while_gru_cell_32_matmul_readvariableop_resource_0:dZ
Hsequential_5_gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0:&
"sequential_5_gru_17_while_identity(
$sequential_5_gru_17_while_identity_1(
$sequential_5_gru_17_while_identity_2(
$sequential_5_gru_17_while_identity_3(
$sequential_5_gru_17_while_identity_4A
=sequential_5_gru_17_while_sequential_5_gru_17_strided_slice_1}
ysequential_5_gru_17_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_17_tensorarrayunstack_tensorlistfromtensorO
=sequential_5_gru_17_while_gru_cell_32_readvariableop_resource:V
Dsequential_5_gru_17_while_gru_cell_32_matmul_readvariableop_resource:dX
Fsequential_5_gru_17_while_gru_cell_32_matmul_1_readvariableop_resource:��;sequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOp�=sequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp�4sequential_5/gru_17/while/gru_cell_32/ReadVariableOp�
Ksequential_5/gru_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
=sequential_5/gru_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_gru_17_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_17_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_gru_17_while_placeholderTsequential_5/gru_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
4sequential_5/gru_17/while/gru_cell_32/ReadVariableOpReadVariableOp?sequential_5_gru_17_while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
-sequential_5/gru_17/while/gru_cell_32/unstackUnpack<sequential_5/gru_17/while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
;sequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOpReadVariableOpFsequential_5_gru_17_while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
,sequential_5/gru_17/while/gru_cell_32/MatMulMatMulDsequential_5/gru_17/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_5/gru_17/while/gru_cell_32/BiasAddBiasAdd6sequential_5/gru_17/while/gru_cell_32/MatMul:product:06sequential_5/gru_17/while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:����������
5sequential_5/gru_17/while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_5/gru_17/while/gru_cell_32/splitSplit>sequential_5/gru_17/while/gru_cell_32/split/split_dim:output:06sequential_5/gru_17/while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
=sequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOpHsequential_5_gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
.sequential_5/gru_17/while/gru_cell_32/MatMul_1MatMul'sequential_5_gru_17_while_placeholder_2Esequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_5/gru_17/while/gru_cell_32/BiasAdd_1BiasAdd8sequential_5/gru_17/while/gru_cell_32/MatMul_1:product:06sequential_5/gru_17/while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      �����
7sequential_5/gru_17/while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_5/gru_17/while/gru_cell_32/split_1SplitV8sequential_5/gru_17/while/gru_cell_32/BiasAdd_1:output:04sequential_5/gru_17/while/gru_cell_32/Const:output:0@sequential_5/gru_17/while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)sequential_5/gru_17/while/gru_cell_32/addAddV24sequential_5/gru_17/while/gru_cell_32/split:output:06sequential_5/gru_17/while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:����������
-sequential_5/gru_17/while/gru_cell_32/SigmoidSigmoid-sequential_5/gru_17/while/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/add_1AddV24sequential_5/gru_17/while/gru_cell_32/split:output:16sequential_5/gru_17/while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:����������
/sequential_5/gru_17/while/gru_cell_32/Sigmoid_1Sigmoid/sequential_5/gru_17/while/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
)sequential_5/gru_17/while/gru_cell_32/mulMul3sequential_5/gru_17/while/gru_cell_32/Sigmoid_1:y:06sequential_5/gru_17/while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/add_2AddV24sequential_5/gru_17/while/gru_cell_32/split:output:2-sequential_5/gru_17/while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:����������
.sequential_5/gru_17/while/gru_cell_32/SoftplusSoftplus/sequential_5/gru_17/while/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/mul_1Mul1sequential_5/gru_17/while/gru_cell_32/Sigmoid:y:0'sequential_5_gru_17_while_placeholder_2*
T0*'
_output_shapes
:���������p
+sequential_5/gru_17/while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_5/gru_17/while/gru_cell_32/subSub4sequential_5/gru_17/while/gru_cell_32/sub/x:output:01sequential_5/gru_17/while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/mul_2Mul-sequential_5/gru_17/while/gru_cell_32/sub:z:0<sequential_5/gru_17/while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
+sequential_5/gru_17/while/gru_cell_32/add_3AddV2/sequential_5/gru_17/while/gru_cell_32/mul_1:z:0/sequential_5/gru_17/while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
>sequential_5/gru_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_gru_17_while_placeholder_1%sequential_5_gru_17_while_placeholder/sequential_5/gru_17/while/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_5/gru_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_17/while/addAddV2%sequential_5_gru_17_while_placeholder(sequential_5/gru_17/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_5/gru_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_17/while/add_1AddV2@sequential_5_gru_17_while_sequential_5_gru_17_while_loop_counter*sequential_5/gru_17/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_5/gru_17/while/IdentityIdentity#sequential_5/gru_17/while/add_1:z:0^sequential_5/gru_17/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_17/while/Identity_1IdentityFsequential_5_gru_17_while_sequential_5_gru_17_while_maximum_iterations^sequential_5/gru_17/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_17/while/Identity_2Identity!sequential_5/gru_17/while/add:z:0^sequential_5/gru_17/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_17/while/Identity_3IdentityNsequential_5/gru_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_5/gru_17/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_17/while/Identity_4Identity/sequential_5/gru_17/while/gru_cell_32/add_3:z:0^sequential_5/gru_17/while/NoOp*
T0*'
_output_shapes
:����������
sequential_5/gru_17/while/NoOpNoOp<^sequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOp>^sequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp5^sequential_5/gru_17/while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_5_gru_17_while_gru_cell_32_matmul_1_readvariableop_resourceHsequential_5_gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0"�
Dsequential_5_gru_17_while_gru_cell_32_matmul_readvariableop_resourceFsequential_5_gru_17_while_gru_cell_32_matmul_readvariableop_resource_0"�
=sequential_5_gru_17_while_gru_cell_32_readvariableop_resource?sequential_5_gru_17_while_gru_cell_32_readvariableop_resource_0"Q
"sequential_5_gru_17_while_identity+sequential_5/gru_17/while/Identity:output:0"U
$sequential_5_gru_17_while_identity_1-sequential_5/gru_17/while/Identity_1:output:0"U
$sequential_5_gru_17_while_identity_2-sequential_5/gru_17/while/Identity_2:output:0"U
$sequential_5_gru_17_while_identity_3-sequential_5/gru_17/while/Identity_3:output:0"U
$sequential_5_gru_17_while_identity_4-sequential_5/gru_17/while/Identity_4:output:0"�
=sequential_5_gru_17_while_sequential_5_gru_17_strided_slice_1?sequential_5_gru_17_while_sequential_5_gru_17_strided_slice_1_0"�
ysequential_5_gru_17_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_17_tensorarrayunstack_tensorlistfromtensor{sequential_5_gru_17_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2z
;sequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOp;sequential_5/gru_17/while/gru_cell_32/MatMul/ReadVariableOp2~
=sequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp=sequential_5/gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp2l
4sequential_5/gru_17/while/gru_cell_32/ReadVariableOp4sequential_5/gru_17/while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�=
�
while_body_2907335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�

�
-__inference_gru_cell_31_layer_call_fn_2908506

inputs
states_0
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903655o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2907074

inputs6
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2906985*
condR
while_cond_2906984*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&sequential_5_gru_17_while_cond_2903157D
@sequential_5_gru_17_while_sequential_5_gru_17_while_loop_counterJ
Fsequential_5_gru_17_while_sequential_5_gru_17_while_maximum_iterations)
%sequential_5_gru_17_while_placeholder+
'sequential_5_gru_17_while_placeholder_1+
'sequential_5_gru_17_while_placeholder_2F
Bsequential_5_gru_17_while_less_sequential_5_gru_17_strided_slice_1]
Ysequential_5_gru_17_while_sequential_5_gru_17_while_cond_2903157___redundant_placeholder0]
Ysequential_5_gru_17_while_sequential_5_gru_17_while_cond_2903157___redundant_placeholder1]
Ysequential_5_gru_17_while_sequential_5_gru_17_while_cond_2903157___redundant_placeholder2]
Ysequential_5_gru_17_while_sequential_5_gru_17_while_cond_2903157___redundant_placeholder3&
"sequential_5_gru_17_while_identity
�
sequential_5/gru_17/while/LessLess%sequential_5_gru_17_while_placeholderBsequential_5_gru_17_while_less_sequential_5_gru_17_strided_slice_1*
T0*
_output_shapes
: s
"sequential_5/gru_17/while/IdentityIdentity"sequential_5/gru_17/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_5_gru_17_while_identity+sequential_5/gru_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2905196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2905196___redundant_placeholder05
1while_while_cond_2905196___redundant_placeholder15
1while_while_cond_2905196___redundant_placeholder25
1while_while_cond_2905196___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�E
�	
gru_16_while_body_2905729*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0E
2gru_16_while_gru_cell_31_readvariableop_resource_0:	�M
9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0:
��N
;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensorC
0gru_16_while_gru_cell_31_readvariableop_resource:	�K
7gru_16_while_gru_cell_31_matmul_readvariableop_resource:
��L
9gru_16_while_gru_cell_31_matmul_1_readvariableop_resource:	d���.gru_16/while/gru_cell_31/MatMul/ReadVariableOp�0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp�'gru_16/while/gru_cell_31/ReadVariableOp�
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
'gru_16/while/gru_cell_31/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_16/while/gru_cell_31/unstackUnpack/gru_16/while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_16/while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
gru_16/while/gru_cell_31/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_16/while/gru_cell_31/BiasAddBiasAdd)gru_16/while/gru_cell_31/MatMul:product:0)gru_16/while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_16/while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/while/gru_cell_31/splitSplit1gru_16/while/gru_cell_31/split/split_dim:output:0)gru_16/while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
!gru_16/while/gru_cell_31/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_16/while/gru_cell_31/BiasAdd_1BiasAdd+gru_16/while/gru_cell_31/MatMul_1:product:0)gru_16/while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_16/while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����u
*gru_16/while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_16/while/gru_cell_31/split_1SplitV+gru_16/while/gru_cell_31/BiasAdd_1:output:0'gru_16/while/gru_cell_31/Const:output:03gru_16/while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_16/while/gru_cell_31/addAddV2'gru_16/while/gru_cell_31/split:output:0)gru_16/while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������d
 gru_16/while/gru_cell_31/SigmoidSigmoid gru_16/while/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_1AddV2'gru_16/while/gru_cell_31/split:output:1)gru_16/while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������d�
"gru_16/while/gru_cell_31/Sigmoid_1Sigmoid"gru_16/while/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mulMul&gru_16/while/gru_cell_31/Sigmoid_1:y:0)gru_16/while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_2AddV2'gru_16/while/gru_cell_31/split:output:2 gru_16/while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������d�
"gru_16/while/gru_cell_31/Sigmoid_2Sigmoid"gru_16/while/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mul_1Mul$gru_16/while/gru_cell_31/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:���������dc
gru_16/while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_16/while/gru_cell_31/subSub'gru_16/while/gru_cell_31/sub/x:output:0$gru_16/while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mul_2Mul gru_16/while/gru_cell_31/sub:z:0&gru_16/while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_3AddV2"gru_16/while/gru_cell_31/mul_1:z:0"gru_16/while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: V
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations^gru_16/while/NoOp*
T0*
_output_shapes
: n
gru_16/while/Identity_2Identitygru_16/while/add:z:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_31/add_3:z:0^gru_16/while/NoOp*
T0*'
_output_shapes
:���������d�
gru_16/while/NoOpNoOp/^gru_16/while/gru_cell_31/MatMul/ReadVariableOp1^gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp(^gru_16/while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_31_matmul_1_readvariableop_resource;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_31_matmul_readvariableop_resource9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_31_readvariableop_resource2gru_16_while_gru_cell_31_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"�
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.gru_16/while/gru_cell_31/MatMul/ReadVariableOp.gru_16/while/gru_cell_31/MatMul/ReadVariableOp2d
0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp2R
'gru_16/while/gru_cell_31/ReadVariableOp'gru_16/while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2904580

inputs6
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:�����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904491*
condR
while_cond_2904490*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:����������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2907927
inputs_05
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907838*
condR
while_cond_2907837*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2904936

inputs5
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904847*
condR
while_cond_2904846*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�

�
-__inference_gru_cell_32_layer_call_fn_2908612

inputs
states_0
unknown:
	unknown_0:d
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2903993o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�
�
while_cond_2907837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907837___redundant_placeholder05
1while_while_cond_2907837___redundant_placeholder15
1while_while_cond_2907837___redundant_placeholder25
1while_while_cond_2907837___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2907640
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907640___redundant_placeholder05
1while_while_cond_2907640___redundant_placeholder15
1while_while_cond_2907640___redundant_placeholder25
1while_while_cond_2907640___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2907334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907334___redundant_placeholder05
1while_while_cond_2907334___redundant_placeholder15
1while_while_cond_2907334___redundant_placeholder25
1while_while_cond_2907334___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
��
�
#__inference__traced_restore_2908941
file_prefix=
*assignvariableop_gru_15_gru_cell_30_kernel:	�J
6assignvariableop_1_gru_15_gru_cell_30_recurrent_kernel:
��=
*assignvariableop_2_gru_15_gru_cell_30_bias:	�@
,assignvariableop_3_gru_16_gru_cell_31_kernel:
��I
6assignvariableop_4_gru_16_gru_cell_31_recurrent_kernel:	d�=
*assignvariableop_5_gru_16_gru_cell_31_bias:	�>
,assignvariableop_6_gru_17_gru_cell_32_kernel:dH
6assignvariableop_7_gru_17_gru_cell_32_recurrent_kernel:<
*assignvariableop_8_gru_17_gru_cell_32_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: G
4assignvariableop_16_adam_gru_15_gru_cell_30_kernel_m:	�R
>assignvariableop_17_adam_gru_15_gru_cell_30_recurrent_kernel_m:
��E
2assignvariableop_18_adam_gru_15_gru_cell_30_bias_m:	�H
4assignvariableop_19_adam_gru_16_gru_cell_31_kernel_m:
��Q
>assignvariableop_20_adam_gru_16_gru_cell_31_recurrent_kernel_m:	d�E
2assignvariableop_21_adam_gru_16_gru_cell_31_bias_m:	�F
4assignvariableop_22_adam_gru_17_gru_cell_32_kernel_m:dP
>assignvariableop_23_adam_gru_17_gru_cell_32_recurrent_kernel_m:D
2assignvariableop_24_adam_gru_17_gru_cell_32_bias_m:G
4assignvariableop_25_adam_gru_15_gru_cell_30_kernel_v:	�R
>assignvariableop_26_adam_gru_15_gru_cell_30_recurrent_kernel_v:
��E
2assignvariableop_27_adam_gru_15_gru_cell_30_bias_v:	�H
4assignvariableop_28_adam_gru_16_gru_cell_31_kernel_v:
��Q
>assignvariableop_29_adam_gru_16_gru_cell_31_recurrent_kernel_v:	d�E
2assignvariableop_30_adam_gru_16_gru_cell_31_bias_v:	�F
4assignvariableop_31_adam_gru_17_gru_cell_32_kernel_v:dP
>assignvariableop_32_adam_gru_17_gru_cell_32_recurrent_kernel_v:D
2assignvariableop_33_adam_gru_17_gru_cell_32_bias_v:
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_gru_15_gru_cell_30_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp6assignvariableop_1_gru_15_gru_cell_30_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_gru_15_gru_cell_30_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_gru_16_gru_cell_31_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_gru_16_gru_cell_31_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_16_gru_cell_31_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_gru_17_gru_cell_32_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_gru_17_gru_cell_32_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_gru_17_gru_cell_32_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_gru_15_gru_cell_30_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_gru_15_gru_cell_30_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_gru_15_gru_cell_30_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_gru_16_gru_cell_31_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_gru_16_gru_cell_31_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_gru_16_gru_cell_31_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_gru_17_gru_cell_32_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_gru_17_gru_cell_32_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_gru_17_gru_cell_32_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_gru_15_gru_cell_30_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_gru_15_gru_cell_30_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_gru_15_gru_cell_30_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_gru_16_gru_cell_31_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_gru_16_gru_cell_31_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_gru_16_gru_cell_31_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_gru_17_gru_cell_32_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_gru_17_gru_cell_32_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_gru_17_gru_cell_32_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�

"__inference__wrapped_model_2903247
gru_15_inputJ
7sequential_5_gru_15_gru_cell_30_readvariableop_resource:	�Q
>sequential_5_gru_15_gru_cell_30_matmul_readvariableop_resource:	�T
@sequential_5_gru_15_gru_cell_30_matmul_1_readvariableop_resource:
��J
7sequential_5_gru_16_gru_cell_31_readvariableop_resource:	�R
>sequential_5_gru_16_gru_cell_31_matmul_readvariableop_resource:
��S
@sequential_5_gru_16_gru_cell_31_matmul_1_readvariableop_resource:	d�I
7sequential_5_gru_17_gru_cell_32_readvariableop_resource:P
>sequential_5_gru_17_gru_cell_32_matmul_readvariableop_resource:dR
@sequential_5_gru_17_gru_cell_32_matmul_1_readvariableop_resource:
identity��5sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOp�7sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOp�.sequential_5/gru_15/gru_cell_30/ReadVariableOp�sequential_5/gru_15/while�5sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOp�7sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOp�.sequential_5/gru_16/gru_cell_31/ReadVariableOp�sequential_5/gru_16/while�5sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOp�7sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOp�.sequential_5/gru_17/gru_cell_32/ReadVariableOp�sequential_5/gru_17/whileU
sequential_5/gru_15/ShapeShapegru_15_input*
T0*
_output_shapes
:q
'sequential_5/gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_5/gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_5/gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_5/gru_15/strided_sliceStridedSlice"sequential_5/gru_15/Shape:output:00sequential_5/gru_15/strided_slice/stack:output:02sequential_5/gru_15/strided_slice/stack_1:output:02sequential_5/gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_5/gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
 sequential_5/gru_15/zeros/packedPack*sequential_5/gru_15/strided_slice:output:0+sequential_5/gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_5/gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_5/gru_15/zerosFill)sequential_5/gru_15/zeros/packed:output:0(sequential_5/gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:����������w
"sequential_5/gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_15/transpose	Transposegru_15_input+sequential_5/gru_15/transpose/perm:output:0*
T0*,
_output_shapes
:����������l
sequential_5/gru_15/Shape_1Shape!sequential_5/gru_15/transpose:y:0*
T0*
_output_shapes
:s
)sequential_5/gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_15/strided_slice_1StridedSlice$sequential_5/gru_15/Shape_1:output:02sequential_5/gru_15/strided_slice_1/stack:output:04sequential_5/gru_15/strided_slice_1/stack_1:output:04sequential_5/gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_5/gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_5/gru_15/TensorArrayV2TensorListReserve8sequential_5/gru_15/TensorArrayV2/element_shape:output:0,sequential_5/gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_5/gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
;sequential_5/gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/gru_15/transpose:y:0Rsequential_5/gru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_5/gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_15/strided_slice_2StridedSlice!sequential_5/gru_15/transpose:y:02sequential_5/gru_15/strided_slice_2/stack:output:04sequential_5/gru_15/strided_slice_2/stack_1:output:04sequential_5/gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
.sequential_5/gru_15/gru_cell_30/ReadVariableOpReadVariableOp7sequential_5_gru_15_gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'sequential_5/gru_15/gru_cell_30/unstackUnpack6sequential_5/gru_15/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
5sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOpReadVariableOp>sequential_5_gru_15_gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
&sequential_5/gru_15/gru_cell_30/MatMulMatMul,sequential_5/gru_15/strided_slice_2:output:0=sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_5/gru_15/gru_cell_30/BiasAddBiasAdd0sequential_5/gru_15/gru_cell_30/MatMul:product:00sequential_5/gru_15/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������z
/sequential_5/gru_15/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_5/gru_15/gru_cell_30/splitSplit8sequential_5/gru_15/gru_cell_30/split/split_dim:output:00sequential_5/gru_15/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
7sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp@sequential_5_gru_15_gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
(sequential_5/gru_15/gru_cell_30/MatMul_1MatMul"sequential_5/gru_15/zeros:output:0?sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)sequential_5/gru_15/gru_cell_30/BiasAdd_1BiasAdd2sequential_5/gru_15/gru_cell_30/MatMul_1:product:00sequential_5/gru_15/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������z
%sequential_5/gru_15/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����|
1sequential_5/gru_15/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_5/gru_15/gru_cell_30/split_1SplitV2sequential_5/gru_15/gru_cell_30/BiasAdd_1:output:0.sequential_5/gru_15/gru_cell_30/Const:output:0:sequential_5/gru_15/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#sequential_5/gru_15/gru_cell_30/addAddV2.sequential_5/gru_15/gru_cell_30/split:output:00sequential_5/gru_15/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:�����������
'sequential_5/gru_15/gru_cell_30/SigmoidSigmoid'sequential_5/gru_15/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
%sequential_5/gru_15/gru_cell_30/add_1AddV2.sequential_5/gru_15/gru_cell_30/split:output:10sequential_5/gru_15/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:�����������
)sequential_5/gru_15/gru_cell_30/Sigmoid_1Sigmoid)sequential_5/gru_15/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
#sequential_5/gru_15/gru_cell_30/mulMul-sequential_5/gru_15/gru_cell_30/Sigmoid_1:y:00sequential_5/gru_15/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
%sequential_5/gru_15/gru_cell_30/add_2AddV2.sequential_5/gru_15/gru_cell_30/split:output:2'sequential_5/gru_15/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:�����������
)sequential_5/gru_15/gru_cell_30/Sigmoid_2Sigmoid)sequential_5/gru_15/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
%sequential_5/gru_15/gru_cell_30/mul_1Mul+sequential_5/gru_15/gru_cell_30/Sigmoid:y:0"sequential_5/gru_15/zeros:output:0*
T0*(
_output_shapes
:����������j
%sequential_5/gru_15/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_5/gru_15/gru_cell_30/subSub.sequential_5/gru_15/gru_cell_30/sub/x:output:0+sequential_5/gru_15/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
%sequential_5/gru_15/gru_cell_30/mul_2Mul'sequential_5/gru_15/gru_cell_30/sub:z:0-sequential_5/gru_15/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
%sequential_5/gru_15/gru_cell_30/add_3AddV2)sequential_5/gru_15/gru_cell_30/mul_1:z:0)sequential_5/gru_15/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
1sequential_5/gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
#sequential_5/gru_15/TensorArrayV2_1TensorListReserve:sequential_5/gru_15/TensorArrayV2_1/element_shape:output:0,sequential_5/gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_5/gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_5/gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_5/gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_5/gru_15/whileWhile/sequential_5/gru_15/while/loop_counter:output:05sequential_5/gru_15/while/maximum_iterations:output:0!sequential_5/gru_15/time:output:0,sequential_5/gru_15/TensorArrayV2_1:handle:0"sequential_5/gru_15/zeros:output:0,sequential_5/gru_15/strided_slice_1:output:0Ksequential_5/gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_5_gru_15_gru_cell_30_readvariableop_resource>sequential_5_gru_15_gru_cell_30_matmul_readvariableop_resource@sequential_5_gru_15_gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&sequential_5_gru_15_while_body_2902860*2
cond*R(
&sequential_5_gru_15_while_cond_2902859*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
Dsequential_5/gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
6sequential_5/gru_15/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/gru_15/while:output:3Msequential_5/gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0|
)sequential_5/gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_5/gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_15/strided_slice_3StridedSlice?sequential_5/gru_15/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/gru_15/strided_slice_3/stack:output:04sequential_5/gru_15/strided_slice_3/stack_1:output:04sequential_5/gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_masky
$sequential_5/gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_15/transpose_1	Transpose?sequential_5/gru_15/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/gru_15/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������o
sequential_5/gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_5/gru_16/ShapeShape#sequential_5/gru_15/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_5/gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_5/gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_5/gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_5/gru_16/strided_sliceStridedSlice"sequential_5/gru_16/Shape:output:00sequential_5/gru_16/strided_slice/stack:output:02sequential_5/gru_16/strided_slice/stack_1:output:02sequential_5/gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_5/gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
 sequential_5/gru_16/zeros/packedPack*sequential_5/gru_16/strided_slice:output:0+sequential_5/gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_5/gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_5/gru_16/zerosFill)sequential_5/gru_16/zeros/packed:output:0(sequential_5/gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������dw
"sequential_5/gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_16/transpose	Transpose#sequential_5/gru_15/transpose_1:y:0+sequential_5/gru_16/transpose/perm:output:0*
T0*-
_output_shapes
:�����������l
sequential_5/gru_16/Shape_1Shape!sequential_5/gru_16/transpose:y:0*
T0*
_output_shapes
:s
)sequential_5/gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_16/strided_slice_1StridedSlice$sequential_5/gru_16/Shape_1:output:02sequential_5/gru_16/strided_slice_1/stack:output:04sequential_5/gru_16/strided_slice_1/stack_1:output:04sequential_5/gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_5/gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_5/gru_16/TensorArrayV2TensorListReserve8sequential_5/gru_16/TensorArrayV2/element_shape:output:0,sequential_5/gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_5/gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
;sequential_5/gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/gru_16/transpose:y:0Rsequential_5/gru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_5/gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_16/strided_slice_2StridedSlice!sequential_5/gru_16/transpose:y:02sequential_5/gru_16/strided_slice_2/stack:output:04sequential_5/gru_16/strided_slice_2/stack_1:output:04sequential_5/gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
.sequential_5/gru_16/gru_cell_31/ReadVariableOpReadVariableOp7sequential_5_gru_16_gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'sequential_5/gru_16/gru_cell_31/unstackUnpack6sequential_5/gru_16/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
5sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOpReadVariableOp>sequential_5_gru_16_gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
&sequential_5/gru_16/gru_cell_31/MatMulMatMul,sequential_5/gru_16/strided_slice_2:output:0=sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_5/gru_16/gru_cell_31/BiasAddBiasAdd0sequential_5/gru_16/gru_cell_31/MatMul:product:00sequential_5/gru_16/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������z
/sequential_5/gru_16/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_5/gru_16/gru_cell_31/splitSplit8sequential_5/gru_16/gru_cell_31/split/split_dim:output:00sequential_5/gru_16/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
7sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp@sequential_5_gru_16_gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
(sequential_5/gru_16/gru_cell_31/MatMul_1MatMul"sequential_5/gru_16/zeros:output:0?sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)sequential_5/gru_16/gru_cell_31/BiasAdd_1BiasAdd2sequential_5/gru_16/gru_cell_31/MatMul_1:product:00sequential_5/gru_16/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������z
%sequential_5/gru_16/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����|
1sequential_5/gru_16/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_5/gru_16/gru_cell_31/split_1SplitV2sequential_5/gru_16/gru_cell_31/BiasAdd_1:output:0.sequential_5/gru_16/gru_cell_31/Const:output:0:sequential_5/gru_16/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#sequential_5/gru_16/gru_cell_31/addAddV2.sequential_5/gru_16/gru_cell_31/split:output:00sequential_5/gru_16/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������d�
'sequential_5/gru_16/gru_cell_31/SigmoidSigmoid'sequential_5/gru_16/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
%sequential_5/gru_16/gru_cell_31/add_1AddV2.sequential_5/gru_16/gru_cell_31/split:output:10sequential_5/gru_16/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������d�
)sequential_5/gru_16/gru_cell_31/Sigmoid_1Sigmoid)sequential_5/gru_16/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
#sequential_5/gru_16/gru_cell_31/mulMul-sequential_5/gru_16/gru_cell_31/Sigmoid_1:y:00sequential_5/gru_16/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
%sequential_5/gru_16/gru_cell_31/add_2AddV2.sequential_5/gru_16/gru_cell_31/split:output:2'sequential_5/gru_16/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������d�
)sequential_5/gru_16/gru_cell_31/Sigmoid_2Sigmoid)sequential_5/gru_16/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
%sequential_5/gru_16/gru_cell_31/mul_1Mul+sequential_5/gru_16/gru_cell_31/Sigmoid:y:0"sequential_5/gru_16/zeros:output:0*
T0*'
_output_shapes
:���������dj
%sequential_5/gru_16/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_5/gru_16/gru_cell_31/subSub.sequential_5/gru_16/gru_cell_31/sub/x:output:0+sequential_5/gru_16/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
%sequential_5/gru_16/gru_cell_31/mul_2Mul'sequential_5/gru_16/gru_cell_31/sub:z:0-sequential_5/gru_16/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
%sequential_5/gru_16/gru_cell_31/add_3AddV2)sequential_5/gru_16/gru_cell_31/mul_1:z:0)sequential_5/gru_16/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1sequential_5/gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
#sequential_5/gru_16/TensorArrayV2_1TensorListReserve:sequential_5/gru_16/TensorArrayV2_1/element_shape:output:0,sequential_5/gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_5/gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_5/gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_5/gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_5/gru_16/whileWhile/sequential_5/gru_16/while/loop_counter:output:05sequential_5/gru_16/while/maximum_iterations:output:0!sequential_5/gru_16/time:output:0,sequential_5/gru_16/TensorArrayV2_1:handle:0"sequential_5/gru_16/zeros:output:0,sequential_5/gru_16/strided_slice_1:output:0Ksequential_5/gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_5_gru_16_gru_cell_31_readvariableop_resource>sequential_5_gru_16_gru_cell_31_matmul_readvariableop_resource@sequential_5_gru_16_gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&sequential_5_gru_16_while_body_2903009*2
cond*R(
&sequential_5_gru_16_while_cond_2903008*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
Dsequential_5/gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
6sequential_5/gru_16/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/gru_16/while:output:3Msequential_5/gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0|
)sequential_5/gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_5/gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_16/strided_slice_3StridedSlice?sequential_5/gru_16/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/gru_16/strided_slice_3/stack:output:04sequential_5/gru_16/strided_slice_3/stack_1:output:04sequential_5/gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_masky
$sequential_5/gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_16/transpose_1	Transpose?sequential_5/gru_16/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/gru_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������do
sequential_5/gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_5/gru_17/ShapeShape#sequential_5/gru_16/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_5/gru_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_5/gru_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_5/gru_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_5/gru_17/strided_sliceStridedSlice"sequential_5/gru_17/Shape:output:00sequential_5/gru_17/strided_slice/stack:output:02sequential_5/gru_17/strided_slice/stack_1:output:02sequential_5/gru_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_5/gru_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
 sequential_5/gru_17/zeros/packedPack*sequential_5/gru_17/strided_slice:output:0+sequential_5/gru_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_5/gru_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_5/gru_17/zerosFill)sequential_5/gru_17/zeros/packed:output:0(sequential_5/gru_17/zeros/Const:output:0*
T0*'
_output_shapes
:���������w
"sequential_5/gru_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_17/transpose	Transpose#sequential_5/gru_16/transpose_1:y:0+sequential_5/gru_17/transpose/perm:output:0*
T0*,
_output_shapes
:����������dl
sequential_5/gru_17/Shape_1Shape!sequential_5/gru_17/transpose:y:0*
T0*
_output_shapes
:s
)sequential_5/gru_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_17/strided_slice_1StridedSlice$sequential_5/gru_17/Shape_1:output:02sequential_5/gru_17/strided_slice_1/stack:output:04sequential_5/gru_17/strided_slice_1/stack_1:output:04sequential_5/gru_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_5/gru_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_5/gru_17/TensorArrayV2TensorListReserve8sequential_5/gru_17/TensorArrayV2/element_shape:output:0,sequential_5/gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_5/gru_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
;sequential_5/gru_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/gru_17/transpose:y:0Rsequential_5/gru_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_5/gru_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/gru_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_17/strided_slice_2StridedSlice!sequential_5/gru_17/transpose:y:02sequential_5/gru_17/strided_slice_2/stack:output:04sequential_5/gru_17/strided_slice_2/stack_1:output:04sequential_5/gru_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
.sequential_5/gru_17/gru_cell_32/ReadVariableOpReadVariableOp7sequential_5_gru_17_gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0�
'sequential_5/gru_17/gru_cell_32/unstackUnpack6sequential_5/gru_17/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
5sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOpReadVariableOp>sequential_5_gru_17_gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
&sequential_5/gru_17/gru_cell_32/MatMulMatMul,sequential_5/gru_17/strided_slice_2:output:0=sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential_5/gru_17/gru_cell_32/BiasAddBiasAdd0sequential_5/gru_17/gru_cell_32/MatMul:product:00sequential_5/gru_17/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������z
/sequential_5/gru_17/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_5/gru_17/gru_cell_32/splitSplit8sequential_5/gru_17/gru_cell_32/split/split_dim:output:00sequential_5/gru_17/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
7sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp@sequential_5_gru_17_gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
(sequential_5/gru_17/gru_cell_32/MatMul_1MatMul"sequential_5/gru_17/zeros:output:0?sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential_5/gru_17/gru_cell_32/BiasAdd_1BiasAdd2sequential_5/gru_17/gru_cell_32/MatMul_1:product:00sequential_5/gru_17/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������z
%sequential_5/gru_17/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����|
1sequential_5/gru_17/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_5/gru_17/gru_cell_32/split_1SplitV2sequential_5/gru_17/gru_cell_32/BiasAdd_1:output:0.sequential_5/gru_17/gru_cell_32/Const:output:0:sequential_5/gru_17/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#sequential_5/gru_17/gru_cell_32/addAddV2.sequential_5/gru_17/gru_cell_32/split:output:00sequential_5/gru_17/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:����������
'sequential_5/gru_17/gru_cell_32/SigmoidSigmoid'sequential_5/gru_17/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
%sequential_5/gru_17/gru_cell_32/add_1AddV2.sequential_5/gru_17/gru_cell_32/split:output:10sequential_5/gru_17/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:����������
)sequential_5/gru_17/gru_cell_32/Sigmoid_1Sigmoid)sequential_5/gru_17/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
#sequential_5/gru_17/gru_cell_32/mulMul-sequential_5/gru_17/gru_cell_32/Sigmoid_1:y:00sequential_5/gru_17/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
%sequential_5/gru_17/gru_cell_32/add_2AddV2.sequential_5/gru_17/gru_cell_32/split:output:2'sequential_5/gru_17/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:����������
(sequential_5/gru_17/gru_cell_32/SoftplusSoftplus)sequential_5/gru_17/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
%sequential_5/gru_17/gru_cell_32/mul_1Mul+sequential_5/gru_17/gru_cell_32/Sigmoid:y:0"sequential_5/gru_17/zeros:output:0*
T0*'
_output_shapes
:���������j
%sequential_5/gru_17/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_5/gru_17/gru_cell_32/subSub.sequential_5/gru_17/gru_cell_32/sub/x:output:0+sequential_5/gru_17/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
%sequential_5/gru_17/gru_cell_32/mul_2Mul'sequential_5/gru_17/gru_cell_32/sub:z:06sequential_5/gru_17/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
%sequential_5/gru_17/gru_cell_32/add_3AddV2)sequential_5/gru_17/gru_cell_32/mul_1:z:0)sequential_5/gru_17/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
1sequential_5/gru_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#sequential_5/gru_17/TensorArrayV2_1TensorListReserve:sequential_5/gru_17/TensorArrayV2_1/element_shape:output:0,sequential_5/gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_5/gru_17/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_5/gru_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_5/gru_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_5/gru_17/whileWhile/sequential_5/gru_17/while/loop_counter:output:05sequential_5/gru_17/while/maximum_iterations:output:0!sequential_5/gru_17/time:output:0,sequential_5/gru_17/TensorArrayV2_1:handle:0"sequential_5/gru_17/zeros:output:0,sequential_5/gru_17/strided_slice_1:output:0Ksequential_5/gru_17/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_5_gru_17_gru_cell_32_readvariableop_resource>sequential_5_gru_17_gru_cell_32_matmul_readvariableop_resource@sequential_5_gru_17_gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&sequential_5_gru_17_while_body_2903158*2
cond*R(
&sequential_5_gru_17_while_cond_2903157*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
Dsequential_5/gru_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6sequential_5/gru_17/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/gru_17/while:output:3Msequential_5/gru_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0|
)sequential_5/gru_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_5/gru_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/gru_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_5/gru_17/strided_slice_3StridedSlice?sequential_5/gru_17/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/gru_17/strided_slice_3/stack:output:04sequential_5/gru_17/strided_slice_3/stack_1:output:04sequential_5/gru_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
$sequential_5/gru_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_5/gru_17/transpose_1	Transpose?sequential_5/gru_17/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/gru_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������o
sequential_5/gru_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    w
IdentityIdentity#sequential_5/gru_17/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp6^sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOp8^sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOp/^sequential_5/gru_15/gru_cell_30/ReadVariableOp^sequential_5/gru_15/while6^sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOp8^sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOp/^sequential_5/gru_16/gru_cell_31/ReadVariableOp^sequential_5/gru_16/while6^sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOp8^sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOp/^sequential_5/gru_17/gru_cell_32/ReadVariableOp^sequential_5/gru_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2n
5sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOp5sequential_5/gru_15/gru_cell_30/MatMul/ReadVariableOp2r
7sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOp7sequential_5/gru_15/gru_cell_30/MatMul_1/ReadVariableOp2`
.sequential_5/gru_15/gru_cell_30/ReadVariableOp.sequential_5/gru_15/gru_cell_30/ReadVariableOp26
sequential_5/gru_15/whilesequential_5/gru_15/while2n
5sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOp5sequential_5/gru_16/gru_cell_31/MatMul/ReadVariableOp2r
7sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOp7sequential_5/gru_16/gru_cell_31/MatMul_1/ReadVariableOp2`
.sequential_5/gru_16/gru_cell_31/ReadVariableOp.sequential_5/gru_16/gru_cell_31/ReadVariableOp26
sequential_5/gru_16/whilesequential_5/gru_16/while2n
5sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOp5sequential_5/gru_17/gru_cell_32/MatMul/ReadVariableOp2r
7sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOp7sequential_5/gru_17/gru_cell_32/MatMul_1/ReadVariableOp2`
.sequential_5/gru_17/gru_cell_32/ReadVariableOp.sequential_5/gru_17/gru_cell_32/ReadVariableOp26
sequential_5/gru_17/whilesequential_5/gru_17/while:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
�=
�
while_body_2907488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903798

inputs

states*
readvariableop_resource:	�2
matmul_readvariableop_resource:
��3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������dQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������dS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
while_cond_2904490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904490___redundant_placeholder05
1while_while_cond_2904490___redundant_placeholder15
1while_while_cond_2904490___redundant_placeholder25
1while_while_cond_2904490___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�E
�	
gru_16_while_body_2906180*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2)
%gru_16_while_gru_16_strided_slice_1_0e
agru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0E
2gru_16_while_gru_cell_31_readvariableop_resource_0:	�M
9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0:
��N
;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
gru_16_while_identity
gru_16_while_identity_1
gru_16_while_identity_2
gru_16_while_identity_3
gru_16_while_identity_4'
#gru_16_while_gru_16_strided_slice_1c
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensorC
0gru_16_while_gru_cell_31_readvariableop_resource:	�K
7gru_16_while_gru_cell_31_matmul_readvariableop_resource:
��L
9gru_16_while_gru_cell_31_matmul_1_readvariableop_resource:	d���.gru_16/while/gru_cell_31/MatMul/ReadVariableOp�0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp�'gru_16/while/gru_cell_31/ReadVariableOp�
>gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
0gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0gru_16_while_placeholderGgru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
'gru_16/while/gru_cell_31/ReadVariableOpReadVariableOp2gru_16_while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_16/while/gru_cell_31/unstackUnpack/gru_16/while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_16/while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
gru_16/while/gru_cell_31/MatMulMatMul7gru_16/while/TensorArrayV2Read/TensorListGetItem:item:06gru_16/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_16/while/gru_cell_31/BiasAddBiasAdd)gru_16/while/gru_cell_31/MatMul:product:0)gru_16/while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_16/while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/while/gru_cell_31/splitSplit1gru_16/while/gru_cell_31/split/split_dim:output:0)gru_16/while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
!gru_16/while/gru_cell_31/MatMul_1MatMulgru_16_while_placeholder_28gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_16/while/gru_cell_31/BiasAdd_1BiasAdd+gru_16/while/gru_cell_31/MatMul_1:product:0)gru_16/while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_16/while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����u
*gru_16/while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_16/while/gru_cell_31/split_1SplitV+gru_16/while/gru_cell_31/BiasAdd_1:output:0'gru_16/while/gru_cell_31/Const:output:03gru_16/while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_16/while/gru_cell_31/addAddV2'gru_16/while/gru_cell_31/split:output:0)gru_16/while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������d
 gru_16/while/gru_cell_31/SigmoidSigmoid gru_16/while/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_1AddV2'gru_16/while/gru_cell_31/split:output:1)gru_16/while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������d�
"gru_16/while/gru_cell_31/Sigmoid_1Sigmoid"gru_16/while/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mulMul&gru_16/while/gru_cell_31/Sigmoid_1:y:0)gru_16/while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_2AddV2'gru_16/while/gru_cell_31/split:output:2 gru_16/while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������d�
"gru_16/while/gru_cell_31/Sigmoid_2Sigmoid"gru_16/while/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mul_1Mul$gru_16/while/gru_cell_31/Sigmoid:y:0gru_16_while_placeholder_2*
T0*'
_output_shapes
:���������dc
gru_16/while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_16/while/gru_cell_31/subSub'gru_16/while/gru_cell_31/sub/x:output:0$gru_16/while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/mul_2Mul gru_16/while/gru_cell_31/sub:z:0&gru_16/while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_16/while/gru_cell_31/add_3AddV2"gru_16/while/gru_cell_31/mul_1:z:0"gru_16/while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_16_while_placeholder_1gru_16_while_placeholder"gru_16/while/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_16/while/addAddV2gru_16_while_placeholdergru_16/while/add/y:output:0*
T0*
_output_shapes
: V
gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_16/while/add_1AddV2&gru_16_while_gru_16_while_loop_countergru_16/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_16/while/IdentityIdentitygru_16/while/add_1:z:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_1Identity,gru_16_while_gru_16_while_maximum_iterations^gru_16/while/NoOp*
T0*
_output_shapes
: n
gru_16/while/Identity_2Identitygru_16/while/add:z:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_3IdentityAgru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_16/while/NoOp*
T0*
_output_shapes
: �
gru_16/while/Identity_4Identity"gru_16/while/gru_cell_31/add_3:z:0^gru_16/while/NoOp*
T0*'
_output_shapes
:���������d�
gru_16/while/NoOpNoOp/^gru_16/while/gru_cell_31/MatMul/ReadVariableOp1^gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp(^gru_16/while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_16_while_gru_16_strided_slice_1%gru_16_while_gru_16_strided_slice_1_0"x
9gru_16_while_gru_cell_31_matmul_1_readvariableop_resource;gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0"t
7gru_16_while_gru_cell_31_matmul_readvariableop_resource9gru_16_while_gru_cell_31_matmul_readvariableop_resource_0"f
0gru_16_while_gru_cell_31_readvariableop_resource2gru_16_while_gru_cell_31_readvariableop_resource_0"7
gru_16_while_identitygru_16/while/Identity:output:0";
gru_16_while_identity_1 gru_16/while/Identity_1:output:0";
gru_16_while_identity_2 gru_16/while/Identity_2:output:0";
gru_16_while_identity_3 gru_16/while/Identity_3:output:0";
gru_16_while_identity_4 gru_16/while/Identity_4:output:0"�
_gru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensoragru_16_while_tensorarrayv2read_tensorlistgetitem_gru_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.gru_16/while/gru_cell_31/MatMul/ReadVariableOp.gru_16/while/gru_cell_31/MatMul/ReadVariableOp2d
0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp0gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp2R
'gru_16/while/gru_cell_31/ReadVariableOp'gru_16/while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�

�
-__inference_gru_cell_30_layer_call_fn_2908414

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903460p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0
�
�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903655

inputs

states*
readvariableop_resource:	�2
matmul_readvariableop_resource:
��3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������dQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������dS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�	
�
gru_15_while_cond_2905579*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2,
(gru_15_while_less_gru_15_strided_slice_1C
?gru_15_while_gru_15_while_cond_2905579___redundant_placeholder0C
?gru_15_while_gru_15_while_cond_2905579___redundant_placeholder1C
?gru_15_while_gru_15_while_cond_2905579___redundant_placeholder2C
?gru_15_while_gru_15_while_cond_2905579___redundant_placeholder3
gru_15_while_identity
~
gru_15/while/LessLessgru_15_while_placeholder(gru_15_while_less_gru_15_strided_slice_1*
T0*
_output_shapes
: Y
gru_15/while/IdentityIdentitygru_15/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_15_while_identitygru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908386

inputs5
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2908297*
condR
while_cond_2908296*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�

�
-__inference_gru_cell_31_layer_call_fn_2908520

inputs
states_0
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
(__inference_gru_16_layer_call_fn_2907118

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2905111t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907577

inputs6
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:�����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907488*
condR
while_cond_2907487*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:����������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�=
�
while_body_2906832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�E
�	
gru_17_while_body_2905878*
&gru_17_while_gru_17_while_loop_counter0
,gru_17_while_gru_17_while_maximum_iterations
gru_17_while_placeholder
gru_17_while_placeholder_1
gru_17_while_placeholder_2)
%gru_17_while_gru_17_strided_slice_1_0e
agru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0D
2gru_17_while_gru_cell_32_readvariableop_resource_0:K
9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0:dM
;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0:
gru_17_while_identity
gru_17_while_identity_1
gru_17_while_identity_2
gru_17_while_identity_3
gru_17_while_identity_4'
#gru_17_while_gru_17_strided_slice_1c
_gru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensorB
0gru_17_while_gru_cell_32_readvariableop_resource:I
7gru_17_while_gru_cell_32_matmul_readvariableop_resource:dK
9gru_17_while_gru_cell_32_matmul_1_readvariableop_resource:��.gru_17/while/gru_cell_32/MatMul/ReadVariableOp�0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp�'gru_17/while/gru_cell_32/ReadVariableOp�
>gru_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
0gru_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0gru_17_while_placeholderGgru_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
'gru_17/while/gru_cell_32/ReadVariableOpReadVariableOp2gru_17_while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 gru_17/while/gru_cell_32/unstackUnpack/gru_17/while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
.gru_17/while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
gru_17/while/gru_cell_32/MatMulMatMul7gru_17/while/TensorArrayV2Read/TensorListGetItem:item:06gru_17/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 gru_17/while/gru_cell_32/BiasAddBiasAdd)gru_17/while/gru_cell_32/MatMul:product:0)gru_17/while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������s
(gru_17/while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/while/gru_cell_32/splitSplit1gru_17/while/gru_cell_32/split/split_dim:output:0)gru_17/while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
!gru_17/while/gru_cell_32/MatMul_1MatMulgru_17_while_placeholder_28gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"gru_17/while/gru_cell_32/BiasAdd_1BiasAdd+gru_17/while/gru_cell_32/MatMul_1:product:0)gru_17/while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������s
gru_17/while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����u
*gru_17/while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_17/while/gru_cell_32/split_1SplitV+gru_17/while/gru_cell_32/BiasAdd_1:output:0'gru_17/while/gru_cell_32/Const:output:03gru_17/while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_17/while/gru_cell_32/addAddV2'gru_17/while/gru_cell_32/split:output:0)gru_17/while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������
 gru_17/while/gru_cell_32/SigmoidSigmoid gru_17/while/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_1AddV2'gru_17/while/gru_cell_32/split:output:1)gru_17/while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:����������
"gru_17/while/gru_cell_32/Sigmoid_1Sigmoid"gru_17/while/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mulMul&gru_17/while/gru_cell_32/Sigmoid_1:y:0)gru_17/while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_2AddV2'gru_17/while/gru_cell_32/split:output:2 gru_17/while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:����������
!gru_17/while/gru_cell_32/SoftplusSoftplus"gru_17/while/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mul_1Mul$gru_17/while/gru_cell_32/Sigmoid:y:0gru_17_while_placeholder_2*
T0*'
_output_shapes
:���������c
gru_17/while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_17/while/gru_cell_32/subSub'gru_17/while/gru_cell_32/sub/x:output:0$gru_17/while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mul_2Mul gru_17/while/gru_cell_32/sub:z:0/gru_17/while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_3AddV2"gru_17/while/gru_cell_32/mul_1:z:0"gru_17/while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
1gru_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_17_while_placeholder_1gru_17_while_placeholder"gru_17/while/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_17/while/addAddV2gru_17_while_placeholdergru_17/while/add/y:output:0*
T0*
_output_shapes
: V
gru_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_17/while/add_1AddV2&gru_17_while_gru_17_while_loop_countergru_17/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_17/while/IdentityIdentitygru_17/while/add_1:z:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_1Identity,gru_17_while_gru_17_while_maximum_iterations^gru_17/while/NoOp*
T0*
_output_shapes
: n
gru_17/while/Identity_2Identitygru_17/while/add:z:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_3IdentityAgru_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_4Identity"gru_17/while/gru_cell_32/add_3:z:0^gru_17/while/NoOp*
T0*'
_output_shapes
:����������
gru_17/while/NoOpNoOp/^gru_17/while/gru_cell_32/MatMul/ReadVariableOp1^gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp(^gru_17/while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_17_while_gru_17_strided_slice_1%gru_17_while_gru_17_strided_slice_1_0"x
9gru_17_while_gru_cell_32_matmul_1_readvariableop_resource;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0"t
7gru_17_while_gru_cell_32_matmul_readvariableop_resource9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0"f
0gru_17_while_gru_cell_32_readvariableop_resource2gru_17_while_gru_cell_32_readvariableop_resource_0"7
gru_17_while_identitygru_17/while/Identity:output:0";
gru_17_while_identity_1 gru_17/while/Identity_1:output:0";
gru_17_while_identity_2 gru_17/while/Identity_2:output:0";
gru_17_while_identity_3 gru_17/while/Identity_3:output:0";
gru_17_while_identity_4 gru_17/while/Identity_4:output:0"�
_gru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensoragru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.gru_17/while/gru_cell_32/MatMul/ReadVariableOp.gru_17/while/gru_cell_32/MatMul/ReadVariableOp2d
0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp2R
'gru_17/while/gru_cell_32/ReadVariableOp'gru_17/while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2906525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2906525___redundant_placeholder05
1while_while_cond_2906525___redundant_placeholder15
1while_while_cond_2906525___redundant_placeholder25
1while_while_cond_2906525___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2906678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2906678___redundant_placeholder05
1while_while_cond_2906678___redundant_placeholder15
1while_while_cond_2906678___redundant_placeholder25
1while_while_cond_2906678___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�=
�
while_body_2905022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�

�
.__inference_sequential_5_layer_call_fn_2905389
gru_15_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	d�
	unknown_5:
	unknown_6:d
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905345t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
�
�
(__inference_gru_17_layer_call_fn_2907763

inputs
unknown:
	unknown_0:d
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904740t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�4
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2904070

inputs%
gru_cell_32_2903994:%
gru_cell_32_2903996:d%
gru_cell_32_2903998:
identity��#gru_cell_32/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
#gru_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_32_2903994gru_cell_32_2903996gru_cell_32_2903998*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2903993n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_32_2903994gru_cell_32_2903996gru_cell_32_2903998*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904006*
condR
while_cond_2904005*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������t
NoOpNoOp$^gru_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2J
#gru_cell_32/StatefulPartitionedCall#gru_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2905286

inputs6
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2905197*
condR
while_cond_2905196*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903317

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:����������c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:����������R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:����������^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:����������Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:����������R
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:����������T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������W
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:����������W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates
� 
�
while_body_2903850
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_31_2903872_0:	�/
while_gru_cell_31_2903874_0:
��.
while_gru_cell_31_2903876_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_31_2903872:	�-
while_gru_cell_31_2903874:
��,
while_gru_cell_31_2903876:	d���)while/gru_cell_31/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
)while/gru_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_31_2903872_0while_gru_cell_31_2903874_0while_gru_cell_31_2903876_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903798�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������dx

while/NoOpNoOp*^while/gru_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_31_2903872while_gru_cell_31_2903872_0"8
while_gru_cell_31_2903874while_gru_cell_31_2903874_0"8
while_gru_cell_31_2903876while_gru_cell_31_2903876_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2V
)while/gru_cell_31/StatefulPartitionedCall)while/gru_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2904650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904650___redundant_placeholder05
1while_while_cond_2904650___redundant_placeholder15
1while_while_cond_2904650___redundant_placeholder25
1while_while_cond_2904650___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
&sequential_5_gru_16_while_cond_2903008D
@sequential_5_gru_16_while_sequential_5_gru_16_while_loop_counterJ
Fsequential_5_gru_16_while_sequential_5_gru_16_while_maximum_iterations)
%sequential_5_gru_16_while_placeholder+
'sequential_5_gru_16_while_placeholder_1+
'sequential_5_gru_16_while_placeholder_2F
Bsequential_5_gru_16_while_less_sequential_5_gru_16_strided_slice_1]
Ysequential_5_gru_16_while_sequential_5_gru_16_while_cond_2903008___redundant_placeholder0]
Ysequential_5_gru_16_while_sequential_5_gru_16_while_cond_2903008___redundant_placeholder1]
Ysequential_5_gru_16_while_sequential_5_gru_16_while_cond_2903008___redundant_placeholder2]
Ysequential_5_gru_16_while_sequential_5_gru_16_while_cond_2903008___redundant_placeholder3&
"sequential_5_gru_16_while_identity
�
sequential_5/gru_16/while/LessLess%sequential_5_gru_16_while_placeholderBsequential_5_gru_16_while_less_sequential_5_gru_16_strided_slice_1*
T0*
_output_shapes
: s
"sequential_5/gru_16/while/IdentityIdentity"sequential_5/gru_16/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_5_gru_16_while_identity+sequential_5/gru_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_gru_15_layer_call_fn_2906440
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2903576}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905439
gru_15_input!
gru_15_2905417:	�!
gru_15_2905419:	�"
gru_15_2905421:
��!
gru_16_2905424:	�"
gru_16_2905426:
��!
gru_16_2905428:	d� 
gru_17_2905431: 
gru_17_2905433:d 
gru_17_2905435:
identity��gru_15/StatefulPartitionedCall�gru_16/StatefulPartitionedCall�gru_17/StatefulPartitionedCall�
gru_15/StatefulPartitionedCallStatefulPartitionedCallgru_15_inputgru_15_2905417gru_15_2905419gru_15_2905421*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2905286�
gru_16/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0gru_16_2905424gru_16_2905426gru_16_2905428*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2905111�
gru_17/StatefulPartitionedCallStatefulPartitionedCall'gru_16/StatefulPartitionedCall:output:0gru_17_2905431gru_17_2905433gru_17_2905435*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904936{
IdentityIdentity'gru_17/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_15/StatefulPartitionedCall^gru_16/StatefulPartitionedCall^gru_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall2@
gru_17/StatefulPartitionedCallgru_17/StatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907424
inputs_06
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907335*
condR
while_cond_2907334*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
��
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905967

inputs=
*gru_15_gru_cell_30_readvariableop_resource:	�D
1gru_15_gru_cell_30_matmul_readvariableop_resource:	�G
3gru_15_gru_cell_30_matmul_1_readvariableop_resource:
��=
*gru_16_gru_cell_31_readvariableop_resource:	�E
1gru_16_gru_cell_31_matmul_readvariableop_resource:
��F
3gru_16_gru_cell_31_matmul_1_readvariableop_resource:	d�<
*gru_17_gru_cell_32_readvariableop_resource:C
1gru_17_gru_cell_32_matmul_readvariableop_resource:dE
3gru_17_gru_cell_32_matmul_1_readvariableop_resource:
identity��(gru_15/gru_cell_30/MatMul/ReadVariableOp�*gru_15/gru_cell_30/MatMul_1/ReadVariableOp�!gru_15/gru_cell_30/ReadVariableOp�gru_15/while�(gru_16/gru_cell_31/MatMul/ReadVariableOp�*gru_16/gru_cell_31/MatMul_1/ReadVariableOp�!gru_16/gru_cell_31/ReadVariableOp�gru_16/while�(gru_17/gru_cell_32/MatMul/ReadVariableOp�*gru_17/gru_cell_32/MatMul_1/ReadVariableOp�!gru_17/gru_cell_32/ReadVariableOp�gru_17/whileB
gru_15/ShapeShapeinputs*
T0*
_output_shapes
:d
gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_sliceStridedSlicegru_15/Shape:output:0#gru_15/strided_slice/stack:output:0%gru_15/strided_slice/stack_1:output:0%gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
gru_15/zeros/packedPackgru_15/strided_slice:output:0gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_15/zerosFillgru_15/zeros/packed:output:0gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:����������j
gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
gru_15/transpose	Transposeinputsgru_15/transpose/perm:output:0*
T0*,
_output_shapes
:����������R
gru_15/Shape_1Shapegru_15/transpose:y:0*
T0*
_output_shapes
:f
gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_1StridedSlicegru_15/Shape_1:output:0%gru_15/strided_slice_1/stack:output:0'gru_15/strided_slice_1/stack_1:output:0'gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/TensorArrayV2TensorListReserve+gru_15/TensorArrayV2/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
.gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_15/transpose:y:0Egru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_2StridedSlicegru_15/transpose:y:0%gru_15/strided_slice_2/stack:output:0'gru_15/strided_slice_2/stack_1:output:0'gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!gru_15/gru_cell_30/ReadVariableOpReadVariableOp*gru_15_gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_15/gru_cell_30/unstackUnpack)gru_15/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_15/gru_cell_30/MatMul/ReadVariableOpReadVariableOp1gru_15_gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_15/gru_cell_30/MatMulMatMulgru_15/strided_slice_2:output:00gru_15/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/BiasAddBiasAdd#gru_15/gru_cell_30/MatMul:product:0#gru_15/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_15/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/gru_cell_30/splitSplit+gru_15/gru_cell_30/split/split_dim:output:0#gru_15/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
*gru_15/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp3gru_15_gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_15/gru_cell_30/MatMul_1MatMulgru_15/zeros:output:02gru_15/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/BiasAdd_1BiasAdd%gru_15/gru_cell_30/MatMul_1:product:0#gru_15/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_15/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����o
$gru_15/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/gru_cell_30/split_1SplitV%gru_15/gru_cell_30/BiasAdd_1:output:0!gru_15/gru_cell_30/Const:output:0-gru_15/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_15/gru_cell_30/addAddV2!gru_15/gru_cell_30/split:output:0#gru_15/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������t
gru_15/gru_cell_30/SigmoidSigmoidgru_15/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_1AddV2!gru_15/gru_cell_30/split:output:1#gru_15/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������x
gru_15/gru_cell_30/Sigmoid_1Sigmoidgru_15/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mulMul gru_15/gru_cell_30/Sigmoid_1:y:0#gru_15/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_2AddV2!gru_15/gru_cell_30/split:output:2gru_15/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������x
gru_15/gru_cell_30/Sigmoid_2Sigmoidgru_15/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mul_1Mulgru_15/gru_cell_30/Sigmoid:y:0gru_15/zeros:output:0*
T0*(
_output_shapes
:����������]
gru_15/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_15/gru_cell_30/subSub!gru_15/gru_cell_30/sub/x:output:0gru_15/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mul_2Mulgru_15/gru_cell_30/sub:z:0 gru_15/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_3AddV2gru_15/gru_cell_30/mul_1:z:0gru_15/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������u
$gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
gru_15/TensorArrayV2_1TensorListReserve-gru_15/TensorArrayV2_1/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_15/whileWhile"gru_15/while/loop_counter:output:0(gru_15/while/maximum_iterations:output:0gru_15/time:output:0gru_15/TensorArrayV2_1:handle:0gru_15/zeros:output:0gru_15/strided_slice_1:output:0>gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_15_gru_cell_30_readvariableop_resource1gru_15_gru_cell_30_matmul_readvariableop_resource3gru_15_gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_15_while_body_2905580*%
condR
gru_15_while_cond_2905579*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
7gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)gru_15/TensorArrayV2Stack/TensorListStackTensorListStackgru_15/while:output:3@gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0o
gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_3StridedSlice2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0%gru_15/strided_slice_3/stack:output:0'gru_15/strided_slice_3/stack_1:output:0'gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskl
gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_15/transpose_1	Transpose2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0 gru_15/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������b
gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_16/ShapeShapegru_15/transpose_1:y:0*
T0*
_output_shapes
:d
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������dj
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_16/transpose	Transposegru_15/transpose_1:y:0gru_16/transpose/perm:output:0*
T0*-
_output_shapes
:�����������R
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:f
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
!gru_16/gru_cell_31/ReadVariableOpReadVariableOp*gru_16_gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_16/gru_cell_31/unstackUnpack)gru_16/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_16/gru_cell_31/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_16/gru_cell_31/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_16/gru_cell_31/BiasAddBiasAdd#gru_16/gru_cell_31/MatMul:product:0#gru_16/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_16/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/gru_cell_31/splitSplit+gru_16/gru_cell_31/split/split_dim:output:0#gru_16/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
*gru_16/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_16/gru_cell_31/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_16/gru_cell_31/BiasAdd_1BiasAdd%gru_16/gru_cell_31/MatMul_1:product:0#gru_16/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_16/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����o
$gru_16/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/gru_cell_31/split_1SplitV%gru_16/gru_cell_31/BiasAdd_1:output:0!gru_16/gru_cell_31/Const:output:0-gru_16/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_16/gru_cell_31/addAddV2!gru_16/gru_cell_31/split:output:0#gru_16/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������ds
gru_16/gru_cell_31/SigmoidSigmoidgru_16/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_1AddV2!gru_16/gru_cell_31/split:output:1#gru_16/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������dw
gru_16/gru_cell_31/Sigmoid_1Sigmoidgru_16/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mulMul gru_16/gru_cell_31/Sigmoid_1:y:0#gru_16/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_2AddV2!gru_16/gru_cell_31/split:output:2gru_16/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������dw
gru_16/gru_cell_31/Sigmoid_2Sigmoidgru_16/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mul_1Mulgru_16/gru_cell_31/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:���������d]
gru_16/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_16/gru_cell_31/subSub!gru_16/gru_cell_31/sub/x:output:0gru_16/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mul_2Mulgru_16/gru_cell_31/sub:z:0 gru_16/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_3AddV2gru_16/gru_cell_31/mul_1:z:0gru_16/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������du
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_31_readvariableop_resource1gru_16_gru_cell_31_matmul_readvariableop_resource3gru_16_gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_16_while_body_2905729*%
condR
gru_16_while_cond_2905728*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0o
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskl
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������db
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_17/ShapeShapegru_16/transpose_1:y:0*
T0*
_output_shapes
:d
gru_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_sliceStridedSlicegru_17/Shape:output:0#gru_17/strided_slice/stack:output:0%gru_17/strided_slice/stack_1:output:0%gru_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
gru_17/zeros/packedPackgru_17/strided_slice:output:0gru_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_17/zerosFillgru_17/zeros/packed:output:0gru_17/zeros/Const:output:0*
T0*'
_output_shapes
:���������j
gru_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_17/transpose	Transposegru_16/transpose_1:y:0gru_17/transpose/perm:output:0*
T0*,
_output_shapes
:����������dR
gru_17/Shape_1Shapegru_17/transpose:y:0*
T0*
_output_shapes
:f
gru_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_1StridedSlicegru_17/Shape_1:output:0%gru_17/strided_slice_1/stack:output:0'gru_17/strided_slice_1/stack_1:output:0'gru_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/TensorArrayV2TensorListReserve+gru_17/TensorArrayV2/element_shape:output:0gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
.gru_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_17/transpose:y:0Egru_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_2StridedSlicegru_17/transpose:y:0%gru_17/strided_slice_2/stack:output:0'gru_17/strided_slice_2/stack_1:output:0'gru_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
!gru_17/gru_cell_32/ReadVariableOpReadVariableOp*gru_17_gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_17/gru_cell_32/unstackUnpack)gru_17/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
(gru_17/gru_cell_32/MatMul/ReadVariableOpReadVariableOp1gru_17_gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_17/gru_cell_32/MatMulMatMulgru_17/strided_slice_2:output:00gru_17/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/BiasAddBiasAdd#gru_17/gru_cell_32/MatMul:product:0#gru_17/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������m
"gru_17/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/gru_cell_32/splitSplit+gru_17/gru_cell_32/split/split_dim:output:0#gru_17/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
*gru_17/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp3gru_17_gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_17/gru_cell_32/MatMul_1MatMulgru_17/zeros:output:02gru_17/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/BiasAdd_1BiasAdd%gru_17/gru_cell_32/MatMul_1:product:0#gru_17/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������m
gru_17/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru_17/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/gru_cell_32/split_1SplitV%gru_17/gru_cell_32/BiasAdd_1:output:0!gru_17/gru_cell_32/Const:output:0-gru_17/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_17/gru_cell_32/addAddV2!gru_17/gru_cell_32/split:output:0#gru_17/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������s
gru_17/gru_cell_32/SigmoidSigmoidgru_17/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_1AddV2!gru_17/gru_cell_32/split:output:1#gru_17/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������w
gru_17/gru_cell_32/Sigmoid_1Sigmoidgru_17/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mulMul gru_17/gru_cell_32/Sigmoid_1:y:0#gru_17/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_2AddV2!gru_17/gru_cell_32/split:output:2gru_17/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������w
gru_17/gru_cell_32/SoftplusSoftplusgru_17/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mul_1Mulgru_17/gru_cell_32/Sigmoid:y:0gru_17/zeros:output:0*
T0*'
_output_shapes
:���������]
gru_17/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_17/gru_cell_32/subSub!gru_17/gru_cell_32/sub/x:output:0gru_17/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mul_2Mulgru_17/gru_cell_32/sub:z:0)gru_17/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_3AddV2gru_17/gru_cell_32/mul_1:z:0gru_17/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������u
$gru_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
gru_17/TensorArrayV2_1TensorListReserve-gru_17/TensorArrayV2_1/element_shape:output:0gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_17/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_17/whileWhile"gru_17/while/loop_counter:output:0(gru_17/while/maximum_iterations:output:0gru_17/time:output:0gru_17/TensorArrayV2_1:handle:0gru_17/zeros:output:0gru_17/strided_slice_1:output:0>gru_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_17_gru_cell_32_readvariableop_resource1gru_17_gru_cell_32_matmul_readvariableop_resource3gru_17_gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_17_while_body_2905878*%
condR
gru_17_while_cond_2905877*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
7gru_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)gru_17/TensorArrayV2Stack/TensorListStackTensorListStackgru_17/while:output:3@gru_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0o
gru_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_3StridedSlice2gru_17/TensorArrayV2Stack/TensorListStack:tensor:0%gru_17/strided_slice_3/stack:output:0'gru_17/strided_slice_3/stack_1:output:0'gru_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskl
gru_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_17/transpose_1	Transpose2gru_17/TensorArrayV2Stack/TensorListStack:tensor:0 gru_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������b
gru_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_17/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp)^gru_15/gru_cell_30/MatMul/ReadVariableOp+^gru_15/gru_cell_30/MatMul_1/ReadVariableOp"^gru_15/gru_cell_30/ReadVariableOp^gru_15/while)^gru_16/gru_cell_31/MatMul/ReadVariableOp+^gru_16/gru_cell_31/MatMul_1/ReadVariableOp"^gru_16/gru_cell_31/ReadVariableOp^gru_16/while)^gru_17/gru_cell_32/MatMul/ReadVariableOp+^gru_17/gru_cell_32/MatMul_1/ReadVariableOp"^gru_17/gru_cell_32/ReadVariableOp^gru_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2T
(gru_15/gru_cell_30/MatMul/ReadVariableOp(gru_15/gru_cell_30/MatMul/ReadVariableOp2X
*gru_15/gru_cell_30/MatMul_1/ReadVariableOp*gru_15/gru_cell_30/MatMul_1/ReadVariableOp2F
!gru_15/gru_cell_30/ReadVariableOp!gru_15/gru_cell_30/ReadVariableOp2
gru_15/whilegru_15/while2T
(gru_16/gru_cell_31/MatMul/ReadVariableOp(gru_16/gru_cell_31/MatMul/ReadVariableOp2X
*gru_16/gru_cell_31/MatMul_1/ReadVariableOp*gru_16/gru_cell_31/MatMul_1/ReadVariableOp2F
!gru_16/gru_cell_31/ReadVariableOp!gru_16/gru_cell_31/ReadVariableOp2
gru_16/whilegru_16/while2T
(gru_17/gru_cell_32/MatMul/ReadVariableOp(gru_17/gru_cell_32/MatMul/ReadVariableOp2X
*gru_17/gru_cell_32/MatMul_1/ReadVariableOp*gru_17/gru_cell_32/MatMul_1/ReadVariableOp2F
!gru_17/gru_cell_32/ReadVariableOp!gru_17/gru_cell_32/ReadVariableOp2
gru_17/whilegru_17/while:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903460

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:����������c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:����������R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:����������^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:����������Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:����������R
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:����������T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������W
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:����������W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
(__inference_gru_15_layer_call_fn_2906451

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2904420u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2906418

inputs=
*gru_15_gru_cell_30_readvariableop_resource:	�D
1gru_15_gru_cell_30_matmul_readvariableop_resource:	�G
3gru_15_gru_cell_30_matmul_1_readvariableop_resource:
��=
*gru_16_gru_cell_31_readvariableop_resource:	�E
1gru_16_gru_cell_31_matmul_readvariableop_resource:
��F
3gru_16_gru_cell_31_matmul_1_readvariableop_resource:	d�<
*gru_17_gru_cell_32_readvariableop_resource:C
1gru_17_gru_cell_32_matmul_readvariableop_resource:dE
3gru_17_gru_cell_32_matmul_1_readvariableop_resource:
identity��(gru_15/gru_cell_30/MatMul/ReadVariableOp�*gru_15/gru_cell_30/MatMul_1/ReadVariableOp�!gru_15/gru_cell_30/ReadVariableOp�gru_15/while�(gru_16/gru_cell_31/MatMul/ReadVariableOp�*gru_16/gru_cell_31/MatMul_1/ReadVariableOp�!gru_16/gru_cell_31/ReadVariableOp�gru_16/while�(gru_17/gru_cell_32/MatMul/ReadVariableOp�*gru_17/gru_cell_32/MatMul_1/ReadVariableOp�!gru_17/gru_cell_32/ReadVariableOp�gru_17/whileB
gru_15/ShapeShapeinputs*
T0*
_output_shapes
:d
gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_sliceStridedSlicegru_15/Shape:output:0#gru_15/strided_slice/stack:output:0%gru_15/strided_slice/stack_1:output:0%gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
gru_15/zeros/packedPackgru_15/strided_slice:output:0gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_15/zerosFillgru_15/zeros/packed:output:0gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:����������j
gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
gru_15/transpose	Transposeinputsgru_15/transpose/perm:output:0*
T0*,
_output_shapes
:����������R
gru_15/Shape_1Shapegru_15/transpose:y:0*
T0*
_output_shapes
:f
gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_1StridedSlicegru_15/Shape_1:output:0%gru_15/strided_slice_1/stack:output:0'gru_15/strided_slice_1/stack_1:output:0'gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/TensorArrayV2TensorListReserve+gru_15/TensorArrayV2/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
.gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_15/transpose:y:0Egru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_2StridedSlicegru_15/transpose:y:0%gru_15/strided_slice_2/stack:output:0'gru_15/strided_slice_2/stack_1:output:0'gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!gru_15/gru_cell_30/ReadVariableOpReadVariableOp*gru_15_gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_15/gru_cell_30/unstackUnpack)gru_15/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_15/gru_cell_30/MatMul/ReadVariableOpReadVariableOp1gru_15_gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_15/gru_cell_30/MatMulMatMulgru_15/strided_slice_2:output:00gru_15/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/BiasAddBiasAdd#gru_15/gru_cell_30/MatMul:product:0#gru_15/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_15/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/gru_cell_30/splitSplit+gru_15/gru_cell_30/split/split_dim:output:0#gru_15/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
*gru_15/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp3gru_15_gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_15/gru_cell_30/MatMul_1MatMulgru_15/zeros:output:02gru_15/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/BiasAdd_1BiasAdd%gru_15/gru_cell_30/MatMul_1:product:0#gru_15/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_15/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����o
$gru_15/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/gru_cell_30/split_1SplitV%gru_15/gru_cell_30/BiasAdd_1:output:0!gru_15/gru_cell_30/Const:output:0-gru_15/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_15/gru_cell_30/addAddV2!gru_15/gru_cell_30/split:output:0#gru_15/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������t
gru_15/gru_cell_30/SigmoidSigmoidgru_15/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_1AddV2!gru_15/gru_cell_30/split:output:1#gru_15/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������x
gru_15/gru_cell_30/Sigmoid_1Sigmoidgru_15/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mulMul gru_15/gru_cell_30/Sigmoid_1:y:0#gru_15/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_2AddV2!gru_15/gru_cell_30/split:output:2gru_15/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������x
gru_15/gru_cell_30/Sigmoid_2Sigmoidgru_15/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mul_1Mulgru_15/gru_cell_30/Sigmoid:y:0gru_15/zeros:output:0*
T0*(
_output_shapes
:����������]
gru_15/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_15/gru_cell_30/subSub!gru_15/gru_cell_30/sub/x:output:0gru_15/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/mul_2Mulgru_15/gru_cell_30/sub:z:0 gru_15/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_15/gru_cell_30/add_3AddV2gru_15/gru_cell_30/mul_1:z:0gru_15/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������u
$gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
gru_15/TensorArrayV2_1TensorListReserve-gru_15/TensorArrayV2_1/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_15/whileWhile"gru_15/while/loop_counter:output:0(gru_15/while/maximum_iterations:output:0gru_15/time:output:0gru_15/TensorArrayV2_1:handle:0gru_15/zeros:output:0gru_15/strided_slice_1:output:0>gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_15_gru_cell_30_readvariableop_resource1gru_15_gru_cell_30_matmul_readvariableop_resource3gru_15_gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_15_while_body_2906031*%
condR
gru_15_while_cond_2906030*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
7gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)gru_15/TensorArrayV2Stack/TensorListStackTensorListStackgru_15/while:output:3@gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0o
gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_15/strided_slice_3StridedSlice2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0%gru_15/strided_slice_3/stack:output:0'gru_15/strided_slice_3/stack_1:output:0'gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskl
gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_15/transpose_1	Transpose2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0 gru_15/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������b
gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_16/ShapeShapegru_15/transpose_1:y:0*
T0*
_output_shapes
:d
gru_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_sliceStridedSlicegru_16/Shape:output:0#gru_16/strided_slice/stack:output:0%gru_16/strided_slice/stack_1:output:0%gru_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
gru_16/zeros/packedPackgru_16/strided_slice:output:0gru_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_16/zerosFillgru_16/zeros/packed:output:0gru_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������dj
gru_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_16/transpose	Transposegru_15/transpose_1:y:0gru_16/transpose/perm:output:0*
T0*-
_output_shapes
:�����������R
gru_16/Shape_1Shapegru_16/transpose:y:0*
T0*
_output_shapes
:f
gru_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_1StridedSlicegru_16/Shape_1:output:0%gru_16/strided_slice_1/stack:output:0'gru_16/strided_slice_1/stack_1:output:0'gru_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/TensorArrayV2TensorListReserve+gru_16/TensorArrayV2/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
.gru_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_16/transpose:y:0Egru_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_2StridedSlicegru_16/transpose:y:0%gru_16/strided_slice_2/stack:output:0'gru_16/strided_slice_2/stack_1:output:0'gru_16/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
!gru_16/gru_cell_31/ReadVariableOpReadVariableOp*gru_16_gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_16/gru_cell_31/unstackUnpack)gru_16/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_16/gru_cell_31/MatMul/ReadVariableOpReadVariableOp1gru_16_gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_16/gru_cell_31/MatMulMatMulgru_16/strided_slice_2:output:00gru_16/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_16/gru_cell_31/BiasAddBiasAdd#gru_16/gru_cell_31/MatMul:product:0#gru_16/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_16/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/gru_cell_31/splitSplit+gru_16/gru_cell_31/split/split_dim:output:0#gru_16/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
*gru_16/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp3gru_16_gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_16/gru_cell_31/MatMul_1MatMulgru_16/zeros:output:02gru_16/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_16/gru_cell_31/BiasAdd_1BiasAdd%gru_16/gru_cell_31/MatMul_1:product:0#gru_16/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_16/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����o
$gru_16/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_16/gru_cell_31/split_1SplitV%gru_16/gru_cell_31/BiasAdd_1:output:0!gru_16/gru_cell_31/Const:output:0-gru_16/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_16/gru_cell_31/addAddV2!gru_16/gru_cell_31/split:output:0#gru_16/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������ds
gru_16/gru_cell_31/SigmoidSigmoidgru_16/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_1AddV2!gru_16/gru_cell_31/split:output:1#gru_16/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������dw
gru_16/gru_cell_31/Sigmoid_1Sigmoidgru_16/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mulMul gru_16/gru_cell_31/Sigmoid_1:y:0#gru_16/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_2AddV2!gru_16/gru_cell_31/split:output:2gru_16/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������dw
gru_16/gru_cell_31/Sigmoid_2Sigmoidgru_16/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mul_1Mulgru_16/gru_cell_31/Sigmoid:y:0gru_16/zeros:output:0*
T0*'
_output_shapes
:���������d]
gru_16/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_16/gru_cell_31/subSub!gru_16/gru_cell_31/sub/x:output:0gru_16/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/mul_2Mulgru_16/gru_cell_31/sub:z:0 gru_16/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_16/gru_cell_31/add_3AddV2gru_16/gru_cell_31/mul_1:z:0gru_16/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������du
$gru_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
gru_16/TensorArrayV2_1TensorListReserve-gru_16/TensorArrayV2_1/element_shape:output:0gru_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_16/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_16/whileWhile"gru_16/while/loop_counter:output:0(gru_16/while/maximum_iterations:output:0gru_16/time:output:0gru_16/TensorArrayV2_1:handle:0gru_16/zeros:output:0gru_16/strided_slice_1:output:0>gru_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_16_gru_cell_31_readvariableop_resource1gru_16_gru_cell_31_matmul_readvariableop_resource3gru_16_gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_16_while_body_2906180*%
condR
gru_16_while_cond_2906179*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
7gru_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)gru_16/TensorArrayV2Stack/TensorListStackTensorListStackgru_16/while:output:3@gru_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0o
gru_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_16/strided_slice_3StridedSlice2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0%gru_16/strided_slice_3/stack:output:0'gru_16/strided_slice_3/stack_1:output:0'gru_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskl
gru_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_16/transpose_1	Transpose2gru_16/TensorArrayV2Stack/TensorListStack:tensor:0 gru_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������db
gru_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_17/ShapeShapegru_16/transpose_1:y:0*
T0*
_output_shapes
:d
gru_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_sliceStridedSlicegru_17/Shape:output:0#gru_17/strided_slice/stack:output:0%gru_17/strided_slice/stack_1:output:0%gru_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
gru_17/zeros/packedPackgru_17/strided_slice:output:0gru_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_17/zerosFillgru_17/zeros/packed:output:0gru_17/zeros/Const:output:0*
T0*'
_output_shapes
:���������j
gru_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_17/transpose	Transposegru_16/transpose_1:y:0gru_17/transpose/perm:output:0*
T0*,
_output_shapes
:����������dR
gru_17/Shape_1Shapegru_17/transpose:y:0*
T0*
_output_shapes
:f
gru_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_1StridedSlicegru_17/Shape_1:output:0%gru_17/strided_slice_1/stack:output:0'gru_17/strided_slice_1/stack_1:output:0'gru_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/TensorArrayV2TensorListReserve+gru_17/TensorArrayV2/element_shape:output:0gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
.gru_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_17/transpose:y:0Egru_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_2StridedSlicegru_17/transpose:y:0%gru_17/strided_slice_2/stack:output:0'gru_17/strided_slice_2/stack_1:output:0'gru_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
!gru_17/gru_cell_32/ReadVariableOpReadVariableOp*gru_17_gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_17/gru_cell_32/unstackUnpack)gru_17/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
(gru_17/gru_cell_32/MatMul/ReadVariableOpReadVariableOp1gru_17_gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_17/gru_cell_32/MatMulMatMulgru_17/strided_slice_2:output:00gru_17/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/BiasAddBiasAdd#gru_17/gru_cell_32/MatMul:product:0#gru_17/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������m
"gru_17/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/gru_cell_32/splitSplit+gru_17/gru_cell_32/split/split_dim:output:0#gru_17/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
*gru_17/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp3gru_17_gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_17/gru_cell_32/MatMul_1MatMulgru_17/zeros:output:02gru_17/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/BiasAdd_1BiasAdd%gru_17/gru_cell_32/MatMul_1:product:0#gru_17/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������m
gru_17/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru_17/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/gru_cell_32/split_1SplitV%gru_17/gru_cell_32/BiasAdd_1:output:0!gru_17/gru_cell_32/Const:output:0-gru_17/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_17/gru_cell_32/addAddV2!gru_17/gru_cell_32/split:output:0#gru_17/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������s
gru_17/gru_cell_32/SigmoidSigmoidgru_17/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_1AddV2!gru_17/gru_cell_32/split:output:1#gru_17/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������w
gru_17/gru_cell_32/Sigmoid_1Sigmoidgru_17/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mulMul gru_17/gru_cell_32/Sigmoid_1:y:0#gru_17/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_2AddV2!gru_17/gru_cell_32/split:output:2gru_17/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������w
gru_17/gru_cell_32/SoftplusSoftplusgru_17/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mul_1Mulgru_17/gru_cell_32/Sigmoid:y:0gru_17/zeros:output:0*
T0*'
_output_shapes
:���������]
gru_17/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_17/gru_cell_32/subSub!gru_17/gru_cell_32/sub/x:output:0gru_17/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/mul_2Mulgru_17/gru_cell_32/sub:z:0)gru_17/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_17/gru_cell_32/add_3AddV2gru_17/gru_cell_32/mul_1:z:0gru_17/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������u
$gru_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
gru_17/TensorArrayV2_1TensorListReserve-gru_17/TensorArrayV2_1/element_shape:output:0gru_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_17/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_17/whileWhile"gru_17/while/loop_counter:output:0(gru_17/while/maximum_iterations:output:0gru_17/time:output:0gru_17/TensorArrayV2_1:handle:0gru_17/zeros:output:0gru_17/strided_slice_1:output:0>gru_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_17_gru_cell_32_readvariableop_resource1gru_17_gru_cell_32_matmul_readvariableop_resource3gru_17_gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_17_while_body_2906329*%
condR
gru_17_while_cond_2906328*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
7gru_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)gru_17/TensorArrayV2Stack/TensorListStackTensorListStackgru_17/while:output:3@gru_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0o
gru_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_17/strided_slice_3StridedSlice2gru_17/TensorArrayV2Stack/TensorListStack:tensor:0%gru_17/strided_slice_3/stack:output:0'gru_17/strided_slice_3/stack_1:output:0'gru_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskl
gru_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_17/transpose_1	Transpose2gru_17/TensorArrayV2Stack/TensorListStack:tensor:0 gru_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������b
gru_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_17/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp)^gru_15/gru_cell_30/MatMul/ReadVariableOp+^gru_15/gru_cell_30/MatMul_1/ReadVariableOp"^gru_15/gru_cell_30/ReadVariableOp^gru_15/while)^gru_16/gru_cell_31/MatMul/ReadVariableOp+^gru_16/gru_cell_31/MatMul_1/ReadVariableOp"^gru_16/gru_cell_31/ReadVariableOp^gru_16/while)^gru_17/gru_cell_32/MatMul/ReadVariableOp+^gru_17/gru_cell_32/MatMul_1/ReadVariableOp"^gru_17/gru_cell_32/ReadVariableOp^gru_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2T
(gru_15/gru_cell_30/MatMul/ReadVariableOp(gru_15/gru_cell_30/MatMul/ReadVariableOp2X
*gru_15/gru_cell_30/MatMul_1/ReadVariableOp*gru_15/gru_cell_30/MatMul_1/ReadVariableOp2F
!gru_15/gru_cell_30/ReadVariableOp!gru_15/gru_cell_30/ReadVariableOp2
gru_15/whilegru_15/while2T
(gru_16/gru_cell_31/MatMul/ReadVariableOp(gru_16/gru_cell_31/MatMul/ReadVariableOp2X
*gru_16/gru_cell_31/MatMul_1/ReadVariableOp*gru_16/gru_cell_31/MatMul_1/ReadVariableOp2F
!gru_16/gru_cell_31/ReadVariableOp!gru_16/gru_cell_31/ReadVariableOp2
gru_16/whilegru_16/while2T
(gru_17/gru_cell_32/MatMul/ReadVariableOp(gru_17/gru_cell_32/MatMul/ReadVariableOp2X
*gru_17/gru_cell_32/MatMul_1/ReadVariableOp*gru_17/gru_cell_32/MatMul_1/ReadVariableOp2F
!gru_17/gru_cell_32/ReadVariableOp!gru_17/gru_cell_32/ReadVariableOp2
gru_17/whilegru_17/while:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
while_body_2904006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_32_2904028_0:-
while_gru_cell_32_2904030_0:d-
while_gru_cell_32_2904032_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_32_2904028:+
while_gru_cell_32_2904030:d+
while_gru_cell_32_2904032:��)while/gru_cell_32/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
)while/gru_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_32_2904028_0while_gru_cell_32_2904030_0while_gru_cell_32_2904032_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2903993�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_32/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������x

while/NoOpNoOp*^while/gru_cell_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_32_2904028while_gru_cell_32_2904028_0"8
while_gru_cell_32_2904030while_gru_cell_32_2904030_0"8
while_gru_cell_32_2904032while_gru_cell_32_2904032_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2V
)while/gru_cell_32/StatefulPartitionedCall)while/gru_cell_32/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907271
inputs_06
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907182*
condR
while_cond_2907181*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908453

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:����������c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:����������R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:����������^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:����������Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:����������R
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:����������V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������W
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:����������W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0
�
�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908598

inputs
states_0*
readvariableop_resource:	�2
matmul_readvariableop_resource:
��3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������dQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������dU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�4
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2903576

inputs&
gru_cell_30_2903500:	�&
gru_cell_30_2903502:	�'
gru_cell_30_2903504:
��
identity��#gru_cell_30/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#gru_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_30_2903500gru_cell_30_2903502gru_cell_30_2903504*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903460n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_30_2903500gru_cell_30_2903502gru_cell_30_2903504*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2903512*
condR
while_cond_2903511*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:�������������������t
NoOpNoOp$^gru_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#gru_cell_30/StatefulPartitionedCall#gru_cell_30/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�=
�
while_body_2906985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�=
�
while_body_2907991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�F
�	
gru_15_while_body_2906031*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2)
%gru_15_while_gru_15_strided_slice_1_0e
agru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0E
2gru_15_while_gru_cell_30_readvariableop_resource_0:	�L
9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0:	�O
;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
gru_15_while_identity
gru_15_while_identity_1
gru_15_while_identity_2
gru_15_while_identity_3
gru_15_while_identity_4'
#gru_15_while_gru_15_strided_slice_1c
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensorC
0gru_15_while_gru_cell_30_readvariableop_resource:	�J
7gru_15_while_gru_cell_30_matmul_readvariableop_resource:	�M
9gru_15_while_gru_cell_30_matmul_1_readvariableop_resource:
����.gru_15/while/gru_cell_30/MatMul/ReadVariableOp�0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp�'gru_15/while/gru_cell_30/ReadVariableOp�
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0gru_15_while_placeholderGgru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'gru_15/while/gru_cell_30/ReadVariableOpReadVariableOp2gru_15_while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_15/while/gru_cell_30/unstackUnpack/gru_15/while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_15/while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
gru_15/while/gru_cell_30/MatMulMatMul7gru_15/while/TensorArrayV2Read/TensorListGetItem:item:06gru_15/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_15/while/gru_cell_30/BiasAddBiasAdd)gru_15/while/gru_cell_30/MatMul:product:0)gru_15/while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_15/while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/while/gru_cell_30/splitSplit1gru_15/while/gru_cell_30/split/split_dim:output:0)gru_15/while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
!gru_15/while/gru_cell_30/MatMul_1MatMulgru_15_while_placeholder_28gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/BiasAdd_1BiasAdd+gru_15/while/gru_cell_30/MatMul_1:product:0)gru_15/while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_15/while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����u
*gru_15/while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_15/while/gru_cell_30/split_1SplitV+gru_15/while/gru_cell_30/BiasAdd_1:output:0'gru_15/while/gru_cell_30/Const:output:03gru_15/while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_15/while/gru_cell_30/addAddV2'gru_15/while/gru_cell_30/split:output:0)gru_15/while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:�����������
 gru_15/while/gru_cell_30/SigmoidSigmoid gru_15/while/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_1AddV2'gru_15/while/gru_cell_30/split:output:1)gru_15/while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/Sigmoid_1Sigmoid"gru_15/while/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mulMul&gru_15/while/gru_cell_30/Sigmoid_1:y:0)gru_15/while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_2AddV2'gru_15/while/gru_cell_30/split:output:2 gru_15/while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/Sigmoid_2Sigmoid"gru_15/while/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mul_1Mul$gru_15/while/gru_cell_30/Sigmoid:y:0gru_15_while_placeholder_2*
T0*(
_output_shapes
:����������c
gru_15/while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_15/while/gru_cell_30/subSub'gru_15/while/gru_cell_30/sub/x:output:0$gru_15/while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mul_2Mul gru_15/while/gru_cell_30/sub:z:0&gru_15/while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_3AddV2"gru_15/while/gru_cell_30/mul_1:z:0"gru_15/while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
1gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_15_while_placeholder_1gru_15_while_placeholder"gru_15/while/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_15/while/addAddV2gru_15_while_placeholdergru_15/while/add/y:output:0*
T0*
_output_shapes
: V
gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_15/while/add_1AddV2&gru_15_while_gru_15_while_loop_countergru_15/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_15/while/IdentityIdentitygru_15/while/add_1:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_1Identity,gru_15_while_gru_15_while_maximum_iterations^gru_15/while/NoOp*
T0*
_output_shapes
: n
gru_15/while/Identity_2Identitygru_15/while/add:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_3IdentityAgru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_4Identity"gru_15/while/gru_cell_30/add_3:z:0^gru_15/while/NoOp*
T0*(
_output_shapes
:�����������
gru_15/while/NoOpNoOp/^gru_15/while/gru_cell_30/MatMul/ReadVariableOp1^gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp(^gru_15/while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_15_while_gru_15_strided_slice_1%gru_15_while_gru_15_strided_slice_1_0"x
9gru_15_while_gru_cell_30_matmul_1_readvariableop_resource;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0"t
7gru_15_while_gru_cell_30_matmul_readvariableop_resource9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0"f
0gru_15_while_gru_cell_30_readvariableop_resource2gru_15_while_gru_cell_30_readvariableop_resource_0"7
gru_15_while_identitygru_15/while/Identity:output:0";
gru_15_while_identity_1 gru_15/while/Identity_1:output:0";
gru_15_while_identity_2 gru_15/while/Identity_2:output:0";
gru_15_while_identity_3 gru_15/while/Identity_3:output:0";
gru_15_while_identity_4 gru_15/while/Identity_4:output:0"�
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensoragru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2`
.gru_15/while/gru_cell_30/MatMul/ReadVariableOp.gru_15/while/gru_cell_30/MatMul/ReadVariableOp2d
0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp2R
'gru_15/while/gru_cell_30/ReadVariableOp'gru_15/while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908665

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������Q
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:���������U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:���������V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905345

inputs!
gru_15_2905323:	�!
gru_15_2905325:	�"
gru_15_2905327:
��!
gru_16_2905330:	�"
gru_16_2905332:
��!
gru_16_2905334:	d� 
gru_17_2905337: 
gru_17_2905339:d 
gru_17_2905341:
identity��gru_15/StatefulPartitionedCall�gru_16/StatefulPartitionedCall�gru_17/StatefulPartitionedCall�
gru_15/StatefulPartitionedCallStatefulPartitionedCallinputsgru_15_2905323gru_15_2905325gru_15_2905327*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2905286�
gru_16/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0gru_16_2905330gru_16_2905332gru_16_2905334*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2905111�
gru_17/StatefulPartitionedCallStatefulPartitionedCall'gru_16/StatefulPartitionedCall:output:0gru_17_2905337gru_17_2905339gru_17_2905341*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904936{
IdentityIdentity'gru_17/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_15/StatefulPartitionedCall^gru_16/StatefulPartitionedCall^gru_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall2@
gru_17/StatefulPartitionedCallgru_17/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
while_cond_2903511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2903511___redundant_placeholder05
1while_while_cond_2903511___redundant_placeholder15
1while_while_cond_2903511___redundant_placeholder25
1while_while_cond_2903511___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2904420

inputs6
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904331*
condR
while_cond_2904330*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_gru_15_layer_call_fn_2906429
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2903394}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_2903667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2903667___redundant_placeholder05
1while_while_cond_2903667___redundant_placeholder15
1while_while_cond_2903667___redundant_placeholder25
1while_while_cond_2903667___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908233

inputs5
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2908144*
condR
while_cond_2908143*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
while_cond_2906984
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2906984___redundant_placeholder05
1while_while_cond_2906984___redundant_placeholder15
1while_while_cond_2906984___redundant_placeholder25
1while_while_cond_2906984___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_gru_16_layer_call_fn_2907085
inputs_0
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2903732|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
while_cond_2904846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904846___redundant_placeholder05
1while_while_cond_2904846___redundant_placeholder15
1while_while_cond_2904846___redundant_placeholder25
1while_while_cond_2904846___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2908296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2908296___redundant_placeholder05
1while_while_cond_2908296___redundant_placeholder15
1while_while_cond_2908296___redundant_placeholder25
1while_while_cond_2908296___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2904740

inputs5
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904651*
condR
while_cond_2904650*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
while_cond_2904187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904187___redundant_placeholder05
1while_while_cond_2904187___redundant_placeholder15
1while_while_cond_2904187___redundant_placeholder25
1while_while_cond_2904187___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908559

inputs
states_0*
readvariableop_resource:	�2
matmul_readvariableop_resource:
��3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������dQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������dU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905414
gru_15_input!
gru_15_2905392:	�!
gru_15_2905394:	�"
gru_15_2905396:
��!
gru_16_2905399:	�"
gru_16_2905401:
��!
gru_16_2905403:	d� 
gru_17_2905406: 
gru_17_2905408:d 
gru_17_2905410:
identity��gru_15/StatefulPartitionedCall�gru_16/StatefulPartitionedCall�gru_17/StatefulPartitionedCall�
gru_15/StatefulPartitionedCallStatefulPartitionedCallgru_15_inputgru_15_2905392gru_15_2905394gru_15_2905396*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2904420�
gru_16/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0gru_16_2905399gru_16_2905401gru_16_2905403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2904580�
gru_17/StatefulPartitionedCallStatefulPartitionedCall'gru_16/StatefulPartitionedCall:output:0gru_17_2905406gru_17_2905408gru_17_2905410*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904740{
IdentityIdentity'gru_17/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_15/StatefulPartitionedCall^gru_16/StatefulPartitionedCall^gru_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall2@
gru_17/StatefulPartitionedCallgru_17/StatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
�=
�
while_body_2904847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�=
�
while_body_2905197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2904330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904330___redundant_placeholder05
1while_while_cond_2904330___redundant_placeholder15
1while_while_cond_2904330___redundant_placeholder25
1while_while_cond_2904330___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
� 
�
while_body_2903330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_30_2903352_0:	�.
while_gru_cell_30_2903354_0:	�/
while_gru_cell_30_2903356_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_30_2903352:	�,
while_gru_cell_30_2903354:	�-
while_gru_cell_30_2903356:
����)while/gru_cell_30/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/gru_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_30_2903352_0while_gru_cell_30_2903354_0while_gru_cell_30_2903356_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903317�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:����������x

while/NoOpNoOp*^while/gru_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_30_2903352while_gru_cell_30_2903352_0"8
while_gru_cell_30_2903354while_gru_cell_30_2903354_0"8
while_gru_cell_30_2903356while_gru_cell_30_2903356_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2V
)while/gru_cell_30/StatefulPartitionedCall)while/gru_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�=
�
while_body_2908297
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906615
inputs_06
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2906526*
condR
while_cond_2906525*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�	
�
gru_17_while_cond_2906328*
&gru_17_while_gru_17_while_loop_counter0
,gru_17_while_gru_17_while_maximum_iterations
gru_17_while_placeholder
gru_17_while_placeholder_1
gru_17_while_placeholder_2,
(gru_17_while_less_gru_17_strided_slice_1C
?gru_17_while_gru_17_while_cond_2906328___redundant_placeholder0C
?gru_17_while_gru_17_while_cond_2906328___redundant_placeholder1C
?gru_17_while_gru_17_while_cond_2906328___redundant_placeholder2C
?gru_17_while_gru_17_while_cond_2906328___redundant_placeholder3
gru_17_while_identity
~
gru_17/while/LessLessgru_17_while_placeholder(gru_17_while_less_gru_17_strided_slice_1*
T0*
_output_shapes
: Y
gru_17/while/IdentityIdentitygru_17/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_17_while_identitygru_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2904749

inputs!
gru_15_2904421:	�!
gru_15_2904423:	�"
gru_15_2904425:
��!
gru_16_2904581:	�"
gru_16_2904583:
��!
gru_16_2904585:	d� 
gru_17_2904741: 
gru_17_2904743:d 
gru_17_2904745:
identity��gru_15/StatefulPartitionedCall�gru_16/StatefulPartitionedCall�gru_17/StatefulPartitionedCall�
gru_15/StatefulPartitionedCallStatefulPartitionedCallinputsgru_15_2904421gru_15_2904423gru_15_2904425*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2904420�
gru_16/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0gru_16_2904581gru_16_2904583gru_16_2904585*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2904580�
gru_17/StatefulPartitionedCallStatefulPartitionedCall'gru_16/StatefulPartitionedCall:output:0gru_17_2904741gru_17_2904743gru_17_2904745*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904740{
IdentityIdentity'gru_17/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_15/StatefulPartitionedCall^gru_16/StatefulPartitionedCall^gru_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall2@
gru_16/StatefulPartitionedCallgru_16/StatefulPartitionedCall2@
gru_17/StatefulPartitionedCallgru_17/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
gru_16_while_cond_2906179*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1C
?gru_16_while_gru_16_while_cond_2906179___redundant_placeholder0C
?gru_16_while_gru_16_while_cond_2906179___redundant_placeholder1C
?gru_16_while_gru_16_while_cond_2906179___redundant_placeholder2C
?gru_16_while_gru_16_while_cond_2906179___redundant_placeholder3
gru_16_while_identity
~
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: Y
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_16_while_identitygru_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
.__inference_sequential_5_layer_call_fn_2905516

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	d�
	unknown_5:
	unknown_6:d
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905345t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
while_cond_2904005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2904005___redundant_placeholder05
1while_while_cond_2904005___redundant_placeholder15
1while_while_cond_2904005___redundant_placeholder25
1while_while_cond_2904005___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2908143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2908143___redundant_placeholder05
1while_while_cond_2908143___redundant_placeholder15
1while_while_cond_2908143___redundant_placeholder25
1while_while_cond_2908143___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�V
�
&sequential_5_gru_15_while_body_2902860D
@sequential_5_gru_15_while_sequential_5_gru_15_while_loop_counterJ
Fsequential_5_gru_15_while_sequential_5_gru_15_while_maximum_iterations)
%sequential_5_gru_15_while_placeholder+
'sequential_5_gru_15_while_placeholder_1+
'sequential_5_gru_15_while_placeholder_2C
?sequential_5_gru_15_while_sequential_5_gru_15_strided_slice_1_0
{sequential_5_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_15_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_5_gru_15_while_gru_cell_30_readvariableop_resource_0:	�Y
Fsequential_5_gru_15_while_gru_cell_30_matmul_readvariableop_resource_0:	�\
Hsequential_5_gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0:
��&
"sequential_5_gru_15_while_identity(
$sequential_5_gru_15_while_identity_1(
$sequential_5_gru_15_while_identity_2(
$sequential_5_gru_15_while_identity_3(
$sequential_5_gru_15_while_identity_4A
=sequential_5_gru_15_while_sequential_5_gru_15_strided_slice_1}
ysequential_5_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_15_tensorarrayunstack_tensorlistfromtensorP
=sequential_5_gru_15_while_gru_cell_30_readvariableop_resource:	�W
Dsequential_5_gru_15_while_gru_cell_30_matmul_readvariableop_resource:	�Z
Fsequential_5_gru_15_while_gru_cell_30_matmul_1_readvariableop_resource:
����;sequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOp�=sequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp�4sequential_5/gru_15/while/gru_cell_30/ReadVariableOp�
Ksequential_5/gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_5/gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_15_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_gru_15_while_placeholderTsequential_5/gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
4sequential_5/gru_15/while/gru_cell_30/ReadVariableOpReadVariableOp?sequential_5_gru_15_while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
-sequential_5/gru_15/while/gru_cell_30/unstackUnpack<sequential_5/gru_15/while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
;sequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOpReadVariableOpFsequential_5_gru_15_while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
,sequential_5/gru_15/while/gru_cell_30/MatMulMatMulDsequential_5/gru_15/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_5/gru_15/while/gru_cell_30/BiasAddBiasAdd6sequential_5/gru_15/while/gru_cell_30/MatMul:product:06sequential_5/gru_15/while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:�����������
5sequential_5/gru_15/while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_5/gru_15/while/gru_cell_30/splitSplit>sequential_5/gru_15/while/gru_cell_30/split/split_dim:output:06sequential_5/gru_15/while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
=sequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOpHsequential_5_gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
.sequential_5/gru_15/while/gru_cell_30/MatMul_1MatMul'sequential_5_gru_15_while_placeholder_2Esequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/sequential_5/gru_15/while/gru_cell_30/BiasAdd_1BiasAdd8sequential_5/gru_15/while/gru_cell_30/MatMul_1:product:06sequential_5/gru_15/while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  �����
7sequential_5/gru_15/while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_5/gru_15/while/gru_cell_30/split_1SplitV8sequential_5/gru_15/while/gru_cell_30/BiasAdd_1:output:04sequential_5/gru_15/while/gru_cell_30/Const:output:0@sequential_5/gru_15/while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)sequential_5/gru_15/while/gru_cell_30/addAddV24sequential_5/gru_15/while/gru_cell_30/split:output:06sequential_5/gru_15/while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:�����������
-sequential_5/gru_15/while/gru_cell_30/SigmoidSigmoid-sequential_5/gru_15/while/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/add_1AddV24sequential_5/gru_15/while/gru_cell_30/split:output:16sequential_5/gru_15/while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:�����������
/sequential_5/gru_15/while/gru_cell_30/Sigmoid_1Sigmoid/sequential_5/gru_15/while/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
)sequential_5/gru_15/while/gru_cell_30/mulMul3sequential_5/gru_15/while/gru_cell_30/Sigmoid_1:y:06sequential_5/gru_15/while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/add_2AddV24sequential_5/gru_15/while/gru_cell_30/split:output:2-sequential_5/gru_15/while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:�����������
/sequential_5/gru_15/while/gru_cell_30/Sigmoid_2Sigmoid/sequential_5/gru_15/while/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/mul_1Mul1sequential_5/gru_15/while/gru_cell_30/Sigmoid:y:0'sequential_5_gru_15_while_placeholder_2*
T0*(
_output_shapes
:����������p
+sequential_5/gru_15/while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_5/gru_15/while/gru_cell_30/subSub4sequential_5/gru_15/while/gru_cell_30/sub/x:output:01sequential_5/gru_15/while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/mul_2Mul-sequential_5/gru_15/while/gru_cell_30/sub:z:03sequential_5/gru_15/while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
+sequential_5/gru_15/while/gru_cell_30/add_3AddV2/sequential_5/gru_15/while/gru_cell_30/mul_1:z:0/sequential_5/gru_15/while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
>sequential_5/gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_gru_15_while_placeholder_1%sequential_5_gru_15_while_placeholder/sequential_5/gru_15/while/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_5/gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_15/while/addAddV2%sequential_5_gru_15_while_placeholder(sequential_5/gru_15/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_5/gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_15/while/add_1AddV2@sequential_5_gru_15_while_sequential_5_gru_15_while_loop_counter*sequential_5/gru_15/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_5/gru_15/while/IdentityIdentity#sequential_5/gru_15/while/add_1:z:0^sequential_5/gru_15/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_15/while/Identity_1IdentityFsequential_5_gru_15_while_sequential_5_gru_15_while_maximum_iterations^sequential_5/gru_15/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_15/while/Identity_2Identity!sequential_5/gru_15/while/add:z:0^sequential_5/gru_15/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_15/while/Identity_3IdentityNsequential_5/gru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_5/gru_15/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_15/while/Identity_4Identity/sequential_5/gru_15/while/gru_cell_30/add_3:z:0^sequential_5/gru_15/while/NoOp*
T0*(
_output_shapes
:�����������
sequential_5/gru_15/while/NoOpNoOp<^sequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOp>^sequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp5^sequential_5/gru_15/while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_5_gru_15_while_gru_cell_30_matmul_1_readvariableop_resourceHsequential_5_gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0"�
Dsequential_5_gru_15_while_gru_cell_30_matmul_readvariableop_resourceFsequential_5_gru_15_while_gru_cell_30_matmul_readvariableop_resource_0"�
=sequential_5_gru_15_while_gru_cell_30_readvariableop_resource?sequential_5_gru_15_while_gru_cell_30_readvariableop_resource_0"Q
"sequential_5_gru_15_while_identity+sequential_5/gru_15/while/Identity:output:0"U
$sequential_5_gru_15_while_identity_1-sequential_5/gru_15/while/Identity_1:output:0"U
$sequential_5_gru_15_while_identity_2-sequential_5/gru_15/while/Identity_2:output:0"U
$sequential_5_gru_15_while_identity_3-sequential_5/gru_15/while/Identity_3:output:0"U
$sequential_5_gru_15_while_identity_4-sequential_5/gru_15/while/Identity_4:output:0"�
=sequential_5_gru_15_while_sequential_5_gru_15_strided_slice_1?sequential_5_gru_15_while_sequential_5_gru_15_strided_slice_1_0"�
ysequential_5_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_15_tensorarrayunstack_tensorlistfromtensor{sequential_5_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2z
;sequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOp;sequential_5/gru_15/while/gru_cell_30/MatMul/ReadVariableOp2~
=sequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp=sequential_5/gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp2l
4sequential_5/gru_15/while/gru_cell_30/ReadVariableOp4sequential_5/gru_15/while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�	
�
gru_16_while_cond_2905728*
&gru_16_while_gru_16_while_loop_counter0
,gru_16_while_gru_16_while_maximum_iterations
gru_16_while_placeholder
gru_16_while_placeholder_1
gru_16_while_placeholder_2,
(gru_16_while_less_gru_16_strided_slice_1C
?gru_16_while_gru_16_while_cond_2905728___redundant_placeholder0C
?gru_16_while_gru_16_while_cond_2905728___redundant_placeholder1C
?gru_16_while_gru_16_while_cond_2905728___redundant_placeholder2C
?gru_16_while_gru_16_while_cond_2905728___redundant_placeholder3
gru_16_while_identity
~
gru_16/while/LessLessgru_16_while_placeholder(gru_16_while_less_gru_16_strided_slice_1*
T0*
_output_shapes
: Y
gru_16/while/IdentityIdentitygru_16/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_16_while_identitygru_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�=
�
while_body_2907182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�=
�
while_body_2907641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2903914

inputs&
gru_cell_31_2903838:	�'
gru_cell_31_2903840:
��&
gru_cell_31_2903842:	d�
identity��#gru_cell_31/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
#gru_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_31_2903838gru_cell_31_2903840gru_cell_31_2903842*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903798n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_31_2903838gru_cell_31_2903840gru_cell_31_2903842*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2903850*
condR
while_cond_2903849*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������dt
NoOpNoOp$^gru_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#gru_cell_31/StatefulPartitionedCall#gru_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�=
�
while_body_2906526
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908704

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������Q
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:���������U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:���������V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�M
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908080
inputs_05
#gru_cell_32_readvariableop_resource:<
*gru_cell_32_matmul_readvariableop_resource:d>
,gru_cell_32_matmul_1_readvariableop_resource:
identity��!gru_cell_32/MatMul/ReadVariableOp�#gru_cell_32/MatMul_1/ReadVariableOp�gru_cell_32/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask~
gru_cell_32/ReadVariableOpReadVariableOp#gru_cell_32_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_32/unstackUnpack"gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_32/MatMul/ReadVariableOpReadVariableOp*gru_cell_32_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_32/MatMulMatMulstrided_slice_2:output:0)gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAddBiasAddgru_cell_32/MatMul:product:0gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/splitSplit$gru_cell_32/split/split_dim:output:0gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_32/MatMul_1MatMulzeros:output:0+gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_32/BiasAdd_1BiasAddgru_cell_32/MatMul_1:product:0gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_32/split_1SplitVgru_cell_32/BiasAdd_1:output:0gru_cell_32/Const:output:0&gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_32/addAddV2gru_cell_32/split:output:0gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_32/SigmoidSigmoidgru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/add_1AddV2gru_cell_32/split:output:1gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_32/Sigmoid_1Sigmoidgru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_32/mulMulgru_cell_32/Sigmoid_1:y:0gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_32/add_2AddV2gru_cell_32/split:output:2gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_32/SoftplusSoftplusgru_cell_32/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_32/mul_1Mulgru_cell_32/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_32/subSubgru_cell_32/sub/x:output:0gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_32/mul_2Mulgru_cell_32/sub:z:0"gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_32/add_3AddV2gru_cell_32/mul_1:z:0gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_32_readvariableop_resource*gru_cell_32_matmul_readvariableop_resource,gru_cell_32_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907991*
condR
while_cond_2907990*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp"^gru_cell_32/MatMul/ReadVariableOp$^gru_cell_32/MatMul_1/ReadVariableOp^gru_cell_32/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2F
!gru_cell_32/MatMul/ReadVariableOp!gru_cell_32/MatMul/ReadVariableOp2J
#gru_cell_32/MatMul_1/ReadVariableOp#gru_cell_32/MatMul_1/ReadVariableOp28
gru_cell_32/ReadVariableOpgru_cell_32/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
�
�
(__inference_gru_15_layer_call_fn_2906462

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_15_layer_call_and_return_conditional_losses_2905286u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�L
�
 __inference__traced_save_2908829
file_prefix8
4savev2_gru_15_gru_cell_30_kernel_read_readvariableopB
>savev2_gru_15_gru_cell_30_recurrent_kernel_read_readvariableop6
2savev2_gru_15_gru_cell_30_bias_read_readvariableop8
4savev2_gru_16_gru_cell_31_kernel_read_readvariableopB
>savev2_gru_16_gru_cell_31_recurrent_kernel_read_readvariableop6
2savev2_gru_16_gru_cell_31_bias_read_readvariableop8
4savev2_gru_17_gru_cell_32_kernel_read_readvariableopB
>savev2_gru_17_gru_cell_32_recurrent_kernel_read_readvariableop6
2savev2_gru_17_gru_cell_32_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_gru_15_gru_cell_30_kernel_m_read_readvariableopI
Esavev2_adam_gru_15_gru_cell_30_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_15_gru_cell_30_bias_m_read_readvariableop?
;savev2_adam_gru_16_gru_cell_31_kernel_m_read_readvariableopI
Esavev2_adam_gru_16_gru_cell_31_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_16_gru_cell_31_bias_m_read_readvariableop?
;savev2_adam_gru_17_gru_cell_32_kernel_m_read_readvariableopI
Esavev2_adam_gru_17_gru_cell_32_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_17_gru_cell_32_bias_m_read_readvariableop?
;savev2_adam_gru_15_gru_cell_30_kernel_v_read_readvariableopI
Esavev2_adam_gru_15_gru_cell_30_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_15_gru_cell_30_bias_v_read_readvariableop?
;savev2_adam_gru_16_gru_cell_31_kernel_v_read_readvariableopI
Esavev2_adam_gru_16_gru_cell_31_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_16_gru_cell_31_bias_v_read_readvariableop?
;savev2_adam_gru_17_gru_cell_32_kernel_v_read_readvariableopI
Esavev2_adam_gru_17_gru_cell_32_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_17_gru_cell_32_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_gru_15_gru_cell_30_kernel_read_readvariableop>savev2_gru_15_gru_cell_30_recurrent_kernel_read_readvariableop2savev2_gru_15_gru_cell_30_bias_read_readvariableop4savev2_gru_16_gru_cell_31_kernel_read_readvariableop>savev2_gru_16_gru_cell_31_recurrent_kernel_read_readvariableop2savev2_gru_16_gru_cell_31_bias_read_readvariableop4savev2_gru_17_gru_cell_32_kernel_read_readvariableop>savev2_gru_17_gru_cell_32_recurrent_kernel_read_readvariableop2savev2_gru_17_gru_cell_32_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_gru_15_gru_cell_30_kernel_m_read_readvariableopEsavev2_adam_gru_15_gru_cell_30_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_15_gru_cell_30_bias_m_read_readvariableop;savev2_adam_gru_16_gru_cell_31_kernel_m_read_readvariableopEsavev2_adam_gru_16_gru_cell_31_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_16_gru_cell_31_bias_m_read_readvariableop;savev2_adam_gru_17_gru_cell_32_kernel_m_read_readvariableopEsavev2_adam_gru_17_gru_cell_32_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_17_gru_cell_32_bias_m_read_readvariableop;savev2_adam_gru_15_gru_cell_30_kernel_v_read_readvariableopEsavev2_adam_gru_15_gru_cell_30_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_15_gru_cell_30_bias_v_read_readvariableop;savev2_adam_gru_16_gru_cell_31_kernel_v_read_readvariableopEsavev2_adam_gru_16_gru_cell_31_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_16_gru_cell_31_bias_v_read_readvariableop;savev2_adam_gru_17_gru_cell_32_kernel_v_read_readvariableopEsavev2_adam_gru_17_gru_cell_32_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_17_gru_cell_32_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:
��:	�:
��:	d�:	�:d::: : : : : : : :	�:
��:	�:
��:	d�:	�:d:::	�:
��:	�:
��:	d�:	�:d::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	d�:%!

_output_shapes
:	�:$ 

_output_shapes

:d:$ 

_output_shapes

::$	 

_output_shapes

::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	d�:%!

_output_shapes
:	�:$ 

_output_shapes

:d:$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	d�:%!

_output_shapes
:	�:$  

_output_shapes

:d:$! 

_output_shapes

::$" 

_output_shapes

::#

_output_shapes
: 
�
�
while_cond_2907990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907990___redundant_placeholder05
1while_while_cond_2907990___redundant_placeholder15
1while_while_cond_2907990___redundant_placeholder25
1while_while_cond_2907990___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�

�
%__inference_signature_wrapper_2905470
gru_15_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	d�
	unknown_5:
	unknown_6:d
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_2903247t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
�
�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2904136

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������Q
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:���������S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:���������V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�

�
.__inference_sequential_5_layer_call_fn_2904770
gru_15_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	d�
	unknown_5:
	unknown_6:d
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_2904749t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_15_input
� 
�
while_body_2904188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_32_2904210_0:-
while_gru_cell_32_2904212_0:d-
while_gru_cell_32_2904214_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_32_2904210:+
while_gru_cell_32_2904212:d+
while_gru_cell_32_2904214:��)while/gru_cell_32/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
)while/gru_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_32_2904210_0while_gru_cell_32_2904212_0while_gru_cell_32_2904214_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2904136�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_32/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������x

while/NoOpNoOp*^while/gru_cell_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_32_2904210while_gru_cell_32_2904210_0"8
while_gru_cell_32_2904212while_gru_cell_32_2904212_0"8
while_gru_cell_32_2904214while_gru_cell_32_2904214_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2V
)while/gru_cell_32/StatefulPartitionedCall)while/gru_cell_32/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
-__inference_gru_cell_30_layer_call_fn_2908400

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:
��
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903317p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0
�
�
(__inference_gru_16_layer_call_fn_2907107

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2904580t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�=
�
while_body_2906679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2903394

inputs&
gru_cell_30_2903318:	�&
gru_cell_30_2903320:	�'
gru_cell_30_2903322:
��
identity��#gru_cell_30/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#gru_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_30_2903318gru_cell_30_2903320gru_cell_30_2903322*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903317n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_30_2903318gru_cell_30_2903320gru_cell_30_2903322*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2903330*
condR
while_cond_2903329*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:�������������������t
NoOpNoOp$^gru_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#gru_cell_30/StatefulPartitionedCall#gru_cell_30/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�=
�
while_body_2908144
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_gru_17_layer_call_fn_2907741
inputs_0
unknown:
	unknown_0:d
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904070|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
�
�
while_cond_2903849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2903849___redundant_placeholder05
1while_while_cond_2903849___redundant_placeholder15
1while_while_cond_2903849___redundant_placeholder25
1while_while_cond_2903849___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_gru_17_layer_call_fn_2907752
inputs_0
unknown:
	unknown_0:d
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904252|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907730

inputs6
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:�����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2907641*
condR
while_cond_2907640*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:����������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
gru_15_while_cond_2906030*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2,
(gru_15_while_less_gru_15_strided_slice_1C
?gru_15_while_gru_15_while_cond_2906030___redundant_placeholder0C
?gru_15_while_gru_15_while_cond_2906030___redundant_placeholder1C
?gru_15_while_gru_15_while_cond_2906030___redundant_placeholder2C
?gru_15_while_gru_15_while_cond_2906030___redundant_placeholder3
gru_15_while_identity
~
gru_15/while/LessLessgru_15_while_placeholder(gru_15_while_less_gru_15_strided_slice_1*
T0*
_output_shapes
: Y
gru_15/while/IdentityIdentitygru_15/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_15_while_identitygru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_gru_17_layer_call_fn_2907774

inputs
unknown:
	unknown_0:d
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_17_layer_call_and_return_conditional_losses_2904936t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
while_cond_2903329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2903329___redundant_placeholder05
1while_while_cond_2903329___redundant_placeholder15
1while_while_cond_2903329___redundant_placeholder25
1while_while_cond_2903329___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�M
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2905111

inputs6
#gru_cell_31_readvariableop_resource:	�>
*gru_cell_31_matmul_readvariableop_resource:
��?
,gru_cell_31_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_31/MatMul/ReadVariableOp�#gru_cell_31/MatMul_1/ReadVariableOp�gru_cell_31/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:�����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask
gru_cell_31/ReadVariableOpReadVariableOp#gru_cell_31_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_31/unstackUnpack"gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_31/MatMul/ReadVariableOpReadVariableOp*gru_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_31/MatMulMatMulstrided_slice_2:output:0)gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAddBiasAddgru_cell_31/MatMul:product:0gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/splitSplit$gru_cell_31/split/split_dim:output:0gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_31/MatMul_1MatMulzeros:output:0+gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_31/BiasAdd_1BiasAddgru_cell_31/MatMul_1:product:0gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_31/split_1SplitVgru_cell_31/BiasAdd_1:output:0gru_cell_31/Const:output:0&gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_31/addAddV2gru_cell_31/split:output:0gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_31/SigmoidSigmoidgru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/add_1AddV2gru_cell_31/split:output:1gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_1Sigmoidgru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_31/mulMulgru_cell_31/Sigmoid_1:y:0gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_31/add_2AddV2gru_cell_31/split:output:2gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_31/Sigmoid_2Sigmoidgru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_31/mul_1Mulgru_cell_31/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_31/subSubgru_cell_31/sub/x:output:0gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/mul_2Mulgru_cell_31/sub:z:0gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_31/add_3AddV2gru_cell_31/mul_1:z:0gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_31_readvariableop_resource*gru_cell_31_matmul_readvariableop_resource,gru_cell_31_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2905022*
condR
while_cond_2905021*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:����������d�
NoOpNoOp"^gru_cell_31/MatMul/ReadVariableOp$^gru_cell_31/MatMul_1/ReadVariableOp^gru_cell_31/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_31/MatMul/ReadVariableOp!gru_cell_31/MatMul/ReadVariableOp2J
#gru_cell_31/MatMul_1/ReadVariableOp#gru_cell_31/MatMul_1/ReadVariableOp28
gru_cell_31/ReadVariableOpgru_cell_31/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�4
�
C__inference_gru_17_layer_call_and_return_conditional_losses_2904252

inputs%
gru_cell_32_2904176:%
gru_cell_32_2904178:d%
gru_cell_32_2904180:
identity��#gru_cell_32/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������dD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
#gru_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_32_2904176gru_cell_32_2904178gru_cell_32_2904180*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2904136n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_32_2904176gru_cell_32_2904178gru_cell_32_2904180*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2904188*
condR
while_cond_2904187*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������t
NoOpNoOp$^gru_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2J
#gru_cell_32/StatefulPartitionedCall#gru_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906768
inputs_06
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2906679*
condR
while_cond_2906678*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�E
�	
gru_17_while_body_2906329*
&gru_17_while_gru_17_while_loop_counter0
,gru_17_while_gru_17_while_maximum_iterations
gru_17_while_placeholder
gru_17_while_placeholder_1
gru_17_while_placeholder_2)
%gru_17_while_gru_17_strided_slice_1_0e
agru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0D
2gru_17_while_gru_cell_32_readvariableop_resource_0:K
9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0:dM
;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0:
gru_17_while_identity
gru_17_while_identity_1
gru_17_while_identity_2
gru_17_while_identity_3
gru_17_while_identity_4'
#gru_17_while_gru_17_strided_slice_1c
_gru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensorB
0gru_17_while_gru_cell_32_readvariableop_resource:I
7gru_17_while_gru_cell_32_matmul_readvariableop_resource:dK
9gru_17_while_gru_cell_32_matmul_1_readvariableop_resource:��.gru_17/while/gru_cell_32/MatMul/ReadVariableOp�0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp�'gru_17/while/gru_cell_32/ReadVariableOp�
>gru_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
0gru_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0gru_17_while_placeholderGgru_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
'gru_17/while/gru_cell_32/ReadVariableOpReadVariableOp2gru_17_while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 gru_17/while/gru_cell_32/unstackUnpack/gru_17/while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
.gru_17/while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
gru_17/while/gru_cell_32/MatMulMatMul7gru_17/while/TensorArrayV2Read/TensorListGetItem:item:06gru_17/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 gru_17/while/gru_cell_32/BiasAddBiasAdd)gru_17/while/gru_cell_32/MatMul:product:0)gru_17/while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������s
(gru_17/while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_17/while/gru_cell_32/splitSplit1gru_17/while/gru_cell_32/split/split_dim:output:0)gru_17/while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
!gru_17/while/gru_cell_32/MatMul_1MatMulgru_17_while_placeholder_28gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"gru_17/while/gru_cell_32/BiasAdd_1BiasAdd+gru_17/while/gru_cell_32/MatMul_1:product:0)gru_17/while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������s
gru_17/while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����u
*gru_17/while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_17/while/gru_cell_32/split_1SplitV+gru_17/while/gru_cell_32/BiasAdd_1:output:0'gru_17/while/gru_cell_32/Const:output:03gru_17/while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_17/while/gru_cell_32/addAddV2'gru_17/while/gru_cell_32/split:output:0)gru_17/while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������
 gru_17/while/gru_cell_32/SigmoidSigmoid gru_17/while/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_1AddV2'gru_17/while/gru_cell_32/split:output:1)gru_17/while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:����������
"gru_17/while/gru_cell_32/Sigmoid_1Sigmoid"gru_17/while/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mulMul&gru_17/while/gru_cell_32/Sigmoid_1:y:0)gru_17/while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_2AddV2'gru_17/while/gru_cell_32/split:output:2 gru_17/while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:����������
!gru_17/while/gru_cell_32/SoftplusSoftplus"gru_17/while/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mul_1Mul$gru_17/while/gru_cell_32/Sigmoid:y:0gru_17_while_placeholder_2*
T0*'
_output_shapes
:���������c
gru_17/while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_17/while/gru_cell_32/subSub'gru_17/while/gru_cell_32/sub/x:output:0$gru_17/while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/mul_2Mul gru_17/while/gru_cell_32/sub:z:0/gru_17/while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_17/while/gru_cell_32/add_3AddV2"gru_17/while/gru_cell_32/mul_1:z:0"gru_17/while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
1gru_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_17_while_placeholder_1gru_17_while_placeholder"gru_17/while/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_17/while/addAddV2gru_17_while_placeholdergru_17/while/add/y:output:0*
T0*
_output_shapes
: V
gru_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_17/while/add_1AddV2&gru_17_while_gru_17_while_loop_countergru_17/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_17/while/IdentityIdentitygru_17/while/add_1:z:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_1Identity,gru_17_while_gru_17_while_maximum_iterations^gru_17/while/NoOp*
T0*
_output_shapes
: n
gru_17/while/Identity_2Identitygru_17/while/add:z:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_3IdentityAgru_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_17/while/NoOp*
T0*
_output_shapes
: �
gru_17/while/Identity_4Identity"gru_17/while/gru_cell_32/add_3:z:0^gru_17/while/NoOp*
T0*'
_output_shapes
:����������
gru_17/while/NoOpNoOp/^gru_17/while/gru_cell_32/MatMul/ReadVariableOp1^gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp(^gru_17/while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_17_while_gru_17_strided_slice_1%gru_17_while_gru_17_strided_slice_1_0"x
9gru_17_while_gru_cell_32_matmul_1_readvariableop_resource;gru_17_while_gru_cell_32_matmul_1_readvariableop_resource_0"t
7gru_17_while_gru_cell_32_matmul_readvariableop_resource9gru_17_while_gru_cell_32_matmul_readvariableop_resource_0"f
0gru_17_while_gru_cell_32_readvariableop_resource2gru_17_while_gru_cell_32_readvariableop_resource_0"7
gru_17_while_identitygru_17/while/Identity:output:0";
gru_17_while_identity_1 gru_17/while/Identity_1:output:0";
gru_17_while_identity_2 gru_17/while/Identity_2:output:0";
gru_17_while_identity_3 gru_17/while/Identity_3:output:0";
gru_17_while_identity_4 gru_17/while/Identity_4:output:0"�
_gru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensoragru_17_while_tensorarrayv2read_tensorlistgetitem_gru_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.gru_17/while/gru_cell_32/MatMul/ReadVariableOp.gru_17/while/gru_cell_32/MatMul/ReadVariableOp2d
0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp0gru_17/while/gru_cell_32/MatMul_1/ReadVariableOp2R
'gru_17/while/gru_cell_32/ReadVariableOp'gru_17/while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
� 
�
while_body_2903512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_30_2903534_0:	�.
while_gru_cell_30_2903536_0:	�/
while_gru_cell_30_2903538_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_30_2903534:	�,
while_gru_cell_30_2903536:	�-
while_gru_cell_30_2903538:
����)while/gru_cell_30/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/gru_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_30_2903534_0while_gru_cell_30_2903536_0while_gru_cell_30_2903538_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2903460�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:����������x

while/NoOpNoOp*^while/gru_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_30_2903534while_gru_cell_30_2903534_0"8
while_gru_cell_30_2903536while_gru_cell_30_2903536_0"8
while_gru_cell_30_2903538while_gru_cell_30_2903538_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2V
)while/gru_cell_30/StatefulPartitionedCall)while/gru_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908492

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:����������c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:����������R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:����������^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:����������Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:����������R
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:����������V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������W
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:����������W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������:����������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0
�=
�
while_body_2904491
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_31_readvariableop_resource_0:	�F
2while_gru_cell_31_matmul_readvariableop_resource_0:
��G
4while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_31_readvariableop_resource:	�D
0while_gru_cell_31_matmul_readvariableop_resource:
��E
2while_gru_cell_31_matmul_1_readvariableop_resource:	d���'while/gru_cell_31/MatMul/ReadVariableOp�)while/gru_cell_31/MatMul_1/ReadVariableOp� while/gru_cell_31/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_31/ReadVariableOpReadVariableOp+while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_31/unstackUnpack(while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_31/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAddBiasAdd"while/gru_cell_31/MatMul:product:0"while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/splitSplit*while/gru_cell_31/split/split_dim:output:0"while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_31/MatMul_1MatMulwhile_placeholder_21while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_31/BiasAdd_1BiasAdd$while/gru_cell_31/MatMul_1:product:0"while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_31/split_1SplitV$while/gru_cell_31/BiasAdd_1:output:0 while/gru_cell_31/Const:output:0,while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_31/addAddV2 while/gru_cell_31/split:output:0"while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_31/SigmoidSigmoidwhile/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_1AddV2 while/gru_cell_31/split:output:1"while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_1Sigmoidwhile/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mulMulwhile/gru_cell_31/Sigmoid_1:y:0"while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_2AddV2 while/gru_cell_31/split:output:2while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_31/Sigmoid_2Sigmoidwhile/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_1Mulwhile/gru_cell_31/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_31/subSub while/gru_cell_31/sub/x:output:0while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/mul_2Mulwhile/gru_cell_31/sub:z:0while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_31/add_3AddV2while/gru_cell_31/mul_1:z:0while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_31/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_31/MatMul/ReadVariableOp*^while/gru_cell_31/MatMul_1/ReadVariableOp!^while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_31_matmul_1_readvariableop_resource4while_gru_cell_31_matmul_1_readvariableop_resource_0"f
0while_gru_cell_31_matmul_readvariableop_resource2while_gru_cell_31_matmul_readvariableop_resource_0"X
)while_gru_cell_31_readvariableop_resource+while_gru_cell_31_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2R
'while/gru_cell_31/MatMul/ReadVariableOp'while/gru_cell_31/MatMul/ReadVariableOp2V
)while/gru_cell_31/MatMul_1/ReadVariableOp)while/gru_cell_31/MatMul_1/ReadVariableOp2D
 while/gru_cell_31/ReadVariableOp while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�	
�
gru_17_while_cond_2905877*
&gru_17_while_gru_17_while_loop_counter0
,gru_17_while_gru_17_while_maximum_iterations
gru_17_while_placeholder
gru_17_while_placeholder_1
gru_17_while_placeholder_2,
(gru_17_while_less_gru_17_strided_slice_1C
?gru_17_while_gru_17_while_cond_2905877___redundant_placeholder0C
?gru_17_while_gru_17_while_cond_2905877___redundant_placeholder1C
?gru_17_while_gru_17_while_cond_2905877___redundant_placeholder2C
?gru_17_while_gru_17_while_cond_2905877___redundant_placeholder3
gru_17_while_identity
~
gru_17/while/LessLessgru_17_while_placeholder(gru_17_while_less_gru_17_strided_slice_1*
T0*
_output_shapes
: Y
gru_17/while/IdentityIdentitygru_17/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_17_while_identitygru_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�=
�
while_body_2904331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_30_readvariableop_resource_0:	�E
2while_gru_cell_30_matmul_readvariableop_resource_0:	�H
4while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_30_readvariableop_resource:	�C
0while_gru_cell_30_matmul_readvariableop_resource:	�F
2while_gru_cell_30_matmul_1_readvariableop_resource:
����'while/gru_cell_30/MatMul/ReadVariableOp�)while/gru_cell_30/MatMul_1/ReadVariableOp� while/gru_cell_30/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_30/ReadVariableOpReadVariableOp+while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/unstackUnpack(while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAddBiasAdd"while/gru_cell_30/MatMul:product:0"while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/splitSplit*while/gru_cell_30/split/split_dim:output:0"while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_30/MatMul_1MatMulwhile_placeholder_21while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/BiasAdd_1BiasAdd$while/gru_cell_30/MatMul_1:product:0"while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_30/split_1SplitV$while/gru_cell_30/BiasAdd_1:output:0 while/gru_cell_30/Const:output:0,while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_30/addAddV2 while/gru_cell_30/split:output:0"while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_30/SigmoidSigmoidwhile/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_1AddV2 while/gru_cell_30/split:output:1"while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_1Sigmoidwhile/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mulMulwhile/gru_cell_30/Sigmoid_1:y:0"while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_2AddV2 while/gru_cell_30/split:output:2while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_30/Sigmoid_2Sigmoidwhile/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_1Mulwhile/gru_cell_30/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_30/subSub while/gru_cell_30/sub/x:output:0while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/mul_2Mulwhile/gru_cell_30/sub:z:0while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_30/add_3AddV2while/gru_cell_30/mul_1:z:0while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_30/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_30/MatMul/ReadVariableOp*^while/gru_cell_30/MatMul_1/ReadVariableOp!^while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_30_matmul_1_readvariableop_resource4while_gru_cell_30_matmul_1_readvariableop_resource_0"f
0while_gru_cell_30_matmul_readvariableop_resource2while_gru_cell_30_matmul_readvariableop_resource_0"X
)while_gru_cell_30_readvariableop_resource+while_gru_cell_30_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2R
'while/gru_cell_30/MatMul/ReadVariableOp'while/gru_cell_30/MatMul/ReadVariableOp2V
)while/gru_cell_30/MatMul_1/ReadVariableOp)while/gru_cell_30/MatMul_1/ReadVariableOp2D
 while/gru_cell_30/ReadVariableOp while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_gru_16_layer_call_and_return_conditional_losses_2903732

inputs&
gru_cell_31_2903656:	�'
gru_cell_31_2903658:
��&
gru_cell_31_2903660:	d�
identity��#gru_cell_31/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
#gru_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_31_2903656gru_cell_31_2903658gru_cell_31_2903660*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903655n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_31_2903656gru_cell_31_2903658gru_cell_31_2903660*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2903668*
condR
while_cond_2903667*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������dt
NoOpNoOp$^gru_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#gru_cell_31/StatefulPartitionedCall#gru_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�M
�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906921

inputs6
#gru_cell_30_readvariableop_resource:	�=
*gru_cell_30_matmul_readvariableop_resource:	�@
,gru_cell_30_matmul_1_readvariableop_resource:
��
identity��!gru_cell_30/MatMul/ReadVariableOp�#gru_cell_30/MatMul_1/ReadVariableOp�gru_cell_30/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
gru_cell_30/ReadVariableOpReadVariableOp#gru_cell_30_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_30/unstackUnpack"gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_30/MatMul/ReadVariableOpReadVariableOp*gru_cell_30_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_30/MatMulMatMulstrided_slice_2:output:0)gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAddBiasAddgru_cell_30/MatMul:product:0gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/splitSplit$gru_cell_30/split/split_dim:output:0gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_30/MatMul_1MatMulzeros:output:0+gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_30/BiasAdd_1BiasAddgru_cell_30/MatMul_1:product:0gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_30/split_1SplitVgru_cell_30/BiasAdd_1:output:0gru_cell_30/Const:output:0&gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_30/addAddV2gru_cell_30/split:output:0gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_30/SigmoidSigmoidgru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/add_1AddV2gru_cell_30/split:output:1gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_1Sigmoidgru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_30/mulMulgru_cell_30/Sigmoid_1:y:0gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_30/add_2AddV2gru_cell_30/split:output:2gru_cell_30/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_30/Sigmoid_2Sigmoidgru_cell_30/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_30/mul_1Mulgru_cell_30/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_30/subSubgru_cell_30/sub/x:output:0gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/mul_2Mulgru_cell_30/sub:z:0gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_30/add_3AddV2gru_cell_30/mul_1:z:0gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_30_readvariableop_resource*gru_cell_30_matmul_readvariableop_resource,gru_cell_30_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :����������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2906832*
condR
while_cond_2906831*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp"^gru_cell_30/MatMul/ReadVariableOp$^gru_cell_30/MatMul_1/ReadVariableOp^gru_cell_30/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_30/MatMul/ReadVariableOp!gru_cell_30/MatMul/ReadVariableOp2J
#gru_cell_30/MatMul_1/ReadVariableOp#gru_cell_30/MatMul_1/ReadVariableOp28
gru_cell_30/ReadVariableOpgru_cell_30/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�V
�
&sequential_5_gru_16_while_body_2903009D
@sequential_5_gru_16_while_sequential_5_gru_16_while_loop_counterJ
Fsequential_5_gru_16_while_sequential_5_gru_16_while_maximum_iterations)
%sequential_5_gru_16_while_placeholder+
'sequential_5_gru_16_while_placeholder_1+
'sequential_5_gru_16_while_placeholder_2C
?sequential_5_gru_16_while_sequential_5_gru_16_strided_slice_1_0
{sequential_5_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_16_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_5_gru_16_while_gru_cell_31_readvariableop_resource_0:	�Z
Fsequential_5_gru_16_while_gru_cell_31_matmul_readvariableop_resource_0:
��[
Hsequential_5_gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0:	d�&
"sequential_5_gru_16_while_identity(
$sequential_5_gru_16_while_identity_1(
$sequential_5_gru_16_while_identity_2(
$sequential_5_gru_16_while_identity_3(
$sequential_5_gru_16_while_identity_4A
=sequential_5_gru_16_while_sequential_5_gru_16_strided_slice_1}
ysequential_5_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_16_tensorarrayunstack_tensorlistfromtensorP
=sequential_5_gru_16_while_gru_cell_31_readvariableop_resource:	�X
Dsequential_5_gru_16_while_gru_cell_31_matmul_readvariableop_resource:
��Y
Fsequential_5_gru_16_while_gru_cell_31_matmul_1_readvariableop_resource:	d���;sequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOp�=sequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp�4sequential_5/gru_16/while/gru_cell_31/ReadVariableOp�
Ksequential_5/gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
=sequential_5/gru_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_16_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_gru_16_while_placeholderTsequential_5/gru_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
4sequential_5/gru_16/while/gru_cell_31/ReadVariableOpReadVariableOp?sequential_5_gru_16_while_gru_cell_31_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
-sequential_5/gru_16/while/gru_cell_31/unstackUnpack<sequential_5/gru_16/while/gru_cell_31/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
;sequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOpReadVariableOpFsequential_5_gru_16_while_gru_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
,sequential_5/gru_16/while/gru_cell_31/MatMulMatMulDsequential_5/gru_16/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_5/gru_16/while/gru_cell_31/BiasAddBiasAdd6sequential_5/gru_16/while/gru_cell_31/MatMul:product:06sequential_5/gru_16/while/gru_cell_31/unstack:output:0*
T0*(
_output_shapes
:�����������
5sequential_5/gru_16/while/gru_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_5/gru_16/while/gru_cell_31/splitSplit>sequential_5/gru_16/while/gru_cell_31/split/split_dim:output:06sequential_5/gru_16/while/gru_cell_31/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
=sequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOpReadVariableOpHsequential_5_gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
.sequential_5/gru_16/while/gru_cell_31/MatMul_1MatMul'sequential_5_gru_16_while_placeholder_2Esequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/sequential_5/gru_16/while/gru_cell_31/BiasAdd_1BiasAdd8sequential_5/gru_16/while/gru_cell_31/MatMul_1:product:06sequential_5/gru_16/while/gru_cell_31/unstack:output:1*
T0*(
_output_shapes
:�����������
+sequential_5/gru_16/while/gru_cell_31/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   �����
7sequential_5/gru_16/while/gru_cell_31/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_5/gru_16/while/gru_cell_31/split_1SplitV8sequential_5/gru_16/while/gru_cell_31/BiasAdd_1:output:04sequential_5/gru_16/while/gru_cell_31/Const:output:0@sequential_5/gru_16/while/gru_cell_31/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)sequential_5/gru_16/while/gru_cell_31/addAddV24sequential_5/gru_16/while/gru_cell_31/split:output:06sequential_5/gru_16/while/gru_cell_31/split_1:output:0*
T0*'
_output_shapes
:���������d�
-sequential_5/gru_16/while/gru_cell_31/SigmoidSigmoid-sequential_5/gru_16/while/gru_cell_31/add:z:0*
T0*'
_output_shapes
:���������d�
+sequential_5/gru_16/while/gru_cell_31/add_1AddV24sequential_5/gru_16/while/gru_cell_31/split:output:16sequential_5/gru_16/while/gru_cell_31/split_1:output:1*
T0*'
_output_shapes
:���������d�
/sequential_5/gru_16/while/gru_cell_31/Sigmoid_1Sigmoid/sequential_5/gru_16/while/gru_cell_31/add_1:z:0*
T0*'
_output_shapes
:���������d�
)sequential_5/gru_16/while/gru_cell_31/mulMul3sequential_5/gru_16/while/gru_cell_31/Sigmoid_1:y:06sequential_5/gru_16/while/gru_cell_31/split_1:output:2*
T0*'
_output_shapes
:���������d�
+sequential_5/gru_16/while/gru_cell_31/add_2AddV24sequential_5/gru_16/while/gru_cell_31/split:output:2-sequential_5/gru_16/while/gru_cell_31/mul:z:0*
T0*'
_output_shapes
:���������d�
/sequential_5/gru_16/while/gru_cell_31/Sigmoid_2Sigmoid/sequential_5/gru_16/while/gru_cell_31/add_2:z:0*
T0*'
_output_shapes
:���������d�
+sequential_5/gru_16/while/gru_cell_31/mul_1Mul1sequential_5/gru_16/while/gru_cell_31/Sigmoid:y:0'sequential_5_gru_16_while_placeholder_2*
T0*'
_output_shapes
:���������dp
+sequential_5/gru_16/while/gru_cell_31/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_5/gru_16/while/gru_cell_31/subSub4sequential_5/gru_16/while/gru_cell_31/sub/x:output:01sequential_5/gru_16/while/gru_cell_31/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
+sequential_5/gru_16/while/gru_cell_31/mul_2Mul-sequential_5/gru_16/while/gru_cell_31/sub:z:03sequential_5/gru_16/while/gru_cell_31/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
+sequential_5/gru_16/while/gru_cell_31/add_3AddV2/sequential_5/gru_16/while/gru_cell_31/mul_1:z:0/sequential_5/gru_16/while/gru_cell_31/mul_2:z:0*
T0*'
_output_shapes
:���������d�
>sequential_5/gru_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_gru_16_while_placeholder_1%sequential_5_gru_16_while_placeholder/sequential_5/gru_16/while/gru_cell_31/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_5/gru_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_16/while/addAddV2%sequential_5_gru_16_while_placeholder(sequential_5/gru_16/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_5/gru_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_5/gru_16/while/add_1AddV2@sequential_5_gru_16_while_sequential_5_gru_16_while_loop_counter*sequential_5/gru_16/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_5/gru_16/while/IdentityIdentity#sequential_5/gru_16/while/add_1:z:0^sequential_5/gru_16/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_16/while/Identity_1IdentityFsequential_5_gru_16_while_sequential_5_gru_16_while_maximum_iterations^sequential_5/gru_16/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_16/while/Identity_2Identity!sequential_5/gru_16/while/add:z:0^sequential_5/gru_16/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_16/while/Identity_3IdentityNsequential_5/gru_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_5/gru_16/while/NoOp*
T0*
_output_shapes
: �
$sequential_5/gru_16/while/Identity_4Identity/sequential_5/gru_16/while/gru_cell_31/add_3:z:0^sequential_5/gru_16/while/NoOp*
T0*'
_output_shapes
:���������d�
sequential_5/gru_16/while/NoOpNoOp<^sequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOp>^sequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp5^sequential_5/gru_16/while/gru_cell_31/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_5_gru_16_while_gru_cell_31_matmul_1_readvariableop_resourceHsequential_5_gru_16_while_gru_cell_31_matmul_1_readvariableop_resource_0"�
Dsequential_5_gru_16_while_gru_cell_31_matmul_readvariableop_resourceFsequential_5_gru_16_while_gru_cell_31_matmul_readvariableop_resource_0"�
=sequential_5_gru_16_while_gru_cell_31_readvariableop_resource?sequential_5_gru_16_while_gru_cell_31_readvariableop_resource_0"Q
"sequential_5_gru_16_while_identity+sequential_5/gru_16/while/Identity:output:0"U
$sequential_5_gru_16_while_identity_1-sequential_5/gru_16/while/Identity_1:output:0"U
$sequential_5_gru_16_while_identity_2-sequential_5/gru_16/while/Identity_2:output:0"U
$sequential_5_gru_16_while_identity_3-sequential_5/gru_16/while/Identity_3:output:0"U
$sequential_5_gru_16_while_identity_4-sequential_5/gru_16/while/Identity_4:output:0"�
=sequential_5_gru_16_while_sequential_5_gru_16_strided_slice_1?sequential_5_gru_16_while_sequential_5_gru_16_strided_slice_1_0"�
ysequential_5_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_16_tensorarrayunstack_tensorlistfromtensor{sequential_5_gru_16_while_tensorarrayv2read_tensorlistgetitem_sequential_5_gru_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2z
;sequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOp;sequential_5/gru_16/while/gru_cell_31/MatMul/ReadVariableOp2~
=sequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp=sequential_5/gru_16/while/gru_cell_31/MatMul_1/ReadVariableOp2l
4sequential_5/gru_16/while/gru_cell_31/ReadVariableOp4sequential_5/gru_16/while/gru_cell_31/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
� 
�
while_body_2903668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_31_2903690_0:	�/
while_gru_cell_31_2903692_0:
��.
while_gru_cell_31_2903694_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_31_2903690:	�-
while_gru_cell_31_2903692:
��,
while_gru_cell_31_2903694:	d���)while/gru_cell_31/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
)while/gru_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_31_2903690_0while_gru_cell_31_2903692_0while_gru_cell_31_2903694_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2903655�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity2while/gru_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������dx

while/NoOpNoOp*^while/gru_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_31_2903690while_gru_cell_31_2903690_0"8
while_gru_cell_31_2903692while_gru_cell_31_2903692_0"8
while_gru_cell_31_2903694while_gru_cell_31_2903694_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2V
)while/gru_cell_31/StatefulPartitionedCall)while/gru_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2906831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2906831___redundant_placeholder05
1while_while_cond_2906831___redundant_placeholder15
1while_while_cond_2906831___redundant_placeholder25
1while_while_cond_2906831___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2903993

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������Q
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:���������S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:���������V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�

�
-__inference_gru_cell_32_layer_call_fn_2908626

inputs
states_0
unknown:
	unknown_0:d
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2904136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������d:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�
�
&sequential_5_gru_15_while_cond_2902859D
@sequential_5_gru_15_while_sequential_5_gru_15_while_loop_counterJ
Fsequential_5_gru_15_while_sequential_5_gru_15_while_maximum_iterations)
%sequential_5_gru_15_while_placeholder+
'sequential_5_gru_15_while_placeholder_1+
'sequential_5_gru_15_while_placeholder_2F
Bsequential_5_gru_15_while_less_sequential_5_gru_15_strided_slice_1]
Ysequential_5_gru_15_while_sequential_5_gru_15_while_cond_2902859___redundant_placeholder0]
Ysequential_5_gru_15_while_sequential_5_gru_15_while_cond_2902859___redundant_placeholder1]
Ysequential_5_gru_15_while_sequential_5_gru_15_while_cond_2902859___redundant_placeholder2]
Ysequential_5_gru_15_while_sequential_5_gru_15_while_cond_2902859___redundant_placeholder3&
"sequential_5_gru_15_while_identity
�
sequential_5/gru_15/while/LessLess%sequential_5_gru_15_while_placeholderBsequential_5_gru_15_while_less_sequential_5_gru_15_strided_slice_1*
T0*
_output_shapes
: s
"sequential_5/gru_15/while/IdentityIdentity"sequential_5/gru_15/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_5_gru_15_while_identity+sequential_5/gru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�

�
.__inference_sequential_5_layer_call_fn_2905493

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	d�
	unknown_5:
	unknown_6:d
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_2904749t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
while_body_2904651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_gru_16_layer_call_fn_2907096
inputs_0
unknown:	�
	unknown_0:
��
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_16_layer_call_and_return_conditional_losses_2903914|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�=
�
while_body_2907838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_32_readvariableop_resource_0:D
2while_gru_cell_32_matmul_readvariableop_resource_0:dF
4while_gru_cell_32_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_32_readvariableop_resource:B
0while_gru_cell_32_matmul_readvariableop_resource:dD
2while_gru_cell_32_matmul_1_readvariableop_resource:��'while/gru_cell_32/MatMul/ReadVariableOp�)while/gru_cell_32/MatMul_1/ReadVariableOp� while/gru_cell_32/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_32/ReadVariableOpReadVariableOp+while_gru_cell_32_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/unstackUnpack(while/gru_cell_32/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_32/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAddBiasAdd"while/gru_cell_32/MatMul:product:0"while/gru_cell_32/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/splitSplit*while/gru_cell_32/split/split_dim:output:0"while/gru_cell_32/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_32/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_32/MatMul_1MatMulwhile_placeholder_21while/gru_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/BiasAdd_1BiasAdd$while/gru_cell_32/MatMul_1:product:0"while/gru_cell_32/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_32/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_32/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_32/split_1SplitV$while/gru_cell_32/BiasAdd_1:output:0 while/gru_cell_32/Const:output:0,while/gru_cell_32/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_32/addAddV2 while/gru_cell_32/split:output:0"while/gru_cell_32/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_32/SigmoidSigmoidwhile/gru_cell_32/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_1AddV2 while/gru_cell_32/split:output:1"while/gru_cell_32/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_32/Sigmoid_1Sigmoidwhile/gru_cell_32/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mulMulwhile/gru_cell_32/Sigmoid_1:y:0"while/gru_cell_32/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_2AddV2 while/gru_cell_32/split:output:2while/gru_cell_32/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_32/SoftplusSoftpluswhile/gru_cell_32/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_1Mulwhile/gru_cell_32/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_32/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_32/subSub while/gru_cell_32/sub/x:output:0while/gru_cell_32/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/mul_2Mulwhile/gru_cell_32/sub:z:0(while/gru_cell_32/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_32/add_3AddV2while/gru_cell_32/mul_1:z:0while/gru_cell_32/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_32/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_32/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_32/MatMul/ReadVariableOp*^while/gru_cell_32/MatMul_1/ReadVariableOp!^while/gru_cell_32/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_32_matmul_1_readvariableop_resource4while_gru_cell_32_matmul_1_readvariableop_resource_0"f
0while_gru_cell_32_matmul_readvariableop_resource2while_gru_cell_32_matmul_readvariableop_resource_0"X
)while_gru_cell_32_readvariableop_resource+while_gru_cell_32_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2R
'while/gru_cell_32/MatMul/ReadVariableOp'while/gru_cell_32/MatMul/ReadVariableOp2V
)while/gru_cell_32/MatMul_1/ReadVariableOp)while/gru_cell_32/MatMul_1/ReadVariableOp2D
 while/gru_cell_32/ReadVariableOp while/gru_cell_32/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�F
�	
gru_15_while_body_2905580*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2)
%gru_15_while_gru_15_strided_slice_1_0e
agru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0E
2gru_15_while_gru_cell_30_readvariableop_resource_0:	�L
9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0:	�O
;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0:
��
gru_15_while_identity
gru_15_while_identity_1
gru_15_while_identity_2
gru_15_while_identity_3
gru_15_while_identity_4'
#gru_15_while_gru_15_strided_slice_1c
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensorC
0gru_15_while_gru_cell_30_readvariableop_resource:	�J
7gru_15_while_gru_cell_30_matmul_readvariableop_resource:	�M
9gru_15_while_gru_cell_30_matmul_1_readvariableop_resource:
����.gru_15/while/gru_cell_30/MatMul/ReadVariableOp�0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp�'gru_15/while/gru_cell_30/ReadVariableOp�
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0gru_15_while_placeholderGgru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'gru_15/while/gru_cell_30/ReadVariableOpReadVariableOp2gru_15_while_gru_cell_30_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_15/while/gru_cell_30/unstackUnpack/gru_15/while/gru_cell_30/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_15/while/gru_cell_30/MatMul/ReadVariableOpReadVariableOp9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
gru_15/while/gru_cell_30/MatMulMatMul7gru_15/while/TensorArrayV2Read/TensorListGetItem:item:06gru_15/while/gru_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_15/while/gru_cell_30/BiasAddBiasAdd)gru_15/while/gru_cell_30/MatMul:product:0)gru_15/while/gru_cell_30/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_15/while/gru_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_15/while/gru_cell_30/splitSplit1gru_15/while/gru_cell_30/split/split_dim:output:0)gru_15/while/gru_cell_30/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOpReadVariableOp;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
!gru_15/while/gru_cell_30/MatMul_1MatMulgru_15_while_placeholder_28gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/BiasAdd_1BiasAdd+gru_15/while/gru_cell_30/MatMul_1:product:0)gru_15/while/gru_cell_30/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_15/while/gru_cell_30/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����u
*gru_15/while/gru_cell_30/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_15/while/gru_cell_30/split_1SplitV+gru_15/while/gru_cell_30/BiasAdd_1:output:0'gru_15/while/gru_cell_30/Const:output:03gru_15/while/gru_cell_30/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_15/while/gru_cell_30/addAddV2'gru_15/while/gru_cell_30/split:output:0)gru_15/while/gru_cell_30/split_1:output:0*
T0*(
_output_shapes
:�����������
 gru_15/while/gru_cell_30/SigmoidSigmoid gru_15/while/gru_cell_30/add:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_1AddV2'gru_15/while/gru_cell_30/split:output:1)gru_15/while/gru_cell_30/split_1:output:1*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/Sigmoid_1Sigmoid"gru_15/while/gru_cell_30/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mulMul&gru_15/while/gru_cell_30/Sigmoid_1:y:0)gru_15/while/gru_cell_30/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_2AddV2'gru_15/while/gru_cell_30/split:output:2 gru_15/while/gru_cell_30/mul:z:0*
T0*(
_output_shapes
:�����������
"gru_15/while/gru_cell_30/Sigmoid_2Sigmoid"gru_15/while/gru_cell_30/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mul_1Mul$gru_15/while/gru_cell_30/Sigmoid:y:0gru_15_while_placeholder_2*
T0*(
_output_shapes
:����������c
gru_15/while/gru_cell_30/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_15/while/gru_cell_30/subSub'gru_15/while/gru_cell_30/sub/x:output:0$gru_15/while/gru_cell_30/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/mul_2Mul gru_15/while/gru_cell_30/sub:z:0&gru_15/while/gru_cell_30/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_15/while/gru_cell_30/add_3AddV2"gru_15/while/gru_cell_30/mul_1:z:0"gru_15/while/gru_cell_30/mul_2:z:0*
T0*(
_output_shapes
:�����������
1gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_15_while_placeholder_1gru_15_while_placeholder"gru_15/while/gru_cell_30/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_15/while/addAddV2gru_15_while_placeholdergru_15/while/add/y:output:0*
T0*
_output_shapes
: V
gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_15/while/add_1AddV2&gru_15_while_gru_15_while_loop_countergru_15/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_15/while/IdentityIdentitygru_15/while/add_1:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_1Identity,gru_15_while_gru_15_while_maximum_iterations^gru_15/while/NoOp*
T0*
_output_shapes
: n
gru_15/while/Identity_2Identitygru_15/while/add:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_3IdentityAgru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_15/while/NoOp*
T0*
_output_shapes
: �
gru_15/while/Identity_4Identity"gru_15/while/gru_cell_30/add_3:z:0^gru_15/while/NoOp*
T0*(
_output_shapes
:�����������
gru_15/while/NoOpNoOp/^gru_15/while/gru_cell_30/MatMul/ReadVariableOp1^gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp(^gru_15/while/gru_cell_30/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_15_while_gru_15_strided_slice_1%gru_15_while_gru_15_strided_slice_1_0"x
9gru_15_while_gru_cell_30_matmul_1_readvariableop_resource;gru_15_while_gru_cell_30_matmul_1_readvariableop_resource_0"t
7gru_15_while_gru_cell_30_matmul_readvariableop_resource9gru_15_while_gru_cell_30_matmul_readvariableop_resource_0"f
0gru_15_while_gru_cell_30_readvariableop_resource2gru_15_while_gru_cell_30_readvariableop_resource_0"7
gru_15_while_identitygru_15/while/Identity:output:0";
gru_15_while_identity_1 gru_15/while/Identity_1:output:0";
gru_15_while_identity_2 gru_15/while/Identity_2:output:0";
gru_15_while_identity_3 gru_15/while/Identity_3:output:0";
gru_15_while_identity_4 gru_15/while/Identity_4:output:0"�
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensoragru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2`
.gru_15/while/gru_cell_30/MatMul/ReadVariableOp.gru_15/while/gru_cell_30/MatMul/ReadVariableOp2d
0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp0gru_15/while/gru_cell_30/MatMul_1/ReadVariableOp2R
'gru_15/while/gru_cell_30/ReadVariableOp'gru_15/while/gru_cell_30/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2907181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2907181___redundant_placeholder05
1while_while_cond_2907181___redundant_placeholder15
1while_while_cond_2907181___redundant_placeholder25
1while_while_cond_2907181___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
J
gru_15_input:
serving_default_gru_15_input:0����������?
gru_175
StatefulPartitionedCall:0����������tensorflow/serving/predict:٣
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_random_generator
&cell
'
state_spec"
_tf_keras_rnn_layer
_
(0
)1
*2
+3
,4
-5
.6
/7
08"
trackable_list_wrapper
_
(0
)1
*2
+3
,4
-5
.6
/7
08"
trackable_list_wrapper
 "
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
6trace_0
7trace_1
8trace_2
9trace_32�
.__inference_sequential_5_layer_call_fn_2904770
.__inference_sequential_5_layer_call_fn_2905493
.__inference_sequential_5_layer_call_fn_2905516
.__inference_sequential_5_layer_call_fn_2905389�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z6trace_0z7trace_1z8trace_2z9trace_3
�
:trace_0
;trace_1
<trace_2
=trace_32�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905967
I__inference_sequential_5_layer_call_and_return_conditional_losses_2906418
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905414
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905439�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z:trace_0z;trace_1z<trace_2z=trace_3
�B�
"__inference__wrapped_model_2903247gru_15_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate(m�)m�*m�+m�,m�-m�.m�/m�0m�(v�)v�*v�+v�,v�-v�.v�/v�0v�"
	optimizer
,
Cserving_default"
signature_map
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Dstates
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32�
(__inference_gru_15_layer_call_fn_2906429
(__inference_gru_15_layer_call_fn_2906440
(__inference_gru_15_layer_call_fn_2906451
(__inference_gru_15_layer_call_fn_2906462�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
�
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_32�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906615
C__inference_gru_15_layer_call_and_return_conditional_losses_2906768
C__inference_gru_15_layer_call_and_return_conditional_losses_2906921
C__inference_gru_15_layer_call_and_return_conditional_losses_2907074�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zNtrace_0zOtrace_1zPtrace_2zQtrace_3
"
_generic_user_object
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
X_random_generator

(kernel
)recurrent_kernel
*bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
_trace_0
`trace_1
atrace_2
btrace_32�
(__inference_gru_16_layer_call_fn_2907085
(__inference_gru_16_layer_call_fn_2907096
(__inference_gru_16_layer_call_fn_2907107
(__inference_gru_16_layer_call_fn_2907118�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z_trace_0z`trace_1zatrace_2zbtrace_3
�
ctrace_0
dtrace_1
etrace_2
ftrace_32�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907271
C__inference_gru_16_layer_call_and_return_conditional_losses_2907424
C__inference_gru_16_layer_call_and_return_conditional_losses_2907577
C__inference_gru_16_layer_call_and_return_conditional_losses_2907730�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zctrace_0zdtrace_1zetrace_2zftrace_3
"
_generic_user_object
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator

+kernel
,recurrent_kernel
-bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
�

nstates
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
ttrace_0
utrace_1
vtrace_2
wtrace_32�
(__inference_gru_17_layer_call_fn_2907741
(__inference_gru_17_layer_call_fn_2907752
(__inference_gru_17_layer_call_fn_2907763
(__inference_gru_17_layer_call_fn_2907774�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
�
xtrace_0
ytrace_1
ztrace_2
{trace_32�
C__inference_gru_17_layer_call_and_return_conditional_losses_2907927
C__inference_gru_17_layer_call_and_return_conditional_losses_2908080
C__inference_gru_17_layer_call_and_return_conditional_losses_2908233
C__inference_gru_17_layer_call_and_return_conditional_losses_2908386�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zxtrace_0zytrace_1zztrace_2z{trace_3
"
_generic_user_object
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

.kernel
/recurrent_kernel
0bias"
_tf_keras_layer
 "
trackable_list_wrapper
,:*	�2gru_15/gru_cell_30/kernel
7:5
��2#gru_15/gru_cell_30/recurrent_kernel
*:(	�2gru_15/gru_cell_30/bias
-:+
��2gru_16/gru_cell_31/kernel
6:4	d�2#gru_16/gru_cell_31/recurrent_kernel
*:(	�2gru_16/gru_cell_31/bias
+:)d2gru_17/gru_cell_32/kernel
5:32#gru_17/gru_cell_32/recurrent_kernel
):'2gru_17/gru_cell_32/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_5_layer_call_fn_2904770gru_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_2905493inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_2905516inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_2905389gru_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905967inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2906418inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905414gru_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905439gru_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_2905470gru_15_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_gru_15_layer_call_fn_2906429inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_15_layer_call_fn_2906440inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_15_layer_call_fn_2906451inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_15_layer_call_fn_2906462inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906615inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906768inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_15_layer_call_and_return_conditional_losses_2906921inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_15_layer_call_and_return_conditional_losses_2907074inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_gru_cell_30_layer_call_fn_2908400
-__inference_gru_cell_30_layer_call_fn_2908414�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908453
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908492�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_gru_16_layer_call_fn_2907085inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_16_layer_call_fn_2907096inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_16_layer_call_fn_2907107inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_16_layer_call_fn_2907118inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907271inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907424inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907577inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_16_layer_call_and_return_conditional_losses_2907730inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_gru_cell_31_layer_call_fn_2908506
-__inference_gru_cell_31_layer_call_fn_2908520�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908559
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908598�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
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
trackable_dict_wrapper
�B�
(__inference_gru_17_layer_call_fn_2907741inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_17_layer_call_fn_2907752inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_17_layer_call_fn_2907763inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_17_layer_call_fn_2907774inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_17_layer_call_and_return_conditional_losses_2907927inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908080inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908233inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_17_layer_call_and_return_conditional_losses_2908386inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_gru_cell_32_layer_call_fn_2908612
-__inference_gru_cell_32_layer_call_fn_2908626�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908665
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908704�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_gru_cell_30_layer_call_fn_2908400inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_gru_cell_30_layer_call_fn_2908414inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908453inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908492inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_gru_cell_31_layer_call_fn_2908506inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_gru_cell_31_layer_call_fn_2908520inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908559inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908598inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_gru_cell_32_layer_call_fn_2908612inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_gru_cell_32_layer_call_fn_2908626inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908665inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908704inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
1:/	�2 Adam/gru_15/gru_cell_30/kernel/m
<::
��2*Adam/gru_15/gru_cell_30/recurrent_kernel/m
/:-	�2Adam/gru_15/gru_cell_30/bias/m
2:0
��2 Adam/gru_16/gru_cell_31/kernel/m
;:9	d�2*Adam/gru_16/gru_cell_31/recurrent_kernel/m
/:-	�2Adam/gru_16/gru_cell_31/bias/m
0:.d2 Adam/gru_17/gru_cell_32/kernel/m
::82*Adam/gru_17/gru_cell_32/recurrent_kernel/m
.:,2Adam/gru_17/gru_cell_32/bias/m
1:/	�2 Adam/gru_15/gru_cell_30/kernel/v
<::
��2*Adam/gru_15/gru_cell_30/recurrent_kernel/v
/:-	�2Adam/gru_15/gru_cell_30/bias/v
2:0
��2 Adam/gru_16/gru_cell_31/kernel/v
;:9	d�2*Adam/gru_16/gru_cell_31/recurrent_kernel/v
/:-	�2Adam/gru_16/gru_cell_31/bias/v
0:.d2 Adam/gru_17/gru_cell_32/kernel/v
::82*Adam/gru_17/gru_cell_32/recurrent_kernel/v
.:,2Adam/gru_17/gru_cell_32/bias/v�
"__inference__wrapped_model_2903247}	*()-+,0./:�7
0�-
+�(
gru_15_input����������
� "4�1
/
gru_17%�"
gru_17�����������
C__inference_gru_15_layer_call_and_return_conditional_losses_2906615�*()O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "3�0
)�&
0�������������������
� �
C__inference_gru_15_layer_call_and_return_conditional_losses_2906768�*()O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "3�0
)�&
0�������������������
� �
C__inference_gru_15_layer_call_and_return_conditional_losses_2906921t*()@�=
6�3
%�"
inputs����������

 
p 

 
� "+�(
!�
0�����������
� �
C__inference_gru_15_layer_call_and_return_conditional_losses_2907074t*()@�=
6�3
%�"
inputs����������

 
p

 
� "+�(
!�
0�����������
� �
(__inference_gru_15_layer_call_fn_2906429~*()O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "&�#��������������������
(__inference_gru_15_layer_call_fn_2906440~*()O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "&�#��������������������
(__inference_gru_15_layer_call_fn_2906451g*()@�=
6�3
%�"
inputs����������

 
p 

 
� "�������������
(__inference_gru_15_layer_call_fn_2906462g*()@�=
6�3
%�"
inputs����������

 
p

 
� "�������������
C__inference_gru_16_layer_call_and_return_conditional_losses_2907271�-+,P�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "2�/
(�%
0������������������d
� �
C__inference_gru_16_layer_call_and_return_conditional_losses_2907424�-+,P�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "2�/
(�%
0������������������d
� �
C__inference_gru_16_layer_call_and_return_conditional_losses_2907577t-+,A�>
7�4
&�#
inputs�����������

 
p 

 
� "*�'
 �
0����������d
� �
C__inference_gru_16_layer_call_and_return_conditional_losses_2907730t-+,A�>
7�4
&�#
inputs�����������

 
p

 
� "*�'
 �
0����������d
� �
(__inference_gru_16_layer_call_fn_2907085~-+,P�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "%�"������������������d�
(__inference_gru_16_layer_call_fn_2907096~-+,P�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "%�"������������������d�
(__inference_gru_16_layer_call_fn_2907107g-+,A�>
7�4
&�#
inputs�����������

 
p 

 
� "�����������d�
(__inference_gru_16_layer_call_fn_2907118g-+,A�>
7�4
&�#
inputs�����������

 
p

 
� "�����������d�
C__inference_gru_17_layer_call_and_return_conditional_losses_2907927�0./O�L
E�B
4�1
/�,
inputs/0������������������d

 
p 

 
� "2�/
(�%
0������������������
� �
C__inference_gru_17_layer_call_and_return_conditional_losses_2908080�0./O�L
E�B
4�1
/�,
inputs/0������������������d

 
p

 
� "2�/
(�%
0������������������
� �
C__inference_gru_17_layer_call_and_return_conditional_losses_2908233s0./@�=
6�3
%�"
inputs����������d

 
p 

 
� "*�'
 �
0����������
� �
C__inference_gru_17_layer_call_and_return_conditional_losses_2908386s0./@�=
6�3
%�"
inputs����������d

 
p

 
� "*�'
 �
0����������
� �
(__inference_gru_17_layer_call_fn_2907741}0./O�L
E�B
4�1
/�,
inputs/0������������������d

 
p 

 
� "%�"�������������������
(__inference_gru_17_layer_call_fn_2907752}0./O�L
E�B
4�1
/�,
inputs/0������������������d

 
p

 
� "%�"�������������������
(__inference_gru_17_layer_call_fn_2907763f0./@�=
6�3
%�"
inputs����������d

 
p 

 
� "������������
(__inference_gru_17_layer_call_fn_2907774f0./@�=
6�3
%�"
inputs����������d

 
p

 
� "������������
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908453�*()]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p 
� "T�Q
J�G
�
0/0����������
%�"
 �
0/1/0����������
� �
H__inference_gru_cell_30_layer_call_and_return_conditional_losses_2908492�*()]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p
� "T�Q
J�G
�
0/0����������
%�"
 �
0/1/0����������
� �
-__inference_gru_cell_30_layer_call_fn_2908400�*()]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p 
� "F�C
�
0����������
#� 
�
1/0�����������
-__inference_gru_cell_30_layer_call_fn_2908414�*()]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p
� "F�C
�
0����������
#� 
�
1/0�����������
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908559�-+,]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������d
p 
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
H__inference_gru_cell_31_layer_call_and_return_conditional_losses_2908598�-+,]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������d
p
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
-__inference_gru_cell_31_layer_call_fn_2908506�-+,]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������d
p 
� "D�A
�
0���������d
"�
�
1/0���������d�
-__inference_gru_cell_31_layer_call_fn_2908520�-+,]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������d
p
� "D�A
�
0���������d
"�
�
1/0���������d�
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908665�0./\�Y
R�O
 �
inputs���������d
'�$
"�
states/0���������
p 
� "R�O
H�E
�
0/0���������
$�!
�
0/1/0���������
� �
H__inference_gru_cell_32_layer_call_and_return_conditional_losses_2908704�0./\�Y
R�O
 �
inputs���������d
'�$
"�
states/0���������
p
� "R�O
H�E
�
0/0���������
$�!
�
0/1/0���������
� �
-__inference_gru_cell_32_layer_call_fn_2908612�0./\�Y
R�O
 �
inputs���������d
'�$
"�
states/0���������
p 
� "D�A
�
0���������
"�
�
1/0����������
-__inference_gru_cell_32_layer_call_fn_2908626�0./\�Y
R�O
 �
inputs���������d
'�$
"�
states/0���������
p
� "D�A
�
0���������
"�
�
1/0����������
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905414{	*()-+,0./B�?
8�5
+�(
gru_15_input����������
p 

 
� "*�'
 �
0����������
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905439{	*()-+,0./B�?
8�5
+�(
gru_15_input����������
p

 
� "*�'
 �
0����������
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_2905967u	*()-+,0./<�9
2�/
%�"
inputs����������
p 

 
� "*�'
 �
0����������
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_2906418u	*()-+,0./<�9
2�/
%�"
inputs����������
p

 
� "*�'
 �
0����������
� �
.__inference_sequential_5_layer_call_fn_2904770n	*()-+,0./B�?
8�5
+�(
gru_15_input����������
p 

 
� "������������
.__inference_sequential_5_layer_call_fn_2905389n	*()-+,0./B�?
8�5
+�(
gru_15_input����������
p

 
� "������������
.__inference_sequential_5_layer_call_fn_2905493h	*()-+,0./<�9
2�/
%�"
inputs����������
p 

 
� "������������
.__inference_sequential_5_layer_call_fn_2905516h	*()-+,0./<�9
2�/
%�"
inputs����������
p

 
� "������������
%__inference_signature_wrapper_2905470�	*()-+,0./J�G
� 
@�=
;
gru_15_input+�(
gru_15_input����������"4�1
/
gru_17%�"
gru_17����������