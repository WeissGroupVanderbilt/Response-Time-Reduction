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
Adam/gru_14/gru_cell_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_14/gru_cell_26/bias/v
�
2Adam/gru_14/gru_cell_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_14/gru_cell_26/bias/v*
_output_shapes

:*
dtype0
�
*Adam/gru_14/gru_cell_26/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_14/gru_cell_26/recurrent_kernel/v
�
>Adam/gru_14/gru_cell_26/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_14/gru_cell_26/recurrent_kernel/v*
_output_shapes

:*
dtype0
�
 Adam/gru_14/gru_cell_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_14/gru_cell_26/kernel/v
�
4Adam/gru_14/gru_cell_26/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_14/gru_cell_26/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/gru_13/gru_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_13/gru_cell_25/bias/v
�
2Adam/gru_13/gru_cell_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_13/gru_cell_25/bias/v*
_output_shapes
:	�*
dtype0
�
*Adam/gru_13/gru_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*;
shared_name,*Adam/gru_13/gru_cell_25/recurrent_kernel/v
�
>Adam/gru_13/gru_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_13/gru_cell_25/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
 Adam/gru_13/gru_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" Adam/gru_13/gru_cell_25/kernel/v
�
4Adam/gru_13/gru_cell_25/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_13/gru_cell_25/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/gru_12/gru_cell_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_12/gru_cell_24/bias/v
�
2Adam/gru_12/gru_cell_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_12/gru_cell_24/bias/v*
_output_shapes
:	�*
dtype0
�
*Adam/gru_12/gru_cell_24/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/gru_12/gru_cell_24/recurrent_kernel/v
�
>Adam/gru_12/gru_cell_24/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_12/gru_cell_24/recurrent_kernel/v* 
_output_shapes
:
��*
dtype0
�
 Adam/gru_12/gru_cell_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_12/gru_cell_24/kernel/v
�
4Adam/gru_12/gru_cell_24/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_12/gru_cell_24/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/gru_14/gru_cell_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_14/gru_cell_26/bias/m
�
2Adam/gru_14/gru_cell_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_14/gru_cell_26/bias/m*
_output_shapes

:*
dtype0
�
*Adam/gru_14/gru_cell_26/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_14/gru_cell_26/recurrent_kernel/m
�
>Adam/gru_14/gru_cell_26/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_14/gru_cell_26/recurrent_kernel/m*
_output_shapes

:*
dtype0
�
 Adam/gru_14/gru_cell_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_14/gru_cell_26/kernel/m
�
4Adam/gru_14/gru_cell_26/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_14/gru_cell_26/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/gru_13/gru_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_13/gru_cell_25/bias/m
�
2Adam/gru_13/gru_cell_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_13/gru_cell_25/bias/m*
_output_shapes
:	�*
dtype0
�
*Adam/gru_13/gru_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*;
shared_name,*Adam/gru_13/gru_cell_25/recurrent_kernel/m
�
>Adam/gru_13/gru_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_13/gru_cell_25/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
 Adam/gru_13/gru_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" Adam/gru_13/gru_cell_25/kernel/m
�
4Adam/gru_13/gru_cell_25/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_13/gru_cell_25/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/gru_12/gru_cell_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_12/gru_cell_24/bias/m
�
2Adam/gru_12/gru_cell_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_12/gru_cell_24/bias/m*
_output_shapes
:	�*
dtype0
�
*Adam/gru_12/gru_cell_24/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/gru_12/gru_cell_24/recurrent_kernel/m
�
>Adam/gru_12/gru_cell_24/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_12/gru_cell_24/recurrent_kernel/m* 
_output_shapes
:
��*
dtype0
�
 Adam/gru_12/gru_cell_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_12/gru_cell_24/kernel/m
�
4Adam/gru_12/gru_cell_24/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_12/gru_cell_24/kernel/m*
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
gru_14/gru_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namegru_14/gru_cell_26/bias
�
+gru_14/gru_cell_26/bias/Read/ReadVariableOpReadVariableOpgru_14/gru_cell_26/bias*
_output_shapes

:*
dtype0
�
#gru_14/gru_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#gru_14/gru_cell_26/recurrent_kernel
�
7gru_14/gru_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_14/gru_cell_26/recurrent_kernel*
_output_shapes

:*
dtype0
�
gru_14/gru_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_namegru_14/gru_cell_26/kernel
�
-gru_14/gru_cell_26/kernel/Read/ReadVariableOpReadVariableOpgru_14/gru_cell_26/kernel*
_output_shapes

:d*
dtype0
�
gru_13/gru_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_13/gru_cell_25/bias
�
+gru_13/gru_cell_25/bias/Read/ReadVariableOpReadVariableOpgru_13/gru_cell_25/bias*
_output_shapes
:	�*
dtype0
�
#gru_13/gru_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*4
shared_name%#gru_13/gru_cell_25/recurrent_kernel
�
7gru_13/gru_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_13/gru_cell_25/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
gru_13/gru_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_namegru_13/gru_cell_25/kernel
�
-gru_13/gru_cell_25/kernel/Read/ReadVariableOpReadVariableOpgru_13/gru_cell_25/kernel* 
_output_shapes
:
��*
dtype0
�
gru_12/gru_cell_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_12/gru_cell_24/bias
�
+gru_12/gru_cell_24/bias/Read/ReadVariableOpReadVariableOpgru_12/gru_cell_24/bias*
_output_shapes
:	�*
dtype0
�
#gru_12/gru_cell_24/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#gru_12/gru_cell_24/recurrent_kernel
�
7gru_12/gru_cell_24/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_12/gru_cell_24/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
gru_12/gru_cell_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_namegru_12/gru_cell_24/kernel
�
-gru_12/gru_cell_24/kernel/Read/ReadVariableOpReadVariableOpgru_12/gru_cell_24/kernel*
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
VARIABLE_VALUEgru_12/gru_cell_24/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_12/gru_cell_24/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_12/gru_cell_24/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_13/gru_cell_25/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_13/gru_cell_25/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_13/gru_cell_25/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_14/gru_cell_26/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_14/gru_cell_26/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_14/gru_cell_26/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE Adam/gru_12/gru_cell_24/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_12/gru_cell_24/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_12/gru_cell_24/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_13/gru_cell_25/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_13/gru_cell_25/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_13/gru_cell_25/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_14/gru_cell_26/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_14/gru_cell_26/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_14/gru_cell_26/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_12/gru_cell_24/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_12/gru_cell_24/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_12/gru_cell_24/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_13/gru_cell_25/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_13/gru_cell_25/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_13/gru_cell_25/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_14/gru_cell_26/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/gru_14/gru_cell_26/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_14/gru_cell_26/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_gru_12_inputPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_12_inputgru_12/gru_cell_24/biasgru_12/gru_cell_24/kernel#gru_12/gru_cell_24/recurrent_kernelgru_13/gru_cell_25/biasgru_13/gru_cell_25/kernel#gru_13/gru_cell_25/recurrent_kernelgru_14/gru_cell_26/biasgru_14/gru_cell_26/kernel#gru_14/gru_cell_26/recurrent_kernel*
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
%__inference_signature_wrapper_2432300
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-gru_12/gru_cell_24/kernel/Read/ReadVariableOp7gru_12/gru_cell_24/recurrent_kernel/Read/ReadVariableOp+gru_12/gru_cell_24/bias/Read/ReadVariableOp-gru_13/gru_cell_25/kernel/Read/ReadVariableOp7gru_13/gru_cell_25/recurrent_kernel/Read/ReadVariableOp+gru_13/gru_cell_25/bias/Read/ReadVariableOp-gru_14/gru_cell_26/kernel/Read/ReadVariableOp7gru_14/gru_cell_26/recurrent_kernel/Read/ReadVariableOp+gru_14/gru_cell_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/gru_12/gru_cell_24/kernel/m/Read/ReadVariableOp>Adam/gru_12/gru_cell_24/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_12/gru_cell_24/bias/m/Read/ReadVariableOp4Adam/gru_13/gru_cell_25/kernel/m/Read/ReadVariableOp>Adam/gru_13/gru_cell_25/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_13/gru_cell_25/bias/m/Read/ReadVariableOp4Adam/gru_14/gru_cell_26/kernel/m/Read/ReadVariableOp>Adam/gru_14/gru_cell_26/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_14/gru_cell_26/bias/m/Read/ReadVariableOp4Adam/gru_12/gru_cell_24/kernel/v/Read/ReadVariableOp>Adam/gru_12/gru_cell_24/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_12/gru_cell_24/bias/v/Read/ReadVariableOp4Adam/gru_13/gru_cell_25/kernel/v/Read/ReadVariableOp>Adam/gru_13/gru_cell_25/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_13/gru_cell_25/bias/v/Read/ReadVariableOp4Adam/gru_14/gru_cell_26/kernel/v/Read/ReadVariableOp>Adam/gru_14/gru_cell_26/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_14/gru_cell_26/bias/v/Read/ReadVariableOpConst*/
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
 __inference__traced_save_2435659
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegru_12/gru_cell_24/kernel#gru_12/gru_cell_24/recurrent_kernelgru_12/gru_cell_24/biasgru_13/gru_cell_25/kernel#gru_13/gru_cell_25/recurrent_kernelgru_13/gru_cell_25/biasgru_14/gru_cell_26/kernel#gru_14/gru_cell_26/recurrent_kernelgru_14/gru_cell_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/gru_12/gru_cell_24/kernel/m*Adam/gru_12/gru_cell_24/recurrent_kernel/mAdam/gru_12/gru_cell_24/bias/m Adam/gru_13/gru_cell_25/kernel/m*Adam/gru_13/gru_cell_25/recurrent_kernel/mAdam/gru_13/gru_cell_25/bias/m Adam/gru_14/gru_cell_26/kernel/m*Adam/gru_14/gru_cell_26/recurrent_kernel/mAdam/gru_14/gru_cell_26/bias/m Adam/gru_12/gru_cell_24/kernel/v*Adam/gru_12/gru_cell_24/recurrent_kernel/vAdam/gru_12/gru_cell_24/bias/v Adam/gru_13/gru_cell_25/kernel/v*Adam/gru_13/gru_cell_25/recurrent_kernel/vAdam/gru_13/gru_cell_25/bias/v Adam/gru_14/gru_cell_26/kernel/v*Adam/gru_14/gru_cell_26/recurrent_kernel/vAdam/gru_14/gru_cell_26/bias/v*.
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
#__inference__traced_restore_2435771��+
�
�
while_cond_2431676
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431676___redundant_placeholder05
1while_while_cond_2431676___redundant_placeholder15
1while_while_cond_2431676___redundant_placeholder25
1while_while_cond_2431676___redundant_placeholder3
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
while_cond_2431017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431017___redundant_placeholder05
1while_while_cond_2431017___redundant_placeholder15
1while_while_cond_2431017___redundant_placeholder25
1while_while_cond_2431017___redundant_placeholder3
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
�=
�
while_body_2431161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
�F
�	
gru_12_while_body_2432410*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2)
%gru_12_while_gru_12_strided_slice_1_0e
agru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0E
2gru_12_while_gru_cell_24_readvariableop_resource_0:	�L
9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0:	�O
;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
gru_12_while_identity
gru_12_while_identity_1
gru_12_while_identity_2
gru_12_while_identity_3
gru_12_while_identity_4'
#gru_12_while_gru_12_strided_slice_1c
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensorC
0gru_12_while_gru_cell_24_readvariableop_resource:	�J
7gru_12_while_gru_cell_24_matmul_readvariableop_resource:	�M
9gru_12_while_gru_cell_24_matmul_1_readvariableop_resource:
����.gru_12/while/gru_cell_24/MatMul/ReadVariableOp�0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp�'gru_12/while/gru_cell_24/ReadVariableOp�
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0gru_12_while_placeholderGgru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'gru_12/while/gru_cell_24/ReadVariableOpReadVariableOp2gru_12_while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_12/while/gru_cell_24/unstackUnpack/gru_12/while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_12/while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
gru_12/while/gru_cell_24/MatMulMatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:06gru_12/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_12/while/gru_cell_24/BiasAddBiasAdd)gru_12/while/gru_cell_24/MatMul:product:0)gru_12/while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_12/while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/while/gru_cell_24/splitSplit1gru_12/while/gru_cell_24/split/split_dim:output:0)gru_12/while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
!gru_12/while/gru_cell_24/MatMul_1MatMulgru_12_while_placeholder_28gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/BiasAdd_1BiasAdd+gru_12/while/gru_cell_24/MatMul_1:product:0)gru_12/while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_12/while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����u
*gru_12/while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_12/while/gru_cell_24/split_1SplitV+gru_12/while/gru_cell_24/BiasAdd_1:output:0'gru_12/while/gru_cell_24/Const:output:03gru_12/while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_12/while/gru_cell_24/addAddV2'gru_12/while/gru_cell_24/split:output:0)gru_12/while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:�����������
 gru_12/while/gru_cell_24/SigmoidSigmoid gru_12/while/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_1AddV2'gru_12/while/gru_cell_24/split:output:1)gru_12/while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/Sigmoid_1Sigmoid"gru_12/while/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mulMul&gru_12/while/gru_cell_24/Sigmoid_1:y:0)gru_12/while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_2AddV2'gru_12/while/gru_cell_24/split:output:2 gru_12/while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/Sigmoid_2Sigmoid"gru_12/while/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mul_1Mul$gru_12/while/gru_cell_24/Sigmoid:y:0gru_12_while_placeholder_2*
T0*(
_output_shapes
:����������c
gru_12/while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_12/while/gru_cell_24/subSub'gru_12/while/gru_cell_24/sub/x:output:0$gru_12/while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mul_2Mul gru_12/while/gru_cell_24/sub:z:0&gru_12/while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_3AddV2"gru_12/while/gru_cell_24/mul_1:z:0"gru_12/while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
1gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_12_while_placeholder_1gru_12_while_placeholder"gru_12/while/gru_cell_24/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_12/while/addAddV2gru_12_while_placeholdergru_12/while/add/y:output:0*
T0*
_output_shapes
: V
gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_12/while/add_1AddV2&gru_12_while_gru_12_while_loop_countergru_12/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_12/while/IdentityIdentitygru_12/while/add_1:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_1Identity,gru_12_while_gru_12_while_maximum_iterations^gru_12/while/NoOp*
T0*
_output_shapes
: n
gru_12/while/Identity_2Identitygru_12/while/add:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_3IdentityAgru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_4Identity"gru_12/while/gru_cell_24/add_3:z:0^gru_12/while/NoOp*
T0*(
_output_shapes
:�����������
gru_12/while/NoOpNoOp/^gru_12/while/gru_cell_24/MatMul/ReadVariableOp1^gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp(^gru_12/while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_12_while_gru_12_strided_slice_1%gru_12_while_gru_12_strided_slice_1_0"x
9gru_12_while_gru_cell_24_matmul_1_readvariableop_resource;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0"t
7gru_12_while_gru_cell_24_matmul_readvariableop_resource9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0"f
0gru_12_while_gru_cell_24_readvariableop_resource2gru_12_while_gru_cell_24_readvariableop_resource_0"7
gru_12_while_identitygru_12/while/Identity:output:0";
gru_12_while_identity_1 gru_12/while/Identity_1:output:0";
gru_12_while_identity_2 gru_12/while/Identity_2:output:0";
gru_12_while_identity_3 gru_12/while/Identity_3:output:0";
gru_12_while_identity_4 gru_12/while/Identity_4:output:0"�
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensoragru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2`
.gru_12/while/gru_cell_24/MatMul/ReadVariableOp.gru_12/while/gru_cell_24/MatMul/ReadVariableOp2d
0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp2R
'gru_12/while/gru_cell_24/ReadVariableOp'gru_12/while/gru_cell_24/ReadVariableOp: 
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

�
-__inference_gru_cell_26_layer_call_fn_2435456

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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430966o
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
�M
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2434407

inputs6
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while;
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2434318*
condR
while_cond_2434317*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
.__inference_sequential_4_layer_call_fn_2431600
gru_12_input
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
StatefulPartitionedCallStatefulPartitionedCallgru_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2431579t
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
_user_specified_namegru_12_input
�M
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2431941

inputs6
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while;
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2431852*
condR
while_cond_2431851*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432244
gru_12_input!
gru_12_2432222:	�!
gru_12_2432224:	�"
gru_12_2432226:
��!
gru_13_2432229:	�"
gru_13_2432231:
��!
gru_13_2432233:	d� 
gru_14_2432236: 
gru_14_2432238:d 
gru_14_2432240:
identity��gru_12/StatefulPartitionedCall�gru_13/StatefulPartitionedCall�gru_14/StatefulPartitionedCall�
gru_12/StatefulPartitionedCallStatefulPartitionedCallgru_12_inputgru_12_2432222gru_12_2432224gru_12_2432226*
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2431250�
gru_13/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0gru_13_2432229gru_13_2432231gru_13_2432233*
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431410�
gru_14/StatefulPartitionedCallStatefulPartitionedCall'gru_13/StatefulPartitionedCall:output:0gru_14_2432236gru_14_2432238gru_14_2432240*
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431570{
IdentityIdentity'gru_14/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_12/StatefulPartitionedCall^gru_13/StatefulPartitionedCall^gru_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2@
gru_13/StatefulPartitionedCallgru_13/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_12_input
�M
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2433598
inputs_06
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while=
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2433509*
condR
while_cond_2433508*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�M
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2431410

inputs6
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while;
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2431321*
condR
while_cond_2431320*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�M
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2431250

inputs6
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while;
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2431161*
condR
while_cond_2431160*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_gru_13_layer_call_fn_2433948

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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431941t
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
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432175

inputs!
gru_12_2432153:	�!
gru_12_2432155:	�"
gru_12_2432157:
��!
gru_13_2432160:	�"
gru_13_2432162:
��!
gru_13_2432164:	d� 
gru_14_2432167: 
gru_14_2432169:d 
gru_14_2432171:
identity��gru_12/StatefulPartitionedCall�gru_13/StatefulPartitionedCall�gru_14/StatefulPartitionedCall�
gru_12/StatefulPartitionedCallStatefulPartitionedCallinputsgru_12_2432153gru_12_2432155gru_12_2432157*
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2432116�
gru_13/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0gru_13_2432160gru_13_2432162gru_13_2432164*
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431941�
gru_14/StatefulPartitionedCallStatefulPartitionedCall'gru_13/StatefulPartitionedCall:output:0gru_14_2432167gru_14_2432169gru_14_2432171*
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431766{
IdentityIdentity'gru_14/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_12/StatefulPartitionedCall^gru_13/StatefulPartitionedCall^gru_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2@
gru_13/StatefulPartitionedCallgru_13/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
.__inference_sequential_4_layer_call_fn_2432219
gru_12_input
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
StatefulPartitionedCallStatefulPartitionedCallgru_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432175t
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
_user_specified_namegru_12_input
�

�
.__inference_sequential_4_layer_call_fn_2432323

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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2431579t
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
� 
�
while_body_2430680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_25_2430702_0:	�/
while_gru_cell_25_2430704_0:
��.
while_gru_cell_25_2430706_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_25_2430702:	�-
while_gru_cell_25_2430704:
��,
while_gru_cell_25_2430706:	d���)while/gru_cell_25/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_2430702_0while_gru_cell_25_2430704_0while_gru_cell_25_2430706_0*
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430628�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������dx

while/NoOpNoOp*^while/gru_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_25_2430702while_gru_cell_25_2430702_0"8
while_gru_cell_25_2430704while_gru_cell_25_2430704_0"8
while_gru_cell_25_2430706while_gru_cell_25_2430706_0")
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
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435495

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
�
�
(__inference_gru_12_layer_call_fn_2433281

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
C__inference_gru_12_layer_call_and_return_conditional_losses_2431250u
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
�
#__inference__traced_restore_2435771
file_prefix=
*assignvariableop_gru_12_gru_cell_24_kernel:	�J
6assignvariableop_1_gru_12_gru_cell_24_recurrent_kernel:
��=
*assignvariableop_2_gru_12_gru_cell_24_bias:	�@
,assignvariableop_3_gru_13_gru_cell_25_kernel:
��I
6assignvariableop_4_gru_13_gru_cell_25_recurrent_kernel:	d�=
*assignvariableop_5_gru_13_gru_cell_25_bias:	�>
,assignvariableop_6_gru_14_gru_cell_26_kernel:dH
6assignvariableop_7_gru_14_gru_cell_26_recurrent_kernel:<
*assignvariableop_8_gru_14_gru_cell_26_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: G
4assignvariableop_16_adam_gru_12_gru_cell_24_kernel_m:	�R
>assignvariableop_17_adam_gru_12_gru_cell_24_recurrent_kernel_m:
��E
2assignvariableop_18_adam_gru_12_gru_cell_24_bias_m:	�H
4assignvariableop_19_adam_gru_13_gru_cell_25_kernel_m:
��Q
>assignvariableop_20_adam_gru_13_gru_cell_25_recurrent_kernel_m:	d�E
2assignvariableop_21_adam_gru_13_gru_cell_25_bias_m:	�F
4assignvariableop_22_adam_gru_14_gru_cell_26_kernel_m:dP
>assignvariableop_23_adam_gru_14_gru_cell_26_recurrent_kernel_m:D
2assignvariableop_24_adam_gru_14_gru_cell_26_bias_m:G
4assignvariableop_25_adam_gru_12_gru_cell_24_kernel_v:	�R
>assignvariableop_26_adam_gru_12_gru_cell_24_recurrent_kernel_v:
��E
2assignvariableop_27_adam_gru_12_gru_cell_24_bias_v:	�H
4assignvariableop_28_adam_gru_13_gru_cell_25_kernel_v:
��Q
>assignvariableop_29_adam_gru_13_gru_cell_25_recurrent_kernel_v:	d�E
2assignvariableop_30_adam_gru_13_gru_cell_25_bias_v:	�F
4assignvariableop_31_adam_gru_14_gru_cell_26_kernel_v:dP
>assignvariableop_32_adam_gru_14_gru_cell_26_recurrent_kernel_v:D
2assignvariableop_33_adam_gru_14_gru_cell_26_bias_v:
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
AssignVariableOpAssignVariableOp*assignvariableop_gru_12_gru_cell_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp6assignvariableop_1_gru_12_gru_cell_24_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_gru_12_gru_cell_24_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_gru_13_gru_cell_25_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_gru_13_gru_cell_25_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_13_gru_cell_25_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_gru_14_gru_cell_26_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_gru_14_gru_cell_26_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_gru_14_gru_cell_26_biasIdentity_8:output:0"/device:CPU:0*
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
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_gru_12_gru_cell_24_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_gru_12_gru_cell_24_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_gru_12_gru_cell_24_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_gru_13_gru_cell_25_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_gru_13_gru_cell_25_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_gru_13_gru_cell_25_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_gru_14_gru_cell_26_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_gru_14_gru_cell_26_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_gru_14_gru_cell_26_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_gru_12_gru_cell_24_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_gru_12_gru_cell_24_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_gru_12_gru_cell_24_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_gru_13_gru_cell_25_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_gru_13_gru_cell_25_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_gru_13_gru_cell_25_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_gru_14_gru_cell_26_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_gru_14_gru_cell_26_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_gru_14_gru_cell_26_bias_vIdentity_33:output:0"/device:CPU:0*
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
� 
�
while_body_2431018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_26_2431040_0:-
while_gru_cell_26_2431042_0:d-
while_gru_cell_26_2431044_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_26_2431040:+
while_gru_cell_26_2431042:d+
while_gru_cell_26_2431044:��)while/gru_cell_26/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_2431040_0while_gru_cell_26_2431042_0while_gru_cell_26_2431044_0*
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430966�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������x

while/NoOpNoOp*^while/gru_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_26_2431040while_gru_cell_26_2431040_0"8
while_gru_cell_26_2431042while_gru_cell_26_2431042_0"8
while_gru_cell_26_2431044while_gru_cell_26_2431044_0")
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
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 
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
while_cond_2434667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434667___redundant_placeholder05
1while_while_cond_2434667___redundant_placeholder15
1while_while_cond_2434667___redundant_placeholder25
1while_while_cond_2434667___redundant_placeholder3
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

�
-__inference_gru_cell_24_layer_call_fn_2435244

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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430290p
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
�M
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2433445
inputs_06
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while=
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2433356*
condR
while_cond_2433355*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
(__inference_gru_12_layer_call_fn_2433292

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
C__inference_gru_12_layer_call_and_return_conditional_losses_2432116u
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
�
�
while_cond_2430835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2430835___redundant_placeholder05
1while_while_cond_2430835___redundant_placeholder15
1while_while_cond_2430835___redundant_placeholder25
1while_while_cond_2430835___redundant_placeholder3
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
.__inference_sequential_4_layer_call_fn_2432346

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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432175t
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
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432269
gru_12_input!
gru_12_2432247:	�!
gru_12_2432249:	�"
gru_12_2432251:
��!
gru_13_2432254:	�"
gru_13_2432256:
��!
gru_13_2432258:	d� 
gru_14_2432261: 
gru_14_2432263:d 
gru_14_2432265:
identity��gru_12/StatefulPartitionedCall�gru_13/StatefulPartitionedCall�gru_14/StatefulPartitionedCall�
gru_12/StatefulPartitionedCallStatefulPartitionedCallgru_12_inputgru_12_2432247gru_12_2432249gru_12_2432251*
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2432116�
gru_13/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0gru_13_2432254gru_13_2432256gru_13_2432258*
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431941�
gru_14/StatefulPartitionedCallStatefulPartitionedCall'gru_13/StatefulPartitionedCall:output:0gru_14_2432261gru_14_2432263gru_14_2432265*
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431766{
IdentityIdentity'gru_14/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_12/StatefulPartitionedCall^gru_13/StatefulPartitionedCall^gru_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2@
gru_13/StatefulPartitionedCallgru_13/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_12_input
�
�
while_cond_2434317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434317___redundant_placeholder05
1while_while_cond_2434317___redundant_placeholder15
1while_while_cond_2434317___redundant_placeholder25
1while_while_cond_2434317___redundant_placeholder3
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
while_cond_2434011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434011___redundant_placeholder05
1while_while_cond_2434011___redundant_placeholder15
1while_while_cond_2434011___redundant_placeholder25
1while_while_cond_2434011___redundant_placeholder3
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
while_cond_2432026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2432026___redundant_placeholder05
1while_while_cond_2432026___redundant_placeholder15
1while_while_cond_2432026___redundant_placeholder25
1while_while_cond_2432026___redundant_placeholder3
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2434757
inputs_05
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while=
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2434668*
condR
while_cond_2434667*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
�	
�
gru_14_while_cond_2433158*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2,
(gru_14_while_less_gru_14_strided_slice_1C
?gru_14_while_gru_14_while_cond_2433158___redundant_placeholder0C
?gru_14_while_gru_14_while_cond_2433158___redundant_placeholder1C
?gru_14_while_gru_14_while_cond_2433158___redundant_placeholder2C
?gru_14_while_gru_14_while_cond_2433158___redundant_placeholder3
gru_14_while_identity
~
gru_14/while/LessLessgru_14_while_placeholder(gru_14_while_less_gru_14_strided_slice_1*
T0*
_output_shapes
: Y
gru_14/while/IdentityIdentitygru_14/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_14_while_identitygru_14/while/Identity:output:0*(
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
�
gru_13_while_cond_2433009*
&gru_13_while_gru_13_while_loop_counter0
,gru_13_while_gru_13_while_maximum_iterations
gru_13_while_placeholder
gru_13_while_placeholder_1
gru_13_while_placeholder_2,
(gru_13_while_less_gru_13_strided_slice_1C
?gru_13_while_gru_13_while_cond_2433009___redundant_placeholder0C
?gru_13_while_gru_13_while_cond_2433009___redundant_placeholder1C
?gru_13_while_gru_13_while_cond_2433009___redundant_placeholder2C
?gru_13_while_gru_13_while_cond_2433009___redundant_placeholder3
gru_13_while_identity
~
gru_13/while/LessLessgru_13_while_placeholder(gru_13_while_less_gru_13_strided_slice_1*
T0*
_output_shapes
: Y
gru_13/while/IdentityIdentitygru_13/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_13_while_identitygru_13/while/Identity:output:0*(
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
while_body_2434012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430147

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
�M
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2433751

inputs6
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while;
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2433662*
condR
while_cond_2433661*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
while_body_2434318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
while_cond_2430341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2430341___redundant_placeholder05
1while_while_cond_2430341___redundant_placeholder15
1while_while_cond_2430341___redundant_placeholder25
1while_while_cond_2430341___redundant_placeholder3
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
while_body_2434668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
(__inference_gru_14_layer_call_fn_2434582
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431082|
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
�	
�
gru_13_while_cond_2432558*
&gru_13_while_gru_13_while_loop_counter0
,gru_13_while_gru_13_while_maximum_iterations
gru_13_while_placeholder
gru_13_while_placeholder_1
gru_13_while_placeholder_2,
(gru_13_while_less_gru_13_strided_slice_1C
?gru_13_while_gru_13_while_cond_2432558___redundant_placeholder0C
?gru_13_while_gru_13_while_cond_2432558___redundant_placeholder1C
?gru_13_while_gru_13_while_cond_2432558___redundant_placeholder2C
?gru_13_while_gru_13_while_cond_2432558___redundant_placeholder3
gru_13_while_identity
~
gru_13/while/LessLessgru_13_while_placeholder(gru_13_while_less_gru_13_strided_slice_1*
T0*
_output_shapes
: Y
gru_13/while/IdentityIdentitygru_13/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_13_while_identitygru_13/while/Identity:output:0*(
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431766

inputs5
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while;
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2431677*
condR
while_cond_2431676*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�E
�	
gru_13_while_body_2432559*
&gru_13_while_gru_13_while_loop_counter0
,gru_13_while_gru_13_while_maximum_iterations
gru_13_while_placeholder
gru_13_while_placeholder_1
gru_13_while_placeholder_2)
%gru_13_while_gru_13_strided_slice_1_0e
agru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0E
2gru_13_while_gru_cell_25_readvariableop_resource_0:	�M
9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0:
��N
;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
gru_13_while_identity
gru_13_while_identity_1
gru_13_while_identity_2
gru_13_while_identity_3
gru_13_while_identity_4'
#gru_13_while_gru_13_strided_slice_1c
_gru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensorC
0gru_13_while_gru_cell_25_readvariableop_resource:	�K
7gru_13_while_gru_cell_25_matmul_readvariableop_resource:
��L
9gru_13_while_gru_cell_25_matmul_1_readvariableop_resource:	d���.gru_13/while/gru_cell_25/MatMul/ReadVariableOp�0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp�'gru_13/while/gru_cell_25/ReadVariableOp�
>gru_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
0gru_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0gru_13_while_placeholderGgru_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
'gru_13/while/gru_cell_25/ReadVariableOpReadVariableOp2gru_13_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_13/while/gru_cell_25/unstackUnpack/gru_13/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_13/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
gru_13/while/gru_cell_25/MatMulMatMul7gru_13/while/TensorArrayV2Read/TensorListGetItem:item:06gru_13/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_13/while/gru_cell_25/BiasAddBiasAdd)gru_13/while/gru_cell_25/MatMul:product:0)gru_13/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_13/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/while/gru_cell_25/splitSplit1gru_13/while/gru_cell_25/split/split_dim:output:0)gru_13/while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
!gru_13/while/gru_cell_25/MatMul_1MatMulgru_13_while_placeholder_28gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_13/while/gru_cell_25/BiasAdd_1BiasAdd+gru_13/while/gru_cell_25/MatMul_1:product:0)gru_13/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_13/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����u
*gru_13/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_13/while/gru_cell_25/split_1SplitV+gru_13/while/gru_cell_25/BiasAdd_1:output:0'gru_13/while/gru_cell_25/Const:output:03gru_13/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_13/while/gru_cell_25/addAddV2'gru_13/while/gru_cell_25/split:output:0)gru_13/while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������d
 gru_13/while/gru_cell_25/SigmoidSigmoid gru_13/while/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_1AddV2'gru_13/while/gru_cell_25/split:output:1)gru_13/while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������d�
"gru_13/while/gru_cell_25/Sigmoid_1Sigmoid"gru_13/while/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mulMul&gru_13/while/gru_cell_25/Sigmoid_1:y:0)gru_13/while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_2AddV2'gru_13/while/gru_cell_25/split:output:2 gru_13/while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������d�
"gru_13/while/gru_cell_25/Sigmoid_2Sigmoid"gru_13/while/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mul_1Mul$gru_13/while/gru_cell_25/Sigmoid:y:0gru_13_while_placeholder_2*
T0*'
_output_shapes
:���������dc
gru_13/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_13/while/gru_cell_25/subSub'gru_13/while/gru_cell_25/sub/x:output:0$gru_13/while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mul_2Mul gru_13/while/gru_cell_25/sub:z:0&gru_13/while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_3AddV2"gru_13/while/gru_cell_25/mul_1:z:0"gru_13/while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1gru_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_13_while_placeholder_1gru_13_while_placeholder"gru_13/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_13/while/addAddV2gru_13_while_placeholdergru_13/while/add/y:output:0*
T0*
_output_shapes
: V
gru_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_13/while/add_1AddV2&gru_13_while_gru_13_while_loop_countergru_13/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_13/while/IdentityIdentitygru_13/while/add_1:z:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_1Identity,gru_13_while_gru_13_while_maximum_iterations^gru_13/while/NoOp*
T0*
_output_shapes
: n
gru_13/while/Identity_2Identitygru_13/while/add:z:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_3IdentityAgru_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_4Identity"gru_13/while/gru_cell_25/add_3:z:0^gru_13/while/NoOp*
T0*'
_output_shapes
:���������d�
gru_13/while/NoOpNoOp/^gru_13/while/gru_cell_25/MatMul/ReadVariableOp1^gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp(^gru_13/while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_13_while_gru_13_strided_slice_1%gru_13_while_gru_13_strided_slice_1_0"x
9gru_13_while_gru_cell_25_matmul_1_readvariableop_resource;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0"t
7gru_13_while_gru_cell_25_matmul_readvariableop_resource9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0"f
0gru_13_while_gru_cell_25_readvariableop_resource2gru_13_while_gru_cell_25_readvariableop_resource_0"7
gru_13_while_identitygru_13/while/Identity:output:0";
gru_13_while_identity_1 gru_13/while/Identity_1:output:0";
gru_13_while_identity_2 gru_13/while/Identity_2:output:0";
gru_13_while_identity_3 gru_13/while/Identity_3:output:0";
gru_13_while_identity_4 gru_13/while/Identity_4:output:0"�
_gru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensoragru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.gru_13/while/gru_cell_25/MatMul/ReadVariableOp.gru_13/while/gru_cell_25/MatMul/ReadVariableOp2d
0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp2R
'gru_13/while/gru_cell_25/ReadVariableOp'gru_13/while/gru_cell_25/ReadVariableOp: 
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
�V
�
&sequential_4_gru_12_while_body_2429690D
@sequential_4_gru_12_while_sequential_4_gru_12_while_loop_counterJ
Fsequential_4_gru_12_while_sequential_4_gru_12_while_maximum_iterations)
%sequential_4_gru_12_while_placeholder+
'sequential_4_gru_12_while_placeholder_1+
'sequential_4_gru_12_while_placeholder_2C
?sequential_4_gru_12_while_sequential_4_gru_12_strided_slice_1_0
{sequential_4_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_12_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_4_gru_12_while_gru_cell_24_readvariableop_resource_0:	�Y
Fsequential_4_gru_12_while_gru_cell_24_matmul_readvariableop_resource_0:	�\
Hsequential_4_gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0:
��&
"sequential_4_gru_12_while_identity(
$sequential_4_gru_12_while_identity_1(
$sequential_4_gru_12_while_identity_2(
$sequential_4_gru_12_while_identity_3(
$sequential_4_gru_12_while_identity_4A
=sequential_4_gru_12_while_sequential_4_gru_12_strided_slice_1}
ysequential_4_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_12_tensorarrayunstack_tensorlistfromtensorP
=sequential_4_gru_12_while_gru_cell_24_readvariableop_resource:	�W
Dsequential_4_gru_12_while_gru_cell_24_matmul_readvariableop_resource:	�Z
Fsequential_4_gru_12_while_gru_cell_24_matmul_1_readvariableop_resource:
����;sequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOp�=sequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp�4sequential_4/gru_12/while/gru_cell_24/ReadVariableOp�
Ksequential_4/gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_4/gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_12_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_gru_12_while_placeholderTsequential_4/gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
4sequential_4/gru_12/while/gru_cell_24/ReadVariableOpReadVariableOp?sequential_4_gru_12_while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
-sequential_4/gru_12/while/gru_cell_24/unstackUnpack<sequential_4/gru_12/while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
;sequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOpReadVariableOpFsequential_4_gru_12_while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
,sequential_4/gru_12/while/gru_cell_24/MatMulMatMulDsequential_4/gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_4/gru_12/while/gru_cell_24/BiasAddBiasAdd6sequential_4/gru_12/while/gru_cell_24/MatMul:product:06sequential_4/gru_12/while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:�����������
5sequential_4/gru_12/while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_4/gru_12/while/gru_cell_24/splitSplit>sequential_4/gru_12/while/gru_cell_24/split/split_dim:output:06sequential_4/gru_12/while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
=sequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
.sequential_4/gru_12/while/gru_cell_24/MatMul_1MatMul'sequential_4_gru_12_while_placeholder_2Esequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/sequential_4/gru_12/while/gru_cell_24/BiasAdd_1BiasAdd8sequential_4/gru_12/while/gru_cell_24/MatMul_1:product:06sequential_4/gru_12/while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  �����
7sequential_4/gru_12/while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_4/gru_12/while/gru_cell_24/split_1SplitV8sequential_4/gru_12/while/gru_cell_24/BiasAdd_1:output:04sequential_4/gru_12/while/gru_cell_24/Const:output:0@sequential_4/gru_12/while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)sequential_4/gru_12/while/gru_cell_24/addAddV24sequential_4/gru_12/while/gru_cell_24/split:output:06sequential_4/gru_12/while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:�����������
-sequential_4/gru_12/while/gru_cell_24/SigmoidSigmoid-sequential_4/gru_12/while/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/add_1AddV24sequential_4/gru_12/while/gru_cell_24/split:output:16sequential_4/gru_12/while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:�����������
/sequential_4/gru_12/while/gru_cell_24/Sigmoid_1Sigmoid/sequential_4/gru_12/while/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
)sequential_4/gru_12/while/gru_cell_24/mulMul3sequential_4/gru_12/while/gru_cell_24/Sigmoid_1:y:06sequential_4/gru_12/while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/add_2AddV24sequential_4/gru_12/while/gru_cell_24/split:output:2-sequential_4/gru_12/while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:�����������
/sequential_4/gru_12/while/gru_cell_24/Sigmoid_2Sigmoid/sequential_4/gru_12/while/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/mul_1Mul1sequential_4/gru_12/while/gru_cell_24/Sigmoid:y:0'sequential_4_gru_12_while_placeholder_2*
T0*(
_output_shapes
:����������p
+sequential_4/gru_12/while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_4/gru_12/while/gru_cell_24/subSub4sequential_4/gru_12/while/gru_cell_24/sub/x:output:01sequential_4/gru_12/while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/mul_2Mul-sequential_4/gru_12/while/gru_cell_24/sub:z:03sequential_4/gru_12/while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
+sequential_4/gru_12/while/gru_cell_24/add_3AddV2/sequential_4/gru_12/while/gru_cell_24/mul_1:z:0/sequential_4/gru_12/while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
>sequential_4/gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_gru_12_while_placeholder_1%sequential_4_gru_12_while_placeholder/sequential_4/gru_12/while/gru_cell_24/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_4/gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_12/while/addAddV2%sequential_4_gru_12_while_placeholder(sequential_4/gru_12/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_12/while/add_1AddV2@sequential_4_gru_12_while_sequential_4_gru_12_while_loop_counter*sequential_4/gru_12/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_4/gru_12/while/IdentityIdentity#sequential_4/gru_12/while/add_1:z:0^sequential_4/gru_12/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_12/while/Identity_1IdentityFsequential_4_gru_12_while_sequential_4_gru_12_while_maximum_iterations^sequential_4/gru_12/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_12/while/Identity_2Identity!sequential_4/gru_12/while/add:z:0^sequential_4/gru_12/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_12/while/Identity_3IdentityNsequential_4/gru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/gru_12/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_12/while/Identity_4Identity/sequential_4/gru_12/while/gru_cell_24/add_3:z:0^sequential_4/gru_12/while/NoOp*
T0*(
_output_shapes
:�����������
sequential_4/gru_12/while/NoOpNoOp<^sequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOp>^sequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp5^sequential_4/gru_12/while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_4_gru_12_while_gru_cell_24_matmul_1_readvariableop_resourceHsequential_4_gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0"�
Dsequential_4_gru_12_while_gru_cell_24_matmul_readvariableop_resourceFsequential_4_gru_12_while_gru_cell_24_matmul_readvariableop_resource_0"�
=sequential_4_gru_12_while_gru_cell_24_readvariableop_resource?sequential_4_gru_12_while_gru_cell_24_readvariableop_resource_0"Q
"sequential_4_gru_12_while_identity+sequential_4/gru_12/while/Identity:output:0"U
$sequential_4_gru_12_while_identity_1-sequential_4/gru_12/while/Identity_1:output:0"U
$sequential_4_gru_12_while_identity_2-sequential_4/gru_12/while/Identity_2:output:0"U
$sequential_4_gru_12_while_identity_3-sequential_4/gru_12/while/Identity_3:output:0"U
$sequential_4_gru_12_while_identity_4-sequential_4/gru_12/while/Identity_4:output:0"�
=sequential_4_gru_12_while_sequential_4_gru_12_strided_slice_1?sequential_4_gru_12_while_sequential_4_gru_12_strided_slice_1_0"�
ysequential_4_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_12_tensorarrayunstack_tensorlistfromtensor{sequential_4_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2z
;sequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOp;sequential_4/gru_12/while/gru_cell_24/MatMul/ReadVariableOp2~
=sequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp=sequential_4/gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp2l
4sequential_4/gru_12/while/gru_cell_24/ReadVariableOp4sequential_4/gru_12/while/gru_cell_24/ReadVariableOp: 
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
�
�
(__inference_gru_14_layer_call_fn_2434593

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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431570t
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
while_cond_2433661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2433661___redundant_placeholder05
1while_while_cond_2433661___redundant_placeholder15
1while_while_cond_2433661___redundant_placeholder25
1while_while_cond_2433661___redundant_placeholder3
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
�V
�
&sequential_4_gru_14_while_body_2429988D
@sequential_4_gru_14_while_sequential_4_gru_14_while_loop_counterJ
Fsequential_4_gru_14_while_sequential_4_gru_14_while_maximum_iterations)
%sequential_4_gru_14_while_placeholder+
'sequential_4_gru_14_while_placeholder_1+
'sequential_4_gru_14_while_placeholder_2C
?sequential_4_gru_14_while_sequential_4_gru_14_strided_slice_1_0
{sequential_4_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_14_tensorarrayunstack_tensorlistfromtensor_0Q
?sequential_4_gru_14_while_gru_cell_26_readvariableop_resource_0:X
Fsequential_4_gru_14_while_gru_cell_26_matmul_readvariableop_resource_0:dZ
Hsequential_4_gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0:&
"sequential_4_gru_14_while_identity(
$sequential_4_gru_14_while_identity_1(
$sequential_4_gru_14_while_identity_2(
$sequential_4_gru_14_while_identity_3(
$sequential_4_gru_14_while_identity_4A
=sequential_4_gru_14_while_sequential_4_gru_14_strided_slice_1}
ysequential_4_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_14_tensorarrayunstack_tensorlistfromtensorO
=sequential_4_gru_14_while_gru_cell_26_readvariableop_resource:V
Dsequential_4_gru_14_while_gru_cell_26_matmul_readvariableop_resource:dX
Fsequential_4_gru_14_while_gru_cell_26_matmul_1_readvariableop_resource:��;sequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOp�=sequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp�4sequential_4/gru_14/while/gru_cell_26/ReadVariableOp�
Ksequential_4/gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
=sequential_4/gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_14_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_gru_14_while_placeholderTsequential_4/gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
4sequential_4/gru_14/while/gru_cell_26/ReadVariableOpReadVariableOp?sequential_4_gru_14_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
-sequential_4/gru_14/while/gru_cell_26/unstackUnpack<sequential_4/gru_14/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
;sequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOpFsequential_4_gru_14_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
,sequential_4/gru_14/while/gru_cell_26/MatMulMatMulDsequential_4/gru_14/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_4/gru_14/while/gru_cell_26/BiasAddBiasAdd6sequential_4/gru_14/while/gru_cell_26/MatMul:product:06sequential_4/gru_14/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:����������
5sequential_4/gru_14/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_4/gru_14/while/gru_cell_26/splitSplit>sequential_4/gru_14/while/gru_cell_26/split/split_dim:output:06sequential_4/gru_14/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
=sequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
.sequential_4/gru_14/while/gru_cell_26/MatMul_1MatMul'sequential_4_gru_14_while_placeholder_2Esequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_4/gru_14/while/gru_cell_26/BiasAdd_1BiasAdd8sequential_4/gru_14/while/gru_cell_26/MatMul_1:product:06sequential_4/gru_14/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      �����
7sequential_4/gru_14/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_4/gru_14/while/gru_cell_26/split_1SplitV8sequential_4/gru_14/while/gru_cell_26/BiasAdd_1:output:04sequential_4/gru_14/while/gru_cell_26/Const:output:0@sequential_4/gru_14/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)sequential_4/gru_14/while/gru_cell_26/addAddV24sequential_4/gru_14/while/gru_cell_26/split:output:06sequential_4/gru_14/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:����������
-sequential_4/gru_14/while/gru_cell_26/SigmoidSigmoid-sequential_4/gru_14/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/add_1AddV24sequential_4/gru_14/while/gru_cell_26/split:output:16sequential_4/gru_14/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:����������
/sequential_4/gru_14/while/gru_cell_26/Sigmoid_1Sigmoid/sequential_4/gru_14/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
)sequential_4/gru_14/while/gru_cell_26/mulMul3sequential_4/gru_14/while/gru_cell_26/Sigmoid_1:y:06sequential_4/gru_14/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/add_2AddV24sequential_4/gru_14/while/gru_cell_26/split:output:2-sequential_4/gru_14/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:����������
.sequential_4/gru_14/while/gru_cell_26/SoftplusSoftplus/sequential_4/gru_14/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/mul_1Mul1sequential_4/gru_14/while/gru_cell_26/Sigmoid:y:0'sequential_4_gru_14_while_placeholder_2*
T0*'
_output_shapes
:���������p
+sequential_4/gru_14/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_4/gru_14/while/gru_cell_26/subSub4sequential_4/gru_14/while/gru_cell_26/sub/x:output:01sequential_4/gru_14/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/mul_2Mul-sequential_4/gru_14/while/gru_cell_26/sub:z:0<sequential_4/gru_14/while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
+sequential_4/gru_14/while/gru_cell_26/add_3AddV2/sequential_4/gru_14/while/gru_cell_26/mul_1:z:0/sequential_4/gru_14/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
>sequential_4/gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_gru_14_while_placeholder_1%sequential_4_gru_14_while_placeholder/sequential_4/gru_14/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_4/gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_14/while/addAddV2%sequential_4_gru_14_while_placeholder(sequential_4/gru_14/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_14/while/add_1AddV2@sequential_4_gru_14_while_sequential_4_gru_14_while_loop_counter*sequential_4/gru_14/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_4/gru_14/while/IdentityIdentity#sequential_4/gru_14/while/add_1:z:0^sequential_4/gru_14/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_14/while/Identity_1IdentityFsequential_4_gru_14_while_sequential_4_gru_14_while_maximum_iterations^sequential_4/gru_14/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_14/while/Identity_2Identity!sequential_4/gru_14/while/add:z:0^sequential_4/gru_14/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_14/while/Identity_3IdentityNsequential_4/gru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/gru_14/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_14/while/Identity_4Identity/sequential_4/gru_14/while/gru_cell_26/add_3:z:0^sequential_4/gru_14/while/NoOp*
T0*'
_output_shapes
:����������
sequential_4/gru_14/while/NoOpNoOp<^sequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOp>^sequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp5^sequential_4/gru_14/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_4_gru_14_while_gru_cell_26_matmul_1_readvariableop_resourceHsequential_4_gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0"�
Dsequential_4_gru_14_while_gru_cell_26_matmul_readvariableop_resourceFsequential_4_gru_14_while_gru_cell_26_matmul_readvariableop_resource_0"�
=sequential_4_gru_14_while_gru_cell_26_readvariableop_resource?sequential_4_gru_14_while_gru_cell_26_readvariableop_resource_0"Q
"sequential_4_gru_14_while_identity+sequential_4/gru_14/while/Identity:output:0"U
$sequential_4_gru_14_while_identity_1-sequential_4/gru_14/while/Identity_1:output:0"U
$sequential_4_gru_14_while_identity_2-sequential_4/gru_14/while/Identity_2:output:0"U
$sequential_4_gru_14_while_identity_3-sequential_4/gru_14/while/Identity_3:output:0"U
$sequential_4_gru_14_while_identity_4-sequential_4/gru_14/while/Identity_4:output:0"�
=sequential_4_gru_14_while_sequential_4_gru_14_strided_slice_1?sequential_4_gru_14_while_sequential_4_gru_14_strided_slice_1_0"�
ysequential_4_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_14_tensorarrayunstack_tensorlistfromtensor{sequential_4_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2z
;sequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOp;sequential_4/gru_14/while/gru_cell_26/MatMul/ReadVariableOp2~
=sequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp=sequential_4/gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp2l
4sequential_4/gru_14/while/gru_cell_26/ReadVariableOp4sequential_4/gru_14/while/gru_cell_26/ReadVariableOp: 
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
��
�

"__inference__wrapped_model_2430077
gru_12_inputJ
7sequential_4_gru_12_gru_cell_24_readvariableop_resource:	�Q
>sequential_4_gru_12_gru_cell_24_matmul_readvariableop_resource:	�T
@sequential_4_gru_12_gru_cell_24_matmul_1_readvariableop_resource:
��J
7sequential_4_gru_13_gru_cell_25_readvariableop_resource:	�R
>sequential_4_gru_13_gru_cell_25_matmul_readvariableop_resource:
��S
@sequential_4_gru_13_gru_cell_25_matmul_1_readvariableop_resource:	d�I
7sequential_4_gru_14_gru_cell_26_readvariableop_resource:P
>sequential_4_gru_14_gru_cell_26_matmul_readvariableop_resource:dR
@sequential_4_gru_14_gru_cell_26_matmul_1_readvariableop_resource:
identity��5sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOp�7sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOp�.sequential_4/gru_12/gru_cell_24/ReadVariableOp�sequential_4/gru_12/while�5sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOp�7sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOp�.sequential_4/gru_13/gru_cell_25/ReadVariableOp�sequential_4/gru_13/while�5sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOp�7sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOp�.sequential_4/gru_14/gru_cell_26/ReadVariableOp�sequential_4/gru_14/whileU
sequential_4/gru_12/ShapeShapegru_12_input*
T0*
_output_shapes
:q
'sequential_4/gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_4/gru_12/strided_sliceStridedSlice"sequential_4/gru_12/Shape:output:00sequential_4/gru_12/strided_slice/stack:output:02sequential_4/gru_12/strided_slice/stack_1:output:02sequential_4/gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_4/gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
 sequential_4/gru_12/zeros/packedPack*sequential_4/gru_12/strided_slice:output:0+sequential_4/gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_4/gru_12/zerosFill)sequential_4/gru_12/zeros/packed:output:0(sequential_4/gru_12/zeros/Const:output:0*
T0*(
_output_shapes
:����������w
"sequential_4/gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_12/transpose	Transposegru_12_input+sequential_4/gru_12/transpose/perm:output:0*
T0*,
_output_shapes
:����������l
sequential_4/gru_12/Shape_1Shape!sequential_4/gru_12/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_12/strided_slice_1StridedSlice$sequential_4/gru_12/Shape_1:output:02sequential_4/gru_12/strided_slice_1/stack:output:04sequential_4/gru_12/strided_slice_1/stack_1:output:04sequential_4/gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_4/gru_12/TensorArrayV2TensorListReserve8sequential_4/gru_12/TensorArrayV2/element_shape:output:0,sequential_4/gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_4/gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
;sequential_4/gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/gru_12/transpose:y:0Rsequential_4/gru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_4/gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_12/strided_slice_2StridedSlice!sequential_4/gru_12/transpose:y:02sequential_4/gru_12/strided_slice_2/stack:output:04sequential_4/gru_12/strided_slice_2/stack_1:output:04sequential_4/gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
.sequential_4/gru_12/gru_cell_24/ReadVariableOpReadVariableOp7sequential_4_gru_12_gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'sequential_4/gru_12/gru_cell_24/unstackUnpack6sequential_4/gru_12/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
5sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOpReadVariableOp>sequential_4_gru_12_gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
&sequential_4/gru_12/gru_cell_24/MatMulMatMul,sequential_4/gru_12/strided_slice_2:output:0=sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_4/gru_12/gru_cell_24/BiasAddBiasAdd0sequential_4/gru_12/gru_cell_24/MatMul:product:00sequential_4/gru_12/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������z
/sequential_4/gru_12/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_4/gru_12/gru_cell_24/splitSplit8sequential_4/gru_12/gru_cell_24/split/split_dim:output:00sequential_4/gru_12/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
7sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_gru_12_gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
(sequential_4/gru_12/gru_cell_24/MatMul_1MatMul"sequential_4/gru_12/zeros:output:0?sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)sequential_4/gru_12/gru_cell_24/BiasAdd_1BiasAdd2sequential_4/gru_12/gru_cell_24/MatMul_1:product:00sequential_4/gru_12/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������z
%sequential_4/gru_12/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����|
1sequential_4/gru_12/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_4/gru_12/gru_cell_24/split_1SplitV2sequential_4/gru_12/gru_cell_24/BiasAdd_1:output:0.sequential_4/gru_12/gru_cell_24/Const:output:0:sequential_4/gru_12/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#sequential_4/gru_12/gru_cell_24/addAddV2.sequential_4/gru_12/gru_cell_24/split:output:00sequential_4/gru_12/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:�����������
'sequential_4/gru_12/gru_cell_24/SigmoidSigmoid'sequential_4/gru_12/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
%sequential_4/gru_12/gru_cell_24/add_1AddV2.sequential_4/gru_12/gru_cell_24/split:output:10sequential_4/gru_12/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:�����������
)sequential_4/gru_12/gru_cell_24/Sigmoid_1Sigmoid)sequential_4/gru_12/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
#sequential_4/gru_12/gru_cell_24/mulMul-sequential_4/gru_12/gru_cell_24/Sigmoid_1:y:00sequential_4/gru_12/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
%sequential_4/gru_12/gru_cell_24/add_2AddV2.sequential_4/gru_12/gru_cell_24/split:output:2'sequential_4/gru_12/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:�����������
)sequential_4/gru_12/gru_cell_24/Sigmoid_2Sigmoid)sequential_4/gru_12/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
%sequential_4/gru_12/gru_cell_24/mul_1Mul+sequential_4/gru_12/gru_cell_24/Sigmoid:y:0"sequential_4/gru_12/zeros:output:0*
T0*(
_output_shapes
:����������j
%sequential_4/gru_12/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_4/gru_12/gru_cell_24/subSub.sequential_4/gru_12/gru_cell_24/sub/x:output:0+sequential_4/gru_12/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
%sequential_4/gru_12/gru_cell_24/mul_2Mul'sequential_4/gru_12/gru_cell_24/sub:z:0-sequential_4/gru_12/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
%sequential_4/gru_12/gru_cell_24/add_3AddV2)sequential_4/gru_12/gru_cell_24/mul_1:z:0)sequential_4/gru_12/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
1sequential_4/gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
#sequential_4/gru_12/TensorArrayV2_1TensorListReserve:sequential_4/gru_12/TensorArrayV2_1/element_shape:output:0,sequential_4/gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_4/gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_4/gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_4/gru_12/whileWhile/sequential_4/gru_12/while/loop_counter:output:05sequential_4/gru_12/while/maximum_iterations:output:0!sequential_4/gru_12/time:output:0,sequential_4/gru_12/TensorArrayV2_1:handle:0"sequential_4/gru_12/zeros:output:0,sequential_4/gru_12/strided_slice_1:output:0Ksequential_4/gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_4_gru_12_gru_cell_24_readvariableop_resource>sequential_4_gru_12_gru_cell_24_matmul_readvariableop_resource@sequential_4_gru_12_gru_cell_24_matmul_1_readvariableop_resource*
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
&sequential_4_gru_12_while_body_2429690*2
cond*R(
&sequential_4_gru_12_while_cond_2429689*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
Dsequential_4/gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
6sequential_4/gru_12/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/gru_12/while:output:3Msequential_4/gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0|
)sequential_4/gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_4/gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_12/strided_slice_3StridedSlice?sequential_4/gru_12/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/gru_12/strided_slice_3/stack:output:04sequential_4/gru_12/strided_slice_3/stack_1:output:04sequential_4/gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_masky
$sequential_4/gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_12/transpose_1	Transpose?sequential_4/gru_12/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/gru_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������o
sequential_4/gru_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_4/gru_13/ShapeShape#sequential_4/gru_12/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_4/gru_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/gru_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/gru_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_4/gru_13/strided_sliceStridedSlice"sequential_4/gru_13/Shape:output:00sequential_4/gru_13/strided_slice/stack:output:02sequential_4/gru_13/strided_slice/stack_1:output:02sequential_4/gru_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/gru_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
 sequential_4/gru_13/zeros/packedPack*sequential_4/gru_13/strided_slice:output:0+sequential_4/gru_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/gru_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_4/gru_13/zerosFill)sequential_4/gru_13/zeros/packed:output:0(sequential_4/gru_13/zeros/Const:output:0*
T0*'
_output_shapes
:���������dw
"sequential_4/gru_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_13/transpose	Transpose#sequential_4/gru_12/transpose_1:y:0+sequential_4/gru_13/transpose/perm:output:0*
T0*-
_output_shapes
:�����������l
sequential_4/gru_13/Shape_1Shape!sequential_4/gru_13/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/gru_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_13/strided_slice_1StridedSlice$sequential_4/gru_13/Shape_1:output:02sequential_4/gru_13/strided_slice_1/stack:output:04sequential_4/gru_13/strided_slice_1/stack_1:output:04sequential_4/gru_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/gru_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_4/gru_13/TensorArrayV2TensorListReserve8sequential_4/gru_13/TensorArrayV2/element_shape:output:0,sequential_4/gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_4/gru_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
;sequential_4/gru_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/gru_13/transpose:y:0Rsequential_4/gru_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_4/gru_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_13/strided_slice_2StridedSlice!sequential_4/gru_13/transpose:y:02sequential_4/gru_13/strided_slice_2/stack:output:04sequential_4/gru_13/strided_slice_2/stack_1:output:04sequential_4/gru_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
.sequential_4/gru_13/gru_cell_25/ReadVariableOpReadVariableOp7sequential_4_gru_13_gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'sequential_4/gru_13/gru_cell_25/unstackUnpack6sequential_4/gru_13/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
5sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOpReadVariableOp>sequential_4_gru_13_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
&sequential_4/gru_13/gru_cell_25/MatMulMatMul,sequential_4/gru_13/strided_slice_2:output:0=sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_4/gru_13/gru_cell_25/BiasAddBiasAdd0sequential_4/gru_13/gru_cell_25/MatMul:product:00sequential_4/gru_13/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������z
/sequential_4/gru_13/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_4/gru_13/gru_cell_25/splitSplit8sequential_4/gru_13/gru_cell_25/split/split_dim:output:00sequential_4/gru_13/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
7sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_gru_13_gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
(sequential_4/gru_13/gru_cell_25/MatMul_1MatMul"sequential_4/gru_13/zeros:output:0?sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)sequential_4/gru_13/gru_cell_25/BiasAdd_1BiasAdd2sequential_4/gru_13/gru_cell_25/MatMul_1:product:00sequential_4/gru_13/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������z
%sequential_4/gru_13/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����|
1sequential_4/gru_13/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_4/gru_13/gru_cell_25/split_1SplitV2sequential_4/gru_13/gru_cell_25/BiasAdd_1:output:0.sequential_4/gru_13/gru_cell_25/Const:output:0:sequential_4/gru_13/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#sequential_4/gru_13/gru_cell_25/addAddV2.sequential_4/gru_13/gru_cell_25/split:output:00sequential_4/gru_13/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������d�
'sequential_4/gru_13/gru_cell_25/SigmoidSigmoid'sequential_4/gru_13/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
%sequential_4/gru_13/gru_cell_25/add_1AddV2.sequential_4/gru_13/gru_cell_25/split:output:10sequential_4/gru_13/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������d�
)sequential_4/gru_13/gru_cell_25/Sigmoid_1Sigmoid)sequential_4/gru_13/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
#sequential_4/gru_13/gru_cell_25/mulMul-sequential_4/gru_13/gru_cell_25/Sigmoid_1:y:00sequential_4/gru_13/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
%sequential_4/gru_13/gru_cell_25/add_2AddV2.sequential_4/gru_13/gru_cell_25/split:output:2'sequential_4/gru_13/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������d�
)sequential_4/gru_13/gru_cell_25/Sigmoid_2Sigmoid)sequential_4/gru_13/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
%sequential_4/gru_13/gru_cell_25/mul_1Mul+sequential_4/gru_13/gru_cell_25/Sigmoid:y:0"sequential_4/gru_13/zeros:output:0*
T0*'
_output_shapes
:���������dj
%sequential_4/gru_13/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_4/gru_13/gru_cell_25/subSub.sequential_4/gru_13/gru_cell_25/sub/x:output:0+sequential_4/gru_13/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
%sequential_4/gru_13/gru_cell_25/mul_2Mul'sequential_4/gru_13/gru_cell_25/sub:z:0-sequential_4/gru_13/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
%sequential_4/gru_13/gru_cell_25/add_3AddV2)sequential_4/gru_13/gru_cell_25/mul_1:z:0)sequential_4/gru_13/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1sequential_4/gru_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
#sequential_4/gru_13/TensorArrayV2_1TensorListReserve:sequential_4/gru_13/TensorArrayV2_1/element_shape:output:0,sequential_4/gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_4/gru_13/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/gru_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_4/gru_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_4/gru_13/whileWhile/sequential_4/gru_13/while/loop_counter:output:05sequential_4/gru_13/while/maximum_iterations:output:0!sequential_4/gru_13/time:output:0,sequential_4/gru_13/TensorArrayV2_1:handle:0"sequential_4/gru_13/zeros:output:0,sequential_4/gru_13/strided_slice_1:output:0Ksequential_4/gru_13/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_4_gru_13_gru_cell_25_readvariableop_resource>sequential_4_gru_13_gru_cell_25_matmul_readvariableop_resource@sequential_4_gru_13_gru_cell_25_matmul_1_readvariableop_resource*
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
&sequential_4_gru_13_while_body_2429839*2
cond*R(
&sequential_4_gru_13_while_cond_2429838*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
Dsequential_4/gru_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
6sequential_4/gru_13/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/gru_13/while:output:3Msequential_4/gru_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0|
)sequential_4/gru_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_4/gru_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_13/strided_slice_3StridedSlice?sequential_4/gru_13/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/gru_13/strided_slice_3/stack:output:04sequential_4/gru_13/strided_slice_3/stack_1:output:04sequential_4/gru_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_masky
$sequential_4/gru_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_13/transpose_1	Transpose?sequential_4/gru_13/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/gru_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������do
sequential_4/gru_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_4/gru_14/ShapeShape#sequential_4/gru_13/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_4/gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_4/gru_14/strided_sliceStridedSlice"sequential_4/gru_14/Shape:output:00sequential_4/gru_14/strided_slice/stack:output:02sequential_4/gru_14/strided_slice/stack_1:output:02sequential_4/gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
 sequential_4/gru_14/zeros/packedPack*sequential_4/gru_14/strided_slice:output:0+sequential_4/gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_4/gru_14/zerosFill)sequential_4/gru_14/zeros/packed:output:0(sequential_4/gru_14/zeros/Const:output:0*
T0*'
_output_shapes
:���������w
"sequential_4/gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_14/transpose	Transpose#sequential_4/gru_13/transpose_1:y:0+sequential_4/gru_14/transpose/perm:output:0*
T0*,
_output_shapes
:����������dl
sequential_4/gru_14/Shape_1Shape!sequential_4/gru_14/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_14/strided_slice_1StridedSlice$sequential_4/gru_14/Shape_1:output:02sequential_4/gru_14/strided_slice_1/stack:output:04sequential_4/gru_14/strided_slice_1/stack_1:output:04sequential_4/gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
!sequential_4/gru_14/TensorArrayV2TensorListReserve8sequential_4/gru_14/TensorArrayV2/element_shape:output:0,sequential_4/gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Isequential_4/gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
;sequential_4/gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/gru_14/transpose:y:0Rsequential_4/gru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
)sequential_4/gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_14/strided_slice_2StridedSlice!sequential_4/gru_14/transpose:y:02sequential_4/gru_14/strided_slice_2/stack:output:04sequential_4/gru_14/strided_slice_2/stack_1:output:04sequential_4/gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
.sequential_4/gru_14/gru_cell_26/ReadVariableOpReadVariableOp7sequential_4_gru_14_gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0�
'sequential_4/gru_14/gru_cell_26/unstackUnpack6sequential_4/gru_14/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
5sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOpReadVariableOp>sequential_4_gru_14_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
&sequential_4/gru_14/gru_cell_26/MatMulMatMul,sequential_4/gru_14/strided_slice_2:output:0=sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential_4/gru_14/gru_cell_26/BiasAddBiasAdd0sequential_4/gru_14/gru_cell_26/MatMul:product:00sequential_4/gru_14/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������z
/sequential_4/gru_14/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_4/gru_14/gru_cell_26/splitSplit8sequential_4/gru_14/gru_cell_26/split/split_dim:output:00sequential_4/gru_14/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
7sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_gru_14_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
(sequential_4/gru_14/gru_cell_26/MatMul_1MatMul"sequential_4/gru_14/zeros:output:0?sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential_4/gru_14/gru_cell_26/BiasAdd_1BiasAdd2sequential_4/gru_14/gru_cell_26/MatMul_1:product:00sequential_4/gru_14/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������z
%sequential_4/gru_14/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����|
1sequential_4/gru_14/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_4/gru_14/gru_cell_26/split_1SplitV2sequential_4/gru_14/gru_cell_26/BiasAdd_1:output:0.sequential_4/gru_14/gru_cell_26/Const:output:0:sequential_4/gru_14/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#sequential_4/gru_14/gru_cell_26/addAddV2.sequential_4/gru_14/gru_cell_26/split:output:00sequential_4/gru_14/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:����������
'sequential_4/gru_14/gru_cell_26/SigmoidSigmoid'sequential_4/gru_14/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
%sequential_4/gru_14/gru_cell_26/add_1AddV2.sequential_4/gru_14/gru_cell_26/split:output:10sequential_4/gru_14/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:����������
)sequential_4/gru_14/gru_cell_26/Sigmoid_1Sigmoid)sequential_4/gru_14/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
#sequential_4/gru_14/gru_cell_26/mulMul-sequential_4/gru_14/gru_cell_26/Sigmoid_1:y:00sequential_4/gru_14/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
%sequential_4/gru_14/gru_cell_26/add_2AddV2.sequential_4/gru_14/gru_cell_26/split:output:2'sequential_4/gru_14/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:����������
(sequential_4/gru_14/gru_cell_26/SoftplusSoftplus)sequential_4/gru_14/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
%sequential_4/gru_14/gru_cell_26/mul_1Mul+sequential_4/gru_14/gru_cell_26/Sigmoid:y:0"sequential_4/gru_14/zeros:output:0*
T0*'
_output_shapes
:���������j
%sequential_4/gru_14/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential_4/gru_14/gru_cell_26/subSub.sequential_4/gru_14/gru_cell_26/sub/x:output:0+sequential_4/gru_14/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
%sequential_4/gru_14/gru_cell_26/mul_2Mul'sequential_4/gru_14/gru_cell_26/sub:z:06sequential_4/gru_14/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
%sequential_4/gru_14/gru_cell_26/add_3AddV2)sequential_4/gru_14/gru_cell_26/mul_1:z:0)sequential_4/gru_14/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
1sequential_4/gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#sequential_4/gru_14/TensorArrayV2_1TensorListReserve:sequential_4/gru_14/TensorArrayV2_1/element_shape:output:0,sequential_4/gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���Z
sequential_4/gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������h
&sequential_4/gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_4/gru_14/whileWhile/sequential_4/gru_14/while/loop_counter:output:05sequential_4/gru_14/while/maximum_iterations:output:0!sequential_4/gru_14/time:output:0,sequential_4/gru_14/TensorArrayV2_1:handle:0"sequential_4/gru_14/zeros:output:0,sequential_4/gru_14/strided_slice_1:output:0Ksequential_4/gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_4_gru_14_gru_cell_26_readvariableop_resource>sequential_4_gru_14_gru_cell_26_matmul_readvariableop_resource@sequential_4_gru_14_gru_cell_26_matmul_1_readvariableop_resource*
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
&sequential_4_gru_14_while_body_2429988*2
cond*R(
&sequential_4_gru_14_while_cond_2429987*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
Dsequential_4/gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6sequential_4/gru_14/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/gru_14/while:output:3Msequential_4/gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0|
)sequential_4/gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������u
+sequential_4/gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_4/gru_14/strided_slice_3StridedSlice?sequential_4/gru_14/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/gru_14/strided_slice_3/stack:output:04sequential_4/gru_14/strided_slice_3/stack_1:output:04sequential_4/gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
$sequential_4/gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_4/gru_14/transpose_1	Transpose?sequential_4/gru_14/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������o
sequential_4/gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    w
IdentityIdentity#sequential_4/gru_14/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp6^sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOp8^sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOp/^sequential_4/gru_12/gru_cell_24/ReadVariableOp^sequential_4/gru_12/while6^sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOp8^sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOp/^sequential_4/gru_13/gru_cell_25/ReadVariableOp^sequential_4/gru_13/while6^sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOp8^sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOp/^sequential_4/gru_14/gru_cell_26/ReadVariableOp^sequential_4/gru_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2n
5sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOp5sequential_4/gru_12/gru_cell_24/MatMul/ReadVariableOp2r
7sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOp7sequential_4/gru_12/gru_cell_24/MatMul_1/ReadVariableOp2`
.sequential_4/gru_12/gru_cell_24/ReadVariableOp.sequential_4/gru_12/gru_cell_24/ReadVariableOp26
sequential_4/gru_12/whilesequential_4/gru_12/while2n
5sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOp5sequential_4/gru_13/gru_cell_25/MatMul/ReadVariableOp2r
7sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOp7sequential_4/gru_13/gru_cell_25/MatMul_1/ReadVariableOp2`
.sequential_4/gru_13/gru_cell_25/ReadVariableOp.sequential_4/gru_13/gru_cell_25/ReadVariableOp26
sequential_4/gru_13/whilesequential_4/gru_13/while2n
5sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOp5sequential_4/gru_14/gru_cell_26/MatMul/ReadVariableOp2r
7sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOp7sequential_4/gru_14/gru_cell_26/MatMul_1/ReadVariableOp2`
.sequential_4/gru_14/gru_cell_26/ReadVariableOp.sequential_4/gru_14/gru_cell_26/ReadVariableOp26
sequential_4/gru_14/whilesequential_4/gru_14/while:Z V
,
_output_shapes
:����������
&
_user_specified_namegru_12_input
�=
�
while_body_2432027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
while_body_2431321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
�
�
(__inference_gru_12_layer_call_fn_2433259
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2430224}
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
�	
�
gru_12_while_cond_2432409*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2,
(gru_12_while_less_gru_12_strided_slice_1C
?gru_12_while_gru_12_while_cond_2432409___redundant_placeholder0C
?gru_12_while_gru_12_while_cond_2432409___redundant_placeholder1C
?gru_12_while_gru_12_while_cond_2432409___redundant_placeholder2C
?gru_12_while_gru_12_while_cond_2432409___redundant_placeholder3
gru_12_while_identity
~
gru_12/while/LessLessgru_12_while_placeholder(gru_12_while_less_gru_12_strided_slice_1*
T0*
_output_shapes
: Y
gru_12/while/IdentityIdentitygru_12/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_12_while_identitygru_12/while/Identity:output:0*(
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
while_cond_2430497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2430497___redundant_placeholder05
1while_while_cond_2430497___redundant_placeholder15
1while_while_cond_2430497___redundant_placeholder25
1while_while_cond_2430497___redundant_placeholder3
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
(__inference_gru_14_layer_call_fn_2434571
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2430900|
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
�
�
&sequential_4_gru_12_while_cond_2429689D
@sequential_4_gru_12_while_sequential_4_gru_12_while_loop_counterJ
Fsequential_4_gru_12_while_sequential_4_gru_12_while_maximum_iterations)
%sequential_4_gru_12_while_placeholder+
'sequential_4_gru_12_while_placeholder_1+
'sequential_4_gru_12_while_placeholder_2F
Bsequential_4_gru_12_while_less_sequential_4_gru_12_strided_slice_1]
Ysequential_4_gru_12_while_sequential_4_gru_12_while_cond_2429689___redundant_placeholder0]
Ysequential_4_gru_12_while_sequential_4_gru_12_while_cond_2429689___redundant_placeholder1]
Ysequential_4_gru_12_while_sequential_4_gru_12_while_cond_2429689___redundant_placeholder2]
Ysequential_4_gru_12_while_sequential_4_gru_12_while_cond_2429689___redundant_placeholder3&
"sequential_4_gru_12_while_identity
�
sequential_4/gru_12/while/LessLess%sequential_4_gru_12_while_placeholderBsequential_4_gru_12_while_less_sequential_4_gru_12_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/gru_12/while/IdentityIdentity"sequential_4/gru_12/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_gru_12_while_identity+sequential_4/gru_12/while/Identity:output:0*(
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430290

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

�
%__inference_signature_wrapper_2432300
gru_12_input
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
StatefulPartitionedCallStatefulPartitionedCallgru_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
"__inference__wrapped_model_2430077t
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
_user_specified_namegru_12_input
�V
�
&sequential_4_gru_13_while_body_2429839D
@sequential_4_gru_13_while_sequential_4_gru_13_while_loop_counterJ
Fsequential_4_gru_13_while_sequential_4_gru_13_while_maximum_iterations)
%sequential_4_gru_13_while_placeholder+
'sequential_4_gru_13_while_placeholder_1+
'sequential_4_gru_13_while_placeholder_2C
?sequential_4_gru_13_while_sequential_4_gru_13_strided_slice_1_0
{sequential_4_gru_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_13_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_4_gru_13_while_gru_cell_25_readvariableop_resource_0:	�Z
Fsequential_4_gru_13_while_gru_cell_25_matmul_readvariableop_resource_0:
��[
Hsequential_4_gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�&
"sequential_4_gru_13_while_identity(
$sequential_4_gru_13_while_identity_1(
$sequential_4_gru_13_while_identity_2(
$sequential_4_gru_13_while_identity_3(
$sequential_4_gru_13_while_identity_4A
=sequential_4_gru_13_while_sequential_4_gru_13_strided_slice_1}
ysequential_4_gru_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_13_tensorarrayunstack_tensorlistfromtensorP
=sequential_4_gru_13_while_gru_cell_25_readvariableop_resource:	�X
Dsequential_4_gru_13_while_gru_cell_25_matmul_readvariableop_resource:
��Y
Fsequential_4_gru_13_while_gru_cell_25_matmul_1_readvariableop_resource:	d���;sequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOp�=sequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp�4sequential_4/gru_13/while/gru_cell_25/ReadVariableOp�
Ksequential_4/gru_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
=sequential_4/gru_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_gru_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_13_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_gru_13_while_placeholderTsequential_4/gru_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
4sequential_4/gru_13/while/gru_cell_25/ReadVariableOpReadVariableOp?sequential_4_gru_13_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
-sequential_4/gru_13/while/gru_cell_25/unstackUnpack<sequential_4/gru_13/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
;sequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpFsequential_4_gru_13_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
,sequential_4/gru_13/while/gru_cell_25/MatMulMatMulDsequential_4/gru_13/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_4/gru_13/while/gru_cell_25/BiasAddBiasAdd6sequential_4/gru_13/while/gru_cell_25/MatMul:product:06sequential_4/gru_13/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:�����������
5sequential_4/gru_13/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+sequential_4/gru_13/while/gru_cell_25/splitSplit>sequential_4/gru_13/while/gru_cell_25/split/split_dim:output:06sequential_4/gru_13/while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
=sequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
.sequential_4/gru_13/while/gru_cell_25/MatMul_1MatMul'sequential_4_gru_13_while_placeholder_2Esequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/sequential_4/gru_13/while/gru_cell_25/BiasAdd_1BiasAdd8sequential_4/gru_13/while/gru_cell_25/MatMul_1:product:06sequential_4/gru_13/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:�����������
+sequential_4/gru_13/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   �����
7sequential_4/gru_13/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-sequential_4/gru_13/while/gru_cell_25/split_1SplitV8sequential_4/gru_13/while/gru_cell_25/BiasAdd_1:output:04sequential_4/gru_13/while/gru_cell_25/Const:output:0@sequential_4/gru_13/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)sequential_4/gru_13/while/gru_cell_25/addAddV24sequential_4/gru_13/while/gru_cell_25/split:output:06sequential_4/gru_13/while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������d�
-sequential_4/gru_13/while/gru_cell_25/SigmoidSigmoid-sequential_4/gru_13/while/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
+sequential_4/gru_13/while/gru_cell_25/add_1AddV24sequential_4/gru_13/while/gru_cell_25/split:output:16sequential_4/gru_13/while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������d�
/sequential_4/gru_13/while/gru_cell_25/Sigmoid_1Sigmoid/sequential_4/gru_13/while/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
)sequential_4/gru_13/while/gru_cell_25/mulMul3sequential_4/gru_13/while/gru_cell_25/Sigmoid_1:y:06sequential_4/gru_13/while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
+sequential_4/gru_13/while/gru_cell_25/add_2AddV24sequential_4/gru_13/while/gru_cell_25/split:output:2-sequential_4/gru_13/while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������d�
/sequential_4/gru_13/while/gru_cell_25/Sigmoid_2Sigmoid/sequential_4/gru_13/while/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
+sequential_4/gru_13/while/gru_cell_25/mul_1Mul1sequential_4/gru_13/while/gru_cell_25/Sigmoid:y:0'sequential_4_gru_13_while_placeholder_2*
T0*'
_output_shapes
:���������dp
+sequential_4/gru_13/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)sequential_4/gru_13/while/gru_cell_25/subSub4sequential_4/gru_13/while/gru_cell_25/sub/x:output:01sequential_4/gru_13/while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
+sequential_4/gru_13/while/gru_cell_25/mul_2Mul-sequential_4/gru_13/while/gru_cell_25/sub:z:03sequential_4/gru_13/while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
+sequential_4/gru_13/while/gru_cell_25/add_3AddV2/sequential_4/gru_13/while/gru_cell_25/mul_1:z:0/sequential_4/gru_13/while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
>sequential_4/gru_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_gru_13_while_placeholder_1%sequential_4_gru_13_while_placeholder/sequential_4/gru_13/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype0:���a
sequential_4/gru_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_13/while/addAddV2%sequential_4_gru_13_while_placeholder(sequential_4/gru_13/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/gru_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_4/gru_13/while/add_1AddV2@sequential_4_gru_13_while_sequential_4_gru_13_while_loop_counter*sequential_4/gru_13/while/add_1/y:output:0*
T0*
_output_shapes
: �
"sequential_4/gru_13/while/IdentityIdentity#sequential_4/gru_13/while/add_1:z:0^sequential_4/gru_13/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_13/while/Identity_1IdentityFsequential_4_gru_13_while_sequential_4_gru_13_while_maximum_iterations^sequential_4/gru_13/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_13/while/Identity_2Identity!sequential_4/gru_13/while/add:z:0^sequential_4/gru_13/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_13/while/Identity_3IdentityNsequential_4/gru_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/gru_13/while/NoOp*
T0*
_output_shapes
: �
$sequential_4/gru_13/while/Identity_4Identity/sequential_4/gru_13/while/gru_cell_25/add_3:z:0^sequential_4/gru_13/while/NoOp*
T0*'
_output_shapes
:���������d�
sequential_4/gru_13/while/NoOpNoOp<^sequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOp>^sequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp5^sequential_4/gru_13/while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Fsequential_4_gru_13_while_gru_cell_25_matmul_1_readvariableop_resourceHsequential_4_gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0"�
Dsequential_4_gru_13_while_gru_cell_25_matmul_readvariableop_resourceFsequential_4_gru_13_while_gru_cell_25_matmul_readvariableop_resource_0"�
=sequential_4_gru_13_while_gru_cell_25_readvariableop_resource?sequential_4_gru_13_while_gru_cell_25_readvariableop_resource_0"Q
"sequential_4_gru_13_while_identity+sequential_4/gru_13/while/Identity:output:0"U
$sequential_4_gru_13_while_identity_1-sequential_4/gru_13/while/Identity_1:output:0"U
$sequential_4_gru_13_while_identity_2-sequential_4/gru_13/while/Identity_2:output:0"U
$sequential_4_gru_13_while_identity_3-sequential_4/gru_13/while/Identity_3:output:0"U
$sequential_4_gru_13_while_identity_4-sequential_4/gru_13/while/Identity_4:output:0"�
=sequential_4_gru_13_while_sequential_4_gru_13_strided_slice_1?sequential_4_gru_13_while_sequential_4_gru_13_strided_slice_1_0"�
ysequential_4_gru_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_13_tensorarrayunstack_tensorlistfromtensor{sequential_4_gru_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2z
;sequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOp;sequential_4/gru_13/while/gru_cell_25/MatMul/ReadVariableOp2~
=sequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp=sequential_4/gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp2l
4sequential_4/gru_13/while/gru_cell_25/ReadVariableOp4sequential_4/gru_13/while/gru_cell_25/ReadVariableOp: 
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
while_body_2430836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_26_2430858_0:-
while_gru_cell_26_2430860_0:d-
while_gru_cell_26_2430862_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_26_2430858:+
while_gru_cell_26_2430860:d+
while_gru_cell_26_2430862:��)while/gru_cell_26/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_2430858_0while_gru_cell_26_2430860_0while_gru_cell_26_2430862_0*
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430823�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������x

while/NoOpNoOp*^while/gru_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_26_2430858while_gru_cell_26_2430858_0"8
while_gru_cell_26_2430860while_gru_cell_26_2430860_0"8
while_gru_cell_26_2430862while_gru_cell_26_2430862_0")
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
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 
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
while_body_2431677
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
(__inference_gru_12_layer_call_fn_2433270
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2430406}
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
�
�
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435428

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
�
�
while_cond_2434164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434164___redundant_placeholder05
1while_while_cond_2434164___redundant_placeholder15
1while_while_cond_2434164___redundant_placeholder25
1while_while_cond_2434164___redundant_placeholder3
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
while_cond_2433355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2433355___redundant_placeholder05
1while_while_cond_2433355___redundant_placeholder15
1while_while_cond_2433355___redundant_placeholder25
1while_while_cond_2433355___redundant_placeholder3
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
�4
�
C__inference_gru_14_layer_call_and_return_conditional_losses_2431082

inputs%
gru_cell_26_2431006:%
gru_cell_26_2431008:d%
gru_cell_26_2431010:
identity��#gru_cell_26/StatefulPartitionedCall�while;
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
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_2431006gru_cell_26_2431008gru_cell_26_2431010*
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430966n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_2431006gru_cell_26_2431008gru_cell_26_2431010*
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
while_body_2431018*
condR
while_cond_2431017*8
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
NoOpNoOp$^gru_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2433248

inputs=
*gru_12_gru_cell_24_readvariableop_resource:	�D
1gru_12_gru_cell_24_matmul_readvariableop_resource:	�G
3gru_12_gru_cell_24_matmul_1_readvariableop_resource:
��=
*gru_13_gru_cell_25_readvariableop_resource:	�E
1gru_13_gru_cell_25_matmul_readvariableop_resource:
��F
3gru_13_gru_cell_25_matmul_1_readvariableop_resource:	d�<
*gru_14_gru_cell_26_readvariableop_resource:C
1gru_14_gru_cell_26_matmul_readvariableop_resource:dE
3gru_14_gru_cell_26_matmul_1_readvariableop_resource:
identity��(gru_12/gru_cell_24/MatMul/ReadVariableOp�*gru_12/gru_cell_24/MatMul_1/ReadVariableOp�!gru_12/gru_cell_24/ReadVariableOp�gru_12/while�(gru_13/gru_cell_25/MatMul/ReadVariableOp�*gru_13/gru_cell_25/MatMul_1/ReadVariableOp�!gru_13/gru_cell_25/ReadVariableOp�gru_13/while�(gru_14/gru_cell_26/MatMul/ReadVariableOp�*gru_14/gru_cell_26/MatMul_1/ReadVariableOp�!gru_14/gru_cell_26/ReadVariableOp�gru_14/whileB
gru_12/ShapeShapeinputs*
T0*
_output_shapes
:d
gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_sliceStridedSlicegru_12/Shape:output:0#gru_12/strided_slice/stack:output:0%gru_12/strided_slice/stack_1:output:0%gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
gru_12/zeros/packedPackgru_12/strided_slice:output:0gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_12/zerosFillgru_12/zeros/packed:output:0gru_12/zeros/Const:output:0*
T0*(
_output_shapes
:����������j
gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
gru_12/transpose	Transposeinputsgru_12/transpose/perm:output:0*
T0*,
_output_shapes
:����������R
gru_12/Shape_1Shapegru_12/transpose:y:0*
T0*
_output_shapes
:f
gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_1StridedSlicegru_12/Shape_1:output:0%gru_12/strided_slice_1/stack:output:0'gru_12/strided_slice_1/stack_1:output:0'gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/TensorArrayV2TensorListReserve+gru_12/TensorArrayV2/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
.gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_12/transpose:y:0Egru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_2StridedSlicegru_12/transpose:y:0%gru_12/strided_slice_2/stack:output:0'gru_12/strided_slice_2/stack_1:output:0'gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!gru_12/gru_cell_24/ReadVariableOpReadVariableOp*gru_12_gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_12/gru_cell_24/unstackUnpack)gru_12/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_12/gru_cell_24/MatMul/ReadVariableOpReadVariableOp1gru_12_gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_12/gru_cell_24/MatMulMatMulgru_12/strided_slice_2:output:00gru_12/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/BiasAddBiasAdd#gru_12/gru_cell_24/MatMul:product:0#gru_12/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_12/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/gru_cell_24/splitSplit+gru_12/gru_cell_24/split/split_dim:output:0#gru_12/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
*gru_12/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp3gru_12_gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_12/gru_cell_24/MatMul_1MatMulgru_12/zeros:output:02gru_12/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/BiasAdd_1BiasAdd%gru_12/gru_cell_24/MatMul_1:product:0#gru_12/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_12/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����o
$gru_12/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/gru_cell_24/split_1SplitV%gru_12/gru_cell_24/BiasAdd_1:output:0!gru_12/gru_cell_24/Const:output:0-gru_12/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_12/gru_cell_24/addAddV2!gru_12/gru_cell_24/split:output:0#gru_12/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������t
gru_12/gru_cell_24/SigmoidSigmoidgru_12/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_1AddV2!gru_12/gru_cell_24/split:output:1#gru_12/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������x
gru_12/gru_cell_24/Sigmoid_1Sigmoidgru_12/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mulMul gru_12/gru_cell_24/Sigmoid_1:y:0#gru_12/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_2AddV2!gru_12/gru_cell_24/split:output:2gru_12/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������x
gru_12/gru_cell_24/Sigmoid_2Sigmoidgru_12/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mul_1Mulgru_12/gru_cell_24/Sigmoid:y:0gru_12/zeros:output:0*
T0*(
_output_shapes
:����������]
gru_12/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_12/gru_cell_24/subSub!gru_12/gru_cell_24/sub/x:output:0gru_12/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mul_2Mulgru_12/gru_cell_24/sub:z:0 gru_12/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_3AddV2gru_12/gru_cell_24/mul_1:z:0gru_12/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:����������u
$gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
gru_12/TensorArrayV2_1TensorListReserve-gru_12/TensorArrayV2_1/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_12/whileWhile"gru_12/while/loop_counter:output:0(gru_12/while/maximum_iterations:output:0gru_12/time:output:0gru_12/TensorArrayV2_1:handle:0gru_12/zeros:output:0gru_12/strided_slice_1:output:0>gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_12_gru_cell_24_readvariableop_resource1gru_12_gru_cell_24_matmul_readvariableop_resource3gru_12_gru_cell_24_matmul_1_readvariableop_resource*
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
gru_12_while_body_2432861*%
condR
gru_12_while_cond_2432860*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
7gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)gru_12/TensorArrayV2Stack/TensorListStackTensorListStackgru_12/while:output:3@gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0o
gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_3StridedSlice2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0%gru_12/strided_slice_3/stack:output:0'gru_12/strided_slice_3/stack_1:output:0'gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskl
gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_12/transpose_1	Transpose2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0 gru_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������b
gru_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_13/ShapeShapegru_12/transpose_1:y:0*
T0*
_output_shapes
:d
gru_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_sliceStridedSlicegru_13/Shape:output:0#gru_13/strided_slice/stack:output:0%gru_13/strided_slice/stack_1:output:0%gru_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
gru_13/zeros/packedPackgru_13/strided_slice:output:0gru_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_13/zerosFillgru_13/zeros/packed:output:0gru_13/zeros/Const:output:0*
T0*'
_output_shapes
:���������dj
gru_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_13/transpose	Transposegru_12/transpose_1:y:0gru_13/transpose/perm:output:0*
T0*-
_output_shapes
:�����������R
gru_13/Shape_1Shapegru_13/transpose:y:0*
T0*
_output_shapes
:f
gru_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_1StridedSlicegru_13/Shape_1:output:0%gru_13/strided_slice_1/stack:output:0'gru_13/strided_slice_1/stack_1:output:0'gru_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/TensorArrayV2TensorListReserve+gru_13/TensorArrayV2/element_shape:output:0gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
.gru_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_13/transpose:y:0Egru_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_2StridedSlicegru_13/transpose:y:0%gru_13/strided_slice_2/stack:output:0'gru_13/strided_slice_2/stack_1:output:0'gru_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
!gru_13/gru_cell_25/ReadVariableOpReadVariableOp*gru_13_gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_13/gru_cell_25/unstackUnpack)gru_13/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_13/gru_cell_25/MatMul/ReadVariableOpReadVariableOp1gru_13_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_13/gru_cell_25/MatMulMatMulgru_13/strided_slice_2:output:00gru_13/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_13/gru_cell_25/BiasAddBiasAdd#gru_13/gru_cell_25/MatMul:product:0#gru_13/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_13/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/gru_cell_25/splitSplit+gru_13/gru_cell_25/split/split_dim:output:0#gru_13/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
*gru_13/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp3gru_13_gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_13/gru_cell_25/MatMul_1MatMulgru_13/zeros:output:02gru_13/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_13/gru_cell_25/BiasAdd_1BiasAdd%gru_13/gru_cell_25/MatMul_1:product:0#gru_13/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_13/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����o
$gru_13/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/gru_cell_25/split_1SplitV%gru_13/gru_cell_25/BiasAdd_1:output:0!gru_13/gru_cell_25/Const:output:0-gru_13/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_13/gru_cell_25/addAddV2!gru_13/gru_cell_25/split:output:0#gru_13/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������ds
gru_13/gru_cell_25/SigmoidSigmoidgru_13/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_1AddV2!gru_13/gru_cell_25/split:output:1#gru_13/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������dw
gru_13/gru_cell_25/Sigmoid_1Sigmoidgru_13/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mulMul gru_13/gru_cell_25/Sigmoid_1:y:0#gru_13/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_2AddV2!gru_13/gru_cell_25/split:output:2gru_13/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������dw
gru_13/gru_cell_25/Sigmoid_2Sigmoidgru_13/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mul_1Mulgru_13/gru_cell_25/Sigmoid:y:0gru_13/zeros:output:0*
T0*'
_output_shapes
:���������d]
gru_13/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_13/gru_cell_25/subSub!gru_13/gru_cell_25/sub/x:output:0gru_13/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mul_2Mulgru_13/gru_cell_25/sub:z:0 gru_13/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_3AddV2gru_13/gru_cell_25/mul_1:z:0gru_13/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������du
$gru_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
gru_13/TensorArrayV2_1TensorListReserve-gru_13/TensorArrayV2_1/element_shape:output:0gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_13/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_13/whileWhile"gru_13/while/loop_counter:output:0(gru_13/while/maximum_iterations:output:0gru_13/time:output:0gru_13/TensorArrayV2_1:handle:0gru_13/zeros:output:0gru_13/strided_slice_1:output:0>gru_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_13_gru_cell_25_readvariableop_resource1gru_13_gru_cell_25_matmul_readvariableop_resource3gru_13_gru_cell_25_matmul_1_readvariableop_resource*
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
gru_13_while_body_2433010*%
condR
gru_13_while_cond_2433009*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
7gru_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)gru_13/TensorArrayV2Stack/TensorListStackTensorListStackgru_13/while:output:3@gru_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0o
gru_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_3StridedSlice2gru_13/TensorArrayV2Stack/TensorListStack:tensor:0%gru_13/strided_slice_3/stack:output:0'gru_13/strided_slice_3/stack_1:output:0'gru_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskl
gru_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_13/transpose_1	Transpose2gru_13/TensorArrayV2Stack/TensorListStack:tensor:0 gru_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������db
gru_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_14/ShapeShapegru_13/transpose_1:y:0*
T0*
_output_shapes
:d
gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_sliceStridedSlicegru_14/Shape:output:0#gru_14/strided_slice/stack:output:0%gru_14/strided_slice/stack_1:output:0%gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
gru_14/zeros/packedPackgru_14/strided_slice:output:0gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_14/zerosFillgru_14/zeros/packed:output:0gru_14/zeros/Const:output:0*
T0*'
_output_shapes
:���������j
gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_14/transpose	Transposegru_13/transpose_1:y:0gru_14/transpose/perm:output:0*
T0*,
_output_shapes
:����������dR
gru_14/Shape_1Shapegru_14/transpose:y:0*
T0*
_output_shapes
:f
gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_1StridedSlicegru_14/Shape_1:output:0%gru_14/strided_slice_1/stack:output:0'gru_14/strided_slice_1/stack_1:output:0'gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/TensorArrayV2TensorListReserve+gru_14/TensorArrayV2/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
.gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_14/transpose:y:0Egru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_2StridedSlicegru_14/transpose:y:0%gru_14/strided_slice_2/stack:output:0'gru_14/strided_slice_2/stack_1:output:0'gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
!gru_14/gru_cell_26/ReadVariableOpReadVariableOp*gru_14_gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_14/gru_cell_26/unstackUnpack)gru_14/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
(gru_14/gru_cell_26/MatMul/ReadVariableOpReadVariableOp1gru_14_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_14/gru_cell_26/MatMulMatMulgru_14/strided_slice_2:output:00gru_14/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/BiasAddBiasAdd#gru_14/gru_cell_26/MatMul:product:0#gru_14/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������m
"gru_14/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/gru_cell_26/splitSplit+gru_14/gru_cell_26/split/split_dim:output:0#gru_14/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
*gru_14/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp3gru_14_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_14/gru_cell_26/MatMul_1MatMulgru_14/zeros:output:02gru_14/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/BiasAdd_1BiasAdd%gru_14/gru_cell_26/MatMul_1:product:0#gru_14/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������m
gru_14/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru_14/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/gru_cell_26/split_1SplitV%gru_14/gru_cell_26/BiasAdd_1:output:0!gru_14/gru_cell_26/Const:output:0-gru_14/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_14/gru_cell_26/addAddV2!gru_14/gru_cell_26/split:output:0#gru_14/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������s
gru_14/gru_cell_26/SigmoidSigmoidgru_14/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_1AddV2!gru_14/gru_cell_26/split:output:1#gru_14/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������w
gru_14/gru_cell_26/Sigmoid_1Sigmoidgru_14/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mulMul gru_14/gru_cell_26/Sigmoid_1:y:0#gru_14/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_2AddV2!gru_14/gru_cell_26/split:output:2gru_14/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������w
gru_14/gru_cell_26/SoftplusSoftplusgru_14/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mul_1Mulgru_14/gru_cell_26/Sigmoid:y:0gru_14/zeros:output:0*
T0*'
_output_shapes
:���������]
gru_14/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_14/gru_cell_26/subSub!gru_14/gru_cell_26/sub/x:output:0gru_14/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mul_2Mulgru_14/gru_cell_26/sub:z:0)gru_14/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_3AddV2gru_14/gru_cell_26/mul_1:z:0gru_14/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:���������u
$gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
gru_14/TensorArrayV2_1TensorListReserve-gru_14/TensorArrayV2_1/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_14/whileWhile"gru_14/while/loop_counter:output:0(gru_14/while/maximum_iterations:output:0gru_14/time:output:0gru_14/TensorArrayV2_1:handle:0gru_14/zeros:output:0gru_14/strided_slice_1:output:0>gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_14_gru_cell_26_readvariableop_resource1gru_14_gru_cell_26_matmul_readvariableop_resource3gru_14_gru_cell_26_matmul_1_readvariableop_resource*
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
gru_14_while_body_2433159*%
condR
gru_14_while_cond_2433158*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
7gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)gru_14/TensorArrayV2Stack/TensorListStackTensorListStackgru_14/while:output:3@gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0o
gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_3StridedSlice2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0%gru_14/strided_slice_3/stack:output:0'gru_14/strided_slice_3/stack_1:output:0'gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskl
gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_14/transpose_1	Transpose2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0 gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������b
gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_14/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp)^gru_12/gru_cell_24/MatMul/ReadVariableOp+^gru_12/gru_cell_24/MatMul_1/ReadVariableOp"^gru_12/gru_cell_24/ReadVariableOp^gru_12/while)^gru_13/gru_cell_25/MatMul/ReadVariableOp+^gru_13/gru_cell_25/MatMul_1/ReadVariableOp"^gru_13/gru_cell_25/ReadVariableOp^gru_13/while)^gru_14/gru_cell_26/MatMul/ReadVariableOp+^gru_14/gru_cell_26/MatMul_1/ReadVariableOp"^gru_14/gru_cell_26/ReadVariableOp^gru_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2T
(gru_12/gru_cell_24/MatMul/ReadVariableOp(gru_12/gru_cell_24/MatMul/ReadVariableOp2X
*gru_12/gru_cell_24/MatMul_1/ReadVariableOp*gru_12/gru_cell_24/MatMul_1/ReadVariableOp2F
!gru_12/gru_cell_24/ReadVariableOp!gru_12/gru_cell_24/ReadVariableOp2
gru_12/whilegru_12/while2T
(gru_13/gru_cell_25/MatMul/ReadVariableOp(gru_13/gru_cell_25/MatMul/ReadVariableOp2X
*gru_13/gru_cell_25/MatMul_1/ReadVariableOp*gru_13/gru_cell_25/MatMul_1/ReadVariableOp2F
!gru_13/gru_cell_25/ReadVariableOp!gru_13/gru_cell_25/ReadVariableOp2
gru_13/whilegru_13/while2T
(gru_14/gru_cell_26/MatMul/ReadVariableOp(gru_14/gru_cell_26/MatMul/ReadVariableOp2X
*gru_14/gru_cell_26/MatMul_1/ReadVariableOp*gru_14/gru_cell_26/MatMul_1/ReadVariableOp2F
!gru_14/gru_cell_26/ReadVariableOp!gru_14/gru_cell_26/ReadVariableOp2
gru_14/whilegru_14/while:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435283

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

�
-__inference_gru_cell_26_layer_call_fn_2435442

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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430823o
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
�=
�
while_body_2431852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
while_body_2435127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
while_cond_2435126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2435126___redundant_placeholder05
1while_while_cond_2435126___redundant_placeholder15
1while_while_cond_2435126___redundant_placeholder25
1while_while_cond_2435126___redundant_placeholder3
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
while_cond_2431320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431320___redundant_placeholder05
1while_while_cond_2431320___redundant_placeholder15
1while_while_cond_2431320___redundant_placeholder25
1while_while_cond_2431320___redundant_placeholder3
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
�4
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2430744

inputs&
gru_cell_25_2430668:	�'
gru_cell_25_2430670:
��&
gru_cell_25_2430672:	d�
identity��#gru_cell_25/StatefulPartitionedCall�while;
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
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_2430668gru_cell_25_2430670gru_cell_25_2430672*
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430628n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_2430668gru_cell_25_2430670gru_cell_25_2430672*
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
while_body_2430680*
condR
while_cond_2430679*8
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
NoOpNoOp$^gru_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435322

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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430628

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
�=
�
while_body_2434165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
while_body_2430342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_24_2430364_0:	�.
while_gru_cell_24_2430366_0:	�/
while_gru_cell_24_2430368_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_24_2430364:	�,
while_gru_cell_24_2430366:	�-
while_gru_cell_24_2430368:
����)while/gru_cell_24/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/gru_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_24_2430364_0while_gru_cell_24_2430366_0while_gru_cell_24_2430368_0*
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430290�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_24/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:����������x

while/NoOpNoOp*^while/gru_cell_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_24_2430364while_gru_cell_24_2430364_0"8
while_gru_cell_24_2430366while_gru_cell_24_2430366_0"8
while_gru_cell_24_2430368while_gru_cell_24_2430368_0")
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
)while/gru_cell_24/StatefulPartitionedCall)while/gru_cell_24/StatefulPartitionedCall: 
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
while_cond_2431851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431851___redundant_placeholder05
1while_while_cond_2431851___redundant_placeholder15
1while_while_cond_2431851___redundant_placeholder25
1while_while_cond_2431851___redundant_placeholder3
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
while_cond_2431160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431160___redundant_placeholder05
1while_while_cond_2431160___redundant_placeholder15
1while_while_cond_2431160___redundant_placeholder25
1while_while_cond_2431160___redundant_placeholder3
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
gru_13_while_body_2433010*
&gru_13_while_gru_13_while_loop_counter0
,gru_13_while_gru_13_while_maximum_iterations
gru_13_while_placeholder
gru_13_while_placeholder_1
gru_13_while_placeholder_2)
%gru_13_while_gru_13_strided_slice_1_0e
agru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0E
2gru_13_while_gru_cell_25_readvariableop_resource_0:	�M
9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0:
��N
;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
gru_13_while_identity
gru_13_while_identity_1
gru_13_while_identity_2
gru_13_while_identity_3
gru_13_while_identity_4'
#gru_13_while_gru_13_strided_slice_1c
_gru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensorC
0gru_13_while_gru_cell_25_readvariableop_resource:	�K
7gru_13_while_gru_cell_25_matmul_readvariableop_resource:
��L
9gru_13_while_gru_cell_25_matmul_1_readvariableop_resource:	d���.gru_13/while/gru_cell_25/MatMul/ReadVariableOp�0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp�'gru_13/while/gru_cell_25/ReadVariableOp�
>gru_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
0gru_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0gru_13_while_placeholderGgru_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
'gru_13/while/gru_cell_25/ReadVariableOpReadVariableOp2gru_13_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_13/while/gru_cell_25/unstackUnpack/gru_13/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_13/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
gru_13/while/gru_cell_25/MatMulMatMul7gru_13/while/TensorArrayV2Read/TensorListGetItem:item:06gru_13/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_13/while/gru_cell_25/BiasAddBiasAdd)gru_13/while/gru_cell_25/MatMul:product:0)gru_13/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_13/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/while/gru_cell_25/splitSplit1gru_13/while/gru_cell_25/split/split_dim:output:0)gru_13/while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
!gru_13/while/gru_cell_25/MatMul_1MatMulgru_13_while_placeholder_28gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_13/while/gru_cell_25/BiasAdd_1BiasAdd+gru_13/while/gru_cell_25/MatMul_1:product:0)gru_13/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_13/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����u
*gru_13/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_13/while/gru_cell_25/split_1SplitV+gru_13/while/gru_cell_25/BiasAdd_1:output:0'gru_13/while/gru_cell_25/Const:output:03gru_13/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_13/while/gru_cell_25/addAddV2'gru_13/while/gru_cell_25/split:output:0)gru_13/while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������d
 gru_13/while/gru_cell_25/SigmoidSigmoid gru_13/while/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_1AddV2'gru_13/while/gru_cell_25/split:output:1)gru_13/while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������d�
"gru_13/while/gru_cell_25/Sigmoid_1Sigmoid"gru_13/while/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mulMul&gru_13/while/gru_cell_25/Sigmoid_1:y:0)gru_13/while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_2AddV2'gru_13/while/gru_cell_25/split:output:2 gru_13/while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������d�
"gru_13/while/gru_cell_25/Sigmoid_2Sigmoid"gru_13/while/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mul_1Mul$gru_13/while/gru_cell_25/Sigmoid:y:0gru_13_while_placeholder_2*
T0*'
_output_shapes
:���������dc
gru_13/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_13/while/gru_cell_25/subSub'gru_13/while/gru_cell_25/sub/x:output:0$gru_13/while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/mul_2Mul gru_13/while/gru_cell_25/sub:z:0&gru_13/while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_13/while/gru_cell_25/add_3AddV2"gru_13/while/gru_cell_25/mul_1:z:0"gru_13/while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
1gru_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_13_while_placeholder_1gru_13_while_placeholder"gru_13/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_13/while/addAddV2gru_13_while_placeholdergru_13/while/add/y:output:0*
T0*
_output_shapes
: V
gru_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_13/while/add_1AddV2&gru_13_while_gru_13_while_loop_countergru_13/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_13/while/IdentityIdentitygru_13/while/add_1:z:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_1Identity,gru_13_while_gru_13_while_maximum_iterations^gru_13/while/NoOp*
T0*
_output_shapes
: n
gru_13/while/Identity_2Identitygru_13/while/add:z:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_3IdentityAgru_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_13/while/NoOp*
T0*
_output_shapes
: �
gru_13/while/Identity_4Identity"gru_13/while/gru_cell_25/add_3:z:0^gru_13/while/NoOp*
T0*'
_output_shapes
:���������d�
gru_13/while/NoOpNoOp/^gru_13/while/gru_cell_25/MatMul/ReadVariableOp1^gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp(^gru_13/while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_13_while_gru_13_strided_slice_1%gru_13_while_gru_13_strided_slice_1_0"x
9gru_13_while_gru_cell_25_matmul_1_readvariableop_resource;gru_13_while_gru_cell_25_matmul_1_readvariableop_resource_0"t
7gru_13_while_gru_cell_25_matmul_readvariableop_resource9gru_13_while_gru_cell_25_matmul_readvariableop_resource_0"f
0gru_13_while_gru_cell_25_readvariableop_resource2gru_13_while_gru_cell_25_readvariableop_resource_0"7
gru_13_while_identitygru_13/while/Identity:output:0";
gru_13_while_identity_1 gru_13/while/Identity_1:output:0";
gru_13_while_identity_2 gru_13/while/Identity_2:output:0";
gru_13_while_identity_3 gru_13/while/Identity_3:output:0";
gru_13_while_identity_4 gru_13/while/Identity_4:output:0"�
_gru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensoragru_13_while_tensorarrayv2read_tensorlistgetitem_gru_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.gru_13/while/gru_cell_25/MatMul/ReadVariableOp.gru_13/while/gru_cell_25/MatMul/ReadVariableOp2d
0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp0gru_13/while/gru_cell_25/MatMul_1/ReadVariableOp2R
'gru_13/while/gru_cell_25/ReadVariableOp'gru_13/while/gru_cell_25/ReadVariableOp: 
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2430900

inputs%
gru_cell_26_2430824:%
gru_cell_26_2430826:d%
gru_cell_26_2430828:
identity��#gru_cell_26/StatefulPartitionedCall�while;
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
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_2430824gru_cell_26_2430826gru_cell_26_2430828*
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430823n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_2430824gru_cell_26_2430826gru_cell_26_2430828*
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
while_body_2430836*
condR
while_cond_2430835*8
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
NoOpNoOp$^gru_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�4
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2430562

inputs&
gru_cell_25_2430486:	�'
gru_cell_25_2430488:
��&
gru_cell_25_2430490:	d�
identity��#gru_cell_25/StatefulPartitionedCall�while;
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
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_2430486gru_cell_25_2430488gru_cell_25_2430490*
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430485n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_2430486gru_cell_25_2430488gru_cell_25_2430490*
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
while_body_2430498*
condR
while_cond_2430497*8
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
NoOpNoOp$^gru_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430485

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
�M
�
C__inference_gru_14_layer_call_and_return_conditional_losses_2435063

inputs5
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while;
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2434974*
condR
while_cond_2434973*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
while_cond_2434973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434973___redundant_placeholder05
1while_while_cond_2434973___redundant_placeholder15
1while_while_cond_2434973___redundant_placeholder25
1while_while_cond_2434973___redundant_placeholder3
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
�4
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2430224

inputs&
gru_cell_24_2430148:	�&
gru_cell_24_2430150:	�'
gru_cell_24_2430152:
��
identity��#gru_cell_24/StatefulPartitionedCall�while;
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
#gru_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_24_2430148gru_cell_24_2430150gru_cell_24_2430152*
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430147n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_24_2430148gru_cell_24_2430150gru_cell_24_2430152*
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
while_body_2430160*
condR
while_cond_2430159*9
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
NoOpNoOp$^gru_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#gru_cell_24/StatefulPartitionedCall#gru_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
� 
�
while_body_2430498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_25_2430520_0:	�/
while_gru_cell_25_2430522_0:
��.
while_gru_cell_25_2430524_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_25_2430520:	�-
while_gru_cell_25_2430522:
��,
while_gru_cell_25_2430524:	d���)while/gru_cell_25/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_2430520_0while_gru_cell_25_2430522_0while_gru_cell_25_2430524_0*
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430485�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������dx

while/NoOpNoOp*^while/gru_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_25_2430520while_gru_cell_25_2430520_0"8
while_gru_cell_25_2430522while_gru_cell_25_2430522_0"8
while_gru_cell_25_2430524while_gru_cell_25_2430524_0")
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
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 
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
�
�
(__inference_gru_13_layer_call_fn_2433915
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2430562|
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
�

�
-__inference_gru_cell_25_layer_call_fn_2435350

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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430628o
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
�L
�
 __inference__traced_save_2435659
file_prefix8
4savev2_gru_12_gru_cell_24_kernel_read_readvariableopB
>savev2_gru_12_gru_cell_24_recurrent_kernel_read_readvariableop6
2savev2_gru_12_gru_cell_24_bias_read_readvariableop8
4savev2_gru_13_gru_cell_25_kernel_read_readvariableopB
>savev2_gru_13_gru_cell_25_recurrent_kernel_read_readvariableop6
2savev2_gru_13_gru_cell_25_bias_read_readvariableop8
4savev2_gru_14_gru_cell_26_kernel_read_readvariableopB
>savev2_gru_14_gru_cell_26_recurrent_kernel_read_readvariableop6
2savev2_gru_14_gru_cell_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_gru_12_gru_cell_24_kernel_m_read_readvariableopI
Esavev2_adam_gru_12_gru_cell_24_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_12_gru_cell_24_bias_m_read_readvariableop?
;savev2_adam_gru_13_gru_cell_25_kernel_m_read_readvariableopI
Esavev2_adam_gru_13_gru_cell_25_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_13_gru_cell_25_bias_m_read_readvariableop?
;savev2_adam_gru_14_gru_cell_26_kernel_m_read_readvariableopI
Esavev2_adam_gru_14_gru_cell_26_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_14_gru_cell_26_bias_m_read_readvariableop?
;savev2_adam_gru_12_gru_cell_24_kernel_v_read_readvariableopI
Esavev2_adam_gru_12_gru_cell_24_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_12_gru_cell_24_bias_v_read_readvariableop?
;savev2_adam_gru_13_gru_cell_25_kernel_v_read_readvariableopI
Esavev2_adam_gru_13_gru_cell_25_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_13_gru_cell_25_bias_v_read_readvariableop?
;savev2_adam_gru_14_gru_cell_26_kernel_v_read_readvariableopI
Esavev2_adam_gru_14_gru_cell_26_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_14_gru_cell_26_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_gru_12_gru_cell_24_kernel_read_readvariableop>savev2_gru_12_gru_cell_24_recurrent_kernel_read_readvariableop2savev2_gru_12_gru_cell_24_bias_read_readvariableop4savev2_gru_13_gru_cell_25_kernel_read_readvariableop>savev2_gru_13_gru_cell_25_recurrent_kernel_read_readvariableop2savev2_gru_13_gru_cell_25_bias_read_readvariableop4savev2_gru_14_gru_cell_26_kernel_read_readvariableop>savev2_gru_14_gru_cell_26_recurrent_kernel_read_readvariableop2savev2_gru_14_gru_cell_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_gru_12_gru_cell_24_kernel_m_read_readvariableopEsavev2_adam_gru_12_gru_cell_24_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_12_gru_cell_24_bias_m_read_readvariableop;savev2_adam_gru_13_gru_cell_25_kernel_m_read_readvariableopEsavev2_adam_gru_13_gru_cell_25_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_13_gru_cell_25_bias_m_read_readvariableop;savev2_adam_gru_14_gru_cell_26_kernel_m_read_readvariableopEsavev2_adam_gru_14_gru_cell_26_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_14_gru_cell_26_bias_m_read_readvariableop;savev2_adam_gru_12_gru_cell_24_kernel_v_read_readvariableopEsavev2_adam_gru_12_gru_cell_24_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_12_gru_cell_24_bias_v_read_readvariableop;savev2_adam_gru_13_gru_cell_25_kernel_v_read_readvariableopEsavev2_adam_gru_13_gru_cell_25_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_13_gru_cell_25_bias_v_read_readvariableop;savev2_adam_gru_14_gru_cell_26_kernel_v_read_readvariableopEsavev2_adam_gru_14_gru_cell_26_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_14_gru_cell_26_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
while_cond_2433814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2433814___redundant_placeholder05
1while_while_cond_2433814___redundant_placeholder15
1while_while_cond_2433814___redundant_placeholder25
1while_while_cond_2433814___redundant_placeholder3
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
while_body_2433509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
while_cond_2433508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2433508___redundant_placeholder05
1while_while_cond_2433508___redundant_placeholder15
1while_while_cond_2433508___redundant_placeholder25
1while_while_cond_2433508___redundant_placeholder3
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
while_cond_2434820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434820___redundant_placeholder05
1while_while_cond_2434820___redundant_placeholder15
1while_while_cond_2434820___redundant_placeholder25
1while_while_cond_2434820___redundant_placeholder3
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
�=
�
while_body_2433815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
�M
�
C__inference_gru_14_layer_call_and_return_conditional_losses_2434910
inputs_05
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while=
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2434821*
condR
while_cond_2434820*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������d
"
_user_specified_name
inputs/0
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432797

inputs=
*gru_12_gru_cell_24_readvariableop_resource:	�D
1gru_12_gru_cell_24_matmul_readvariableop_resource:	�G
3gru_12_gru_cell_24_matmul_1_readvariableop_resource:
��=
*gru_13_gru_cell_25_readvariableop_resource:	�E
1gru_13_gru_cell_25_matmul_readvariableop_resource:
��F
3gru_13_gru_cell_25_matmul_1_readvariableop_resource:	d�<
*gru_14_gru_cell_26_readvariableop_resource:C
1gru_14_gru_cell_26_matmul_readvariableop_resource:dE
3gru_14_gru_cell_26_matmul_1_readvariableop_resource:
identity��(gru_12/gru_cell_24/MatMul/ReadVariableOp�*gru_12/gru_cell_24/MatMul_1/ReadVariableOp�!gru_12/gru_cell_24/ReadVariableOp�gru_12/while�(gru_13/gru_cell_25/MatMul/ReadVariableOp�*gru_13/gru_cell_25/MatMul_1/ReadVariableOp�!gru_13/gru_cell_25/ReadVariableOp�gru_13/while�(gru_14/gru_cell_26/MatMul/ReadVariableOp�*gru_14/gru_cell_26/MatMul_1/ReadVariableOp�!gru_14/gru_cell_26/ReadVariableOp�gru_14/whileB
gru_12/ShapeShapeinputs*
T0*
_output_shapes
:d
gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_sliceStridedSlicegru_12/Shape:output:0#gru_12/strided_slice/stack:output:0%gru_12/strided_slice/stack_1:output:0%gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
gru_12/zeros/packedPackgru_12/strided_slice:output:0gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_12/zerosFillgru_12/zeros/packed:output:0gru_12/zeros/Const:output:0*
T0*(
_output_shapes
:����������j
gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
gru_12/transpose	Transposeinputsgru_12/transpose/perm:output:0*
T0*,
_output_shapes
:����������R
gru_12/Shape_1Shapegru_12/transpose:y:0*
T0*
_output_shapes
:f
gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_1StridedSlicegru_12/Shape_1:output:0%gru_12/strided_slice_1/stack:output:0'gru_12/strided_slice_1/stack_1:output:0'gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/TensorArrayV2TensorListReserve+gru_12/TensorArrayV2/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
.gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_12/transpose:y:0Egru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_2StridedSlicegru_12/transpose:y:0%gru_12/strided_slice_2/stack:output:0'gru_12/strided_slice_2/stack_1:output:0'gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!gru_12/gru_cell_24/ReadVariableOpReadVariableOp*gru_12_gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_12/gru_cell_24/unstackUnpack)gru_12/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_12/gru_cell_24/MatMul/ReadVariableOpReadVariableOp1gru_12_gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_12/gru_cell_24/MatMulMatMulgru_12/strided_slice_2:output:00gru_12/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/BiasAddBiasAdd#gru_12/gru_cell_24/MatMul:product:0#gru_12/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_12/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/gru_cell_24/splitSplit+gru_12/gru_cell_24/split/split_dim:output:0#gru_12/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
*gru_12/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp3gru_12_gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_12/gru_cell_24/MatMul_1MatMulgru_12/zeros:output:02gru_12/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/BiasAdd_1BiasAdd%gru_12/gru_cell_24/MatMul_1:product:0#gru_12/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_12/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����o
$gru_12/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/gru_cell_24/split_1SplitV%gru_12/gru_cell_24/BiasAdd_1:output:0!gru_12/gru_cell_24/Const:output:0-gru_12/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_12/gru_cell_24/addAddV2!gru_12/gru_cell_24/split:output:0#gru_12/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������t
gru_12/gru_cell_24/SigmoidSigmoidgru_12/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_1AddV2!gru_12/gru_cell_24/split:output:1#gru_12/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������x
gru_12/gru_cell_24/Sigmoid_1Sigmoidgru_12/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mulMul gru_12/gru_cell_24/Sigmoid_1:y:0#gru_12/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_2AddV2!gru_12/gru_cell_24/split:output:2gru_12/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������x
gru_12/gru_cell_24/Sigmoid_2Sigmoidgru_12/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mul_1Mulgru_12/gru_cell_24/Sigmoid:y:0gru_12/zeros:output:0*
T0*(
_output_shapes
:����������]
gru_12/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_12/gru_cell_24/subSub!gru_12/gru_cell_24/sub/x:output:0gru_12/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/mul_2Mulgru_12/gru_cell_24/sub:z:0 gru_12/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_12/gru_cell_24/add_3AddV2gru_12/gru_cell_24/mul_1:z:0gru_12/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:����������u
$gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
gru_12/TensorArrayV2_1TensorListReserve-gru_12/TensorArrayV2_1/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_12/whileWhile"gru_12/while/loop_counter:output:0(gru_12/while/maximum_iterations:output:0gru_12/time:output:0gru_12/TensorArrayV2_1:handle:0gru_12/zeros:output:0gru_12/strided_slice_1:output:0>gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_12_gru_cell_24_readvariableop_resource1gru_12_gru_cell_24_matmul_readvariableop_resource3gru_12_gru_cell_24_matmul_1_readvariableop_resource*
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
gru_12_while_body_2432410*%
condR
gru_12_while_cond_2432409*9
output_shapes(
&: : : : :����������: : : : : *
parallel_iterations �
7gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)gru_12/TensorArrayV2Stack/TensorListStackTensorListStackgru_12/while:output:3@gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0o
gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_12/strided_slice_3StridedSlice2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0%gru_12/strided_slice_3/stack:output:0'gru_12/strided_slice_3/stack_1:output:0'gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskl
gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_12/transpose_1	Transpose2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0 gru_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:�����������b
gru_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_13/ShapeShapegru_12/transpose_1:y:0*
T0*
_output_shapes
:d
gru_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_sliceStridedSlicegru_13/Shape:output:0#gru_13/strided_slice/stack:output:0%gru_13/strided_slice/stack_1:output:0%gru_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
gru_13/zeros/packedPackgru_13/strided_slice:output:0gru_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_13/zerosFillgru_13/zeros/packed:output:0gru_13/zeros/Const:output:0*
T0*'
_output_shapes
:���������dj
gru_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_13/transpose	Transposegru_12/transpose_1:y:0gru_13/transpose/perm:output:0*
T0*-
_output_shapes
:�����������R
gru_13/Shape_1Shapegru_13/transpose:y:0*
T0*
_output_shapes
:f
gru_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_1StridedSlicegru_13/Shape_1:output:0%gru_13/strided_slice_1/stack:output:0'gru_13/strided_slice_1/stack_1:output:0'gru_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/TensorArrayV2TensorListReserve+gru_13/TensorArrayV2/element_shape:output:0gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
.gru_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_13/transpose:y:0Egru_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_2StridedSlicegru_13/transpose:y:0%gru_13/strided_slice_2/stack:output:0'gru_13/strided_slice_2/stack_1:output:0'gru_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
!gru_13/gru_cell_25/ReadVariableOpReadVariableOp*gru_13_gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_13/gru_cell_25/unstackUnpack)gru_13/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(gru_13/gru_cell_25/MatMul/ReadVariableOpReadVariableOp1gru_13_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_13/gru_cell_25/MatMulMatMulgru_13/strided_slice_2:output:00gru_13/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_13/gru_cell_25/BiasAddBiasAdd#gru_13/gru_cell_25/MatMul:product:0#gru_13/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������m
"gru_13/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/gru_cell_25/splitSplit+gru_13/gru_cell_25/split/split_dim:output:0#gru_13/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
*gru_13/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp3gru_13_gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_13/gru_cell_25/MatMul_1MatMulgru_13/zeros:output:02gru_13/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_13/gru_cell_25/BiasAdd_1BiasAdd%gru_13/gru_cell_25/MatMul_1:product:0#gru_13/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������m
gru_13/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����o
$gru_13/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_13/gru_cell_25/split_1SplitV%gru_13/gru_cell_25/BiasAdd_1:output:0!gru_13/gru_cell_25/Const:output:0-gru_13/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_13/gru_cell_25/addAddV2!gru_13/gru_cell_25/split:output:0#gru_13/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������ds
gru_13/gru_cell_25/SigmoidSigmoidgru_13/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_1AddV2!gru_13/gru_cell_25/split:output:1#gru_13/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������dw
gru_13/gru_cell_25/Sigmoid_1Sigmoidgru_13/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mulMul gru_13/gru_cell_25/Sigmoid_1:y:0#gru_13/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_2AddV2!gru_13/gru_cell_25/split:output:2gru_13/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������dw
gru_13/gru_cell_25/Sigmoid_2Sigmoidgru_13/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mul_1Mulgru_13/gru_cell_25/Sigmoid:y:0gru_13/zeros:output:0*
T0*'
_output_shapes
:���������d]
gru_13/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_13/gru_cell_25/subSub!gru_13/gru_cell_25/sub/x:output:0gru_13/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/mul_2Mulgru_13/gru_cell_25/sub:z:0 gru_13/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
gru_13/gru_cell_25/add_3AddV2gru_13/gru_cell_25/mul_1:z:0gru_13/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������du
$gru_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
gru_13/TensorArrayV2_1TensorListReserve-gru_13/TensorArrayV2_1/element_shape:output:0gru_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_13/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_13/whileWhile"gru_13/while/loop_counter:output:0(gru_13/while/maximum_iterations:output:0gru_13/time:output:0gru_13/TensorArrayV2_1:handle:0gru_13/zeros:output:0gru_13/strided_slice_1:output:0>gru_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_13_gru_cell_25_readvariableop_resource1gru_13_gru_cell_25_matmul_readvariableop_resource3gru_13_gru_cell_25_matmul_1_readvariableop_resource*
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
gru_13_while_body_2432559*%
condR
gru_13_while_cond_2432558*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
7gru_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)gru_13/TensorArrayV2Stack/TensorListStackTensorListStackgru_13/while:output:3@gru_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������d*
element_dtype0o
gru_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_13/strided_slice_3StridedSlice2gru_13/TensorArrayV2Stack/TensorListStack:tensor:0%gru_13/strided_slice_3/stack:output:0'gru_13/strided_slice_3/stack_1:output:0'gru_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskl
gru_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_13/transpose_1	Transpose2gru_13/TensorArrayV2Stack/TensorListStack:tensor:0 gru_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������db
gru_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_14/ShapeShapegru_13/transpose_1:y:0*
T0*
_output_shapes
:d
gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_sliceStridedSlicegru_14/Shape:output:0#gru_14/strided_slice/stack:output:0%gru_14/strided_slice/stack_1:output:0%gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
gru_14/zeros/packedPackgru_14/strided_slice:output:0gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gru_14/zerosFillgru_14/zeros/packed:output:0gru_14/zeros/Const:output:0*
T0*'
_output_shapes
:���������j
gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_14/transpose	Transposegru_13/transpose_1:y:0gru_14/transpose/perm:output:0*
T0*,
_output_shapes
:����������dR
gru_14/Shape_1Shapegru_14/transpose:y:0*
T0*
_output_shapes
:f
gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_1StridedSlicegru_14/Shape_1:output:0%gru_14/strided_slice_1/stack:output:0'gru_14/strided_slice_1/stack_1:output:0'gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/TensorArrayV2TensorListReserve+gru_14/TensorArrayV2/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
.gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_14/transpose:y:0Egru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_2StridedSlicegru_14/transpose:y:0%gru_14/strided_slice_2/stack:output:0'gru_14/strided_slice_2/stack_1:output:0'gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask�
!gru_14/gru_cell_26/ReadVariableOpReadVariableOp*gru_14_gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_14/gru_cell_26/unstackUnpack)gru_14/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
(gru_14/gru_cell_26/MatMul/ReadVariableOpReadVariableOp1gru_14_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_14/gru_cell_26/MatMulMatMulgru_14/strided_slice_2:output:00gru_14/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/BiasAddBiasAdd#gru_14/gru_cell_26/MatMul:product:0#gru_14/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������m
"gru_14/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/gru_cell_26/splitSplit+gru_14/gru_cell_26/split/split_dim:output:0#gru_14/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
*gru_14/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp3gru_14_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_14/gru_cell_26/MatMul_1MatMulgru_14/zeros:output:02gru_14/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/BiasAdd_1BiasAdd%gru_14/gru_cell_26/MatMul_1:product:0#gru_14/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������m
gru_14/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru_14/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/gru_cell_26/split_1SplitV%gru_14/gru_cell_26/BiasAdd_1:output:0!gru_14/gru_cell_26/Const:output:0-gru_14/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_14/gru_cell_26/addAddV2!gru_14/gru_cell_26/split:output:0#gru_14/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������s
gru_14/gru_cell_26/SigmoidSigmoidgru_14/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_1AddV2!gru_14/gru_cell_26/split:output:1#gru_14/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������w
gru_14/gru_cell_26/Sigmoid_1Sigmoidgru_14/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mulMul gru_14/gru_cell_26/Sigmoid_1:y:0#gru_14/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_2AddV2!gru_14/gru_cell_26/split:output:2gru_14/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������w
gru_14/gru_cell_26/SoftplusSoftplusgru_14/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mul_1Mulgru_14/gru_cell_26/Sigmoid:y:0gru_14/zeros:output:0*
T0*'
_output_shapes
:���������]
gru_14/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_14/gru_cell_26/subSub!gru_14/gru_cell_26/sub/x:output:0gru_14/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/mul_2Mulgru_14/gru_cell_26/sub:z:0)gru_14/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_14/gru_cell_26/add_3AddV2gru_14/gru_cell_26/mul_1:z:0gru_14/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:���������u
$gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
gru_14/TensorArrayV2_1TensorListReserve-gru_14/TensorArrayV2_1/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���M
gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
gru_14/whileWhile"gru_14/while/loop_counter:output:0(gru_14/while/maximum_iterations:output:0gru_14/time:output:0gru_14/TensorArrayV2_1:handle:0gru_14/zeros:output:0gru_14/strided_slice_1:output:0>gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_14_gru_cell_26_readvariableop_resource1gru_14_gru_cell_26_matmul_readvariableop_resource3gru_14_gru_cell_26_matmul_1_readvariableop_resource*
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
gru_14_while_body_2432708*%
condR
gru_14_while_cond_2432707*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
7gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)gru_14/TensorArrayV2Stack/TensorListStackTensorListStackgru_14/while:output:3@gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0o
gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_14/strided_slice_3StridedSlice2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0%gru_14/strided_slice_3/stack:output:0'gru_14/strided_slice_3/stack_1:output:0'gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskl
gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru_14/transpose_1	Transpose2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0 gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������b
gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_14/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp)^gru_12/gru_cell_24/MatMul/ReadVariableOp+^gru_12/gru_cell_24/MatMul_1/ReadVariableOp"^gru_12/gru_cell_24/ReadVariableOp^gru_12/while)^gru_13/gru_cell_25/MatMul/ReadVariableOp+^gru_13/gru_cell_25/MatMul_1/ReadVariableOp"^gru_13/gru_cell_25/ReadVariableOp^gru_13/while)^gru_14/gru_cell_26/MatMul/ReadVariableOp+^gru_14/gru_cell_26/MatMul_1/ReadVariableOp"^gru_14/gru_cell_26/ReadVariableOp^gru_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2T
(gru_12/gru_cell_24/MatMul/ReadVariableOp(gru_12/gru_cell_24/MatMul/ReadVariableOp2X
*gru_12/gru_cell_24/MatMul_1/ReadVariableOp*gru_12/gru_cell_24/MatMul_1/ReadVariableOp2F
!gru_12/gru_cell_24/ReadVariableOp!gru_12/gru_cell_24/ReadVariableOp2
gru_12/whilegru_12/while2T
(gru_13/gru_cell_25/MatMul/ReadVariableOp(gru_13/gru_cell_25/MatMul/ReadVariableOp2X
*gru_13/gru_cell_25/MatMul_1/ReadVariableOp*gru_13/gru_cell_25/MatMul_1/ReadVariableOp2F
!gru_13/gru_cell_25/ReadVariableOp!gru_13/gru_cell_25/ReadVariableOp2
gru_13/whilegru_13/while2T
(gru_14/gru_cell_26/MatMul/ReadVariableOp(gru_14/gru_cell_26/MatMul/ReadVariableOp2X
*gru_14/gru_cell_26/MatMul_1/ReadVariableOp*gru_14/gru_cell_26/MatMul_1/ReadVariableOp2F
!gru_14/gru_cell_26/ReadVariableOp!gru_14/gru_cell_26/ReadVariableOp2
gru_14/whilegru_14/while:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
while_body_2434821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
�E
�	
gru_14_while_body_2432708*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2)
%gru_14_while_gru_14_strided_slice_1_0e
agru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0D
2gru_14_while_gru_cell_26_readvariableop_resource_0:K
9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0:dM
;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0:
gru_14_while_identity
gru_14_while_identity_1
gru_14_while_identity_2
gru_14_while_identity_3
gru_14_while_identity_4'
#gru_14_while_gru_14_strided_slice_1c
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensorB
0gru_14_while_gru_cell_26_readvariableop_resource:I
7gru_14_while_gru_cell_26_matmul_readvariableop_resource:dK
9gru_14_while_gru_cell_26_matmul_1_readvariableop_resource:��.gru_14/while/gru_cell_26/MatMul/ReadVariableOp�0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp�'gru_14/while/gru_cell_26/ReadVariableOp�
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
0gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0gru_14_while_placeholderGgru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
'gru_14/while/gru_cell_26/ReadVariableOpReadVariableOp2gru_14_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 gru_14/while/gru_cell_26/unstackUnpack/gru_14/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
.gru_14/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
gru_14/while/gru_cell_26/MatMulMatMul7gru_14/while/TensorArrayV2Read/TensorListGetItem:item:06gru_14/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 gru_14/while/gru_cell_26/BiasAddBiasAdd)gru_14/while/gru_cell_26/MatMul:product:0)gru_14/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������s
(gru_14/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/while/gru_cell_26/splitSplit1gru_14/while/gru_cell_26/split/split_dim:output:0)gru_14/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
!gru_14/while/gru_cell_26/MatMul_1MatMulgru_14_while_placeholder_28gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"gru_14/while/gru_cell_26/BiasAdd_1BiasAdd+gru_14/while/gru_cell_26/MatMul_1:product:0)gru_14/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������s
gru_14/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����u
*gru_14/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_14/while/gru_cell_26/split_1SplitV+gru_14/while/gru_cell_26/BiasAdd_1:output:0'gru_14/while/gru_cell_26/Const:output:03gru_14/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_14/while/gru_cell_26/addAddV2'gru_14/while/gru_cell_26/split:output:0)gru_14/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������
 gru_14/while/gru_cell_26/SigmoidSigmoid gru_14/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_1AddV2'gru_14/while/gru_cell_26/split:output:1)gru_14/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:����������
"gru_14/while/gru_cell_26/Sigmoid_1Sigmoid"gru_14/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mulMul&gru_14/while/gru_cell_26/Sigmoid_1:y:0)gru_14/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_2AddV2'gru_14/while/gru_cell_26/split:output:2 gru_14/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:����������
!gru_14/while/gru_cell_26/SoftplusSoftplus"gru_14/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mul_1Mul$gru_14/while/gru_cell_26/Sigmoid:y:0gru_14_while_placeholder_2*
T0*'
_output_shapes
:���������c
gru_14/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_14/while/gru_cell_26/subSub'gru_14/while/gru_cell_26/sub/x:output:0$gru_14/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mul_2Mul gru_14/while/gru_cell_26/sub:z:0/gru_14/while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_3AddV2"gru_14/while/gru_cell_26/mul_1:z:0"gru_14/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
1gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_14_while_placeholder_1gru_14_while_placeholder"gru_14/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_14/while/addAddV2gru_14_while_placeholdergru_14/while/add/y:output:0*
T0*
_output_shapes
: V
gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_14/while/add_1AddV2&gru_14_while_gru_14_while_loop_countergru_14/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_14/while/IdentityIdentitygru_14/while/add_1:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_1Identity,gru_14_while_gru_14_while_maximum_iterations^gru_14/while/NoOp*
T0*
_output_shapes
: n
gru_14/while/Identity_2Identitygru_14/while/add:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_3IdentityAgru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_4Identity"gru_14/while/gru_cell_26/add_3:z:0^gru_14/while/NoOp*
T0*'
_output_shapes
:����������
gru_14/while/NoOpNoOp/^gru_14/while/gru_cell_26/MatMul/ReadVariableOp1^gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp(^gru_14/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_14_while_gru_14_strided_slice_1%gru_14_while_gru_14_strided_slice_1_0"x
9gru_14_while_gru_cell_26_matmul_1_readvariableop_resource;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0"t
7gru_14_while_gru_cell_26_matmul_readvariableop_resource9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0"f
0gru_14_while_gru_cell_26_readvariableop_resource2gru_14_while_gru_cell_26_readvariableop_resource_0"7
gru_14_while_identitygru_14/while/Identity:output:0";
gru_14_while_identity_1 gru_14/while/Identity_1:output:0";
gru_14_while_identity_2 gru_14/while/Identity_2:output:0";
gru_14_while_identity_3 gru_14/while/Identity_3:output:0";
gru_14_while_identity_4 gru_14/while/Identity_4:output:0"�
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensoragru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.gru_14/while/gru_cell_26/MatMul/ReadVariableOp.gru_14/while/gru_cell_26/MatMul/ReadVariableOp2d
0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp2R
'gru_14/while/gru_cell_26/ReadVariableOp'gru_14/while/gru_cell_26/ReadVariableOp: 
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431570

inputs5
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while;
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2431481*
condR
while_cond_2431480*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
(__inference_gru_13_layer_call_fn_2433926
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2430744|
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
while_cond_2430679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2430679___redundant_placeholder05
1while_while_cond_2430679___redundant_placeholder15
1while_while_cond_2430679___redundant_placeholder25
1while_while_cond_2430679___redundant_placeholder3
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2432116

inputs6
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while;
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2432027*
condR
while_cond_2432026*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430823

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
�M
�
C__inference_gru_14_layer_call_and_return_conditional_losses_2435216

inputs5
#gru_cell_26_readvariableop_resource:<
*gru_cell_26_matmul_readvariableop_resource:d>
,gru_cell_26_matmul_1_readvariableop_resource:
identity��!gru_cell_26/MatMul/ReadVariableOp�#gru_cell_26/MatMul_1/ReadVariableOp�gru_cell_26/ReadVariableOp�while;
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
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������f
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������f
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����h
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:���������}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������i
gru_cell_26/SoftplusSoftplusgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:���������s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0"gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:���������z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
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
while_body_2435127*
condR
while_cond_2435126*8
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
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������d: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
(__inference_gru_13_layer_call_fn_2433937

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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431410t
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
� 
�
while_body_2430160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_24_2430182_0:	�.
while_gru_cell_24_2430184_0:	�/
while_gru_cell_24_2430186_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_24_2430182:	�,
while_gru_cell_24_2430184:	�-
while_gru_cell_24_2430186:
����)while/gru_cell_24/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/gru_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_24_2430182_0while_gru_cell_24_2430184_0while_gru_cell_24_2430186_0*
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430147�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_24/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:����������x

while/NoOpNoOp*^while/gru_cell_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_24_2430182while_gru_cell_24_2430182_0"8
while_gru_cell_24_2430184while_gru_cell_24_2430184_0"8
while_gru_cell_24_2430186while_gru_cell_24_2430186_0")
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
)while/gru_cell_24/StatefulPartitionedCall)while/gru_cell_24/StatefulPartitionedCall: 
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435389

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
�=
�
while_body_2433356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
while_body_2433662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_24_readvariableop_resource_0:	�E
2while_gru_cell_24_matmul_readvariableop_resource_0:	�H
4while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_24_readvariableop_resource:	�C
0while_gru_cell_24_matmul_readvariableop_resource:	�F
2while_gru_cell_24_matmul_1_readvariableop_resource:
����'while/gru_cell_24/MatMul/ReadVariableOp�)while/gru_cell_24/MatMul_1/ReadVariableOp� while/gru_cell_24/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 while/gru_cell_24/ReadVariableOpReadVariableOp+while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/unstackUnpack(while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAddBiasAdd"while/gru_cell_24/MatMul:product:0"while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/splitSplit*while/gru_cell_24/split/split_dim:output:0"while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
)while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_24/MatMul_1MatMulwhile_placeholder_21while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/BiasAdd_1BiasAdd$while/gru_cell_24/MatMul_1:product:0"while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����n
#while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_24/split_1SplitV$while/gru_cell_24/BiasAdd_1:output:0 while/gru_cell_24/Const:output:0,while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
while/gru_cell_24/addAddV2 while/gru_cell_24/split:output:0"while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������r
while/gru_cell_24/SigmoidSigmoidwhile/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_1AddV2 while/gru_cell_24/split:output:1"while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_1Sigmoidwhile/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mulMulwhile/gru_cell_24/Sigmoid_1:y:0"while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_2AddV2 while/gru_cell_24/split:output:2while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������v
while/gru_cell_24/Sigmoid_2Sigmoidwhile/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_1Mulwhile/gru_cell_24/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:����������\
while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_24/subSub while/gru_cell_24/sub/x:output:0while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/mul_2Mulwhile/gru_cell_24/sub:z:0while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
while/gru_cell_24/add_3AddV2while/gru_cell_24/mul_1:z:0while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_24/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_24/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp(^while/gru_cell_24/MatMul/ReadVariableOp*^while/gru_cell_24/MatMul_1/ReadVariableOp!^while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_24_matmul_1_readvariableop_resource4while_gru_cell_24_matmul_1_readvariableop_resource_0"f
0while_gru_cell_24_matmul_readvariableop_resource2while_gru_cell_24_matmul_readvariableop_resource_0"X
)while_gru_cell_24_readvariableop_resource+while_gru_cell_24_readvariableop_resource_0")
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
'while/gru_cell_24/MatMul/ReadVariableOp'while/gru_cell_24/MatMul/ReadVariableOp2V
)while/gru_cell_24/MatMul_1/ReadVariableOp)while/gru_cell_24/MatMul_1/ReadVariableOp2D
 while/gru_cell_24/ReadVariableOp while/gru_cell_24/ReadVariableOp: 
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
�M
�
C__inference_gru_13_layer_call_and_return_conditional_losses_2434254
inputs_06
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while=
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2434165*
condR
while_cond_2434164*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435534

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
�
�
(__inference_gru_14_layer_call_fn_2434604

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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431766t
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
while_cond_2434470
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2434470___redundant_placeholder05
1while_while_cond_2434470___redundant_placeholder15
1while_while_cond_2434470___redundant_placeholder25
1while_while_cond_2434470___redundant_placeholder3
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
�=
�
while_body_2434974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
-__inference_gru_cell_25_layer_call_fn_2435336

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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2430485o
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
�=
�
while_body_2431481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_26_readvariableop_resource_0:D
2while_gru_cell_26_matmul_readvariableop_resource_0:dF
4while_gru_cell_26_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_26_readvariableop_resource:B
0while_gru_cell_26_matmul_readvariableop_resource:dD
2while_gru_cell_26_matmul_1_readvariableop_resource:��'while/gru_cell_26/MatMul/ReadVariableOp�)while/gru_cell_26/MatMul_1/ReadVariableOp� while/gru_cell_26/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������l
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������l
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����n
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:���������u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:���������u
while/gru_cell_26/SoftplusSoftpluswhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0(while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
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
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 
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
while_body_2434471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	�F
2while_gru_cell_25_matmul_readvariableop_resource_0:
��G
4while_gru_cell_25_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	�D
0while_gru_cell_25_matmul_readvariableop_resource:
��E
2while_gru_cell_25_matmul_1_readvariableop_resource:	d���'while/gru_cell_25/MatMul/ReadVariableOp�)while/gru_cell_25/MatMul_1/ReadVariableOp� while/gru_cell_25/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������l
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������l
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����n
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������dq
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������du
while/gru_cell_25/Sigmoid_2Sigmoidwhile/gru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������d\
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
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
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 
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
�F
�	
gru_12_while_body_2432861*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2)
%gru_12_while_gru_12_strided_slice_1_0e
agru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0E
2gru_12_while_gru_cell_24_readvariableop_resource_0:	�L
9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0:	�O
;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0:
��
gru_12_while_identity
gru_12_while_identity_1
gru_12_while_identity_2
gru_12_while_identity_3
gru_12_while_identity_4'
#gru_12_while_gru_12_strided_slice_1c
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensorC
0gru_12_while_gru_cell_24_readvariableop_resource:	�J
7gru_12_while_gru_cell_24_matmul_readvariableop_resource:	�M
9gru_12_while_gru_cell_24_matmul_1_readvariableop_resource:
����.gru_12/while/gru_cell_24/MatMul/ReadVariableOp�0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp�'gru_12/while/gru_cell_24/ReadVariableOp�
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0gru_12_while_placeholderGgru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'gru_12/while/gru_cell_24/ReadVariableOpReadVariableOp2gru_12_while_gru_cell_24_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 gru_12/while/gru_cell_24/unstackUnpack/gru_12/while/gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.gru_12/while/gru_cell_24/MatMul/ReadVariableOpReadVariableOp9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
gru_12/while/gru_cell_24/MatMulMatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:06gru_12/while/gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 gru_12/while/gru_cell_24/BiasAddBiasAdd)gru_12/while/gru_cell_24/MatMul:product:0)gru_12/while/gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������s
(gru_12/while/gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_12/while/gru_cell_24/splitSplit1gru_12/while/gru_cell_24/split/split_dim:output:0)gru_12/while/gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
!gru_12/while/gru_cell_24/MatMul_1MatMulgru_12_while_placeholder_28gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/BiasAdd_1BiasAdd+gru_12/while/gru_cell_24/MatMul_1:product:0)gru_12/while/gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������s
gru_12/while/gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����u
*gru_12/while/gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_12/while/gru_cell_24/split_1SplitV+gru_12/while/gru_cell_24/BiasAdd_1:output:0'gru_12/while/gru_cell_24/Const:output:03gru_12/while/gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_12/while/gru_cell_24/addAddV2'gru_12/while/gru_cell_24/split:output:0)gru_12/while/gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:�����������
 gru_12/while/gru_cell_24/SigmoidSigmoid gru_12/while/gru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_1AddV2'gru_12/while/gru_cell_24/split:output:1)gru_12/while/gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/Sigmoid_1Sigmoid"gru_12/while/gru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mulMul&gru_12/while/gru_cell_24/Sigmoid_1:y:0)gru_12/while/gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_2AddV2'gru_12/while/gru_cell_24/split:output:2 gru_12/while/gru_cell_24/mul:z:0*
T0*(
_output_shapes
:�����������
"gru_12/while/gru_cell_24/Sigmoid_2Sigmoid"gru_12/while/gru_cell_24/add_2:z:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mul_1Mul$gru_12/while/gru_cell_24/Sigmoid:y:0gru_12_while_placeholder_2*
T0*(
_output_shapes
:����������c
gru_12/while/gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_12/while/gru_cell_24/subSub'gru_12/while/gru_cell_24/sub/x:output:0$gru_12/while/gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/mul_2Mul gru_12/while/gru_cell_24/sub:z:0&gru_12/while/gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:�����������
gru_12/while/gru_cell_24/add_3AddV2"gru_12/while/gru_cell_24/mul_1:z:0"gru_12/while/gru_cell_24/mul_2:z:0*
T0*(
_output_shapes
:�����������
1gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_12_while_placeholder_1gru_12_while_placeholder"gru_12/while/gru_cell_24/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_12/while/addAddV2gru_12_while_placeholdergru_12/while/add/y:output:0*
T0*
_output_shapes
: V
gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_12/while/add_1AddV2&gru_12_while_gru_12_while_loop_countergru_12/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_12/while/IdentityIdentitygru_12/while/add_1:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_1Identity,gru_12_while_gru_12_while_maximum_iterations^gru_12/while/NoOp*
T0*
_output_shapes
: n
gru_12/while/Identity_2Identitygru_12/while/add:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_3IdentityAgru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_12/while/NoOp*
T0*
_output_shapes
: �
gru_12/while/Identity_4Identity"gru_12/while/gru_cell_24/add_3:z:0^gru_12/while/NoOp*
T0*(
_output_shapes
:�����������
gru_12/while/NoOpNoOp/^gru_12/while/gru_cell_24/MatMul/ReadVariableOp1^gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp(^gru_12/while/gru_cell_24/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_12_while_gru_12_strided_slice_1%gru_12_while_gru_12_strided_slice_1_0"x
9gru_12_while_gru_cell_24_matmul_1_readvariableop_resource;gru_12_while_gru_cell_24_matmul_1_readvariableop_resource_0"t
7gru_12_while_gru_cell_24_matmul_readvariableop_resource9gru_12_while_gru_cell_24_matmul_readvariableop_resource_0"f
0gru_12_while_gru_cell_24_readvariableop_resource2gru_12_while_gru_cell_24_readvariableop_resource_0"7
gru_12_while_identitygru_12/while/Identity:output:0";
gru_12_while_identity_1 gru_12/while/Identity_1:output:0";
gru_12_while_identity_2 gru_12/while/Identity_2:output:0";
gru_12_while_identity_3 gru_12/while/Identity_3:output:0";
gru_12_while_identity_4 gru_12/while/Identity_4:output:0"�
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensoragru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :����������: : : : : 2`
.gru_12/while/gru_cell_24/MatMul/ReadVariableOp.gru_12/while/gru_cell_24/MatMul/ReadVariableOp2d
0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp0gru_12/while/gru_cell_24/MatMul_1/ReadVariableOp2R
'gru_12/while/gru_cell_24/ReadVariableOp'gru_12/while/gru_cell_24/ReadVariableOp: 
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
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_2431579

inputs!
gru_12_2431251:	�!
gru_12_2431253:	�"
gru_12_2431255:
��!
gru_13_2431411:	�"
gru_13_2431413:
��!
gru_13_2431415:	d� 
gru_14_2431571: 
gru_14_2431573:d 
gru_14_2431575:
identity��gru_12/StatefulPartitionedCall�gru_13/StatefulPartitionedCall�gru_14/StatefulPartitionedCall�
gru_12/StatefulPartitionedCallStatefulPartitionedCallinputsgru_12_2431251gru_12_2431253gru_12_2431255*
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2431250�
gru_13/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0gru_13_2431411gru_13_2431413gru_13_2431415*
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2431410�
gru_14/StatefulPartitionedCallStatefulPartitionedCall'gru_13/StatefulPartitionedCall:output:0gru_14_2431571gru_14_2431573gru_14_2431575*
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2431570{
IdentityIdentity'gru_14/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^gru_12/StatefulPartitionedCall^gru_13/StatefulPartitionedCall^gru_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2@
gru_13/StatefulPartitionedCallgru_13/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�E
�	
gru_14_while_body_2433159*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2)
%gru_14_while_gru_14_strided_slice_1_0e
agru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0D
2gru_14_while_gru_cell_26_readvariableop_resource_0:K
9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0:dM
;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0:
gru_14_while_identity
gru_14_while_identity_1
gru_14_while_identity_2
gru_14_while_identity_3
gru_14_while_identity_4'
#gru_14_while_gru_14_strided_slice_1c
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensorB
0gru_14_while_gru_cell_26_readvariableop_resource:I
7gru_14_while_gru_cell_26_matmul_readvariableop_resource:dK
9gru_14_while_gru_cell_26_matmul_1_readvariableop_resource:��.gru_14/while/gru_cell_26/MatMul/ReadVariableOp�0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp�'gru_14/while/gru_cell_26/ReadVariableOp�
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
0gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0gru_14_while_placeholderGgru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������d*
element_dtype0�
'gru_14/while/gru_cell_26/ReadVariableOpReadVariableOp2gru_14_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 gru_14/while/gru_cell_26/unstackUnpack/gru_14/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num�
.gru_14/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
gru_14/while/gru_cell_26/MatMulMatMul7gru_14/while/TensorArrayV2Read/TensorListGetItem:item:06gru_14/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 gru_14/while/gru_cell_26/BiasAddBiasAdd)gru_14/while/gru_cell_26/MatMul:product:0)gru_14/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:���������s
(gru_14/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_14/while/gru_cell_26/splitSplit1gru_14/while/gru_cell_26/split/split_dim:output:0)gru_14/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
!gru_14/while/gru_cell_26/MatMul_1MatMulgru_14_while_placeholder_28gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"gru_14/while/gru_cell_26/BiasAdd_1BiasAdd+gru_14/while/gru_cell_26/MatMul_1:product:0)gru_14/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:���������s
gru_14/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����u
*gru_14/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gru_14/while/gru_cell_26/split_1SplitV+gru_14/while/gru_cell_26/BiasAdd_1:output:0'gru_14/while/gru_cell_26/Const:output:03gru_14/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
gru_14/while/gru_cell_26/addAddV2'gru_14/while/gru_cell_26/split:output:0)gru_14/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:���������
 gru_14/while/gru_cell_26/SigmoidSigmoid gru_14/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_1AddV2'gru_14/while/gru_cell_26/split:output:1)gru_14/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:����������
"gru_14/while/gru_cell_26/Sigmoid_1Sigmoid"gru_14/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mulMul&gru_14/while/gru_cell_26/Sigmoid_1:y:0)gru_14/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_2AddV2'gru_14/while/gru_cell_26/split:output:2 gru_14/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:����������
!gru_14/while/gru_cell_26/SoftplusSoftplus"gru_14/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mul_1Mul$gru_14/while/gru_cell_26/Sigmoid:y:0gru_14_while_placeholder_2*
T0*'
_output_shapes
:���������c
gru_14/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_14/while/gru_cell_26/subSub'gru_14/while/gru_cell_26/sub/x:output:0$gru_14/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/mul_2Mul gru_14/while/gru_cell_26/sub:z:0/gru_14/while/gru_cell_26/Softplus:activations:0*
T0*'
_output_shapes
:����������
gru_14/while/gru_cell_26/add_3AddV2"gru_14/while/gru_cell_26/mul_1:z:0"gru_14/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:����������
1gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_14_while_placeholder_1gru_14_while_placeholder"gru_14/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:���T
gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_14/while/addAddV2gru_14_while_placeholdergru_14/while/add/y:output:0*
T0*
_output_shapes
: V
gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
gru_14/while/add_1AddV2&gru_14_while_gru_14_while_loop_countergru_14/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_14/while/IdentityIdentitygru_14/while/add_1:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_1Identity,gru_14_while_gru_14_while_maximum_iterations^gru_14/while/NoOp*
T0*
_output_shapes
: n
gru_14/while/Identity_2Identitygru_14/while/add:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_3IdentityAgru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_14/while/NoOp*
T0*
_output_shapes
: �
gru_14/while/Identity_4Identity"gru_14/while/gru_cell_26/add_3:z:0^gru_14/while/NoOp*
T0*'
_output_shapes
:����������
gru_14/while/NoOpNoOp/^gru_14/while/gru_cell_26/MatMul/ReadVariableOp1^gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp(^gru_14/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_14_while_gru_14_strided_slice_1%gru_14_while_gru_14_strided_slice_1_0"x
9gru_14_while_gru_cell_26_matmul_1_readvariableop_resource;gru_14_while_gru_cell_26_matmul_1_readvariableop_resource_0"t
7gru_14_while_gru_cell_26_matmul_readvariableop_resource9gru_14_while_gru_cell_26_matmul_readvariableop_resource_0"f
0gru_14_while_gru_cell_26_readvariableop_resource2gru_14_while_gru_cell_26_readvariableop_resource_0"7
gru_14_while_identitygru_14/while/Identity:output:0";
gru_14_while_identity_1 gru_14/while/Identity_1:output:0";
gru_14_while_identity_2 gru_14/while/Identity_2:output:0";
gru_14_while_identity_3 gru_14/while/Identity_3:output:0";
gru_14_while_identity_4 gru_14/while/Identity_4:output:0"�
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensoragru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.gru_14/while/gru_cell_26/MatMul/ReadVariableOp.gru_14/while/gru_cell_26/MatMul/ReadVariableOp2d
0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp0gru_14/while/gru_cell_26/MatMul_1/ReadVariableOp2R
'gru_14/while/gru_cell_26/ReadVariableOp'gru_14/while/gru_cell_26/ReadVariableOp: 
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434101
inputs_06
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while=
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2434012*
condR
while_cond_2434011*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�	
�
gru_14_while_cond_2432707*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2,
(gru_14_while_less_gru_14_strided_slice_1C
?gru_14_while_gru_14_while_cond_2432707___redundant_placeholder0C
?gru_14_while_gru_14_while_cond_2432707___redundant_placeholder1C
?gru_14_while_gru_14_while_cond_2432707___redundant_placeholder2C
?gru_14_while_gru_14_while_cond_2432707___redundant_placeholder3
gru_14_while_identity
~
gru_14/while/LessLessgru_14_while_placeholder(gru_14_while_less_gru_14_strided_slice_1*
T0*
_output_shapes
: Y
gru_14/while/IdentityIdentitygru_14/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_14_while_identitygru_14/while/Identity:output:0*(
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
&sequential_4_gru_13_while_cond_2429838D
@sequential_4_gru_13_while_sequential_4_gru_13_while_loop_counterJ
Fsequential_4_gru_13_while_sequential_4_gru_13_while_maximum_iterations)
%sequential_4_gru_13_while_placeholder+
'sequential_4_gru_13_while_placeholder_1+
'sequential_4_gru_13_while_placeholder_2F
Bsequential_4_gru_13_while_less_sequential_4_gru_13_strided_slice_1]
Ysequential_4_gru_13_while_sequential_4_gru_13_while_cond_2429838___redundant_placeholder0]
Ysequential_4_gru_13_while_sequential_4_gru_13_while_cond_2429838___redundant_placeholder1]
Ysequential_4_gru_13_while_sequential_4_gru_13_while_cond_2429838___redundant_placeholder2]
Ysequential_4_gru_13_while_sequential_4_gru_13_while_cond_2429838___redundant_placeholder3&
"sequential_4_gru_13_while_identity
�
sequential_4/gru_13/while/LessLess%sequential_4_gru_13_while_placeholderBsequential_4_gru_13_while_less_sequential_4_gru_13_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/gru_13/while/IdentityIdentity"sequential_4/gru_13/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_gru_13_while_identity+sequential_4/gru_13/while/Identity:output:0*(
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
�4
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2430406

inputs&
gru_cell_24_2430330:	�&
gru_cell_24_2430332:	�'
gru_cell_24_2430334:
��
identity��#gru_cell_24/StatefulPartitionedCall�while;
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
#gru_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_24_2430330gru_cell_24_2430332gru_cell_24_2430334*
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430290n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_24_2430330gru_cell_24_2430332gru_cell_24_2430334*
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
while_body_2430342*
condR
while_cond_2430341*9
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
NoOpNoOp$^gru_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#gru_cell_24/StatefulPartitionedCall#gru_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�M
�
C__inference_gru_12_layer_call_and_return_conditional_losses_2433904

inputs6
#gru_cell_24_readvariableop_resource:	�=
*gru_cell_24_matmul_readvariableop_resource:	�@
,gru_cell_24_matmul_1_readvariableop_resource:
��
identity��!gru_cell_24/MatMul/ReadVariableOp�#gru_cell_24/MatMul_1/ReadVariableOp�gru_cell_24/ReadVariableOp�while;
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
gru_cell_24/ReadVariableOpReadVariableOp#gru_cell_24_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_24/unstackUnpack"gru_cell_24/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_24/MatMul/ReadVariableOpReadVariableOp*gru_cell_24_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell_24/MatMulMatMulstrided_slice_2:output:0)gru_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAddBiasAddgru_cell_24/MatMul:product:0gru_cell_24/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/splitSplit$gru_cell_24/split/split_dim:output:0gru_cell_24/BiasAdd:output:0*
T0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
#gru_cell_24/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_24/MatMul_1MatMulzeros:output:0+gru_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_24/BiasAdd_1BiasAddgru_cell_24/MatMul_1:product:0gru_cell_24/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_24/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  ����h
gru_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_24/split_1SplitVgru_cell_24/BiasAdd_1:output:0gru_cell_24/Const:output:0&gru_cell_24/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:����������:����������:����������*
	num_split�
gru_cell_24/addAddV2gru_cell_24/split:output:0gru_cell_24/split_1:output:0*
T0*(
_output_shapes
:����������f
gru_cell_24/SigmoidSigmoidgru_cell_24/add:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/add_1AddV2gru_cell_24/split:output:1gru_cell_24/split_1:output:1*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_1Sigmoidgru_cell_24/add_1:z:0*
T0*(
_output_shapes
:�����������
gru_cell_24/mulMulgru_cell_24/Sigmoid_1:y:0gru_cell_24/split_1:output:2*
T0*(
_output_shapes
:����������~
gru_cell_24/add_2AddV2gru_cell_24/split:output:2gru_cell_24/mul:z:0*
T0*(
_output_shapes
:����������j
gru_cell_24/Sigmoid_2Sigmoidgru_cell_24/add_2:z:0*
T0*(
_output_shapes
:����������t
gru_cell_24/mul_1Mulgru_cell_24/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:����������V
gru_cell_24/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
gru_cell_24/subSubgru_cell_24/sub/x:output:0gru_cell_24/Sigmoid:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/mul_2Mulgru_cell_24/sub:z:0gru_cell_24/Sigmoid_2:y:0*
T0*(
_output_shapes
:����������{
gru_cell_24/add_3AddV2gru_cell_24/mul_1:z:0gru_cell_24/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_24_readvariableop_resource*gru_cell_24_matmul_readvariableop_resource,gru_cell_24_matmul_1_readvariableop_resource*
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
while_body_2433815*
condR
while_cond_2433814*9
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
NoOpNoOp"^gru_cell_24/MatMul/ReadVariableOp$^gru_cell_24/MatMul_1/ReadVariableOp^gru_cell_24/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2F
!gru_cell_24/MatMul/ReadVariableOp!gru_cell_24/MatMul/ReadVariableOp2J
#gru_cell_24/MatMul_1/ReadVariableOp#gru_cell_24/MatMul_1/ReadVariableOp28
gru_cell_24/ReadVariableOpgru_cell_24/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&sequential_4_gru_14_while_cond_2429987D
@sequential_4_gru_14_while_sequential_4_gru_14_while_loop_counterJ
Fsequential_4_gru_14_while_sequential_4_gru_14_while_maximum_iterations)
%sequential_4_gru_14_while_placeholder+
'sequential_4_gru_14_while_placeholder_1+
'sequential_4_gru_14_while_placeholder_2F
Bsequential_4_gru_14_while_less_sequential_4_gru_14_strided_slice_1]
Ysequential_4_gru_14_while_sequential_4_gru_14_while_cond_2429987___redundant_placeholder0]
Ysequential_4_gru_14_while_sequential_4_gru_14_while_cond_2429987___redundant_placeholder1]
Ysequential_4_gru_14_while_sequential_4_gru_14_while_cond_2429987___redundant_placeholder2]
Ysequential_4_gru_14_while_sequential_4_gru_14_while_cond_2429987___redundant_placeholder3&
"sequential_4_gru_14_while_identity
�
sequential_4/gru_14/while/LessLess%sequential_4_gru_14_while_placeholderBsequential_4_gru_14_while_less_sequential_4_gru_14_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/gru_14/while/IdentityIdentity"sequential_4/gru_14/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_gru_14_while_identity+sequential_4/gru_14/while/Identity:output:0*(
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434560

inputs6
#gru_cell_25_readvariableop_resource:	�>
*gru_cell_25_matmul_readvariableop_resource:
��?
,gru_cell_25_matmul_1_readvariableop_resource:	d�
identity��!gru_cell_25/MatMul/ReadVariableOp�#gru_cell_25/MatMul_1/ReadVariableOp�gru_cell_25/ReadVariableOp�while;
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
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	�*
dtype0y
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:����������f
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:����������f
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����h
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*'
_output_shapes
:���������de
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������d�
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*'
_output_shapes
:���������d}
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*'
_output_shapes
:���������di
gru_cell_25/Sigmoid_2Sigmoidgru_cell_25/add_2:z:0*
T0*'
_output_shapes
:���������ds
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dV
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������dz
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
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
while_body_2434471*
condR
while_cond_2434470*8
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
NoOpNoOp"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
while_cond_2431480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2431480___redundant_placeholder05
1while_while_cond_2431480___redundant_placeholder15
1while_while_cond_2431480___redundant_placeholder25
1while_while_cond_2431480___redundant_placeholder3
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2430966

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
�
�
while_cond_2430159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2430159___redundant_placeholder05
1while_while_cond_2430159___redundant_placeholder15
1while_while_cond_2430159___redundant_placeholder25
1while_while_cond_2430159___redundant_placeholder3
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
�
gru_12_while_cond_2432860*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2,
(gru_12_while_less_gru_12_strided_slice_1C
?gru_12_while_gru_12_while_cond_2432860___redundant_placeholder0C
?gru_12_while_gru_12_while_cond_2432860___redundant_placeholder1C
?gru_12_while_gru_12_while_cond_2432860___redundant_placeholder2C
?gru_12_while_gru_12_while_cond_2432860___redundant_placeholder3
gru_12_while_identity
~
gru_12/while/LessLessgru_12_while_placeholder(gru_12_while_less_gru_12_strided_slice_1*
T0*
_output_shapes
: Y
gru_12/while/IdentityIdentitygru_12/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_12_while_identitygru_12/while/Identity:output:0*(
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

�
-__inference_gru_cell_24_layer_call_fn_2435230

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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2430147p
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
states/0"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
J
gru_12_input:
serving_default_gru_12_input:0����������?
gru_145
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
.__inference_sequential_4_layer_call_fn_2431600
.__inference_sequential_4_layer_call_fn_2432323
.__inference_sequential_4_layer_call_fn_2432346
.__inference_sequential_4_layer_call_fn_2432219�
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432797
I__inference_sequential_4_layer_call_and_return_conditional_losses_2433248
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432244
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432269�
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
"__inference__wrapped_model_2430077gru_12_input"�
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
(__inference_gru_12_layer_call_fn_2433259
(__inference_gru_12_layer_call_fn_2433270
(__inference_gru_12_layer_call_fn_2433281
(__inference_gru_12_layer_call_fn_2433292�
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433445
C__inference_gru_12_layer_call_and_return_conditional_losses_2433598
C__inference_gru_12_layer_call_and_return_conditional_losses_2433751
C__inference_gru_12_layer_call_and_return_conditional_losses_2433904�
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
(__inference_gru_13_layer_call_fn_2433915
(__inference_gru_13_layer_call_fn_2433926
(__inference_gru_13_layer_call_fn_2433937
(__inference_gru_13_layer_call_fn_2433948�
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434101
C__inference_gru_13_layer_call_and_return_conditional_losses_2434254
C__inference_gru_13_layer_call_and_return_conditional_losses_2434407
C__inference_gru_13_layer_call_and_return_conditional_losses_2434560�
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
(__inference_gru_14_layer_call_fn_2434571
(__inference_gru_14_layer_call_fn_2434582
(__inference_gru_14_layer_call_fn_2434593
(__inference_gru_14_layer_call_fn_2434604�
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2434757
C__inference_gru_14_layer_call_and_return_conditional_losses_2434910
C__inference_gru_14_layer_call_and_return_conditional_losses_2435063
C__inference_gru_14_layer_call_and_return_conditional_losses_2435216�
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
,:*	�2gru_12/gru_cell_24/kernel
7:5
��2#gru_12/gru_cell_24/recurrent_kernel
*:(	�2gru_12/gru_cell_24/bias
-:+
��2gru_13/gru_cell_25/kernel
6:4	d�2#gru_13/gru_cell_25/recurrent_kernel
*:(	�2gru_13/gru_cell_25/bias
+:)d2gru_14/gru_cell_26/kernel
5:32#gru_14/gru_cell_26/recurrent_kernel
):'2gru_14/gru_cell_26/bias
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
.__inference_sequential_4_layer_call_fn_2431600gru_12_input"�
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
.__inference_sequential_4_layer_call_fn_2432323inputs"�
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
.__inference_sequential_4_layer_call_fn_2432346inputs"�
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
.__inference_sequential_4_layer_call_fn_2432219gru_12_input"�
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432797inputs"�
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2433248inputs"�
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432244gru_12_input"�
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432269gru_12_input"�
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
%__inference_signature_wrapper_2432300gru_12_input"�
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
(__inference_gru_12_layer_call_fn_2433259inputs/0"�
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
(__inference_gru_12_layer_call_fn_2433270inputs/0"�
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
(__inference_gru_12_layer_call_fn_2433281inputs"�
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
(__inference_gru_12_layer_call_fn_2433292inputs"�
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433445inputs/0"�
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433598inputs/0"�
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433751inputs"�
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433904inputs"�
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
-__inference_gru_cell_24_layer_call_fn_2435230
-__inference_gru_cell_24_layer_call_fn_2435244�
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435283
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435322�
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
(__inference_gru_13_layer_call_fn_2433915inputs/0"�
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
(__inference_gru_13_layer_call_fn_2433926inputs/0"�
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
(__inference_gru_13_layer_call_fn_2433937inputs"�
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
(__inference_gru_13_layer_call_fn_2433948inputs"�
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434101inputs/0"�
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434254inputs/0"�
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434407inputs"�
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434560inputs"�
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
-__inference_gru_cell_25_layer_call_fn_2435336
-__inference_gru_cell_25_layer_call_fn_2435350�
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435389
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435428�
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
(__inference_gru_14_layer_call_fn_2434571inputs/0"�
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
(__inference_gru_14_layer_call_fn_2434582inputs/0"�
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
(__inference_gru_14_layer_call_fn_2434593inputs"�
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
(__inference_gru_14_layer_call_fn_2434604inputs"�
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2434757inputs/0"�
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2434910inputs/0"�
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2435063inputs"�
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2435216inputs"�
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
-__inference_gru_cell_26_layer_call_fn_2435442
-__inference_gru_cell_26_layer_call_fn_2435456�
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435495
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435534�
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
-__inference_gru_cell_24_layer_call_fn_2435230inputsstates/0"�
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
-__inference_gru_cell_24_layer_call_fn_2435244inputsstates/0"�
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435283inputsstates/0"�
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435322inputsstates/0"�
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
-__inference_gru_cell_25_layer_call_fn_2435336inputsstates/0"�
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
-__inference_gru_cell_25_layer_call_fn_2435350inputsstates/0"�
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435389inputsstates/0"�
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435428inputsstates/0"�
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
-__inference_gru_cell_26_layer_call_fn_2435442inputsstates/0"�
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
-__inference_gru_cell_26_layer_call_fn_2435456inputsstates/0"�
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435495inputsstates/0"�
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435534inputsstates/0"�
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
1:/	�2 Adam/gru_12/gru_cell_24/kernel/m
<::
��2*Adam/gru_12/gru_cell_24/recurrent_kernel/m
/:-	�2Adam/gru_12/gru_cell_24/bias/m
2:0
��2 Adam/gru_13/gru_cell_25/kernel/m
;:9	d�2*Adam/gru_13/gru_cell_25/recurrent_kernel/m
/:-	�2Adam/gru_13/gru_cell_25/bias/m
0:.d2 Adam/gru_14/gru_cell_26/kernel/m
::82*Adam/gru_14/gru_cell_26/recurrent_kernel/m
.:,2Adam/gru_14/gru_cell_26/bias/m
1:/	�2 Adam/gru_12/gru_cell_24/kernel/v
<::
��2*Adam/gru_12/gru_cell_24/recurrent_kernel/v
/:-	�2Adam/gru_12/gru_cell_24/bias/v
2:0
��2 Adam/gru_13/gru_cell_25/kernel/v
;:9	d�2*Adam/gru_13/gru_cell_25/recurrent_kernel/v
/:-	�2Adam/gru_13/gru_cell_25/bias/v
0:.d2 Adam/gru_14/gru_cell_26/kernel/v
::82*Adam/gru_14/gru_cell_26/recurrent_kernel/v
.:,2Adam/gru_14/gru_cell_26/bias/v�
"__inference__wrapped_model_2430077}	*()-+,0./:�7
0�-
+�(
gru_12_input����������
� "4�1
/
gru_14%�"
gru_14�����������
C__inference_gru_12_layer_call_and_return_conditional_losses_2433445�*()O�L
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433598�*()O�L
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433751t*()@�=
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
C__inference_gru_12_layer_call_and_return_conditional_losses_2433904t*()@�=
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
(__inference_gru_12_layer_call_fn_2433259~*()O�L
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
(__inference_gru_12_layer_call_fn_2433270~*()O�L
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
(__inference_gru_12_layer_call_fn_2433281g*()@�=
6�3
%�"
inputs����������

 
p 

 
� "�������������
(__inference_gru_12_layer_call_fn_2433292g*()@�=
6�3
%�"
inputs����������

 
p

 
� "�������������
C__inference_gru_13_layer_call_and_return_conditional_losses_2434101�-+,P�M
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434254�-+,P�M
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434407t-+,A�>
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
C__inference_gru_13_layer_call_and_return_conditional_losses_2434560t-+,A�>
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
(__inference_gru_13_layer_call_fn_2433915~-+,P�M
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
(__inference_gru_13_layer_call_fn_2433926~-+,P�M
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
(__inference_gru_13_layer_call_fn_2433937g-+,A�>
7�4
&�#
inputs�����������

 
p 

 
� "�����������d�
(__inference_gru_13_layer_call_fn_2433948g-+,A�>
7�4
&�#
inputs�����������

 
p

 
� "�����������d�
C__inference_gru_14_layer_call_and_return_conditional_losses_2434757�0./O�L
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2434910�0./O�L
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2435063s0./@�=
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
C__inference_gru_14_layer_call_and_return_conditional_losses_2435216s0./@�=
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
(__inference_gru_14_layer_call_fn_2434571}0./O�L
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
(__inference_gru_14_layer_call_fn_2434582}0./O�L
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
(__inference_gru_14_layer_call_fn_2434593f0./@�=
6�3
%�"
inputs����������d

 
p 

 
� "������������
(__inference_gru_14_layer_call_fn_2434604f0./@�=
6�3
%�"
inputs����������d

 
p

 
� "������������
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435283�*()]�Z
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
H__inference_gru_cell_24_layer_call_and_return_conditional_losses_2435322�*()]�Z
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
-__inference_gru_cell_24_layer_call_fn_2435230�*()]�Z
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
-__inference_gru_cell_24_layer_call_fn_2435244�*()]�Z
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435389�-+,]�Z
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
H__inference_gru_cell_25_layer_call_and_return_conditional_losses_2435428�-+,]�Z
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
-__inference_gru_cell_25_layer_call_fn_2435336�-+,]�Z
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
-__inference_gru_cell_25_layer_call_fn_2435350�-+,]�Z
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435495�0./\�Y
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
H__inference_gru_cell_26_layer_call_and_return_conditional_losses_2435534�0./\�Y
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
-__inference_gru_cell_26_layer_call_fn_2435442�0./\�Y
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
-__inference_gru_cell_26_layer_call_fn_2435456�0./\�Y
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432244{	*()-+,0./B�?
8�5
+�(
gru_12_input����������
p 

 
� "*�'
 �
0����������
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432269{	*()-+,0./B�?
8�5
+�(
gru_12_input����������
p

 
� "*�'
 �
0����������
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_2432797u	*()-+,0./<�9
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
I__inference_sequential_4_layer_call_and_return_conditional_losses_2433248u	*()-+,0./<�9
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
.__inference_sequential_4_layer_call_fn_2431600n	*()-+,0./B�?
8�5
+�(
gru_12_input����������
p 

 
� "������������
.__inference_sequential_4_layer_call_fn_2432219n	*()-+,0./B�?
8�5
+�(
gru_12_input����������
p

 
� "������������
.__inference_sequential_4_layer_call_fn_2432323h	*()-+,0./<�9
2�/
%�"
inputs����������
p 

 
� "������������
.__inference_sequential_4_layer_call_fn_2432346h	*()-+,0./<�9
2�/
%�"
inputs����������
p

 
� "������������
%__inference_signature_wrapper_2432300�	*()-+,0./J�G
� 
@�=
;
gru_12_input+�(
gru_12_input����������"4�1
/
gru_14%�"
gru_14����������