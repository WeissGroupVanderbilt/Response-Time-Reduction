нч/
ёЦ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.9.12v2.9.0-18-gd8ce9f9c3018ЩІ-

Adam/gru_11/gru_cell_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_11/gru_cell_20/bias/v

2Adam/gru_11/gru_cell_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_11/gru_cell_20/bias/v*
_output_shapes

:*
dtype0
А
*Adam/gru_11/gru_cell_20/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_11/gru_cell_20/recurrent_kernel/v
Љ
>Adam/gru_11/gru_cell_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_11/gru_cell_20/recurrent_kernel/v*
_output_shapes

:*
dtype0

 Adam/gru_11/gru_cell_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_11/gru_cell_20/kernel/v

4Adam/gru_11/gru_cell_20/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_11/gru_cell_20/kernel/v*
_output_shapes

:d*
dtype0

Adam/gru_10/gru_cell_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*/
shared_name Adam/gru_10/gru_cell_19/bias/v

2Adam/gru_10/gru_cell_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_10/gru_cell_19/bias/v*
_output_shapes
:	Ќ*
dtype0
Б
*Adam/gru_10/gru_cell_19/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dЌ*;
shared_name,*Adam/gru_10/gru_cell_19/recurrent_kernel/v
Њ
>Adam/gru_10/gru_cell_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_10/gru_cell_19/recurrent_kernel/v*
_output_shapes
:	dЌ*
dtype0

 Adam/gru_10/gru_cell_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЌЌ*1
shared_name" Adam/gru_10/gru_cell_19/kernel/v

4Adam/gru_10/gru_cell_19/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_10/gru_cell_19/kernel/v* 
_output_shapes
:
ЌЌ*
dtype0

Adam/gru_9/gru_cell_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameAdam/gru_9/gru_cell_18/bias/v

1Adam/gru_9/gru_cell_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_18/bias/v*
_output_shapes
:	*
dtype0
А
)Adam/gru_9/gru_cell_18/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ќ*:
shared_name+)Adam/gru_9/gru_cell_18/recurrent_kernel/v
Љ
=Adam/gru_9/gru_cell_18/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Adam/gru_9/gru_cell_18/recurrent_kernel/v* 
_output_shapes
:
Ќ*
dtype0

Adam/gru_9/gru_cell_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/gru_9/gru_cell_18/kernel/v

3Adam/gru_9/gru_cell_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_18/kernel/v*
_output_shapes
:	*
dtype0

Adam/gru_11/gru_cell_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/gru_11/gru_cell_20/bias/m

2Adam/gru_11/gru_cell_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_11/gru_cell_20/bias/m*
_output_shapes

:*
dtype0
А
*Adam/gru_11/gru_cell_20/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/gru_11/gru_cell_20/recurrent_kernel/m
Љ
>Adam/gru_11/gru_cell_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_11/gru_cell_20/recurrent_kernel/m*
_output_shapes

:*
dtype0

 Adam/gru_11/gru_cell_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" Adam/gru_11/gru_cell_20/kernel/m

4Adam/gru_11/gru_cell_20/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_11/gru_cell_20/kernel/m*
_output_shapes

:d*
dtype0

Adam/gru_10/gru_cell_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*/
shared_name Adam/gru_10/gru_cell_19/bias/m

2Adam/gru_10/gru_cell_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_10/gru_cell_19/bias/m*
_output_shapes
:	Ќ*
dtype0
Б
*Adam/gru_10/gru_cell_19/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dЌ*;
shared_name,*Adam/gru_10/gru_cell_19/recurrent_kernel/m
Њ
>Adam/gru_10/gru_cell_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_10/gru_cell_19/recurrent_kernel/m*
_output_shapes
:	dЌ*
dtype0

 Adam/gru_10/gru_cell_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЌЌ*1
shared_name" Adam/gru_10/gru_cell_19/kernel/m

4Adam/gru_10/gru_cell_19/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_10/gru_cell_19/kernel/m* 
_output_shapes
:
ЌЌ*
dtype0

Adam/gru_9/gru_cell_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameAdam/gru_9/gru_cell_18/bias/m

1Adam/gru_9/gru_cell_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_18/bias/m*
_output_shapes
:	*
dtype0
А
)Adam/gru_9/gru_cell_18/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ќ*:
shared_name+)Adam/gru_9/gru_cell_18/recurrent_kernel/m
Љ
=Adam/gru_9/gru_cell_18/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Adam/gru_9/gru_cell_18/recurrent_kernel/m* 
_output_shapes
:
Ќ*
dtype0

Adam/gru_9/gru_cell_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/gru_9/gru_cell_18/kernel/m

3Adam/gru_9/gru_cell_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_18/kernel/m*
_output_shapes
:	*
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

gru_11/gru_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namegru_11/gru_cell_20/bias

+gru_11/gru_cell_20/bias/Read/ReadVariableOpReadVariableOpgru_11/gru_cell_20/bias*
_output_shapes

:*
dtype0
Ђ
#gru_11/gru_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#gru_11/gru_cell_20/recurrent_kernel

7gru_11/gru_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_11/gru_cell_20/recurrent_kernel*
_output_shapes

:*
dtype0

gru_11/gru_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_namegru_11/gru_cell_20/kernel

-gru_11/gru_cell_20/kernel/Read/ReadVariableOpReadVariableOpgru_11/gru_cell_20/kernel*
_output_shapes

:d*
dtype0

gru_10/gru_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*(
shared_namegru_10/gru_cell_19/bias

+gru_10/gru_cell_19/bias/Read/ReadVariableOpReadVariableOpgru_10/gru_cell_19/bias*
_output_shapes
:	Ќ*
dtype0
Ѓ
#gru_10/gru_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dЌ*4
shared_name%#gru_10/gru_cell_19/recurrent_kernel

7gru_10/gru_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_10/gru_cell_19/recurrent_kernel*
_output_shapes
:	dЌ*
dtype0

gru_10/gru_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЌЌ**
shared_namegru_10/gru_cell_19/kernel

-gru_10/gru_cell_19/kernel/Read/ReadVariableOpReadVariableOpgru_10/gru_cell_19/kernel* 
_output_shapes
:
ЌЌ*
dtype0

gru_9/gru_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_namegru_9/gru_cell_18/bias

*gru_9/gru_cell_18/bias/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_18/bias*
_output_shapes
:	*
dtype0
Ђ
"gru_9/gru_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ќ*3
shared_name$"gru_9/gru_cell_18/recurrent_kernel

6gru_9/gru_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_9/gru_cell_18/recurrent_kernel* 
_output_shapes
:
Ќ*
dtype0

gru_9/gru_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namegru_9/gru_cell_18/kernel

,gru_9/gru_cell_18/kernel/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_18/kernel*
_output_shapes
:	*
dtype0

NoOpNoOp
I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*вH
valueШHBХH BОH
С
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
С
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
С
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
С
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
А
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
ј
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate(mЃ)mЄ*mЅ+mІ,mЇ-mЈ.mЉ/mЊ0mЋ(vЌ)v­*vЎ+vЏ,vА-vБ.vВ/vГ0vД*
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


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
г
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


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
г
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


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
ж
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

.kernel
/recurrent_kernel
0bias*
* 
XR
VARIABLE_VALUEgru_9/gru_cell_18/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_9/gru_cell_18/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_9/gru_cell_18/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_10/gru_cell_19/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_10/gru_cell_19/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_10/gru_cell_19/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgru_11/gru_cell_20/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#gru_11/gru_cell_20/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_11/gru_cell_20/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

0*
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
<
	variables
 	keras_api

Ёtotal

Ђcount*
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
Ё0
Ђ1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_9/gru_cell_18/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_9/gru_cell_18/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_9/gru_cell_18/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_10/gru_cell_19/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/gru_10/gru_cell_19/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_10/gru_cell_19/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_11/gru_cell_20/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/gru_11/gru_cell_20/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_11/gru_cell_20/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_9/gru_cell_18/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_9/gru_cell_18/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_9/gru_cell_18/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_10/gru_cell_19/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/gru_10/gru_cell_19/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_10/gru_cell_19/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/gru_11/gru_cell_20/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/gru_11/gru_cell_20/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_11/gru_cell_20/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_gru_9_inputPlaceholder*,
_output_shapes
:џџџџџџџџџњ*
dtype0*!
shape:џџџџџџџџџњ
й
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_9_inputgru_9/gru_cell_18/biasgru_9/gru_cell_18/kernel"gru_9/gru_cell_18/recurrent_kernelgru_10/gru_cell_19/biasgru_10/gru_cell_19/kernel#gru_10/gru_cell_19/recurrent_kernelgru_11/gru_cell_20/biasgru_11/gru_cell_20/kernel#gru_11/gru_cell_20/recurrent_kernel*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1959771
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
в
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,gru_9/gru_cell_18/kernel/Read/ReadVariableOp6gru_9/gru_cell_18/recurrent_kernel/Read/ReadVariableOp*gru_9/gru_cell_18/bias/Read/ReadVariableOp-gru_10/gru_cell_19/kernel/Read/ReadVariableOp7gru_10/gru_cell_19/recurrent_kernel/Read/ReadVariableOp+gru_10/gru_cell_19/bias/Read/ReadVariableOp-gru_11/gru_cell_20/kernel/Read/ReadVariableOp7gru_11/gru_cell_20/recurrent_kernel/Read/ReadVariableOp+gru_11/gru_cell_20/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/gru_9/gru_cell_18/kernel/m/Read/ReadVariableOp=Adam/gru_9/gru_cell_18/recurrent_kernel/m/Read/ReadVariableOp1Adam/gru_9/gru_cell_18/bias/m/Read/ReadVariableOp4Adam/gru_10/gru_cell_19/kernel/m/Read/ReadVariableOp>Adam/gru_10/gru_cell_19/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_10/gru_cell_19/bias/m/Read/ReadVariableOp4Adam/gru_11/gru_cell_20/kernel/m/Read/ReadVariableOp>Adam/gru_11/gru_cell_20/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_11/gru_cell_20/bias/m/Read/ReadVariableOp3Adam/gru_9/gru_cell_18/kernel/v/Read/ReadVariableOp=Adam/gru_9/gru_cell_18/recurrent_kernel/v/Read/ReadVariableOp1Adam/gru_9/gru_cell_18/bias/v/Read/ReadVariableOp4Adam/gru_10/gru_cell_19/kernel/v/Read/ReadVariableOp>Adam/gru_10/gru_cell_19/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_10/gru_cell_19/bias/v/Read/ReadVariableOp4Adam/gru_11/gru_cell_20/kernel/v/Read/ReadVariableOp>Adam/gru_11/gru_cell_20/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_11/gru_cell_20/bias/v/Read/ReadVariableOpConst*/
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_1963130
Ѕ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegru_9/gru_cell_18/kernel"gru_9/gru_cell_18/recurrent_kernelgru_9/gru_cell_18/biasgru_10/gru_cell_19/kernel#gru_10/gru_cell_19/recurrent_kernelgru_10/gru_cell_19/biasgru_11/gru_cell_20/kernel#gru_11/gru_cell_20/recurrent_kernelgru_11/gru_cell_20/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/gru_9/gru_cell_18/kernel/m)Adam/gru_9/gru_cell_18/recurrent_kernel/mAdam/gru_9/gru_cell_18/bias/m Adam/gru_10/gru_cell_19/kernel/m*Adam/gru_10/gru_cell_19/recurrent_kernel/mAdam/gru_10/gru_cell_19/bias/m Adam/gru_11/gru_cell_20/kernel/m*Adam/gru_11/gru_cell_20/recurrent_kernel/mAdam/gru_11/gru_cell_20/bias/mAdam/gru_9/gru_cell_18/kernel/v)Adam/gru_9/gru_cell_18/recurrent_kernel/vAdam/gru_9/gru_cell_18/bias/v Adam/gru_10/gru_cell_19/kernel/v*Adam/gru_10/gru_cell_19/recurrent_kernel/vAdam/gru_10/gru_cell_19/bias/v Adam/gru_11/gru_cell_20/kernel/v*Adam/gru_11/gru_cell_20/recurrent_kernel/vAdam/gru_11/gru_cell_20/bias/v*.
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1963242Эя+
 
О
while_body_1958151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_19_1958173_0:	Ќ/
while_gru_cell_19_1958175_0:
ЌЌ.
while_gru_cell_19_1958177_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_19_1958173:	Ќ-
while_gru_cell_19_1958175:
ЌЌ,
while_gru_cell_19_1958177:	dЌЂ)while/gru_cell_19/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
)while/gru_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_19_1958173_0while_gru_cell_19_1958175_0while_gru_cell_19_1958177_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1958099л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdx

while/NoOpNoOp*^while/gru_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_19_1958173while_gru_cell_19_1958173_0"8
while_gru_cell_19_1958175while_gru_cell_19_1958175_0"8
while_gru_cell_19_1958177while_gru_cell_19_1958177_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2V
)while/gru_cell_19/StatefulPartitionedCall)while/gru_cell_19/StatefulPartitionedCall: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
 
О
while_body_1957969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_19_1957991_0:	Ќ/
while_gru_cell_19_1957993_0:
ЌЌ.
while_gru_cell_19_1957995_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_19_1957991:	Ќ-
while_gru_cell_19_1957993:
ЌЌ,
while_gru_cell_19_1957995:	dЌЂ)while/gru_cell_19/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
)while/gru_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_19_1957991_0while_gru_cell_19_1957993_0while_gru_cell_19_1957995_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1957956л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdx

while/NoOpNoOp*^while/gru_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_19_1957991while_gru_cell_19_1957991_0"8
while_gru_cell_19_1957993while_gru_cell_19_1957993_0"8
while_gru_cell_19_1957995while_gru_cell_19_1957995_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2V
)while/gru_cell_19/StatefulPartitionedCall)while/gru_cell_19/StatefulPartitionedCall: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 

й
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1962966

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpf
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
:џџџџџџџџџh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџQ
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
п
Џ
while_cond_1962291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1962291___redundant_placeholder05
1while_while_cond_1962291___redundant_placeholder15
1while_while_cond_1962291___redundant_placeholder25
1while_while_cond_1962291___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

Й
'__inference_gru_9_layer_call_fn_1960763

inputs
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1959587u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Г
М
(__inference_gru_10_layer_call_fn_1961397
inputs_0
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1958215|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
V
л
&sequential_3_gru_11_while_body_1957459D
@sequential_3_gru_11_while_sequential_3_gru_11_while_loop_counterJ
Fsequential_3_gru_11_while_sequential_3_gru_11_while_maximum_iterations)
%sequential_3_gru_11_while_placeholder+
'sequential_3_gru_11_while_placeholder_1+
'sequential_3_gru_11_while_placeholder_2C
?sequential_3_gru_11_while_sequential_3_gru_11_strided_slice_1_0
{sequential_3_gru_11_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_11_tensorarrayunstack_tensorlistfromtensor_0Q
?sequential_3_gru_11_while_gru_cell_20_readvariableop_resource_0:X
Fsequential_3_gru_11_while_gru_cell_20_matmul_readvariableop_resource_0:dZ
Hsequential_3_gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0:&
"sequential_3_gru_11_while_identity(
$sequential_3_gru_11_while_identity_1(
$sequential_3_gru_11_while_identity_2(
$sequential_3_gru_11_while_identity_3(
$sequential_3_gru_11_while_identity_4A
=sequential_3_gru_11_while_sequential_3_gru_11_strided_slice_1}
ysequential_3_gru_11_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_11_tensorarrayunstack_tensorlistfromtensorO
=sequential_3_gru_11_while_gru_cell_20_readvariableop_resource:V
Dsequential_3_gru_11_while_gru_cell_20_matmul_readvariableop_resource:dX
Fsequential_3_gru_11_while_gru_cell_20_matmul_1_readvariableop_resource:Ђ;sequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOpЂ=sequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpЂ4sequential_3/gru_11/while/gru_cell_20/ReadVariableOp
Ksequential_3/gru_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   
=sequential_3/gru_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_3_gru_11_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_11_tensorarrayunstack_tensorlistfromtensor_0%sequential_3_gru_11_while_placeholderTsequential_3/gru_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0Д
4sequential_3/gru_11/while/gru_cell_20/ReadVariableOpReadVariableOp?sequential_3_gru_11_while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0Ћ
-sequential_3/gru_11/while/gru_cell_20/unstackUnpack<sequential_3/gru_11/while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numТ
;sequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOpFsequential_3_gru_11_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0ѓ
,sequential_3/gru_11/while/gru_cell_20/MatMulMatMulDsequential_3/gru_11/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџк
-sequential_3/gru_11/while/gru_cell_20/BiasAddBiasAdd6sequential_3/gru_11/while/gru_cell_20/MatMul:product:06sequential_3/gru_11/while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
5sequential_3/gru_11/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
+sequential_3/gru_11/while/gru_cell_20/splitSplit>sequential_3/gru_11/while/gru_cell_20/split/split_dim:output:06sequential_3/gru_11/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitЦ
=sequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOpHsequential_3_gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0к
.sequential_3/gru_11/while/gru_cell_20/MatMul_1MatMul'sequential_3_gru_11_while_placeholder_2Esequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџо
/sequential_3/gru_11/while/gru_cell_20/BiasAdd_1BiasAdd8sequential_3/gru_11/while/gru_cell_20/MatMul_1:product:06sequential_3/gru_11/while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
+sequential_3/gru_11/while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ
7sequential_3/gru_11/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
-sequential_3/gru_11/while/gru_cell_20/split_1SplitV8sequential_3/gru_11/while/gru_cell_20/BiasAdd_1:output:04sequential_3/gru_11/while/gru_cell_20/Const:output:0@sequential_3/gru_11/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitв
)sequential_3/gru_11/while/gru_cell_20/addAddV24sequential_3/gru_11/while/gru_cell_20/split:output:06sequential_3/gru_11/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
-sequential_3/gru_11/while/gru_cell_20/SigmoidSigmoid-sequential_3/gru_11/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџд
+sequential_3/gru_11/while/gru_cell_20/add_1AddV24sequential_3/gru_11/while/gru_cell_20/split:output:16sequential_3/gru_11/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
/sequential_3/gru_11/while/gru_cell_20/Sigmoid_1Sigmoid/sequential_3/gru_11/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЯ
)sequential_3/gru_11/while/gru_cell_20/mulMul3sequential_3/gru_11/while/gru_cell_20/Sigmoid_1:y:06sequential_3/gru_11/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџЫ
+sequential_3/gru_11/while/gru_cell_20/add_2AddV24sequential_3/gru_11/while/gru_cell_20/split:output:2-sequential_3/gru_11/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
.sequential_3/gru_11/while/gru_cell_20/SoftplusSoftplus/sequential_3/gru_11/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџР
+sequential_3/gru_11/while/gru_cell_20/mul_1Mul1sequential_3/gru_11/while/gru_cell_20/Sigmoid:y:0'sequential_3_gru_11_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџp
+sequential_3/gru_11/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ы
)sequential_3/gru_11/while/gru_cell_20/subSub4sequential_3/gru_11/while/gru_cell_20/sub/x:output:01sequential_3/gru_11/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџб
+sequential_3/gru_11/while/gru_cell_20/mul_2Mul-sequential_3/gru_11/while/gru_cell_20/sub:z:0<sequential_3/gru_11/while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџШ
+sequential_3/gru_11/while/gru_cell_20/add_3AddV2/sequential_3/gru_11/while/gru_cell_20/mul_1:z:0/sequential_3/gru_11/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
>sequential_3/gru_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_3_gru_11_while_placeholder_1%sequential_3_gru_11_while_placeholder/sequential_3/gru_11/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвa
sequential_3/gru_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_3/gru_11/while/addAddV2%sequential_3_gru_11_while_placeholder(sequential_3/gru_11/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_3/gru_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
sequential_3/gru_11/while/add_1AddV2@sequential_3_gru_11_while_sequential_3_gru_11_while_loop_counter*sequential_3/gru_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_3/gru_11/while/IdentityIdentity#sequential_3/gru_11/while/add_1:z:0^sequential_3/gru_11/while/NoOp*
T0*
_output_shapes
: К
$sequential_3/gru_11/while/Identity_1IdentityFsequential_3_gru_11_while_sequential_3_gru_11_while_maximum_iterations^sequential_3/gru_11/while/NoOp*
T0*
_output_shapes
: 
$sequential_3/gru_11/while/Identity_2Identity!sequential_3/gru_11/while/add:z:0^sequential_3/gru_11/while/NoOp*
T0*
_output_shapes
: Т
$sequential_3/gru_11/while/Identity_3IdentityNsequential_3/gru_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_3/gru_11/while/NoOp*
T0*
_output_shapes
: Д
$sequential_3/gru_11/while/Identity_4Identity/sequential_3/gru_11/while/gru_cell_20/add_3:z:0^sequential_3/gru_11/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_3/gru_11/while/NoOpNoOp<^sequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOp>^sequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp5^sequential_3/gru_11/while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Fsequential_3_gru_11_while_gru_cell_20_matmul_1_readvariableop_resourceHsequential_3_gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0"
Dsequential_3_gru_11_while_gru_cell_20_matmul_readvariableop_resourceFsequential_3_gru_11_while_gru_cell_20_matmul_readvariableop_resource_0"
=sequential_3_gru_11_while_gru_cell_20_readvariableop_resource?sequential_3_gru_11_while_gru_cell_20_readvariableop_resource_0"Q
"sequential_3_gru_11_while_identity+sequential_3/gru_11/while/Identity:output:0"U
$sequential_3_gru_11_while_identity_1-sequential_3/gru_11/while/Identity_1:output:0"U
$sequential_3_gru_11_while_identity_2-sequential_3/gru_11/while/Identity_2:output:0"U
$sequential_3_gru_11_while_identity_3-sequential_3/gru_11/while/Identity_3:output:0"U
$sequential_3_gru_11_while_identity_4-sequential_3/gru_11/while/Identity_4:output:0"
=sequential_3_gru_11_while_sequential_3_gru_11_strided_slice_1?sequential_3_gru_11_while_sequential_3_gru_11_strided_slice_1_0"ј
ysequential_3_gru_11_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_11_tensorarrayunstack_tensorlistfromtensor{sequential_3_gru_11_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2z
;sequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOp;sequential_3/gru_11/while/gru_cell_20/MatMul/ReadVariableOp2~
=sequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp=sequential_3/gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp2l
4sequential_3/gru_11/while/gru_cell_20/ReadVariableOp4sequential_3/gru_11/while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
В

ї
.__inference_sequential_3_layer_call_fn_1959690
gru_9_input
unknown:	
	unknown_0:	
	unknown_1:
Ќ
	unknown_2:	Ќ
	unknown_3:
ЌЌ
	unknown_4:	dЌ
	unknown_5:
	unknown_6:d
	unknown_7:
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959646t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input
эE
е	
gru_11_while_body_1960179*
&gru_11_while_gru_11_while_loop_counter0
,gru_11_while_gru_11_while_maximum_iterations
gru_11_while_placeholder
gru_11_while_placeholder_1
gru_11_while_placeholder_2)
%gru_11_while_gru_11_strided_slice_1_0e
agru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0D
2gru_11_while_gru_cell_20_readvariableop_resource_0:K
9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0:dM
;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0:
gru_11_while_identity
gru_11_while_identity_1
gru_11_while_identity_2
gru_11_while_identity_3
gru_11_while_identity_4'
#gru_11_while_gru_11_strided_slice_1c
_gru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensorB
0gru_11_while_gru_cell_20_readvariableop_resource:I
7gru_11_while_gru_cell_20_matmul_readvariableop_resource:dK
9gru_11_while_gru_cell_20_matmul_1_readvariableop_resource:Ђ.gru_11/while/gru_cell_20/MatMul/ReadVariableOpЂ0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpЂ'gru_11/while/gru_cell_20/ReadVariableOp
>gru_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Щ
0gru_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0gru_11_while_placeholderGgru_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
'gru_11/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_11_while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
 gru_11/while/gru_cell_20/unstackUnpack/gru_11/while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numЈ
.gru_11/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0Ь
gru_11/while/gru_cell_20/MatMulMatMul7gru_11/while/TensorArrayV2Read/TensorListGetItem:item:06gru_11/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџГ
 gru_11/while/gru_cell_20/BiasAddBiasAdd)gru_11/while/gru_cell_20/MatMul:product:0)gru_11/while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
(gru_11/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_11/while/gru_cell_20/splitSplit1gru_11/while/gru_cell_20/split/split_dim:output:0)gru_11/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitЌ
0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Г
!gru_11/while/gru_cell_20/MatMul_1MatMulgru_11_while_placeholder_28gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЗ
"gru_11/while/gru_cell_20/BiasAdd_1BiasAdd+gru_11/while/gru_cell_20/MatMul_1:product:0)gru_11/while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџs
gru_11/while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџu
*gru_11/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЊ
 gru_11/while/gru_cell_20/split_1SplitV+gru_11/while/gru_cell_20/BiasAdd_1:output:0'gru_11/while/gru_cell_20/Const:output:03gru_11/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitЋ
gru_11/while/gru_cell_20/addAddV2'gru_11/while/gru_cell_20/split:output:0)gru_11/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 gru_11/while/gru_cell_20/SigmoidSigmoid gru_11/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ­
gru_11/while/gru_cell_20/add_1AddV2'gru_11/while/gru_cell_20/split:output:1)gru_11/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
"gru_11/while/gru_cell_20/Sigmoid_1Sigmoid"gru_11/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
gru_11/while/gru_cell_20/mulMul&gru_11/while/gru_cell_20/Sigmoid_1:y:0)gru_11/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџЄ
gru_11/while/gru_cell_20/add_2AddV2'gru_11/while/gru_cell_20/split:output:2 gru_11/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gru_11/while/gru_cell_20/SoftplusSoftplus"gru_11/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/while/gru_cell_20/mul_1Mul$gru_11/while/gru_cell_20/Sigmoid:y:0gru_11_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџc
gru_11/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
gru_11/while/gru_cell_20/subSub'gru_11/while/gru_cell_20/sub/x:output:0$gru_11/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
gru_11/while/gru_cell_20/mul_2Mul gru_11/while/gru_cell_20/sub:z:0/gru_11/while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
gru_11/while/gru_cell_20/add_3AddV2"gru_11/while/gru_cell_20/mul_1:z:0"gru_11/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџр
1gru_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_11_while_placeholder_1gru_11_while_placeholder"gru_11/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвT
gru_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_11/while/addAddV2gru_11_while_placeholdergru_11/while/add/y:output:0*
T0*
_output_shapes
: V
gru_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_11/while/add_1AddV2&gru_11_while_gru_11_while_loop_countergru_11/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_11/while/IdentityIdentitygru_11/while/add_1:z:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_1Identity,gru_11_while_gru_11_while_maximum_iterations^gru_11/while/NoOp*
T0*
_output_shapes
: n
gru_11/while/Identity_2Identitygru_11/while/add:z:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_3IdentityAgru_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_4Identity"gru_11/while/gru_cell_20/add_3:z:0^gru_11/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџс
gru_11/while/NoOpNoOp/^gru_11/while/gru_cell_20/MatMul/ReadVariableOp1^gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp(^gru_11/while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_11_while_gru_11_strided_slice_1%gru_11_while_gru_11_strided_slice_1_0"x
9gru_11_while_gru_cell_20_matmul_1_readvariableop_resource;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_11_while_gru_cell_20_matmul_readvariableop_resource9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_11_while_gru_cell_20_readvariableop_resource2gru_11_while_gru_cell_20_readvariableop_resource_0"7
gru_11_while_identitygru_11/while/Identity:output:0";
gru_11_while_identity_1 gru_11/while/Identity_1:output:0";
gru_11_while_identity_2 gru_11/while/Identity_2:output:0";
gru_11_while_identity_3 gru_11/while/Identity_3:output:0";
gru_11_while_identity_4 gru_11/while/Identity_4:output:0"Ф
_gru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensoragru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2`
.gru_11/while/gru_cell_20/MatMul/ReadVariableOp.gru_11/while/gru_cell_20/MatMul/ReadVariableOp2d
0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp2R
'gru_11/while/gru_cell_20/ReadVariableOp'gru_11/while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
їE
н	
gru_10_while_body_1960481*
&gru_10_while_gru_10_while_loop_counter0
,gru_10_while_gru_10_while_maximum_iterations
gru_10_while_placeholder
gru_10_while_placeholder_1
gru_10_while_placeholder_2)
%gru_10_while_gru_10_strided_slice_1_0e
agru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0E
2gru_10_while_gru_cell_19_readvariableop_resource_0:	ЌM
9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌN
;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
gru_10_while_identity
gru_10_while_identity_1
gru_10_while_identity_2
gru_10_while_identity_3
gru_10_while_identity_4'
#gru_10_while_gru_10_strided_slice_1c
_gru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensorC
0gru_10_while_gru_cell_19_readvariableop_resource:	ЌK
7gru_10_while_gru_cell_19_matmul_readvariableop_resource:
ЌЌL
9gru_10_while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ.gru_10/while/gru_cell_19/MatMul/ReadVariableOpЂ0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpЂ'gru_10/while/gru_cell_19/ReadVariableOp
>gru_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ъ
0gru_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0gru_10_while_placeholderGgru_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
'gru_10/while/gru_cell_19/ReadVariableOpReadVariableOp2gru_10_while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
 gru_10/while/gru_cell_19/unstackUnpack/gru_10/while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numЊ
.gru_10/while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0Э
gru_10/while/gru_cell_19/MatMulMatMul7gru_10/while/TensorArrayV2Read/TensorListGetItem:item:06gru_10/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌД
 gru_10/while/gru_cell_19/BiasAddBiasAdd)gru_10/while/gru_cell_19/MatMul:product:0)gru_10/while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌs
(gru_10/while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_10/while/gru_cell_19/splitSplit1gru_10/while/gru_cell_19/split/split_dim:output:0)gru_10/while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split­
0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0Д
!gru_10/while/gru_cell_19/MatMul_1MatMulgru_10_while_placeholder_28gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌИ
"gru_10/while/gru_cell_19/BiasAdd_1BiasAdd+gru_10/while/gru_cell_19/MatMul_1:product:0)gru_10/while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌs
gru_10/while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџu
*gru_10/while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЊ
 gru_10/while/gru_cell_19/split_1SplitV+gru_10/while/gru_cell_19/BiasAdd_1:output:0'gru_10/while/gru_cell_19/Const:output:03gru_10/while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitЋ
gru_10/while/gru_cell_19/addAddV2'gru_10/while/gru_cell_19/split:output:0)gru_10/while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
 gru_10/while/gru_cell_19/SigmoidSigmoid gru_10/while/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd­
gru_10/while/gru_cell_19/add_1AddV2'gru_10/while/gru_cell_19/split:output:1)gru_10/while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
"gru_10/while/gru_cell_19/Sigmoid_1Sigmoid"gru_10/while/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
gru_10/while/gru_cell_19/mulMul&gru_10/while/gru_cell_19/Sigmoid_1:y:0)gru_10/while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЄ
gru_10/while/gru_cell_19/add_2AddV2'gru_10/while/gru_cell_19/split:output:2 gru_10/while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
"gru_10/while/gru_cell_19/Sigmoid_2Sigmoid"gru_10/while/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/while/gru_cell_19/mul_1Mul$gru_10/while/gru_cell_19/Sigmoid:y:0gru_10_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџdc
gru_10/while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
gru_10/while/gru_cell_19/subSub'gru_10/while/gru_cell_19/sub/x:output:0$gru_10/while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЁ
gru_10/while/gru_cell_19/mul_2Mul gru_10/while/gru_cell_19/sub:z:0&gru_10/while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЁ
gru_10/while/gru_cell_19/add_3AddV2"gru_10/while/gru_cell_19/mul_1:z:0"gru_10/while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdр
1gru_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_10_while_placeholder_1gru_10_while_placeholder"gru_10/while/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвT
gru_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_10/while/addAddV2gru_10_while_placeholdergru_10/while/add/y:output:0*
T0*
_output_shapes
: V
gru_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_10/while/add_1AddV2&gru_10_while_gru_10_while_loop_countergru_10/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_10/while/IdentityIdentitygru_10/while/add_1:z:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_1Identity,gru_10_while_gru_10_while_maximum_iterations^gru_10/while/NoOp*
T0*
_output_shapes
: n
gru_10/while/Identity_2Identitygru_10/while/add:z:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_3IdentityAgru_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_4Identity"gru_10/while/gru_cell_19/add_3:z:0^gru_10/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdс
gru_10/while/NoOpNoOp/^gru_10/while/gru_cell_19/MatMul/ReadVariableOp1^gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp(^gru_10/while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_10_while_gru_10_strided_slice_1%gru_10_while_gru_10_strided_slice_1_0"x
9gru_10_while_gru_cell_19_matmul_1_readvariableop_resource;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0"t
7gru_10_while_gru_cell_19_matmul_readvariableop_resource9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0"f
0gru_10_while_gru_cell_19_readvariableop_resource2gru_10_while_gru_cell_19_readvariableop_resource_0"7
gru_10_while_identitygru_10/while/Identity:output:0";
gru_10_while_identity_1 gru_10/while/Identity_1:output:0";
gru_10_while_identity_2 gru_10/while/Identity_2:output:0";
gru_10_while_identity_3 gru_10/while/Identity_3:output:0";
gru_10_while_identity_4 gru_10/while/Identity_4:output:0"Ф
_gru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensoragru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2`
.gru_10/while/gru_cell_19/MatMul/ReadVariableOp.gru_10/while/gru_cell_19/MatMul/ReadVariableOp2d
0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp2R
'gru_10/while/gru_cell_19/ReadVariableOp'gru_10/while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
Б
Л
'__inference_gru_9_layer_call_fn_1960730
inputs_0
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1957695}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ћ4

B__inference_gru_9_layer_call_and_return_conditional_losses_1957877

inputs&
gru_cell_18_1957801:	&
gru_cell_18_1957803:	'
gru_cell_18_1957805:
Ќ
identityЂ#gru_cell_18/StatefulPartitionedCallЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskв
#gru_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_18_1957801gru_cell_18_1957803gru_cell_18_1957805*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957761n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_18_1957801gru_cell_18_1957803gru_cell_18_1957805*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1957813*
condR
while_cond_1957812*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌt
NoOpNoOp$^gru_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#gru_cell_18/StatefulPartitionedCall#gru_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
=

while_body_1962598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

л
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1957956

inputs

states*
readvariableop_resource:	Ќ2
matmul_readvariableop_resource:
ЌЌ3
 matmul_1_readvariableop_resource:	dЌ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	Ќ*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
	
Ё
gru_9_while_cond_1959880(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1A
=gru_9_while_gru_9_while_cond_1959880___redundant_placeholder0A
=gru_9_while_gru_9_while_cond_1959880___redundant_placeholder1A
=gru_9_while_gru_9_while_cond_1959880___redundant_placeholder2A
=gru_9_while_gru_9_while_cond_1959880___redundant_placeholder3
gru_9_while_identity
z
gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: W
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_9_while_identitygru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
Р

н
-__inference_gru_cell_18_layer_call_fn_1962715

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957761p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџЌ
"
_user_specified_name
states/0
Ћ4

B__inference_gru_9_layer_call_and_return_conditional_losses_1957695

inputs&
gru_cell_18_1957619:	&
gru_cell_18_1957621:	'
gru_cell_18_1957623:
Ќ
identityЂ#gru_cell_18/StatefulPartitionedCallЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskв
#gru_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_18_1957619gru_cell_18_1957621gru_cell_18_1957623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957618n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_18_1957619gru_cell_18_1957621gru_cell_18_1957623*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1957631*
condR
while_cond_1957630*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌt
NoOpNoOp$^gru_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#gru_cell_18/StatefulPartitionedCall#gru_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
н
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962754

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
Ќ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџІ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:џџџџџџџџџЌJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџЌ
"
_user_specified_name
states/0
Ж

й
-__inference_gru_cell_20_layer_call_fn_1962927

inputs
states_0
unknown:
	unknown_0:d
	unknown_1:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958437o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ЄL
є
 __inference__traced_save_1963130
file_prefix7
3savev2_gru_9_gru_cell_18_kernel_read_readvariableopA
=savev2_gru_9_gru_cell_18_recurrent_kernel_read_readvariableop5
1savev2_gru_9_gru_cell_18_bias_read_readvariableop8
4savev2_gru_10_gru_cell_19_kernel_read_readvariableopB
>savev2_gru_10_gru_cell_19_recurrent_kernel_read_readvariableop6
2savev2_gru_10_gru_cell_19_bias_read_readvariableop8
4savev2_gru_11_gru_cell_20_kernel_read_readvariableopB
>savev2_gru_11_gru_cell_20_recurrent_kernel_read_readvariableop6
2savev2_gru_11_gru_cell_20_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_gru_9_gru_cell_18_kernel_m_read_readvariableopH
Dsavev2_adam_gru_9_gru_cell_18_recurrent_kernel_m_read_readvariableop<
8savev2_adam_gru_9_gru_cell_18_bias_m_read_readvariableop?
;savev2_adam_gru_10_gru_cell_19_kernel_m_read_readvariableopI
Esavev2_adam_gru_10_gru_cell_19_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_10_gru_cell_19_bias_m_read_readvariableop?
;savev2_adam_gru_11_gru_cell_20_kernel_m_read_readvariableopI
Esavev2_adam_gru_11_gru_cell_20_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_11_gru_cell_20_bias_m_read_readvariableop>
:savev2_adam_gru_9_gru_cell_18_kernel_v_read_readvariableopH
Dsavev2_adam_gru_9_gru_cell_18_recurrent_kernel_v_read_readvariableop<
8savev2_adam_gru_9_gru_cell_18_bias_v_read_readvariableop?
;savev2_adam_gru_10_gru_cell_19_kernel_v_read_readvariableopI
Esavev2_adam_gru_10_gru_cell_19_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_10_gru_cell_19_bias_v_read_readvariableop?
;savev2_adam_gru_11_gru_cell_20_kernel_v_read_readvariableopI
Esavev2_adam_gru_11_gru_cell_20_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_11_gru_cell_20_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Д
valueЊBЇ#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_gru_9_gru_cell_18_kernel_read_readvariableop=savev2_gru_9_gru_cell_18_recurrent_kernel_read_readvariableop1savev2_gru_9_gru_cell_18_bias_read_readvariableop4savev2_gru_10_gru_cell_19_kernel_read_readvariableop>savev2_gru_10_gru_cell_19_recurrent_kernel_read_readvariableop2savev2_gru_10_gru_cell_19_bias_read_readvariableop4savev2_gru_11_gru_cell_20_kernel_read_readvariableop>savev2_gru_11_gru_cell_20_recurrent_kernel_read_readvariableop2savev2_gru_11_gru_cell_20_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_gru_9_gru_cell_18_kernel_m_read_readvariableopDsavev2_adam_gru_9_gru_cell_18_recurrent_kernel_m_read_readvariableop8savev2_adam_gru_9_gru_cell_18_bias_m_read_readvariableop;savev2_adam_gru_10_gru_cell_19_kernel_m_read_readvariableopEsavev2_adam_gru_10_gru_cell_19_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_10_gru_cell_19_bias_m_read_readvariableop;savev2_adam_gru_11_gru_cell_20_kernel_m_read_readvariableopEsavev2_adam_gru_11_gru_cell_20_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_11_gru_cell_20_bias_m_read_readvariableop:savev2_adam_gru_9_gru_cell_18_kernel_v_read_readvariableopDsavev2_adam_gru_9_gru_cell_18_recurrent_kernel_v_read_readvariableop8savev2_adam_gru_9_gru_cell_18_bias_v_read_readvariableop;savev2_adam_gru_10_gru_cell_19_kernel_v_read_readvariableopEsavev2_adam_gru_10_gru_cell_19_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_10_gru_cell_19_bias_v_read_readvariableop;savev2_adam_gru_11_gru_cell_20_kernel_v_read_readvariableopEsavev2_adam_gru_11_gru_cell_20_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_11_gru_cell_20_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Э
_input_shapesЛ
И: :	:
Ќ:	:
ЌЌ:	dЌ:	Ќ:d::: : : : : : : :	:
Ќ:	:
ЌЌ:	dЌ:	Ќ:d:::	:
Ќ:	:
ЌЌ:	dЌ:	Ќ:d::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:&"
 
_output_shapes
:
Ќ:%!

_output_shapes
:	:&"
 
_output_shapes
:
ЌЌ:%!

_output_shapes
:	dЌ:%!

_output_shapes
:	Ќ:$ 

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
:	:&"
 
_output_shapes
:
Ќ:%!

_output_shapes
:	:&"
 
_output_shapes
:
ЌЌ:%!

_output_shapes
:	dЌ:%!

_output_shapes
:	Ќ:$ 

_output_shapes

:d:$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	:&"
 
_output_shapes
:
Ќ:%!

_output_shapes
:	:&"
 
_output_shapes
:
ЌЌ:%!

_output_shapes
:	dЌ:%!

_output_shapes
:	Ќ:$  

_output_shapes

:d:$! 

_output_shapes

::$" 

_output_shapes

::#

_output_shapes
: 
=

while_body_1961942
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
M

C__inference_gru_11_layer_call_and_return_conditional_losses_1962534

inputs5
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1962445*
condR
while_cond_1962444*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
оM

C__inference_gru_10_layer_call_and_return_conditional_losses_1961725
inputs_06
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961636*
condR
while_cond_1961635*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
­
И
(__inference_gru_11_layer_call_fn_1962053
inputs_0
unknown:
	unknown_0:d
	unknown_1:
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1958553|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
"
_user_specified_name
inputs/0
с
Џ
while_cond_1958631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958631___redundant_placeholder05
1while_while_cond_1958631___redundant_placeholder15
1while_while_cond_1958631___redundant_placeholder25
1while_while_cond_1958631___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
 
О
while_body_1957813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_18_1957835_0:	.
while_gru_cell_18_1957837_0:	/
while_gru_cell_18_1957839_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_18_1957835:	,
while_gru_cell_18_1957837:	-
while_gru_cell_18_1957839:
ЌЂ)while/gru_cell_18/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
)while/gru_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_18_1957835_0while_gru_cell_18_1957837_0while_gru_cell_18_1957839_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957761л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌx

while/NoOpNoOp*^while/gru_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_18_1957835while_gru_cell_18_1957835_0"8
while_gru_cell_18_1957837while_gru_cell_18_1957837_0"8
while_gru_cell_18_1957839while_gru_cell_18_1957839_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2V
)while/gru_cell_18/StatefulPartitionedCall)while/gru_cell_18/StatefulPartitionedCall: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
п
Џ
while_cond_1958951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958951___redundant_placeholder05
1while_while_cond_1958951___redundant_placeholder15
1while_while_cond_1958951___redundant_placeholder25
1while_while_cond_1958951___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

л
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1958099

inputs

states*
readvariableop_resource:	Ќ2
matmul_readvariableop_resource:
ЌЌ3
 matmul_1_readvariableop_resource:	dЌ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	Ќ*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
п
Џ
while_cond_1961941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961941___redundant_placeholder05
1while_while_cond_1961941___redundant_placeholder15
1while_while_cond_1961941___redundant_placeholder25
1while_while_cond_1961941___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
І4

C__inference_gru_10_layer_call_and_return_conditional_losses_1958033

inputs&
gru_cell_19_1957957:	Ќ'
gru_cell_19_1957959:
ЌЌ&
gru_cell_19_1957961:	dЌ
identityЂ#gru_cell_19/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskа
#gru_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_19_1957957gru_cell_19_1957959gru_cell_19_1957961*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1957956n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_19_1957957gru_cell_19_1957959gru_cell_19_1957961*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1957969*
condR
while_cond_1957968*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdt
NoOpNoOp$^gru_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2J
#gru_cell_19/StatefulPartitionedCall#gru_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
с
Џ
while_cond_1957812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1957812___redundant_placeholder05
1while_while_cond_1957812___redundant_placeholder15
1while_while_cond_1957812___redundant_placeholder25
1while_while_cond_1957812___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
	
Д
gru_11_while_cond_1960178*
&gru_11_while_gru_11_while_loop_counter0
,gru_11_while_gru_11_while_maximum_iterations
gru_11_while_placeholder
gru_11_while_placeholder_1
gru_11_while_placeholder_2,
(gru_11_while_less_gru_11_strided_slice_1C
?gru_11_while_gru_11_while_cond_1960178___redundant_placeholder0C
?gru_11_while_gru_11_while_cond_1960178___redundant_placeholder1C
?gru_11_while_gru_11_while_cond_1960178___redundant_placeholder2C
?gru_11_while_gru_11_while_cond_1960178___redundant_placeholder3
gru_11_while_identity
~
gru_11/while/LessLessgru_11_while_placeholder(gru_11_while_less_gru_11_strided_slice_1*
T0*
_output_shapes
: Y
gru_11/while/IdentityIdentitygru_11/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_11_while_identitygru_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
рњ
ш
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960268

inputs<
)gru_9_gru_cell_18_readvariableop_resource:	C
0gru_9_gru_cell_18_matmul_readvariableop_resource:	F
2gru_9_gru_cell_18_matmul_1_readvariableop_resource:
Ќ=
*gru_10_gru_cell_19_readvariableop_resource:	ЌE
1gru_10_gru_cell_19_matmul_readvariableop_resource:
ЌЌF
3gru_10_gru_cell_19_matmul_1_readvariableop_resource:	dЌ<
*gru_11_gru_cell_20_readvariableop_resource:C
1gru_11_gru_cell_20_matmul_readvariableop_resource:dE
3gru_11_gru_cell_20_matmul_1_readvariableop_resource:
identityЂ(gru_10/gru_cell_19/MatMul/ReadVariableOpЂ*gru_10/gru_cell_19/MatMul_1/ReadVariableOpЂ!gru_10/gru_cell_19/ReadVariableOpЂgru_10/whileЂ(gru_11/gru_cell_20/MatMul/ReadVariableOpЂ*gru_11/gru_cell_20/MatMul_1/ReadVariableOpЂ!gru_11/gru_cell_20/ReadVariableOpЂgru_11/whileЂ'gru_9/gru_cell_18/MatMul/ReadVariableOpЂ)gru_9/gru_cell_18/MatMul_1/ReadVariableOpЂ gru_9/gru_cell_18/ReadVariableOpЂgru_9/whileA
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ќ
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџP
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:e
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ђ
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвe
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
 gru_9/gru_cell_18/ReadVariableOpReadVariableOp)gru_9_gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_9/gru_cell_18/unstackUnpack(gru_9/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'gru_9/gru_cell_18/MatMul/ReadVariableOpReadVariableOp0gru_9_gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0І
gru_9/gru_cell_18/MatMulMatMulgru_9/strided_slice_2:output:0/gru_9/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_9/gru_cell_18/BiasAddBiasAdd"gru_9/gru_cell_18/MatMul:product:0"gru_9/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!gru_9/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_9/gru_cell_18/splitSplit*gru_9/gru_cell_18/split/split_dim:output:0"gru_9/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
)gru_9/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp2gru_9_gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0 
gru_9/gru_cell_18/MatMul_1MatMulgru_9/zeros:output:01gru_9/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
gru_9/gru_cell_18/BiasAdd_1BiasAdd$gru_9/gru_cell_18/MatMul_1:product:0"gru_9/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
gru_9/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#gru_9/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_9/gru_cell_18/split_1SplitV$gru_9/gru_cell_18/BiasAdd_1:output:0 gru_9/gru_cell_18/Const:output:0,gru_9/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_9/gru_cell_18/addAddV2 gru_9/gru_cell_18/split:output:0"gru_9/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
gru_9/gru_cell_18/SigmoidSigmoidgru_9/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_1AddV2 gru_9/gru_cell_18/split:output:1"gru_9/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
gru_9/gru_cell_18/Sigmoid_1Sigmoidgru_9/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mulMulgru_9/gru_cell_18/Sigmoid_1:y:0"gru_9/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_2AddV2 gru_9/gru_cell_18/split:output:2gru_9/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
gru_9/gru_cell_18/Sigmoid_2Sigmoidgru_9/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mul_1Mulgru_9/gru_cell_18/Sigmoid:y:0gru_9/zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ\
gru_9/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_9/gru_cell_18/subSub gru_9/gru_cell_18/sub/x:output:0gru_9/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mul_2Mulgru_9/gru_cell_18/sub:z:0gru_9/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_3AddV2gru_9/gru_cell_18/mul_1:z:0gru_9/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ъ
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвL

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџZ
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_9_gru_cell_18_readvariableop_resource0gru_9_gru_cell_18_matmul_readvariableop_resource2gru_9_gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_9_while_body_1959881*$
condR
gru_9_while_cond_1959880*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  ж
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0n
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџg
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskk
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌa
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Q
gru_10/ShapeShapegru_9/transpose_1:y:0*
T0*
_output_shapes
:d
gru_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
gru_10/strided_sliceStridedSlicegru_10/Shape:output:0#gru_10/strided_slice/stack:output:0%gru_10/strided_slice/stack_1:output:0%gru_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_10/zeros/packedPackgru_10/strided_slice:output:0gru_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_10/zerosFillgru_10/zeros/packed:output:0gru_10/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdj
gru_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_10/transpose	Transposegru_9/transpose_1:y:0gru_10/transpose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌR
gru_10/Shape_1Shapegru_10/transpose:y:0*
T0*
_output_shapes
:f
gru_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
gru_10/strided_slice_1StridedSlicegru_10/Shape_1:output:0%gru_10/strided_slice_1/stack:output:0'gru_10/strided_slice_1/stack_1:output:0'gru_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
gru_10/TensorArrayV2TensorListReserve+gru_10/TensorArrayV2/element_shape:output:0gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<gru_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  ѕ
.gru_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_10/transpose:y:0Egru_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
gru_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_10/strided_slice_2StridedSlicegru_10/transpose:y:0%gru_10/strided_slice_2/stack:output:0'gru_10/strided_slice_2/stack_1:output:0'gru_10/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
!gru_10/gru_cell_19/ReadVariableOpReadVariableOp*gru_10_gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0
gru_10/gru_cell_19/unstackUnpack)gru_10/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
(gru_10/gru_cell_19/MatMul/ReadVariableOpReadVariableOp1gru_10_gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0Љ
gru_10/gru_cell_19/MatMulMatMulgru_10/strided_slice_2:output:00gru_10/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЂ
gru_10/gru_cell_19/BiasAddBiasAdd#gru_10/gru_cell_19/MatMul:product:0#gru_10/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
"gru_10/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_10/gru_cell_19/splitSplit+gru_10/gru_cell_19/split/split_dim:output:0#gru_10/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
*gru_10/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp3gru_10_gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0Ѓ
gru_10/gru_cell_19/MatMul_1MatMulgru_10/zeros:output:02gru_10/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌІ
gru_10/gru_cell_19/BiasAdd_1BiasAdd%gru_10/gru_cell_19/MatMul_1:product:0#gru_10/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌm
gru_10/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџo
$gru_10/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_10/gru_cell_19/split_1SplitV%gru_10/gru_cell_19/BiasAdd_1:output:0!gru_10/gru_cell_19/Const:output:0-gru_10/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_10/gru_cell_19/addAddV2!gru_10/gru_cell_19/split:output:0#gru_10/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_10/gru_cell_19/SigmoidSigmoidgru_10/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_1AddV2!gru_10/gru_cell_19/split:output:1#gru_10/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
gru_10/gru_cell_19/Sigmoid_1Sigmoidgru_10/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mulMul gru_10/gru_cell_19/Sigmoid_1:y:0#gru_10/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_2AddV2!gru_10/gru_cell_19/split:output:2gru_10/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdw
gru_10/gru_cell_19/Sigmoid_2Sigmoidgru_10/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mul_1Mulgru_10/gru_cell_19/Sigmoid:y:0gru_10/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџd]
gru_10/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_10/gru_cell_19/subSub!gru_10/gru_cell_19/sub/x:output:0gru_10/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mul_2Mulgru_10/gru_cell_19/sub:z:0 gru_10/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_3AddV2gru_10/gru_cell_19/mul_1:z:0gru_10/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
$gru_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Э
gru_10/TensorArrayV2_1TensorListReserve-gru_10/TensorArrayV2_1/element_shape:output:0gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
gru_10/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
gru_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_10/whileWhile"gru_10/while/loop_counter:output:0(gru_10/while/maximum_iterations:output:0gru_10/time:output:0gru_10/TensorArrayV2_1:handle:0gru_10/zeros:output:0gru_10/strided_slice_1:output:0>gru_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_10_gru_cell_19_readvariableop_resource1gru_10_gru_cell_19_matmul_readvariableop_resource3gru_10_gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_10_while_body_1960030*%
condR
gru_10_while_cond_1960029*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
7gru_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   и
)gru_10/TensorArrayV2Stack/TensorListStackTensorListStackgru_10/while:output:3@gru_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0o
gru_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
gru_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
gru_10/strided_slice_3StridedSlice2gru_10/TensorArrayV2Stack/TensorListStack:tensor:0%gru_10/strided_slice_3/stack:output:0'gru_10/strided_slice_3/stack_1:output:0'gru_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskl
gru_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
gru_10/transpose_1	Transpose2gru_10/TensorArrayV2Stack/TensorListStack:tensor:0 gru_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњdb
gru_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_11/ShapeShapegru_10/transpose_1:y:0*
T0*
_output_shapes
:d
gru_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
gru_11/strided_sliceStridedSlicegru_11/Shape:output:0#gru_11/strided_slice/stack:output:0%gru_11/strided_slice/stack_1:output:0%gru_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
gru_11/zeros/packedPackgru_11/strided_slice:output:0gru_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_11/zerosFillgru_11/zeros/packed:output:0gru_11/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
gru_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_11/transpose	Transposegru_10/transpose_1:y:0gru_11/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdR
gru_11/Shape_1Shapegru_11/transpose:y:0*
T0*
_output_shapes
:f
gru_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
gru_11/strided_slice_1StridedSlicegru_11/Shape_1:output:0%gru_11/strided_slice_1/stack:output:0'gru_11/strided_slice_1/stack_1:output:0'gru_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
gru_11/TensorArrayV2TensorListReserve+gru_11/TensorArrayV2/element_shape:output:0gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<gru_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ѕ
.gru_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_11/transpose:y:0Egru_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
gru_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_11/strided_slice_2StridedSlicegru_11/transpose:y:0%gru_11/strided_slice_2/stack:output:0'gru_11/strided_slice_2/stack_1:output:0'gru_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask
!gru_11/gru_cell_20/ReadVariableOpReadVariableOp*gru_11_gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0
gru_11/gru_cell_20/unstackUnpack)gru_11/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
(gru_11/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_11_gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ј
gru_11/gru_cell_20/MatMulMatMulgru_11/strided_slice_2:output:00gru_11/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
gru_11/gru_cell_20/BiasAddBiasAdd#gru_11/gru_cell_20/MatMul:product:0#gru_11/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
"gru_11/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_11/gru_cell_20/splitSplit+gru_11/gru_cell_20/split/split_dim:output:0#gru_11/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
*gru_11/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_11_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ђ
gru_11/gru_cell_20/MatMul_1MatMulgru_11/zeros:output:02gru_11/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЅ
gru_11/gru_cell_20/BiasAdd_1BiasAdd%gru_11/gru_cell_20/MatMul_1:product:0#gru_11/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџm
gru_11/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџo
$gru_11/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_11/gru_cell_20/split_1SplitV%gru_11/gru_cell_20/BiasAdd_1:output:0!gru_11/gru_cell_20/Const:output:0-gru_11/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_11/gru_cell_20/addAddV2!gru_11/gru_cell_20/split:output:0#gru_11/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_11/gru_cell_20/SigmoidSigmoidgru_11/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_1AddV2!gru_11/gru_cell_20/split:output:1#gru_11/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџw
gru_11/gru_cell_20/Sigmoid_1Sigmoidgru_11/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mulMul gru_11/gru_cell_20/Sigmoid_1:y:0#gru_11/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_2AddV2!gru_11/gru_cell_20/split:output:2gru_11/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџw
gru_11/gru_cell_20/SoftplusSoftplusgru_11/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mul_1Mulgru_11/gru_cell_20/Sigmoid:y:0gru_11/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]
gru_11/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_11/gru_cell_20/subSub!gru_11/gru_cell_20/sub/x:output:0gru_11/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mul_2Mulgru_11/gru_cell_20/sub:z:0)gru_11/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_3AddV2gru_11/gru_cell_20/mul_1:z:0gru_11/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
$gru_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Э
gru_11/TensorArrayV2_1TensorListReserve-gru_11/TensorArrayV2_1/element_shape:output:0gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
gru_11/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
gru_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_11/whileWhile"gru_11/while/loop_counter:output:0(gru_11/while/maximum_iterations:output:0gru_11/time:output:0gru_11/TensorArrayV2_1:handle:0gru_11/zeros:output:0gru_11/strided_slice_1:output:0>gru_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_11_gru_cell_20_readvariableop_resource1gru_11_gru_cell_20_matmul_readvariableop_resource3gru_11_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_11_while_body_1960179*%
condR
gru_11_while_cond_1960178*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
7gru_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
)gru_11/TensorArrayV2Stack/TensorListStackTensorListStackgru_11/while:output:3@gru_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0o
gru_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
gru_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
gru_11/strided_slice_3StridedSlice2gru_11/TensorArrayV2Stack/TensorListStack:tensor:0%gru_11/strided_slice_3/stack:output:0'gru_11/strided_slice_3/stack_1:output:0'gru_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskl
gru_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
gru_11/transpose_1	Transpose2gru_11/TensorArrayV2Stack/TensorListStack:tensor:0 gru_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњb
gru_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_11/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњу
NoOpNoOp)^gru_10/gru_cell_19/MatMul/ReadVariableOp+^gru_10/gru_cell_19/MatMul_1/ReadVariableOp"^gru_10/gru_cell_19/ReadVariableOp^gru_10/while)^gru_11/gru_cell_20/MatMul/ReadVariableOp+^gru_11/gru_cell_20/MatMul_1/ReadVariableOp"^gru_11/gru_cell_20/ReadVariableOp^gru_11/while(^gru_9/gru_cell_18/MatMul/ReadVariableOp*^gru_9/gru_cell_18/MatMul_1/ReadVariableOp!^gru_9/gru_cell_18/ReadVariableOp^gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2T
(gru_10/gru_cell_19/MatMul/ReadVariableOp(gru_10/gru_cell_19/MatMul/ReadVariableOp2X
*gru_10/gru_cell_19/MatMul_1/ReadVariableOp*gru_10/gru_cell_19/MatMul_1/ReadVariableOp2F
!gru_10/gru_cell_19/ReadVariableOp!gru_10/gru_cell_19/ReadVariableOp2
gru_10/whilegru_10/while2T
(gru_11/gru_cell_20/MatMul/ReadVariableOp(gru_11/gru_cell_20/MatMul/ReadVariableOp2X
*gru_11/gru_cell_20/MatMul_1/ReadVariableOp*gru_11/gru_cell_20/MatMul_1/ReadVariableOp2F
!gru_11/gru_cell_20/ReadVariableOp!gru_11/gru_cell_20/ReadVariableOp2
gru_11/whilegru_11/while2R
'gru_9/gru_cell_18/MatMul/ReadVariableOp'gru_9/gru_cell_18/MatMul/ReadVariableOp2V
)gru_9/gru_cell_18/MatMul_1/ReadVariableOp)gru_9/gru_cell_18/MatMul_1/ReadVariableOp2D
 gru_9/gru_cell_18/ReadVariableOp gru_9/gru_cell_18/ReadVariableOp2
gru_9/whilegru_9/while:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
 
Ж
while_body_1958307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_20_1958329_0:-
while_gru_cell_20_1958331_0:d-
while_gru_cell_20_1958333_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_20_1958329:+
while_gru_cell_20_1958331:d+
while_gru_cell_20_1958333:Ђ)while/gru_cell_20/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
)while/gru_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_20_1958329_0while_gru_cell_20_1958331_0while_gru_cell_20_1958333_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958294л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

while/NoOpNoOp*^while/gru_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_20_1958329while_gru_cell_20_1958329_0"8
while_gru_cell_20_1958331while_gru_cell_20_1958331_0"8
while_gru_cell_20_1958333while_gru_cell_20_1958333_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2V
)while/gru_cell_20/StatefulPartitionedCall)while/gru_cell_20/StatefulPartitionedCall: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
=

while_body_1962292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
КM

B__inference_gru_9_layer_call_and_return_conditional_losses_1961375

inputs6
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961286*
condR
while_cond_1961285*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ф
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ѓ

ђ
.__inference_sequential_3_layer_call_fn_1959794

inputs
unknown:	
	unknown_0:	
	unknown_1:
Ќ
	unknown_2:	Ќ
	unknown_3:
ЌЌ
	unknown_4:	dЌ
	unknown_5:
	unknown_6:d
	unknown_7:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959050t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ѕ
л
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957761

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
Ќ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџІ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:џџџџџџџџџЌJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_namestates
ёЌ
з

"__inference__wrapped_model_1957548
gru_9_inputI
6sequential_3_gru_9_gru_cell_18_readvariableop_resource:	P
=sequential_3_gru_9_gru_cell_18_matmul_readvariableop_resource:	S
?sequential_3_gru_9_gru_cell_18_matmul_1_readvariableop_resource:
ЌJ
7sequential_3_gru_10_gru_cell_19_readvariableop_resource:	ЌR
>sequential_3_gru_10_gru_cell_19_matmul_readvariableop_resource:
ЌЌS
@sequential_3_gru_10_gru_cell_19_matmul_1_readvariableop_resource:	dЌI
7sequential_3_gru_11_gru_cell_20_readvariableop_resource:P
>sequential_3_gru_11_gru_cell_20_matmul_readvariableop_resource:dR
@sequential_3_gru_11_gru_cell_20_matmul_1_readvariableop_resource:
identityЂ5sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOpЂ7sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOpЂ.sequential_3/gru_10/gru_cell_19/ReadVariableOpЂsequential_3/gru_10/whileЂ5sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOpЂ7sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOpЂ.sequential_3/gru_11/gru_cell_20/ReadVariableOpЂsequential_3/gru_11/whileЂ4sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOpЂ6sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOpЂ-sequential_3/gru_9/gru_cell_18/ReadVariableOpЂsequential_3/gru_9/whileS
sequential_3/gru_9/ShapeShapegru_9_input*
T0*
_output_shapes
:p
&sequential_3/gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_3/gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_3/gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 sequential_3/gru_9/strided_sliceStridedSlice!sequential_3/gru_9/Shape:output:0/sequential_3/gru_9/strided_slice/stack:output:01sequential_3/gru_9/strided_slice/stack_1:output:01sequential_3/gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
!sequential_3/gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ЌЌ
sequential_3/gru_9/zeros/packedPack)sequential_3/gru_9/strided_slice:output:0*sequential_3/gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_3/gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
sequential_3/gru_9/zerosFill(sequential_3/gru_9/zeros/packed:output:0'sequential_3/gru_9/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
!sequential_3/gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_3/gru_9/transpose	Transposegru_9_input*sequential_3/gru_9/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџj
sequential_3/gru_9/Shape_1Shape sequential_3/gru_9/transpose:y:0*
T0*
_output_shapes
:r
(sequential_3/gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_3/gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_3/gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"sequential_3/gru_9/strided_slice_1StridedSlice#sequential_3/gru_9/Shape_1:output:01sequential_3/gru_9/strided_slice_1/stack:output:03sequential_3/gru_9/strided_slice_1/stack_1:output:03sequential_3/gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_3/gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџэ
 sequential_3/gru_9/TensorArrayV2TensorListReserve7sequential_3/gru_9/TensorArrayV2/element_shape:output:0+sequential_3/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Hsequential_3/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
:sequential_3/gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_3/gru_9/transpose:y:0Qsequential_3/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвr
(sequential_3/gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_3/gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_3/gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
"sequential_3/gru_9/strided_slice_2StridedSlice sequential_3/gru_9/transpose:y:01sequential_3/gru_9/strided_slice_2/stack:output:03sequential_3/gru_9/strided_slice_2/stack_1:output:03sequential_3/gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЅ
-sequential_3/gru_9/gru_cell_18/ReadVariableOpReadVariableOp6sequential_3_gru_9_gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0
&sequential_3/gru_9/gru_cell_18/unstackUnpack5sequential_3/gru_9/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numГ
4sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOpReadVariableOp=sequential_3_gru_9_gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Э
%sequential_3/gru_9/gru_cell_18/MatMulMatMul+sequential_3/gru_9/strided_slice_2:output:0<sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЦ
&sequential_3/gru_9/gru_cell_18/BiasAddBiasAdd/sequential_3/gru_9/gru_cell_18/MatMul:product:0/sequential_3/gru_9/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
.sequential_3/gru_9/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
$sequential_3/gru_9/gru_cell_18/splitSplit7sequential_3/gru_9/gru_cell_18/split/split_dim:output:0/sequential_3/gru_9/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitИ
6sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp?sequential_3_gru_9_gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0Ч
'sequential_3/gru_9/gru_cell_18/MatMul_1MatMul!sequential_3/gru_9/zeros:output:0>sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
(sequential_3/gru_9/gru_cell_18/BiasAdd_1BiasAdd1sequential_3/gru_9/gru_cell_18/MatMul_1:product:0/sequential_3/gru_9/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџy
$sequential_3/gru_9/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ{
0sequential_3/gru_9/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџХ
&sequential_3/gru_9/gru_cell_18/split_1SplitV1sequential_3/gru_9/gru_cell_18/BiasAdd_1:output:0-sequential_3/gru_9/gru_cell_18/Const:output:09sequential_3/gru_9/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitО
"sequential_3/gru_9/gru_cell_18/addAddV2-sequential_3/gru_9/gru_cell_18/split:output:0/sequential_3/gru_9/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
&sequential_3/gru_9/gru_cell_18/SigmoidSigmoid&sequential_3/gru_9/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌР
$sequential_3/gru_9/gru_cell_18/add_1AddV2-sequential_3/gru_9/gru_cell_18/split:output:1/sequential_3/gru_9/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌ
(sequential_3/gru_9/gru_cell_18/Sigmoid_1Sigmoid(sequential_3/gru_9/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌЛ
"sequential_3/gru_9/gru_cell_18/mulMul,sequential_3/gru_9/gru_cell_18/Sigmoid_1:y:0/sequential_3/gru_9/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌЗ
$sequential_3/gru_9/gru_cell_18/add_2AddV2-sequential_3/gru_9/gru_cell_18/split:output:2&sequential_3/gru_9/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
(sequential_3/gru_9/gru_cell_18/Sigmoid_2Sigmoid(sequential_3/gru_9/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ­
$sequential_3/gru_9/gru_cell_18/mul_1Mul*sequential_3/gru_9/gru_cell_18/Sigmoid:y:0!sequential_3/gru_9/zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
$sequential_3/gru_9/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
"sequential_3/gru_9/gru_cell_18/subSub-sequential_3/gru_9/gru_cell_18/sub/x:output:0*sequential_3/gru_9/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌД
$sequential_3/gru_9/gru_cell_18/mul_2Mul&sequential_3/gru_9/gru_cell_18/sub:z:0,sequential_3/gru_9/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌД
$sequential_3/gru_9/gru_cell_18/add_3AddV2(sequential_3/gru_9/gru_cell_18/mul_1:z:0(sequential_3/gru_9/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
0sequential_3/gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  ё
"sequential_3/gru_9/TensorArrayV2_1TensorListReserve9sequential_3/gru_9/TensorArrayV2_1/element_shape:output:0+sequential_3/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвY
sequential_3/gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_3/gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџg
%sequential_3/gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
sequential_3/gru_9/whileWhile.sequential_3/gru_9/while/loop_counter:output:04sequential_3/gru_9/while/maximum_iterations:output:0 sequential_3/gru_9/time:output:0+sequential_3/gru_9/TensorArrayV2_1:handle:0!sequential_3/gru_9/zeros:output:0+sequential_3/gru_9/strided_slice_1:output:0Jsequential_3/gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_3_gru_9_gru_cell_18_readvariableop_resource=sequential_3_gru_9_gru_cell_18_matmul_readvariableop_resource?sequential_3_gru_9_gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%sequential_3_gru_9_while_body_1957161*1
cond)R'
%sequential_3_gru_9_while_cond_1957160*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
Csequential_3/gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  §
5sequential_3/gru_9/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_3/gru_9/while:output:3Lsequential_3/gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0{
(sequential_3/gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџt
*sequential_3/gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_3/gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
"sequential_3/gru_9/strided_slice_3StridedSlice>sequential_3/gru_9/TensorArrayV2Stack/TensorListStack:tensor:01sequential_3/gru_9/strided_slice_3/stack:output:03sequential_3/gru_9/strided_slice_3/stack_1:output:03sequential_3/gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskx
#sequential_3/gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          б
sequential_3/gru_9/transpose_1	Transpose>sequential_3/gru_9/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_3/gru_9/transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌn
sequential_3/gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
sequential_3/gru_10/ShapeShape"sequential_3/gru_9/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_3/gru_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_3/gru_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_3/gru_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!sequential_3/gru_10/strided_sliceStridedSlice"sequential_3/gru_10/Shape:output:00sequential_3/gru_10/strided_slice/stack:output:02sequential_3/gru_10/strided_slice/stack_1:output:02sequential_3/gru_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_3/gru_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЏ
 sequential_3/gru_10/zeros/packedPack*sequential_3/gru_10/strided_slice:output:0+sequential_3/gru_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_3/gru_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ј
sequential_3/gru_10/zerosFill)sequential_3/gru_10/zeros/packed:output:0(sequential_3/gru_10/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdw
"sequential_3/gru_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
sequential_3/gru_10/transpose	Transpose"sequential_3/gru_9/transpose_1:y:0+sequential_3/gru_10/transpose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌl
sequential_3/gru_10/Shape_1Shape!sequential_3/gru_10/transpose:y:0*
T0*
_output_shapes
:s
)sequential_3/gru_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_3/gru_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#sequential_3/gru_10/strided_slice_1StridedSlice$sequential_3/gru_10/Shape_1:output:02sequential_3/gru_10/strided_slice_1/stack:output:04sequential_3/gru_10/strided_slice_1/stack_1:output:04sequential_3/gru_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_3/gru_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ№
!sequential_3/gru_10/TensorArrayV2TensorListReserve8sequential_3/gru_10/TensorArrayV2/element_shape:output:0,sequential_3/gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Isequential_3/gru_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  
;sequential_3/gru_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_3/gru_10/transpose:y:0Rsequential_3/gru_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвs
)sequential_3/gru_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_3/gru_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
#sequential_3/gru_10/strided_slice_2StridedSlice!sequential_3/gru_10/transpose:y:02sequential_3/gru_10/strided_slice_2/stack:output:04sequential_3/gru_10/strided_slice_2/stack_1:output:04sequential_3/gru_10/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskЇ
.sequential_3/gru_10/gru_cell_19/ReadVariableOpReadVariableOp7sequential_3_gru_10_gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0Ё
'sequential_3/gru_10/gru_cell_19/unstackUnpack6sequential_3/gru_10/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numЖ
5sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOpReadVariableOp>sequential_3_gru_10_gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0а
&sequential_3/gru_10/gru_cell_19/MatMulMatMul,sequential_3/gru_10/strided_slice_2:output:0=sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЩ
'sequential_3/gru_10/gru_cell_19/BiasAddBiasAdd0sequential_3/gru_10/gru_cell_19/MatMul:product:00sequential_3/gru_10/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌz
/sequential_3/gru_10/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
%sequential_3/gru_10/gru_cell_19/splitSplit8sequential_3/gru_10/gru_cell_19/split/split_dim:output:00sequential_3/gru_10/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitЙ
7sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp@sequential_3_gru_10_gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0Ъ
(sequential_3/gru_10/gru_cell_19/MatMul_1MatMul"sequential_3/gru_10/zeros:output:0?sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЭ
)sequential_3/gru_10/gru_cell_19/BiasAdd_1BiasAdd2sequential_3/gru_10/gru_cell_19/MatMul_1:product:00sequential_3/gru_10/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌz
%sequential_3/gru_10/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ|
1sequential_3/gru_10/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
'sequential_3/gru_10/gru_cell_19/split_1SplitV2sequential_3/gru_10/gru_cell_19/BiasAdd_1:output:0.sequential_3/gru_10/gru_cell_19/Const:output:0:sequential_3/gru_10/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitР
#sequential_3/gru_10/gru_cell_19/addAddV2.sequential_3/gru_10/gru_cell_19/split:output:00sequential_3/gru_10/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
'sequential_3/gru_10/gru_cell_19/SigmoidSigmoid'sequential_3/gru_10/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџdТ
%sequential_3/gru_10/gru_cell_19/add_1AddV2.sequential_3/gru_10/gru_cell_19/split:output:10sequential_3/gru_10/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
)sequential_3/gru_10/gru_cell_19/Sigmoid_1Sigmoid)sequential_3/gru_10/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdН
#sequential_3/gru_10/gru_cell_19/mulMul-sequential_3/gru_10/gru_cell_19/Sigmoid_1:y:00sequential_3/gru_10/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЙ
%sequential_3/gru_10/gru_cell_19/add_2AddV2.sequential_3/gru_10/gru_cell_19/split:output:2'sequential_3/gru_10/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
)sequential_3/gru_10/gru_cell_19/Sigmoid_2Sigmoid)sequential_3/gru_10/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЏ
%sequential_3/gru_10/gru_cell_19/mul_1Mul+sequential_3/gru_10/gru_cell_19/Sigmoid:y:0"sequential_3/gru_10/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdj
%sequential_3/gru_10/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
#sequential_3/gru_10/gru_cell_19/subSub.sequential_3/gru_10/gru_cell_19/sub/x:output:0+sequential_3/gru_10/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЖ
%sequential_3/gru_10/gru_cell_19/mul_2Mul'sequential_3/gru_10/gru_cell_19/sub:z:0-sequential_3/gru_10/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЖ
%sequential_3/gru_10/gru_cell_19/add_3AddV2)sequential_3/gru_10/gru_cell_19/mul_1:z:0)sequential_3/gru_10/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
1sequential_3/gru_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   є
#sequential_3/gru_10/TensorArrayV2_1TensorListReserve:sequential_3/gru_10/TensorArrayV2_1/element_shape:output:0,sequential_3/gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвZ
sequential_3/gru_10/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_3/gru_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџh
&sequential_3/gru_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ф
sequential_3/gru_10/whileWhile/sequential_3/gru_10/while/loop_counter:output:05sequential_3/gru_10/while/maximum_iterations:output:0!sequential_3/gru_10/time:output:0,sequential_3/gru_10/TensorArrayV2_1:handle:0"sequential_3/gru_10/zeros:output:0,sequential_3/gru_10/strided_slice_1:output:0Ksequential_3/gru_10/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_3_gru_10_gru_cell_19_readvariableop_resource>sequential_3_gru_10_gru_cell_19_matmul_readvariableop_resource@sequential_3_gru_10_gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&sequential_3_gru_10_while_body_1957310*2
cond*R(
&sequential_3_gru_10_while_cond_1957309*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
Dsequential_3/gru_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   џ
6sequential_3/gru_10/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_3/gru_10/while:output:3Msequential_3/gru_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0|
)sequential_3/gru_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџu
+sequential_3/gru_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
#sequential_3/gru_10/strided_slice_3StridedSlice?sequential_3/gru_10/TensorArrayV2Stack/TensorListStack:tensor:02sequential_3/gru_10/strided_slice_3/stack:output:04sequential_3/gru_10/strided_slice_3/stack_1:output:04sequential_3/gru_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_masky
$sequential_3/gru_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
sequential_3/gru_10/transpose_1	Transpose?sequential_3/gru_10/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_3/gru_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњdo
sequential_3/gru_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_3/gru_11/ShapeShape#sequential_3/gru_10/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_3/gru_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_3/gru_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_3/gru_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!sequential_3/gru_11/strided_sliceStridedSlice"sequential_3/gru_11/Shape:output:00sequential_3/gru_11/strided_slice/stack:output:02sequential_3/gru_11/strided_slice/stack_1:output:02sequential_3/gru_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_3/gru_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Џ
 sequential_3/gru_11/zeros/packedPack*sequential_3/gru_11/strided_slice:output:0+sequential_3/gru_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_3/gru_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ј
sequential_3/gru_11/zerosFill)sequential_3/gru_11/zeros/packed:output:0(sequential_3/gru_11/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџw
"sequential_3/gru_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
sequential_3/gru_11/transpose	Transpose#sequential_3/gru_10/transpose_1:y:0+sequential_3/gru_11/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdl
sequential_3/gru_11/Shape_1Shape!sequential_3/gru_11/transpose:y:0*
T0*
_output_shapes
:s
)sequential_3/gru_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_3/gru_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#sequential_3/gru_11/strided_slice_1StridedSlice$sequential_3/gru_11/Shape_1:output:02sequential_3/gru_11/strided_slice_1/stack:output:04sequential_3/gru_11/strided_slice_1/stack_1:output:04sequential_3/gru_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_3/gru_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ№
!sequential_3/gru_11/TensorArrayV2TensorListReserve8sequential_3/gru_11/TensorArrayV2/element_shape:output:0,sequential_3/gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Isequential_3/gru_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   
;sequential_3/gru_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_3/gru_11/transpose:y:0Rsequential_3/gru_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвs
)sequential_3/gru_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_3/gru_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Э
#sequential_3/gru_11/strided_slice_2StridedSlice!sequential_3/gru_11/transpose:y:02sequential_3/gru_11/strided_slice_2/stack:output:04sequential_3/gru_11/strided_slice_2/stack_1:output:04sequential_3/gru_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskІ
.sequential_3/gru_11/gru_cell_20/ReadVariableOpReadVariableOp7sequential_3_gru_11_gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0
'sequential_3/gru_11/gru_cell_20/unstackUnpack6sequential_3/gru_11/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numД
5sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOpReadVariableOp>sequential_3_gru_11_gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Я
&sequential_3/gru_11/gru_cell_20/MatMulMatMul,sequential_3/gru_11/strided_slice_2:output:0=sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџШ
'sequential_3/gru_11/gru_cell_20/BiasAddBiasAdd0sequential_3/gru_11/gru_cell_20/MatMul:product:00sequential_3/gru_11/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
/sequential_3/gru_11/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
%sequential_3/gru_11/gru_cell_20/splitSplit8sequential_3/gru_11/gru_cell_20/split/split_dim:output:00sequential_3/gru_11/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitИ
7sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp@sequential_3_gru_11_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
(sequential_3/gru_11/gru_cell_20/MatMul_1MatMul"sequential_3/gru_11/zeros:output:0?sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЬ
)sequential_3/gru_11/gru_cell_20/BiasAdd_1BiasAdd2sequential_3/gru_11/gru_cell_20/MatMul_1:product:00sequential_3/gru_11/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџz
%sequential_3/gru_11/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ|
1sequential_3/gru_11/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
'sequential_3/gru_11/gru_cell_20/split_1SplitV2sequential_3/gru_11/gru_cell_20/BiasAdd_1:output:0.sequential_3/gru_11/gru_cell_20/Const:output:0:sequential_3/gru_11/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitР
#sequential_3/gru_11/gru_cell_20/addAddV2.sequential_3/gru_11/gru_cell_20/split:output:00sequential_3/gru_11/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
'sequential_3/gru_11/gru_cell_20/SigmoidSigmoid'sequential_3/gru_11/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџТ
%sequential_3/gru_11/gru_cell_20/add_1AddV2.sequential_3/gru_11/gru_cell_20/split:output:10sequential_3/gru_11/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
)sequential_3/gru_11/gru_cell_20/Sigmoid_1Sigmoid)sequential_3/gru_11/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџН
#sequential_3/gru_11/gru_cell_20/mulMul-sequential_3/gru_11/gru_cell_20/Sigmoid_1:y:00sequential_3/gru_11/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџЙ
%sequential_3/gru_11/gru_cell_20/add_2AddV2.sequential_3/gru_11/gru_cell_20/split:output:2'sequential_3/gru_11/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
(sequential_3/gru_11/gru_cell_20/SoftplusSoftplus)sequential_3/gru_11/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЏ
%sequential_3/gru_11/gru_cell_20/mul_1Mul+sequential_3/gru_11/gru_cell_20/Sigmoid:y:0"sequential_3/gru_11/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
%sequential_3/gru_11/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
#sequential_3/gru_11/gru_cell_20/subSub.sequential_3/gru_11/gru_cell_20/sub/x:output:0+sequential_3/gru_11/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџП
%sequential_3/gru_11/gru_cell_20/mul_2Mul'sequential_3/gru_11/gru_cell_20/sub:z:06sequential_3/gru_11/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
%sequential_3/gru_11/gru_cell_20/add_3AddV2)sequential_3/gru_11/gru_cell_20/mul_1:z:0)sequential_3/gru_11/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
1sequential_3/gru_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
#sequential_3/gru_11/TensorArrayV2_1TensorListReserve:sequential_3/gru_11/TensorArrayV2_1/element_shape:output:0,sequential_3/gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвZ
sequential_3/gru_11/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_3/gru_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџh
&sequential_3/gru_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ф
sequential_3/gru_11/whileWhile/sequential_3/gru_11/while/loop_counter:output:05sequential_3/gru_11/while/maximum_iterations:output:0!sequential_3/gru_11/time:output:0,sequential_3/gru_11/TensorArrayV2_1:handle:0"sequential_3/gru_11/zeros:output:0,sequential_3/gru_11/strided_slice_1:output:0Ksequential_3/gru_11/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_3_gru_11_gru_cell_20_readvariableop_resource>sequential_3_gru_11_gru_cell_20_matmul_readvariableop_resource@sequential_3_gru_11_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&sequential_3_gru_11_while_body_1957459*2
cond*R(
&sequential_3_gru_11_while_cond_1957458*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
Dsequential_3/gru_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   џ
6sequential_3/gru_11/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_3/gru_11/while:output:3Msequential_3/gru_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0|
)sequential_3/gru_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџu
+sequential_3/gru_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_3/gru_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
#sequential_3/gru_11/strided_slice_3StridedSlice?sequential_3/gru_11/TensorArrayV2Stack/TensorListStack:tensor:02sequential_3/gru_11/strided_slice_3/stack:output:04sequential_3/gru_11/strided_slice_3/stack_1:output:04sequential_3/gru_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_masky
$sequential_3/gru_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
sequential_3/gru_11/transpose_1	Transpose?sequential_3/gru_11/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_3/gru_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњo
sequential_3/gru_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    w
IdentityIdentity#sequential_3/gru_11/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњџ
NoOpNoOp6^sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOp8^sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOp/^sequential_3/gru_10/gru_cell_19/ReadVariableOp^sequential_3/gru_10/while6^sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOp8^sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOp/^sequential_3/gru_11/gru_cell_20/ReadVariableOp^sequential_3/gru_11/while5^sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOp7^sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOp.^sequential_3/gru_9/gru_cell_18/ReadVariableOp^sequential_3/gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2n
5sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOp5sequential_3/gru_10/gru_cell_19/MatMul/ReadVariableOp2r
7sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOp7sequential_3/gru_10/gru_cell_19/MatMul_1/ReadVariableOp2`
.sequential_3/gru_10/gru_cell_19/ReadVariableOp.sequential_3/gru_10/gru_cell_19/ReadVariableOp26
sequential_3/gru_10/whilesequential_3/gru_10/while2n
5sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOp5sequential_3/gru_11/gru_cell_20/MatMul/ReadVariableOp2r
7sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOp7sequential_3/gru_11/gru_cell_20/MatMul_1/ReadVariableOp2`
.sequential_3/gru_11/gru_cell_20/ReadVariableOp.sequential_3/gru_11/gru_cell_20/ReadVariableOp26
sequential_3/gru_11/whilesequential_3/gru_11/while2l
4sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOp4sequential_3/gru_9/gru_cell_18/MatMul/ReadVariableOp2p
6sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOp6sequential_3/gru_9/gru_cell_18/MatMul_1/ReadVariableOp2^
-sequential_3/gru_9/gru_cell_18/ReadVariableOp-sequential_3/gru_9/gru_cell_18/ReadVariableOp24
sequential_3/gru_9/whilesequential_3/gru_9/while:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input
ІM

C__inference_gru_10_layer_call_and_return_conditional_losses_1958881

inputs6
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958792*
condR
while_cond_1958791*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs

н
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962899

inputs
states_0*
readvariableop_resource:	Ќ2
matmul_readvariableop_resource:
ЌЌ3
 matmul_1_readvariableop_resource:	dЌ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	Ќ*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0
щD
П	
gru_9_while_body_1960332(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0D
1gru_9_while_gru_cell_18_readvariableop_resource_0:	K
8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0:	N
:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensorB
/gru_9_while_gru_cell_18_readvariableop_resource:	I
6gru_9_while_gru_cell_18_matmul_readvariableop_resource:	L
8gru_9_while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ-gru_9/while/gru_cell_18/MatMul/ReadVariableOpЂ/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpЂ&gru_9/while/gru_cell_18/ReadVariableOp
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ф
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
&gru_9/while/gru_cell_18/ReadVariableOpReadVariableOp1gru_9_while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_9/while/gru_cell_18/unstackUnpack.gru_9/while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numЇ
-gru_9/while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ъ
gru_9/while/gru_cell_18/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:05gru_9/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџБ
gru_9/while/gru_cell_18/BiasAddBiasAdd(gru_9/while/gru_cell_18/MatMul:product:0(gru_9/while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџr
'gru_9/while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_9/while/gru_cell_18/splitSplit0gru_9/while/gru_cell_18/split/split_dim:output:0(gru_9/while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitЌ
/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0Б
 gru_9/while/gru_cell_18/MatMul_1MatMulgru_9_while_placeholder_27gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
!gru_9/while/gru_cell_18/BiasAdd_1BiasAdd*gru_9/while/gru_cell_18/MatMul_1:product:0(gru_9/while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџr
gru_9/while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџt
)gru_9/while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЉ
gru_9/while/gru_cell_18/split_1SplitV*gru_9/while/gru_cell_18/BiasAdd_1:output:0&gru_9/while/gru_cell_18/Const:output:02gru_9/while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitЉ
gru_9/while/gru_cell_18/addAddV2&gru_9/while/gru_cell_18/split:output:0(gru_9/while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_9/while/gru_cell_18/SigmoidSigmoidgru_9/while/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌЋ
gru_9/while/gru_cell_18/add_1AddV2&gru_9/while/gru_cell_18/split:output:1(gru_9/while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌ
!gru_9/while/gru_cell_18/Sigmoid_1Sigmoid!gru_9/while/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌІ
gru_9/while/gru_cell_18/mulMul%gru_9/while/gru_cell_18/Sigmoid_1:y:0(gru_9/while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌЂ
gru_9/while/gru_cell_18/add_2AddV2&gru_9/while/gru_cell_18/split:output:2gru_9/while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
!gru_9/while/gru_cell_18/Sigmoid_2Sigmoid!gru_9/while/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/mul_1Mul#gru_9/while/gru_cell_18/Sigmoid:y:0gru_9_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌb
gru_9/while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
gru_9/while/gru_cell_18/subSub&gru_9/while/gru_cell_18/sub/x:output:0#gru_9/while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/mul_2Mulgru_9/while/gru_cell_18/sub:z:0%gru_9/while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/add_3AddV2!gru_9/while/gru_cell_18/mul_1:z:0!gru_9/while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌм
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder!gru_9/while/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвS
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: U
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations^gru_9/while/NoOp*
T0*
_output_shapes
: k
gru_9/while/Identity_2Identitygru_9/while/add:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_4Identity!gru_9/while/gru_cell_18/add_3:z:0^gru_9/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌн
gru_9/while/NoOpNoOp.^gru_9/while/gru_cell_18/MatMul/ReadVariableOp0^gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp'^gru_9/while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"v
8gru_9_while_gru_cell_18_matmul_1_readvariableop_resource:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0"r
6gru_9_while_gru_cell_18_matmul_readvariableop_resource8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0"d
/gru_9_while_gru_cell_18_readvariableop_resource1gru_9_while_gru_cell_18_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"Р
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2^
-gru_9/while/gru_cell_18/MatMul/ReadVariableOp-gru_9/while/gru_cell_18/MatMul/ReadVariableOp2b
/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp2P
&gru_9/while/gru_cell_18/ReadVariableOp&gru_9/while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
рњ
ш
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960719

inputs<
)gru_9_gru_cell_18_readvariableop_resource:	C
0gru_9_gru_cell_18_matmul_readvariableop_resource:	F
2gru_9_gru_cell_18_matmul_1_readvariableop_resource:
Ќ=
*gru_10_gru_cell_19_readvariableop_resource:	ЌE
1gru_10_gru_cell_19_matmul_readvariableop_resource:
ЌЌF
3gru_10_gru_cell_19_matmul_1_readvariableop_resource:	dЌ<
*gru_11_gru_cell_20_readvariableop_resource:C
1gru_11_gru_cell_20_matmul_readvariableop_resource:dE
3gru_11_gru_cell_20_matmul_1_readvariableop_resource:
identityЂ(gru_10/gru_cell_19/MatMul/ReadVariableOpЂ*gru_10/gru_cell_19/MatMul_1/ReadVariableOpЂ!gru_10/gru_cell_19/ReadVariableOpЂgru_10/whileЂ(gru_11/gru_cell_20/MatMul/ReadVariableOpЂ*gru_11/gru_cell_20/MatMul_1/ReadVariableOpЂ!gru_11/gru_cell_20/ReadVariableOpЂgru_11/whileЂ'gru_9/gru_cell_18/MatMul/ReadVariableOpЂ)gru_9/gru_cell_18/MatMul_1/ReadVariableOpЂ gru_9/gru_cell_18/ReadVariableOpЂgru_9/whileA
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ќ
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџP
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:e
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ђ
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвe
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
 gru_9/gru_cell_18/ReadVariableOpReadVariableOp)gru_9_gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_9/gru_cell_18/unstackUnpack(gru_9/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'gru_9/gru_cell_18/MatMul/ReadVariableOpReadVariableOp0gru_9_gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0І
gru_9/gru_cell_18/MatMulMatMulgru_9/strided_slice_2:output:0/gru_9/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_9/gru_cell_18/BiasAddBiasAdd"gru_9/gru_cell_18/MatMul:product:0"gru_9/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!gru_9/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_9/gru_cell_18/splitSplit*gru_9/gru_cell_18/split/split_dim:output:0"gru_9/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
)gru_9/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp2gru_9_gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0 
gru_9/gru_cell_18/MatMul_1MatMulgru_9/zeros:output:01gru_9/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
gru_9/gru_cell_18/BiasAdd_1BiasAdd$gru_9/gru_cell_18/MatMul_1:product:0"gru_9/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
gru_9/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#gru_9/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_9/gru_cell_18/split_1SplitV$gru_9/gru_cell_18/BiasAdd_1:output:0 gru_9/gru_cell_18/Const:output:0,gru_9/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_9/gru_cell_18/addAddV2 gru_9/gru_cell_18/split:output:0"gru_9/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
gru_9/gru_cell_18/SigmoidSigmoidgru_9/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_1AddV2 gru_9/gru_cell_18/split:output:1"gru_9/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
gru_9/gru_cell_18/Sigmoid_1Sigmoidgru_9/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mulMulgru_9/gru_cell_18/Sigmoid_1:y:0"gru_9/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_2AddV2 gru_9/gru_cell_18/split:output:2gru_9/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
gru_9/gru_cell_18/Sigmoid_2Sigmoidgru_9/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mul_1Mulgru_9/gru_cell_18/Sigmoid:y:0gru_9/zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ\
gru_9/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_9/gru_cell_18/subSub gru_9/gru_cell_18/sub/x:output:0gru_9/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/mul_2Mulgru_9/gru_cell_18/sub:z:0gru_9/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/gru_cell_18/add_3AddV2gru_9/gru_cell_18/mul_1:z:0gru_9/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ъ
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвL

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџZ
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_9_gru_cell_18_readvariableop_resource0gru_9_gru_cell_18_matmul_readvariableop_resource2gru_9_gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_9_while_body_1960332*$
condR
gru_9_while_cond_1960331*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  ж
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0n
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџg
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskk
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌa
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Q
gru_10/ShapeShapegru_9/transpose_1:y:0*
T0*
_output_shapes
:d
gru_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
gru_10/strided_sliceStridedSlicegru_10/Shape:output:0#gru_10/strided_slice/stack:output:0%gru_10/strided_slice/stack_1:output:0%gru_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_10/zeros/packedPackgru_10/strided_slice:output:0gru_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_10/zerosFillgru_10/zeros/packed:output:0gru_10/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdj
gru_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_10/transpose	Transposegru_9/transpose_1:y:0gru_10/transpose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌR
gru_10/Shape_1Shapegru_10/transpose:y:0*
T0*
_output_shapes
:f
gru_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
gru_10/strided_slice_1StridedSlicegru_10/Shape_1:output:0%gru_10/strided_slice_1/stack:output:0'gru_10/strided_slice_1/stack_1:output:0'gru_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
gru_10/TensorArrayV2TensorListReserve+gru_10/TensorArrayV2/element_shape:output:0gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<gru_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  ѕ
.gru_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_10/transpose:y:0Egru_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
gru_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_10/strided_slice_2StridedSlicegru_10/transpose:y:0%gru_10/strided_slice_2/stack:output:0'gru_10/strided_slice_2/stack_1:output:0'gru_10/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
!gru_10/gru_cell_19/ReadVariableOpReadVariableOp*gru_10_gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0
gru_10/gru_cell_19/unstackUnpack)gru_10/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
(gru_10/gru_cell_19/MatMul/ReadVariableOpReadVariableOp1gru_10_gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0Љ
gru_10/gru_cell_19/MatMulMatMulgru_10/strided_slice_2:output:00gru_10/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЂ
gru_10/gru_cell_19/BiasAddBiasAdd#gru_10/gru_cell_19/MatMul:product:0#gru_10/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
"gru_10/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_10/gru_cell_19/splitSplit+gru_10/gru_cell_19/split/split_dim:output:0#gru_10/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
*gru_10/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp3gru_10_gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0Ѓ
gru_10/gru_cell_19/MatMul_1MatMulgru_10/zeros:output:02gru_10/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌІ
gru_10/gru_cell_19/BiasAdd_1BiasAdd%gru_10/gru_cell_19/MatMul_1:product:0#gru_10/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌm
gru_10/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџo
$gru_10/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_10/gru_cell_19/split_1SplitV%gru_10/gru_cell_19/BiasAdd_1:output:0!gru_10/gru_cell_19/Const:output:0-gru_10/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_10/gru_cell_19/addAddV2!gru_10/gru_cell_19/split:output:0#gru_10/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_10/gru_cell_19/SigmoidSigmoidgru_10/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_1AddV2!gru_10/gru_cell_19/split:output:1#gru_10/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
gru_10/gru_cell_19/Sigmoid_1Sigmoidgru_10/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mulMul gru_10/gru_cell_19/Sigmoid_1:y:0#gru_10/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_2AddV2!gru_10/gru_cell_19/split:output:2gru_10/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdw
gru_10/gru_cell_19/Sigmoid_2Sigmoidgru_10/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mul_1Mulgru_10/gru_cell_19/Sigmoid:y:0gru_10/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџd]
gru_10/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_10/gru_cell_19/subSub!gru_10/gru_cell_19/sub/x:output:0gru_10/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/mul_2Mulgru_10/gru_cell_19/sub:z:0 gru_10/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/gru_cell_19/add_3AddV2gru_10/gru_cell_19/mul_1:z:0gru_10/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
$gru_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Э
gru_10/TensorArrayV2_1TensorListReserve-gru_10/TensorArrayV2_1/element_shape:output:0gru_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
gru_10/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
gru_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_10/whileWhile"gru_10/while/loop_counter:output:0(gru_10/while/maximum_iterations:output:0gru_10/time:output:0gru_10/TensorArrayV2_1:handle:0gru_10/zeros:output:0gru_10/strided_slice_1:output:0>gru_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_10_gru_cell_19_readvariableop_resource1gru_10_gru_cell_19_matmul_readvariableop_resource3gru_10_gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_10_while_body_1960481*%
condR
gru_10_while_cond_1960480*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
7gru_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   и
)gru_10/TensorArrayV2Stack/TensorListStackTensorListStackgru_10/while:output:3@gru_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0o
gru_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
gru_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
gru_10/strided_slice_3StridedSlice2gru_10/TensorArrayV2Stack/TensorListStack:tensor:0%gru_10/strided_slice_3/stack:output:0'gru_10/strided_slice_3/stack_1:output:0'gru_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskl
gru_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
gru_10/transpose_1	Transpose2gru_10/TensorArrayV2Stack/TensorListStack:tensor:0 gru_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњdb
gru_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
gru_11/ShapeShapegru_10/transpose_1:y:0*
T0*
_output_shapes
:d
gru_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
gru_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
gru_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
gru_11/strided_sliceStridedSlicegru_11/Shape:output:0#gru_11/strided_slice/stack:output:0%gru_11/strided_slice/stack_1:output:0%gru_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
gru_11/zeros/packedPackgru_11/strided_slice:output:0gru_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
gru_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_11/zerosFillgru_11/zeros/packed:output:0gru_11/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
gru_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_11/transpose	Transposegru_10/transpose_1:y:0gru_11/transpose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdR
gru_11/Shape_1Shapegru_11/transpose:y:0*
T0*
_output_shapes
:f
gru_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
gru_11/strided_slice_1StridedSlicegru_11/Shape_1:output:0%gru_11/strided_slice_1/stack:output:0'gru_11/strided_slice_1/stack_1:output:0'gru_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"gru_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
gru_11/TensorArrayV2TensorListReserve+gru_11/TensorArrayV2/element_shape:output:0gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<gru_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ѕ
.gru_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_11/transpose:y:0Egru_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
gru_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
gru_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_11/strided_slice_2StridedSlicegru_11/transpose:y:0%gru_11/strided_slice_2/stack:output:0'gru_11/strided_slice_2/stack_1:output:0'gru_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask
!gru_11/gru_cell_20/ReadVariableOpReadVariableOp*gru_11_gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0
gru_11/gru_cell_20/unstackUnpack)gru_11/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
(gru_11/gru_cell_20/MatMul/ReadVariableOpReadVariableOp1gru_11_gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ј
gru_11/gru_cell_20/MatMulMatMulgru_11/strided_slice_2:output:00gru_11/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
gru_11/gru_cell_20/BiasAddBiasAdd#gru_11/gru_cell_20/MatMul:product:0#gru_11/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
"gru_11/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
gru_11/gru_cell_20/splitSplit+gru_11/gru_cell_20/split/split_dim:output:0#gru_11/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
*gru_11/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp3gru_11_gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ђ
gru_11/gru_cell_20/MatMul_1MatMulgru_11/zeros:output:02gru_11/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЅ
gru_11/gru_cell_20/BiasAdd_1BiasAdd%gru_11/gru_cell_20/MatMul_1:product:0#gru_11/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџm
gru_11/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџo
$gru_11/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gru_11/gru_cell_20/split_1SplitV%gru_11/gru_cell_20/BiasAdd_1:output:0!gru_11/gru_cell_20/Const:output:0-gru_11/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_11/gru_cell_20/addAddV2!gru_11/gru_cell_20/split:output:0#gru_11/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_11/gru_cell_20/SigmoidSigmoidgru_11/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_1AddV2!gru_11/gru_cell_20/split:output:1#gru_11/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџw
gru_11/gru_cell_20/Sigmoid_1Sigmoidgru_11/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mulMul gru_11/gru_cell_20/Sigmoid_1:y:0#gru_11/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_2AddV2!gru_11/gru_cell_20/split:output:2gru_11/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџw
gru_11/gru_cell_20/SoftplusSoftplusgru_11/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mul_1Mulgru_11/gru_cell_20/Sigmoid:y:0gru_11/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]
gru_11/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_11/gru_cell_20/subSub!gru_11/gru_cell_20/sub/x:output:0gru_11/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/mul_2Mulgru_11/gru_cell_20/sub:z:0)gru_11/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/gru_cell_20/add_3AddV2gru_11/gru_cell_20/mul_1:z:0gru_11/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
$gru_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Э
gru_11/TensorArrayV2_1TensorListReserve-gru_11/TensorArrayV2_1/element_shape:output:0gru_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
gru_11/timeConst*
_output_shapes
: *
dtype0*
value	B : j
gru_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
gru_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_11/whileWhile"gru_11/while/loop_counter:output:0(gru_11/while/maximum_iterations:output:0gru_11/time:output:0gru_11/TensorArrayV2_1:handle:0gru_11/zeros:output:0gru_11/strided_slice_1:output:0>gru_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_11_gru_cell_20_readvariableop_resource1gru_11_gru_cell_20_matmul_readvariableop_resource3gru_11_gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_11_while_body_1960630*%
condR
gru_11_while_cond_1960629*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
7gru_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
)gru_11/TensorArrayV2Stack/TensorListStackTensorListStackgru_11/while:output:3@gru_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0o
gru_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
gru_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
gru_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
gru_11/strided_slice_3StridedSlice2gru_11/TensorArrayV2Stack/TensorListStack:tensor:0%gru_11/strided_slice_3/stack:output:0'gru_11/strided_slice_3/stack_1:output:0'gru_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskl
gru_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
gru_11/transpose_1	Transpose2gru_11/TensorArrayV2Stack/TensorListStack:tensor:0 gru_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњb
gru_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentitygru_11/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњу
NoOpNoOp)^gru_10/gru_cell_19/MatMul/ReadVariableOp+^gru_10/gru_cell_19/MatMul_1/ReadVariableOp"^gru_10/gru_cell_19/ReadVariableOp^gru_10/while)^gru_11/gru_cell_20/MatMul/ReadVariableOp+^gru_11/gru_cell_20/MatMul_1/ReadVariableOp"^gru_11/gru_cell_20/ReadVariableOp^gru_11/while(^gru_9/gru_cell_18/MatMul/ReadVariableOp*^gru_9/gru_cell_18/MatMul_1/ReadVariableOp!^gru_9/gru_cell_18/ReadVariableOp^gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2T
(gru_10/gru_cell_19/MatMul/ReadVariableOp(gru_10/gru_cell_19/MatMul/ReadVariableOp2X
*gru_10/gru_cell_19/MatMul_1/ReadVariableOp*gru_10/gru_cell_19/MatMul_1/ReadVariableOp2F
!gru_10/gru_cell_19/ReadVariableOp!gru_10/gru_cell_19/ReadVariableOp2
gru_10/whilegru_10/while2T
(gru_11/gru_cell_20/MatMul/ReadVariableOp(gru_11/gru_cell_20/MatMul/ReadVariableOp2X
*gru_11/gru_cell_20/MatMul_1/ReadVariableOp*gru_11/gru_cell_20/MatMul_1/ReadVariableOp2F
!gru_11/gru_cell_20/ReadVariableOp!gru_11/gru_cell_20/ReadVariableOp2
gru_11/whilegru_11/while2R
'gru_9/gru_cell_18/MatMul/ReadVariableOp'gru_9/gru_cell_18/MatMul/ReadVariableOp2V
)gru_9/gru_cell_18/MatMul_1/ReadVariableOp)gru_9/gru_cell_18/MatMul_1/ReadVariableOp2D
 gru_9/gru_cell_18/ReadVariableOp gru_9/gru_cell_18/ReadVariableOp2
gru_9/whilegru_9/while:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
	
Д
gru_10_while_cond_1960029*
&gru_10_while_gru_10_while_loop_counter0
,gru_10_while_gru_10_while_maximum_iterations
gru_10_while_placeholder
gru_10_while_placeholder_1
gru_10_while_placeholder_2,
(gru_10_while_less_gru_10_strided_slice_1C
?gru_10_while_gru_10_while_cond_1960029___redundant_placeholder0C
?gru_10_while_gru_10_while_cond_1960029___redundant_placeholder1C
?gru_10_while_gru_10_while_cond_1960029___redundant_placeholder2C
?gru_10_while_gru_10_while_cond_1960029___redundant_placeholder3
gru_10_while_identity
~
gru_10/while/LessLessgru_10_while_placeholder(gru_10_while_less_gru_10_strided_slice_1*
T0*
_output_shapes
: Y
gru_10/while/IdentityIdentitygru_10/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_10_while_identitygru_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
ќ
Ћ
&sequential_3_gru_11_while_cond_1957458D
@sequential_3_gru_11_while_sequential_3_gru_11_while_loop_counterJ
Fsequential_3_gru_11_while_sequential_3_gru_11_while_maximum_iterations)
%sequential_3_gru_11_while_placeholder+
'sequential_3_gru_11_while_placeholder_1+
'sequential_3_gru_11_while_placeholder_2F
Bsequential_3_gru_11_while_less_sequential_3_gru_11_strided_slice_1]
Ysequential_3_gru_11_while_sequential_3_gru_11_while_cond_1957458___redundant_placeholder0]
Ysequential_3_gru_11_while_sequential_3_gru_11_while_cond_1957458___redundant_placeholder1]
Ysequential_3_gru_11_while_sequential_3_gru_11_while_cond_1957458___redundant_placeholder2]
Ysequential_3_gru_11_while_sequential_3_gru_11_while_cond_1957458___redundant_placeholder3&
"sequential_3_gru_11_while_identity
В
sequential_3/gru_11/while/LessLess%sequential_3_gru_11_while_placeholderBsequential_3_gru_11_while_less_sequential_3_gru_11_strided_slice_1*
T0*
_output_shapes
: s
"sequential_3/gru_11/while/IdentityIdentity"sequential_3/gru_11/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_3_gru_11_while_identity+sequential_3/gru_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
U
Х
%sequential_3_gru_9_while_body_1957161B
>sequential_3_gru_9_while_sequential_3_gru_9_while_loop_counterH
Dsequential_3_gru_9_while_sequential_3_gru_9_while_maximum_iterations(
$sequential_3_gru_9_while_placeholder*
&sequential_3_gru_9_while_placeholder_1*
&sequential_3_gru_9_while_placeholder_2A
=sequential_3_gru_9_while_sequential_3_gru_9_strided_slice_1_0}
ysequential_3_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_9_tensorarrayunstack_tensorlistfromtensor_0Q
>sequential_3_gru_9_while_gru_cell_18_readvariableop_resource_0:	X
Esequential_3_gru_9_while_gru_cell_18_matmul_readvariableop_resource_0:	[
Gsequential_3_gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ%
!sequential_3_gru_9_while_identity'
#sequential_3_gru_9_while_identity_1'
#sequential_3_gru_9_while_identity_2'
#sequential_3_gru_9_while_identity_3'
#sequential_3_gru_9_while_identity_4?
;sequential_3_gru_9_while_sequential_3_gru_9_strided_slice_1{
wsequential_3_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_9_tensorarrayunstack_tensorlistfromtensorO
<sequential_3_gru_9_while_gru_cell_18_readvariableop_resource:	V
Csequential_3_gru_9_while_gru_cell_18_matmul_readvariableop_resource:	Y
Esequential_3_gru_9_while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ:sequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOpЂ<sequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpЂ3sequential_3/gru_9/while/gru_cell_18/ReadVariableOp
Jsequential_3/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
<sequential_3/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_3_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_9_tensorarrayunstack_tensorlistfromtensor_0$sequential_3_gru_9_while_placeholderSsequential_3/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Г
3sequential_3/gru_9/while/gru_cell_18/ReadVariableOpReadVariableOp>sequential_3_gru_9_while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ћ
,sequential_3/gru_9/while/gru_cell_18/unstackUnpack;sequential_3/gru_9/while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numС
:sequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOpReadVariableOpEsequential_3_gru_9_while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ё
+sequential_3/gru_9/while/gru_cell_18/MatMulMatMulCsequential_3/gru_9/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџи
,sequential_3/gru_9/while/gru_cell_18/BiasAddBiasAdd5sequential_3/gru_9/while/gru_cell_18/MatMul:product:05sequential_3/gru_9/while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
4sequential_3/gru_9/while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
*sequential_3/gru_9/while/gru_cell_18/splitSplit=sequential_3/gru_9/while/gru_cell_18/split/split_dim:output:05sequential_3/gru_9/while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitЦ
<sequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOpGsequential_3_gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0и
-sequential_3/gru_9/while/gru_cell_18/MatMul_1MatMul&sequential_3_gru_9_while_placeholder_2Dsequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџм
.sequential_3/gru_9/while/gru_cell_18/BiasAdd_1BiasAdd7sequential_3/gru_9/while/gru_cell_18/MatMul_1:product:05sequential_3/gru_9/while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ
*sequential_3/gru_9/while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ
6sequential_3/gru_9/while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџн
,sequential_3/gru_9/while/gru_cell_18/split_1SplitV7sequential_3/gru_9/while/gru_cell_18/BiasAdd_1:output:03sequential_3/gru_9/while/gru_cell_18/Const:output:0?sequential_3/gru_9/while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitа
(sequential_3/gru_9/while/gru_cell_18/addAddV23sequential_3/gru_9/while/gru_cell_18/split:output:05sequential_3/gru_9/while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
,sequential_3/gru_9/while/gru_cell_18/SigmoidSigmoid,sequential_3/gru_9/while/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌв
*sequential_3/gru_9/while/gru_cell_18/add_1AddV23sequential_3/gru_9/while/gru_cell_18/split:output:15sequential_3/gru_9/while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌ
.sequential_3/gru_9/while/gru_cell_18/Sigmoid_1Sigmoid.sequential_3/gru_9/while/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌЭ
(sequential_3/gru_9/while/gru_cell_18/mulMul2sequential_3/gru_9/while/gru_cell_18/Sigmoid_1:y:05sequential_3/gru_9/while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌЩ
*sequential_3/gru_9/while/gru_cell_18/add_2AddV23sequential_3/gru_9/while/gru_cell_18/split:output:2,sequential_3/gru_9/while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
.sequential_3/gru_9/while/gru_cell_18/Sigmoid_2Sigmoid.sequential_3/gru_9/while/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌО
*sequential_3/gru_9/while/gru_cell_18/mul_1Mul0sequential_3/gru_9/while/gru_cell_18/Sigmoid:y:0&sequential_3_gru_9_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌo
*sequential_3/gru_9/while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
(sequential_3/gru_9/while/gru_cell_18/subSub3sequential_3/gru_9/while/gru_cell_18/sub/x:output:00sequential_3/gru_9/while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌЦ
*sequential_3/gru_9/while/gru_cell_18/mul_2Mul,sequential_3/gru_9/while/gru_cell_18/sub:z:02sequential_3/gru_9/while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌЦ
*sequential_3/gru_9/while/gru_cell_18/add_3AddV2.sequential_3/gru_9/while/gru_cell_18/mul_1:z:0.sequential_3/gru_9/while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
=sequential_3/gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_3_gru_9_while_placeholder_1$sequential_3_gru_9_while_placeholder.sequential_3/gru_9/while/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшв`
sequential_3/gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_3/gru_9/while/addAddV2$sequential_3_gru_9_while_placeholder'sequential_3/gru_9/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_3/gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Г
sequential_3/gru_9/while/add_1AddV2>sequential_3_gru_9_while_sequential_3_gru_9_while_loop_counter)sequential_3/gru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_3/gru_9/while/IdentityIdentity"sequential_3/gru_9/while/add_1:z:0^sequential_3/gru_9/while/NoOp*
T0*
_output_shapes
: Ж
#sequential_3/gru_9/while/Identity_1IdentityDsequential_3_gru_9_while_sequential_3_gru_9_while_maximum_iterations^sequential_3/gru_9/while/NoOp*
T0*
_output_shapes
: 
#sequential_3/gru_9/while/Identity_2Identity sequential_3/gru_9/while/add:z:0^sequential_3/gru_9/while/NoOp*
T0*
_output_shapes
: П
#sequential_3/gru_9/while/Identity_3IdentityMsequential_3/gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_3/gru_9/while/NoOp*
T0*
_output_shapes
: В
#sequential_3/gru_9/while/Identity_4Identity.sequential_3/gru_9/while/gru_cell_18/add_3:z:0^sequential_3/gru_9/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ
sequential_3/gru_9/while/NoOpNoOp;^sequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOp=^sequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp4^sequential_3/gru_9/while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Esequential_3_gru_9_while_gru_cell_18_matmul_1_readvariableop_resourceGsequential_3_gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0"
Csequential_3_gru_9_while_gru_cell_18_matmul_readvariableop_resourceEsequential_3_gru_9_while_gru_cell_18_matmul_readvariableop_resource_0"~
<sequential_3_gru_9_while_gru_cell_18_readvariableop_resource>sequential_3_gru_9_while_gru_cell_18_readvariableop_resource_0"O
!sequential_3_gru_9_while_identity*sequential_3/gru_9/while/Identity:output:0"S
#sequential_3_gru_9_while_identity_1,sequential_3/gru_9/while/Identity_1:output:0"S
#sequential_3_gru_9_while_identity_2,sequential_3/gru_9/while/Identity_2:output:0"S
#sequential_3_gru_9_while_identity_3,sequential_3/gru_9/while/Identity_3:output:0"S
#sequential_3_gru_9_while_identity_4,sequential_3/gru_9/while/Identity_4:output:0"|
;sequential_3_gru_9_while_sequential_3_gru_9_strided_slice_1=sequential_3_gru_9_while_sequential_3_gru_9_strided_slice_1_0"є
wsequential_3_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_9_tensorarrayunstack_tensorlistfromtensorysequential_3_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2x
:sequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOp:sequential_3/gru_9/while/gru_cell_18/MatMul/ReadVariableOp2|
<sequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp<sequential_3/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp2j
3sequential_3/gru_9/while/gru_cell_18/ReadVariableOp3sequential_3/gru_9/while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
п
Џ
while_cond_1961482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961482___redundant_placeholder05
1while_while_cond_1961482___redundant_placeholder15
1while_while_cond_1961482___redundant_placeholder25
1while_while_cond_1961482___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1959322
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1959322___redundant_placeholder05
1while_while_cond_1959322___redundant_placeholder15
1while_while_cond_1959322___redundant_placeholder25
1while_while_cond_1959322___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1958150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958150___redundant_placeholder05
1while_while_cond_1958150___redundant_placeholder15
1while_while_cond_1958150___redundant_placeholder25
1while_while_cond_1958150___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
ІM

C__inference_gru_10_layer_call_and_return_conditional_losses_1961878

inputs6
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961789*
condR
while_cond_1961788*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs
КM

B__inference_gru_9_layer_call_and_return_conditional_losses_1959587

inputs6
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1959498*
condR
while_cond_1959497*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ф
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
=

while_body_1958792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
їE
н	
gru_10_while_body_1960030*
&gru_10_while_gru_10_while_loop_counter0
,gru_10_while_gru_10_while_maximum_iterations
gru_10_while_placeholder
gru_10_while_placeholder_1
gru_10_while_placeholder_2)
%gru_10_while_gru_10_strided_slice_1_0e
agru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0E
2gru_10_while_gru_cell_19_readvariableop_resource_0:	ЌM
9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌN
;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
gru_10_while_identity
gru_10_while_identity_1
gru_10_while_identity_2
gru_10_while_identity_3
gru_10_while_identity_4'
#gru_10_while_gru_10_strided_slice_1c
_gru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensorC
0gru_10_while_gru_cell_19_readvariableop_resource:	ЌK
7gru_10_while_gru_cell_19_matmul_readvariableop_resource:
ЌЌL
9gru_10_while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ.gru_10/while/gru_cell_19/MatMul/ReadVariableOpЂ0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpЂ'gru_10/while/gru_cell_19/ReadVariableOp
>gru_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ъ
0gru_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0gru_10_while_placeholderGgru_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
'gru_10/while/gru_cell_19/ReadVariableOpReadVariableOp2gru_10_while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
 gru_10/while/gru_cell_19/unstackUnpack/gru_10/while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numЊ
.gru_10/while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0Э
gru_10/while/gru_cell_19/MatMulMatMul7gru_10/while/TensorArrayV2Read/TensorListGetItem:item:06gru_10/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌД
 gru_10/while/gru_cell_19/BiasAddBiasAdd)gru_10/while/gru_cell_19/MatMul:product:0)gru_10/while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌs
(gru_10/while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_10/while/gru_cell_19/splitSplit1gru_10/while/gru_cell_19/split/split_dim:output:0)gru_10/while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split­
0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0Д
!gru_10/while/gru_cell_19/MatMul_1MatMulgru_10_while_placeholder_28gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌИ
"gru_10/while/gru_cell_19/BiasAdd_1BiasAdd+gru_10/while/gru_cell_19/MatMul_1:product:0)gru_10/while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌs
gru_10/while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџu
*gru_10/while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЊ
 gru_10/while/gru_cell_19/split_1SplitV+gru_10/while/gru_cell_19/BiasAdd_1:output:0'gru_10/while/gru_cell_19/Const:output:03gru_10/while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitЋ
gru_10/while/gru_cell_19/addAddV2'gru_10/while/gru_cell_19/split:output:0)gru_10/while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
 gru_10/while/gru_cell_19/SigmoidSigmoid gru_10/while/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd­
gru_10/while/gru_cell_19/add_1AddV2'gru_10/while/gru_cell_19/split:output:1)gru_10/while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
"gru_10/while/gru_cell_19/Sigmoid_1Sigmoid"gru_10/while/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
gru_10/while/gru_cell_19/mulMul&gru_10/while/gru_cell_19/Sigmoid_1:y:0)gru_10/while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЄ
gru_10/while/gru_cell_19/add_2AddV2'gru_10/while/gru_cell_19/split:output:2 gru_10/while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
"gru_10/while/gru_cell_19/Sigmoid_2Sigmoid"gru_10/while/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_10/while/gru_cell_19/mul_1Mul$gru_10/while/gru_cell_19/Sigmoid:y:0gru_10_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџdc
gru_10/while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
gru_10/while/gru_cell_19/subSub'gru_10/while/gru_cell_19/sub/x:output:0$gru_10/while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЁ
gru_10/while/gru_cell_19/mul_2Mul gru_10/while/gru_cell_19/sub:z:0&gru_10/while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdЁ
gru_10/while/gru_cell_19/add_3AddV2"gru_10/while/gru_cell_19/mul_1:z:0"gru_10/while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdр
1gru_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_10_while_placeholder_1gru_10_while_placeholder"gru_10/while/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвT
gru_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_10/while/addAddV2gru_10_while_placeholdergru_10/while/add/y:output:0*
T0*
_output_shapes
: V
gru_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_10/while/add_1AddV2&gru_10_while_gru_10_while_loop_countergru_10/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_10/while/IdentityIdentitygru_10/while/add_1:z:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_1Identity,gru_10_while_gru_10_while_maximum_iterations^gru_10/while/NoOp*
T0*
_output_shapes
: n
gru_10/while/Identity_2Identitygru_10/while/add:z:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_3IdentityAgru_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_10/while/NoOp*
T0*
_output_shapes
: 
gru_10/while/Identity_4Identity"gru_10/while/gru_cell_19/add_3:z:0^gru_10/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdс
gru_10/while/NoOpNoOp/^gru_10/while/gru_cell_19/MatMul/ReadVariableOp1^gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp(^gru_10/while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_10_while_gru_10_strided_slice_1%gru_10_while_gru_10_strided_slice_1_0"x
9gru_10_while_gru_cell_19_matmul_1_readvariableop_resource;gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0"t
7gru_10_while_gru_cell_19_matmul_readvariableop_resource9gru_10_while_gru_cell_19_matmul_readvariableop_resource_0"f
0gru_10_while_gru_cell_19_readvariableop_resource2gru_10_while_gru_cell_19_readvariableop_resource_0"7
gru_10_while_identitygru_10/while/Identity:output:0";
gru_10_while_identity_1 gru_10/while/Identity_1:output:0";
gru_10_while_identity_2 gru_10/while/Identity_2:output:0";
gru_10_while_identity_3 gru_10/while/Identity_3:output:0";
gru_10_while_identity_4 gru_10/while/Identity_4:output:0"Ф
_gru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensoragru_10_while_tensorarrayv2read_tensorlistgetitem_gru_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2`
.gru_10/while/gru_cell_19/MatMul/ReadVariableOp.gru_10/while/gru_cell_19/MatMul/ReadVariableOp2d
0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp0gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp2R
'gru_10/while/gru_cell_19/ReadVariableOp'gru_10/while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
с
Џ
while_cond_1960979
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1960979___redundant_placeholder05
1while_while_cond_1960979___redundant_placeholder15
1while_while_cond_1960979___redundant_placeholder25
1while_while_cond_1960979___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
ђM

B__inference_gru_9_layer_call_and_return_conditional_losses_1960916
inputs_06
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile=
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1960827*
condR
while_cond_1960826*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ѕ=

while_body_1960827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 

К
(__inference_gru_10_layer_call_fn_1961419

inputs
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1959412t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs
=

while_body_1961789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
с
Џ
while_cond_1960826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1960826___redundant_placeholder05
1while_while_cond_1960826___redundant_placeholder15
1while_while_cond_1960826___redundant_placeholder25
1while_while_cond_1960826___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
ќ
Ћ
&sequential_3_gru_10_while_cond_1957309D
@sequential_3_gru_10_while_sequential_3_gru_10_while_loop_counterJ
Fsequential_3_gru_10_while_sequential_3_gru_10_while_maximum_iterations)
%sequential_3_gru_10_while_placeholder+
'sequential_3_gru_10_while_placeholder_1+
'sequential_3_gru_10_while_placeholder_2F
Bsequential_3_gru_10_while_less_sequential_3_gru_10_strided_slice_1]
Ysequential_3_gru_10_while_sequential_3_gru_10_while_cond_1957309___redundant_placeholder0]
Ysequential_3_gru_10_while_sequential_3_gru_10_while_cond_1957309___redundant_placeholder1]
Ysequential_3_gru_10_while_sequential_3_gru_10_while_cond_1957309___redundant_placeholder2]
Ysequential_3_gru_10_while_sequential_3_gru_10_while_cond_1957309___redundant_placeholder3&
"sequential_3_gru_10_while_identity
В
sequential_3/gru_10/while/LessLess%sequential_3_gru_10_while_placeholderBsequential_3_gru_10_while_less_sequential_3_gru_10_strided_slice_1*
T0*
_output_shapes
: s
"sequential_3/gru_10/while/IdentityIdentity"sequential_3/gru_10/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_3_gru_10_while_identity+sequential_3/gru_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
M

C__inference_gru_11_layer_call_and_return_conditional_losses_1959041

inputs5
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958952*
condR
while_cond_1958951*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
п
Џ
while_cond_1962597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1962597___redundant_placeholder05
1while_while_cond_1962597___redundant_placeholder15
1while_while_cond_1962597___redundant_placeholder25
1while_while_cond_1962597___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ѓ

I__inference_sequential_3_layer_call_and_return_conditional_losses_1959050

inputs 
gru_9_1958722:	 
gru_9_1958724:	!
gru_9_1958726:
Ќ!
gru_10_1958882:	Ќ"
gru_10_1958884:
ЌЌ!
gru_10_1958886:	dЌ 
gru_11_1959042: 
gru_11_1959044:d 
gru_11_1959046:
identityЂgru_10/StatefulPartitionedCallЂgru_11/StatefulPartitionedCallЂgru_9/StatefulPartitionedCall
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_1958722gru_9_1958724gru_9_1958726*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1958721Ѕ
gru_10/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0gru_10_1958882gru_10_1958884gru_10_1958886*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1958881І
gru_11/StatefulPartitionedCallStatefulPartitionedCall'gru_10/StatefulPartitionedCall:output:0gru_11_1959042gru_11_1959044gru_11_1959046*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959041{
IdentityIdentity'gru_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЈ
NoOpNoOp^gru_10/StatefulPartitionedCall^gru_11/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2@
gru_10/StatefulPartitionedCallgru_10/StatefulPartitionedCall2@
gru_11/StatefulPartitionedCallgru_11/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ІM

C__inference_gru_10_layer_call_and_return_conditional_losses_1962031

inputs6
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961942*
condR
while_cond_1961941*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs
Ѕ=

while_body_1961286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
M

C__inference_gru_11_layer_call_and_return_conditional_losses_1962687

inputs5
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1962598*
condR
while_cond_1962597*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
п
Џ
while_cond_1962138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1962138___redundant_placeholder05
1while_while_cond_1962138___redundant_placeholder15
1while_while_cond_1962138___redundant_placeholder25
1while_while_cond_1962138___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ѕ=

while_body_1958632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
4

C__inference_gru_11_layer_call_and_return_conditional_losses_1958553

inputs%
gru_cell_20_1958477:%
gru_cell_20_1958479:d%
gru_cell_20_1958481:
identityЂ#gru_cell_20/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskа
#gru_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_20_1958477gru_cell_20_1958479gru_cell_20_1958481*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958437n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_20_1958477gru_cell_20_1958479gru_cell_20_1958481*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958489*
condR
while_cond_1958488*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
NoOpNoOp$^gru_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 2J
#gru_cell_20/StatefulPartitionedCall#gru_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
 
_user_specified_nameinputs
с
Џ
while_cond_1957630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1957630___redundant_placeholder05
1while_while_cond_1957630___redundant_placeholder15
1while_while_cond_1957630___redundant_placeholder25
1while_while_cond_1957630___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
Б
Л
'__inference_gru_9_layer_call_fn_1960741
inputs_0
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1957877}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
КM

B__inference_gru_9_layer_call_and_return_conditional_losses_1958721

inputs6
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958632*
condR
while_cond_1958631*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ф
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
=

while_body_1959323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
	
Ё
gru_9_while_cond_1960331(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1A
=gru_9_while_gru_9_while_cond_1960331___redundant_placeholder0A
=gru_9_while_gru_9_while_cond_1960331___redundant_placeholder1A
=gru_9_while_gru_9_while_cond_1960331___redundant_placeholder2A
=gru_9_while_gru_9_while_cond_1960331___redundant_placeholder3
gru_9_while_identity
z
gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: W
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_9_while_identitygru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
В

I__inference_sequential_3_layer_call_and_return_conditional_losses_1959740
gru_9_input 
gru_9_1959718:	 
gru_9_1959720:	!
gru_9_1959722:
Ќ!
gru_10_1959725:	Ќ"
gru_10_1959727:
ЌЌ!
gru_10_1959729:	dЌ 
gru_11_1959732: 
gru_11_1959734:d 
gru_11_1959736:
identityЂgru_10/StatefulPartitionedCallЂgru_11/StatefulPartitionedCallЂgru_9/StatefulPartitionedCall
gru_9/StatefulPartitionedCallStatefulPartitionedCallgru_9_inputgru_9_1959718gru_9_1959720gru_9_1959722*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1959587Ѕ
gru_10/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0gru_10_1959725gru_10_1959727gru_10_1959729*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1959412І
gru_11/StatefulPartitionedCallStatefulPartitionedCall'gru_10/StatefulPartitionedCall:output:0gru_11_1959732gru_11_1959734gru_11_1959736*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959237{
IdentityIdentity'gru_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЈ
NoOpNoOp^gru_10/StatefulPartitionedCall^gru_11/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2@
gru_10/StatefulPartitionedCallgru_10/StatefulPartitionedCall2@
gru_11/StatefulPartitionedCallgru_11/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input
с
Џ
while_cond_1961285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961285___redundant_placeholder05
1while_while_cond_1961285___redundant_placeholder15
1while_while_cond_1961285___redundant_placeholder25
1while_while_cond_1961285___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
ЂV
у
&sequential_3_gru_10_while_body_1957310D
@sequential_3_gru_10_while_sequential_3_gru_10_while_loop_counterJ
Fsequential_3_gru_10_while_sequential_3_gru_10_while_maximum_iterations)
%sequential_3_gru_10_while_placeholder+
'sequential_3_gru_10_while_placeholder_1+
'sequential_3_gru_10_while_placeholder_2C
?sequential_3_gru_10_while_sequential_3_gru_10_strided_slice_1_0
{sequential_3_gru_10_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_10_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_3_gru_10_while_gru_cell_19_readvariableop_resource_0:	ЌZ
Fsequential_3_gru_10_while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌ[
Hsequential_3_gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ&
"sequential_3_gru_10_while_identity(
$sequential_3_gru_10_while_identity_1(
$sequential_3_gru_10_while_identity_2(
$sequential_3_gru_10_while_identity_3(
$sequential_3_gru_10_while_identity_4A
=sequential_3_gru_10_while_sequential_3_gru_10_strided_slice_1}
ysequential_3_gru_10_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_10_tensorarrayunstack_tensorlistfromtensorP
=sequential_3_gru_10_while_gru_cell_19_readvariableop_resource:	ЌX
Dsequential_3_gru_10_while_gru_cell_19_matmul_readvariableop_resource:
ЌЌY
Fsequential_3_gru_10_while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ;sequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOpЂ=sequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpЂ4sequential_3/gru_10/while/gru_cell_19/ReadVariableOp
Ksequential_3/gru_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  
=sequential_3/gru_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_3_gru_10_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_10_tensorarrayunstack_tensorlistfromtensor_0%sequential_3_gru_10_while_placeholderTsequential_3/gru_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0Е
4sequential_3/gru_10/while/gru_cell_19/ReadVariableOpReadVariableOp?sequential_3_gru_10_while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0­
-sequential_3/gru_10/while/gru_cell_19/unstackUnpack<sequential_3/gru_10/while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numФ
;sequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOpReadVariableOpFsequential_3_gru_10_while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0є
,sequential_3/gru_10/while/gru_cell_19/MatMulMatMulDsequential_3/gru_10/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌл
-sequential_3/gru_10/while/gru_cell_19/BiasAddBiasAdd6sequential_3/gru_10/while/gru_cell_19/MatMul:product:06sequential_3/gru_10/while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
5sequential_3/gru_10/while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
+sequential_3/gru_10/while/gru_cell_19/splitSplit>sequential_3/gru_10/while/gru_cell_19/split/split_dim:output:06sequential_3/gru_10/while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitЧ
=sequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOpHsequential_3_gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0л
.sequential_3/gru_10/while/gru_cell_19/MatMul_1MatMul'sequential_3_gru_10_while_placeholder_2Esequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌп
/sequential_3/gru_10/while/gru_cell_19/BiasAdd_1BiasAdd8sequential_3/gru_10/while/gru_cell_19/MatMul_1:product:06sequential_3/gru_10/while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌ
+sequential_3/gru_10/while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ
7sequential_3/gru_10/while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
-sequential_3/gru_10/while/gru_cell_19/split_1SplitV8sequential_3/gru_10/while/gru_cell_19/BiasAdd_1:output:04sequential_3/gru_10/while/gru_cell_19/Const:output:0@sequential_3/gru_10/while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitв
)sequential_3/gru_10/while/gru_cell_19/addAddV24sequential_3/gru_10/while/gru_cell_19/split:output:06sequential_3/gru_10/while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
-sequential_3/gru_10/while/gru_cell_19/SigmoidSigmoid-sequential_3/gru_10/while/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџdд
+sequential_3/gru_10/while/gru_cell_19/add_1AddV24sequential_3/gru_10/while/gru_cell_19/split:output:16sequential_3/gru_10/while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
/sequential_3/gru_10/while/gru_cell_19/Sigmoid_1Sigmoid/sequential_3/gru_10/while/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЯ
)sequential_3/gru_10/while/gru_cell_19/mulMul3sequential_3/gru_10/while/gru_cell_19/Sigmoid_1:y:06sequential_3/gru_10/while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЫ
+sequential_3/gru_10/while/gru_cell_19/add_2AddV24sequential_3/gru_10/while/gru_cell_19/split:output:2-sequential_3/gru_10/while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
/sequential_3/gru_10/while/gru_cell_19/Sigmoid_2Sigmoid/sequential_3/gru_10/while/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdР
+sequential_3/gru_10/while/gru_cell_19/mul_1Mul1sequential_3/gru_10/while/gru_cell_19/Sigmoid:y:0'sequential_3_gru_10_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџdp
+sequential_3/gru_10/while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ы
)sequential_3/gru_10/while/gru_cell_19/subSub4sequential_3/gru_10/while/gru_cell_19/sub/x:output:01sequential_3/gru_10/while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdШ
+sequential_3/gru_10/while/gru_cell_19/mul_2Mul-sequential_3/gru_10/while/gru_cell_19/sub:z:03sequential_3/gru_10/while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdШ
+sequential_3/gru_10/while/gru_cell_19/add_3AddV2/sequential_3/gru_10/while/gru_cell_19/mul_1:z:0/sequential_3/gru_10/while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
>sequential_3/gru_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_3_gru_10_while_placeholder_1%sequential_3_gru_10_while_placeholder/sequential_3/gru_10/while/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвa
sequential_3/gru_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_3/gru_10/while/addAddV2%sequential_3_gru_10_while_placeholder(sequential_3/gru_10/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_3/gru_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
sequential_3/gru_10/while/add_1AddV2@sequential_3_gru_10_while_sequential_3_gru_10_while_loop_counter*sequential_3/gru_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_3/gru_10/while/IdentityIdentity#sequential_3/gru_10/while/add_1:z:0^sequential_3/gru_10/while/NoOp*
T0*
_output_shapes
: К
$sequential_3/gru_10/while/Identity_1IdentityFsequential_3_gru_10_while_sequential_3_gru_10_while_maximum_iterations^sequential_3/gru_10/while/NoOp*
T0*
_output_shapes
: 
$sequential_3/gru_10/while/Identity_2Identity!sequential_3/gru_10/while/add:z:0^sequential_3/gru_10/while/NoOp*
T0*
_output_shapes
: Т
$sequential_3/gru_10/while/Identity_3IdentityNsequential_3/gru_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_3/gru_10/while/NoOp*
T0*
_output_shapes
: Д
$sequential_3/gru_10/while/Identity_4Identity/sequential_3/gru_10/while/gru_cell_19/add_3:z:0^sequential_3/gru_10/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
sequential_3/gru_10/while/NoOpNoOp<^sequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOp>^sequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp5^sequential_3/gru_10/while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Fsequential_3_gru_10_while_gru_cell_19_matmul_1_readvariableop_resourceHsequential_3_gru_10_while_gru_cell_19_matmul_1_readvariableop_resource_0"
Dsequential_3_gru_10_while_gru_cell_19_matmul_readvariableop_resourceFsequential_3_gru_10_while_gru_cell_19_matmul_readvariableop_resource_0"
=sequential_3_gru_10_while_gru_cell_19_readvariableop_resource?sequential_3_gru_10_while_gru_cell_19_readvariableop_resource_0"Q
"sequential_3_gru_10_while_identity+sequential_3/gru_10/while/Identity:output:0"U
$sequential_3_gru_10_while_identity_1-sequential_3/gru_10/while/Identity_1:output:0"U
$sequential_3_gru_10_while_identity_2-sequential_3/gru_10/while/Identity_2:output:0"U
$sequential_3_gru_10_while_identity_3-sequential_3/gru_10/while/Identity_3:output:0"U
$sequential_3_gru_10_while_identity_4-sequential_3/gru_10/while/Identity_4:output:0"
=sequential_3_gru_10_while_sequential_3_gru_10_strided_slice_1?sequential_3_gru_10_while_sequential_3_gru_10_strided_slice_1_0"ј
ysequential_3_gru_10_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_10_tensorarrayunstack_tensorlistfromtensor{sequential_3_gru_10_while_tensorarrayv2read_tensorlistgetitem_sequential_3_gru_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2z
;sequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOp;sequential_3/gru_10/while/gru_cell_19/MatMul/ReadVariableOp2~
=sequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp=sequential_3/gru_10/while/gru_cell_19/MatMul_1/ReadVariableOp2l
4sequential_3/gru_10/while/gru_cell_19/ReadVariableOp4sequential_3/gru_10/while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
Ѕ=

while_body_1960980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
жM

C__inference_gru_11_layer_call_and_return_conditional_losses_1962228
inputs_05
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1962139*
condR
while_cond_1962138*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
"
_user_specified_name
inputs/0
эE
е	
gru_11_while_body_1960630*
&gru_11_while_gru_11_while_loop_counter0
,gru_11_while_gru_11_while_maximum_iterations
gru_11_while_placeholder
gru_11_while_placeholder_1
gru_11_while_placeholder_2)
%gru_11_while_gru_11_strided_slice_1_0e
agru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0D
2gru_11_while_gru_cell_20_readvariableop_resource_0:K
9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0:dM
;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0:
gru_11_while_identity
gru_11_while_identity_1
gru_11_while_identity_2
gru_11_while_identity_3
gru_11_while_identity_4'
#gru_11_while_gru_11_strided_slice_1c
_gru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensorB
0gru_11_while_gru_cell_20_readvariableop_resource:I
7gru_11_while_gru_cell_20_matmul_readvariableop_resource:dK
9gru_11_while_gru_cell_20_matmul_1_readvariableop_resource:Ђ.gru_11/while/gru_cell_20/MatMul/ReadVariableOpЂ0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpЂ'gru_11/while/gru_cell_20/ReadVariableOp
>gru_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Щ
0gru_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0gru_11_while_placeholderGgru_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
'gru_11/while/gru_cell_20/ReadVariableOpReadVariableOp2gru_11_while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
 gru_11/while/gru_cell_20/unstackUnpack/gru_11/while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numЈ
.gru_11/while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0Ь
gru_11/while/gru_cell_20/MatMulMatMul7gru_11/while/TensorArrayV2Read/TensorListGetItem:item:06gru_11/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџГ
 gru_11/while/gru_cell_20/BiasAddBiasAdd)gru_11/while/gru_cell_20/MatMul:product:0)gru_11/while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
(gru_11/while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_11/while/gru_cell_20/splitSplit1gru_11/while/gru_cell_20/split/split_dim:output:0)gru_11/while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitЌ
0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Г
!gru_11/while/gru_cell_20/MatMul_1MatMulgru_11_while_placeholder_28gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЗ
"gru_11/while/gru_cell_20/BiasAdd_1BiasAdd+gru_11/while/gru_cell_20/MatMul_1:product:0)gru_11/while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџs
gru_11/while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџu
*gru_11/while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЊ
 gru_11/while/gru_cell_20/split_1SplitV+gru_11/while/gru_cell_20/BiasAdd_1:output:0'gru_11/while/gru_cell_20/Const:output:03gru_11/while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitЋ
gru_11/while/gru_cell_20/addAddV2'gru_11/while/gru_cell_20/split:output:0)gru_11/while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 gru_11/while/gru_cell_20/SigmoidSigmoid gru_11/while/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ­
gru_11/while/gru_cell_20/add_1AddV2'gru_11/while/gru_cell_20/split:output:1)gru_11/while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
"gru_11/while/gru_cell_20/Sigmoid_1Sigmoid"gru_11/while/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
gru_11/while/gru_cell_20/mulMul&gru_11/while/gru_cell_20/Sigmoid_1:y:0)gru_11/while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџЄ
gru_11/while/gru_cell_20/add_2AddV2'gru_11/while/gru_cell_20/split:output:2 gru_11/while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gru_11/while/gru_cell_20/SoftplusSoftplus"gru_11/while/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_11/while/gru_cell_20/mul_1Mul$gru_11/while/gru_cell_20/Sigmoid:y:0gru_11_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџc
gru_11/while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
gru_11/while/gru_cell_20/subSub'gru_11/while/gru_cell_20/sub/x:output:0$gru_11/while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
gru_11/while/gru_cell_20/mul_2Mul gru_11/while/gru_cell_20/sub:z:0/gru_11/while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
gru_11/while/gru_cell_20/add_3AddV2"gru_11/while/gru_cell_20/mul_1:z:0"gru_11/while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџр
1gru_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_11_while_placeholder_1gru_11_while_placeholder"gru_11/while/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвT
gru_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
gru_11/while/addAddV2gru_11_while_placeholdergru_11/while/add/y:output:0*
T0*
_output_shapes
: V
gru_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_11/while/add_1AddV2&gru_11_while_gru_11_while_loop_countergru_11/while/add_1/y:output:0*
T0*
_output_shapes
: n
gru_11/while/IdentityIdentitygru_11/while/add_1:z:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_1Identity,gru_11_while_gru_11_while_maximum_iterations^gru_11/while/NoOp*
T0*
_output_shapes
: n
gru_11/while/Identity_2Identitygru_11/while/add:z:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_3IdentityAgru_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_11/while/NoOp*
T0*
_output_shapes
: 
gru_11/while/Identity_4Identity"gru_11/while/gru_cell_20/add_3:z:0^gru_11/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџс
gru_11/while/NoOpNoOp/^gru_11/while/gru_cell_20/MatMul/ReadVariableOp1^gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp(^gru_11/while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "L
#gru_11_while_gru_11_strided_slice_1%gru_11_while_gru_11_strided_slice_1_0"x
9gru_11_while_gru_cell_20_matmul_1_readvariableop_resource;gru_11_while_gru_cell_20_matmul_1_readvariableop_resource_0"t
7gru_11_while_gru_cell_20_matmul_readvariableop_resource9gru_11_while_gru_cell_20_matmul_readvariableop_resource_0"f
0gru_11_while_gru_cell_20_readvariableop_resource2gru_11_while_gru_cell_20_readvariableop_resource_0"7
gru_11_while_identitygru_11/while/Identity:output:0";
gru_11_while_identity_1 gru_11/while/Identity_1:output:0";
gru_11_while_identity_2 gru_11/while/Identity_2:output:0";
gru_11_while_identity_3 gru_11/while/Identity_3:output:0";
gru_11_while_identity_4 gru_11/while/Identity_4:output:0"Ф
_gru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensoragru_11_while_tensorarrayv2read_tensorlistgetitem_gru_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2`
.gru_11/while/gru_cell_20/MatMul/ReadVariableOp.gru_11/while/gru_cell_20/MatMul/ReadVariableOp2d
0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp0gru_11/while/gru_cell_20/MatMul_1/ReadVariableOp2R
'gru_11/while/gru_cell_20/ReadVariableOp'gru_11/while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

Й
'__inference_gru_9_layer_call_fn_1960752

inputs
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1958721u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
­
н
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962793

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
Ќ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџІ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:џџџџџџџџџЌJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџЌ
"
_user_specified_name
states/0
Ѓ

I__inference_sequential_3_layer_call_and_return_conditional_losses_1959646

inputs 
gru_9_1959624:	 
gru_9_1959626:	!
gru_9_1959628:
Ќ!
gru_10_1959631:	Ќ"
gru_10_1959633:
ЌЌ!
gru_10_1959635:	dЌ 
gru_11_1959638: 
gru_11_1959640:d 
gru_11_1959642:
identityЂgru_10/StatefulPartitionedCallЂgru_11/StatefulPartitionedCallЂgru_9/StatefulPartitionedCall
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_1959624gru_9_1959626gru_9_1959628*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1959587Ѕ
gru_10/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0gru_10_1959631gru_10_1959633gru_10_1959635*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1959412І
gru_11/StatefulPartitionedCallStatefulPartitionedCall'gru_10/StatefulPartitionedCall:output:0gru_11_1959638gru_11_1959640gru_11_1959642*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959237{
IdentityIdentity'gru_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЈ
NoOpNoOp^gru_10/StatefulPartitionedCall^gru_11/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2@
gru_10/StatefulPartitionedCallgru_10/StatefulPartitionedCall2@
gru_11/StatefulPartitionedCallgru_11/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
у

%sequential_3_gru_9_while_cond_1957160B
>sequential_3_gru_9_while_sequential_3_gru_9_while_loop_counterH
Dsequential_3_gru_9_while_sequential_3_gru_9_while_maximum_iterations(
$sequential_3_gru_9_while_placeholder*
&sequential_3_gru_9_while_placeholder_1*
&sequential_3_gru_9_while_placeholder_2D
@sequential_3_gru_9_while_less_sequential_3_gru_9_strided_slice_1[
Wsequential_3_gru_9_while_sequential_3_gru_9_while_cond_1957160___redundant_placeholder0[
Wsequential_3_gru_9_while_sequential_3_gru_9_while_cond_1957160___redundant_placeholder1[
Wsequential_3_gru_9_while_sequential_3_gru_9_while_cond_1957160___redundant_placeholder2[
Wsequential_3_gru_9_while_sequential_3_gru_9_while_cond_1957160___redundant_placeholder3%
!sequential_3_gru_9_while_identity
Ў
sequential_3/gru_9/while/LessLess$sequential_3_gru_9_while_placeholder@sequential_3_gru_9_while_less_sequential_3_gru_9_strided_slice_1*
T0*
_output_shapes
: q
!sequential_3/gru_9/while/IdentityIdentity!sequential_3/gru_9/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_3_gru_9_while_identity*sequential_3/gru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:
M

C__inference_gru_11_layer_call_and_return_conditional_losses_1959237

inputs5
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1959148*
condR
while_cond_1959147*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
Р

н
-__inference_gru_cell_18_layer_call_fn_1962701

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:
Ќ
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957618p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџЌ
"
_user_specified_name
states/0
­
И
(__inference_gru_11_layer_call_fn_1962042
inputs_0
unknown:
	unknown_0:d
	unknown_1:
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1958371|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
"
_user_specified_name
inputs/0
М

н
-__inference_gru_cell_19_layer_call_fn_1962807

inputs
states_0
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1957956o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0


ю
%__inference_signature_wrapper_1959771
gru_9_input
unknown:	
	unknown_0:	
	unknown_1:
Ќ
	unknown_2:	Ќ
	unknown_3:
ЌЌ
	unknown_4:	dЌ
	unknown_5:
	unknown_6:d
	unknown_7:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1957548t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input
Ѓ

ђ
.__inference_sequential_3_layer_call_fn_1959817

inputs
unknown:	
	unknown_0:	
	unknown_1:
Ќ
	unknown_2:	Ќ
	unknown_3:
ЌЌ
	unknown_4:	dЌ
	unknown_5:
	unknown_6:d
	unknown_7:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959646t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Г
М
(__inference_gru_10_layer_call_fn_1961386
inputs_0
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1958033|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
с
Џ
while_cond_1959497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1959497___redundant_placeholder05
1while_while_cond_1959497___redundant_placeholder15
1while_while_cond_1959497___redundant_placeholder25
1while_while_cond_1959497___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:

Ж
(__inference_gru_11_layer_call_fn_1962075

inputs
unknown:
	unknown_0:d
	unknown_1:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959237t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
Ж

й
-__inference_gru_cell_20_layer_call_fn_1962913

inputs
states_0
unknown:
	unknown_0:d
	unknown_1:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0

Ж
(__inference_gru_11_layer_call_fn_1962064

inputs
unknown:
	unknown_0:d
	unknown_1:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959041t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњd
 
_user_specified_nameinputs
п
Џ
while_cond_1961635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961635___redundant_placeholder05
1while_while_cond_1961635___redundant_placeholder15
1while_while_cond_1961635___redundant_placeholder25
1while_while_cond_1961635___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
	
Д
gru_10_while_cond_1960480*
&gru_10_while_gru_10_while_loop_counter0
,gru_10_while_gru_10_while_maximum_iterations
gru_10_while_placeholder
gru_10_while_placeholder_1
gru_10_while_placeholder_2,
(gru_10_while_less_gru_10_strided_slice_1C
?gru_10_while_gru_10_while_cond_1960480___redundant_placeholder0C
?gru_10_while_gru_10_while_cond_1960480___redundant_placeholder1C
?gru_10_while_gru_10_while_cond_1960480___redundant_placeholder2C
?gru_10_while_gru_10_while_cond_1960480___redundant_placeholder3
gru_10_while_identity
~
gru_10/while/LessLessgru_10_while_placeholder(gru_10_while_less_gru_10_strided_slice_1*
T0*
_output_shapes
: Y
gru_10/while/IdentityIdentitygru_10/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_10_while_identitygru_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
В

I__inference_sequential_3_layer_call_and_return_conditional_losses_1959715
gru_9_input 
gru_9_1959693:	 
gru_9_1959695:	!
gru_9_1959697:
Ќ!
gru_10_1959700:	Ќ"
gru_10_1959702:
ЌЌ!
gru_10_1959704:	dЌ 
gru_11_1959707: 
gru_11_1959709:d 
gru_11_1959711:
identityЂgru_10/StatefulPartitionedCallЂgru_11/StatefulPartitionedCallЂgru_9/StatefulPartitionedCall
gru_9/StatefulPartitionedCallStatefulPartitionedCallgru_9_inputgru_9_1959693gru_9_1959695gru_9_1959697*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџњЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1958721Ѕ
gru_10/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0gru_10_1959700gru_10_1959702gru_10_1959704*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1958881І
gru_11/StatefulPartitionedCallStatefulPartitionedCall'gru_10/StatefulPartitionedCall:output:0gru_11_1959707gru_11_1959709gru_11_1959711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_11_layer_call_and_return_conditional_losses_1959041{
IdentityIdentity'gru_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњЈ
NoOpNoOp^gru_10/StatefulPartitionedCall^gru_11/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 2@
gru_10/StatefulPartitionedCallgru_10/StatefulPartitionedCall2@
gru_11/StatefulPartitionedCallgru_11/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input

К
(__inference_gru_10_layer_call_fn_1961408

inputs
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_10_layer_call_and_return_conditional_losses_1958881t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs
=

while_body_1961636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
Ѕ=

while_body_1959498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
І4

C__inference_gru_10_layer_call_and_return_conditional_losses_1958215

inputs&
gru_cell_19_1958139:	Ќ'
gru_cell_19_1958141:
ЌЌ&
gru_cell_19_1958143:	dЌ
identityЂ#gru_cell_19/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maskа
#gru_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_19_1958139gru_cell_19_1958141gru_cell_19_1958143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1958099n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_19_1958139gru_cell_19_1958141gru_cell_19_1958143*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958151*
condR
while_cond_1958150*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdt
NoOpNoOp$^gru_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2J
#gru_cell_19/StatefulPartitionedCall#gru_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
Ѕ=

while_body_1961133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_18_readvariableop_resource_0:	E
2while_gru_cell_18_matmul_readvariableop_resource_0:	H
4while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_18_readvariableop_resource:	C
0while_gru_cell_18_matmul_readvariableop_resource:	F
2while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ'while/gru_cell_18/MatMul/ReadVariableOpЂ)while/gru_cell_18/MatMul_1/ReadVariableOpЂ while/gru_cell_18/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 while/gru_cell_18/ReadVariableOpReadVariableOp+while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_18/unstackUnpack(while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0И
while/gru_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/gru_cell_18/BiasAddBiasAdd"while/gru_cell_18/MatMul:product:0"while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџl
!while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџм
while/gru_cell_18/splitSplit*while/gru_cell_18/split/split_dim:output:0"while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split 
)while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0
while/gru_cell_18/MatMul_1MatMulwhile_placeholder_21while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
while/gru_cell_18/BiasAdd_1BiasAdd$while/gru_cell_18/MatMul_1:product:0"while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџl
while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџn
#while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_18/split_1SplitV$while/gru_cell_18/BiasAdd_1:output:0 while/gru_cell_18/Const:output:0,while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
while/gru_cell_18/addAddV2 while/gru_cell_18/split:output:0"while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌr
while/gru_cell_18/SigmoidSigmoidwhile/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_1AddV2 while/gru_cell_18/split:output:1"while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_1Sigmoidwhile/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mulMulwhile/gru_cell_18/Sigmoid_1:y:0"while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_2AddV2 while/gru_cell_18/split:output:2while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌv
while/gru_cell_18/Sigmoid_2Sigmoidwhile/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_1Mulwhile/gru_cell_18/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌ\
while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_18/subSub while/gru_cell_18/sub/x:output:0while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/mul_2Mulwhile/gru_cell_18/sub:z:0while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_18/add_3AddV2while/gru_cell_18/mul_1:z:0while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/gru_cell_18/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌХ

while/NoOpNoOp(^while/gru_cell_18/MatMul/ReadVariableOp*^while/gru_cell_18/MatMul_1/ReadVariableOp!^while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_18_matmul_1_readvariableop_resource4while_gru_cell_18_matmul_1_readvariableop_resource_0"f
0while_gru_cell_18_matmul_readvariableop_resource2while_gru_cell_18_matmul_readvariableop_resource_0"X
)while_gru_cell_18_readvariableop_resource+while_gru_cell_18_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2R
'while/gru_cell_18/MatMul/ReadVariableOp'while/gru_cell_18/MatMul/ReadVariableOp2V
)while/gru_cell_18/MatMul_1/ReadVariableOp)while/gru_cell_18/MatMul_1/ReadVariableOp2D
 while/gru_cell_18/ReadVariableOp while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
п
Џ
while_cond_1958488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958488___redundant_placeholder05
1while_while_cond_1958488___redundant_placeholder15
1while_while_cond_1958488___redundant_placeholder25
1while_while_cond_1958488___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
4

C__inference_gru_11_layer_call_and_return_conditional_losses_1958371

inputs%
gru_cell_20_1958295:%
gru_cell_20_1958297:d%
gru_cell_20_1958299:
identityЂ#gru_cell_20/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskа
#gru_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_20_1958295gru_cell_20_1958297gru_cell_20_1958299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958294n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_20_1958295gru_cell_20_1958297gru_cell_20_1958299*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1958307*
condR
while_cond_1958306*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
NoOpNoOp$^gru_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 2J
#gru_cell_20/StatefulPartitionedCall#gru_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
 
_user_specified_nameinputs
=

while_body_1962139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
В

ї
.__inference_sequential_3_layer_call_fn_1959071
gru_9_input
unknown:	
	unknown_0:	
	unknown_1:
Ќ
	unknown_2:	Ќ
	unknown_3:
ЌЌ
	unknown_4:	dЌ
	unknown_5:
	unknown_6:d
	unknown_7:
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959050t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:џџџџџџџџџњ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_namegru_9_input
=

while_body_1958952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
п
Џ
while_cond_1958306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958306___redundant_placeholder05
1while_while_cond_1958306___redundant_placeholder15
1while_while_cond_1958306___redundant_placeholder25
1while_while_cond_1958306___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
 
Ж
while_body_1958489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_20_1958511_0:-
while_gru_cell_20_1958513_0:d-
while_gru_cell_20_1958515_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_20_1958511:+
while_gru_cell_20_1958513:d+
while_gru_cell_20_1958515:Ђ)while/gru_cell_20/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
)while/gru_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_20_1958511_0while_gru_cell_20_1958513_0while_gru_cell_20_1958515_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958437л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

while/NoOpNoOp*^while/gru_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_20_1958511while_gru_cell_20_1958511_0"8
while_gru_cell_20_1958513while_gru_cell_20_1958513_0"8
while_gru_cell_20_1958515while_gru_cell_20_1958515_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2V
)while/gru_cell_20/StatefulPartitionedCall)while/gru_cell_20/StatefulPartitionedCall: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
=

while_body_1962445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
п
Џ
while_cond_1961788
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961788___redundant_placeholder05
1while_while_cond_1961788___redundant_placeholder15
1while_while_cond_1961788___redundant_placeholder25
1while_while_cond_1961788___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1958791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1958791___redundant_placeholder05
1while_while_cond_1958791___redundant_placeholder15
1while_while_cond_1958791___redundant_placeholder25
1while_while_cond_1958791___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1957968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1957968___redundant_placeholder05
1while_while_cond_1957968___redundant_placeholder15
1while_while_cond_1957968___redundant_placeholder25
1while_while_cond_1957968___redundant_placeholder3
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
-: : : : :џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
=

while_body_1959148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_20_readvariableop_resource_0:D
2while_gru_cell_20_matmul_readvariableop_resource_0:dF
4while_gru_cell_20_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_20_readvariableop_resource:B
0while_gru_cell_20_matmul_readvariableop_resource:dD
2while_gru_cell_20_matmul_1_readvariableop_resource:Ђ'while/gru_cell_20/MatMul/ReadVariableOpЂ)while/gru_cell_20/MatMul_1/ReadVariableOpЂ while/gru_cell_20/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџd*
element_dtype0
 while/gru_cell_20/ReadVariableOpReadVariableOp+while_gru_cell_20_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/unstackUnpack(while/gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_20/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_20_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0З
while/gru_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/BiasAddBiasAdd"while/gru_cell_20/MatMul:product:0"while/gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
!while/gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_20/splitSplit*while/gru_cell_20/split/split_dim:output:0"while/gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
)while/gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_20/MatMul_1MatMulwhile_placeholder_21while/gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
while/gru_cell_20/BiasAdd_1BiasAdd$while/gru_cell_20/MatMul_1:product:0"while/gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
while/gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџn
#while/gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_20/split_1SplitV$while/gru_cell_20/BiasAdd_1:output:0 while/gru_cell_20/Const:output:0,while/gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
while/gru_cell_20/addAddV2 while/gru_cell_20/split:output:0"while/gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
while/gru_cell_20/SigmoidSigmoidwhile/gru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_1AddV2 while/gru_cell_20/split:output:1"while/gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/Sigmoid_1Sigmoidwhile/gru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mulMulwhile/gru_cell_20/Sigmoid_1:y:0"while/gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_2AddV2 while/gru_cell_20/split:output:2while/gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
while/gru_cell_20/SoftplusSoftpluswhile/gru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_1Mulwhile/gru_cell_20/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ\
while/gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_20/subSub while/gru_cell_20/sub/x:output:0while/gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/mul_2Mulwhile/gru_cell_20/sub:z:0(while/gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/gru_cell_20/add_3AddV2while/gru_cell_20/mul_1:z:0while/gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_20/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_20/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ

while/NoOpNoOp(^while/gru_cell_20/MatMul/ReadVariableOp*^while/gru_cell_20/MatMul_1/ReadVariableOp!^while/gru_cell_20/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_20_matmul_1_readvariableop_resource4while_gru_cell_20_matmul_1_readvariableop_resource_0"f
0while_gru_cell_20_matmul_readvariableop_resource2while_gru_cell_20_matmul_readvariableop_resource_0"X
)while_gru_cell_20_readvariableop_resource+while_gru_cell_20_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2R
'while/gru_cell_20/MatMul/ReadVariableOp'while/gru_cell_20/MatMul/ReadVariableOp2V
)while/gru_cell_20/MatMul_1/ReadVariableOp)while/gru_cell_20/MatMul_1/ReadVariableOp2D
 while/gru_cell_20/ReadVariableOp while/gru_cell_20/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
	
Д
gru_11_while_cond_1960629*
&gru_11_while_gru_11_while_loop_counter0
,gru_11_while_gru_11_while_maximum_iterations
gru_11_while_placeholder
gru_11_while_placeholder_1
gru_11_while_placeholder_2,
(gru_11_while_less_gru_11_strided_slice_1C
?gru_11_while_gru_11_while_cond_1960629___redundant_placeholder0C
?gru_11_while_gru_11_while_cond_1960629___redundant_placeholder1C
?gru_11_while_gru_11_while_cond_1960629___redundant_placeholder2C
?gru_11_while_gru_11_while_cond_1960629___redundant_placeholder3
gru_11_while_identity
~
gru_11/while/LessLessgru_11_while_placeholder(gru_11_while_less_gru_11_strided_slice_1*
T0*
_output_shapes
: Y
gru_11/while/IdentityIdentitygru_11/while/Less:z:0*
T0
*
_output_shapes
: "7
gru_11_while_identitygru_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1962444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1962444___redundant_placeholder05
1while_while_cond_1962444___redundant_placeholder15
1while_while_cond_1962444___redundant_placeholder25
1while_while_cond_1962444___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

й
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1963005

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpf
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
:џџџџџџџџџh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџQ
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
п
Џ
while_cond_1959147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1959147___redundant_placeholder05
1while_while_cond_1959147___redundant_placeholder15
1while_while_cond_1959147___redundant_placeholder25
1while_while_cond_1959147___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ђM

B__inference_gru_9_layer_call_and_return_conditional_losses_1961069
inputs_06
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile=
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1960980*
condR
while_cond_1960979*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ъ
ѓ
#__inference__traced_restore_1963242
file_prefix<
)assignvariableop_gru_9_gru_cell_18_kernel:	I
5assignvariableop_1_gru_9_gru_cell_18_recurrent_kernel:
Ќ<
)assignvariableop_2_gru_9_gru_cell_18_bias:	@
,assignvariableop_3_gru_10_gru_cell_19_kernel:
ЌЌI
6assignvariableop_4_gru_10_gru_cell_19_recurrent_kernel:	dЌ=
*assignvariableop_5_gru_10_gru_cell_19_bias:	Ќ>
,assignvariableop_6_gru_11_gru_cell_20_kernel:dH
6assignvariableop_7_gru_11_gru_cell_20_recurrent_kernel:<
*assignvariableop_8_gru_11_gru_cell_20_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: F
3assignvariableop_16_adam_gru_9_gru_cell_18_kernel_m:	Q
=assignvariableop_17_adam_gru_9_gru_cell_18_recurrent_kernel_m:
ЌD
1assignvariableop_18_adam_gru_9_gru_cell_18_bias_m:	H
4assignvariableop_19_adam_gru_10_gru_cell_19_kernel_m:
ЌЌQ
>assignvariableop_20_adam_gru_10_gru_cell_19_recurrent_kernel_m:	dЌE
2assignvariableop_21_adam_gru_10_gru_cell_19_bias_m:	ЌF
4assignvariableop_22_adam_gru_11_gru_cell_20_kernel_m:dP
>assignvariableop_23_adam_gru_11_gru_cell_20_recurrent_kernel_m:D
2assignvariableop_24_adam_gru_11_gru_cell_20_bias_m:F
3assignvariableop_25_adam_gru_9_gru_cell_18_kernel_v:	Q
=assignvariableop_26_adam_gru_9_gru_cell_18_recurrent_kernel_v:
ЌD
1assignvariableop_27_adam_gru_9_gru_cell_18_bias_v:	H
4assignvariableop_28_adam_gru_10_gru_cell_19_kernel_v:
ЌЌQ
>assignvariableop_29_adam_gru_10_gru_cell_19_recurrent_kernel_v:	dЌE
2assignvariableop_30_adam_gru_10_gru_cell_19_bias_v:	ЌF
4assignvariableop_31_adam_gru_11_gru_cell_20_kernel_v:dP
>assignvariableop_32_adam_gru_11_gru_cell_20_recurrent_kernel_v:D
2assignvariableop_33_adam_gru_11_gru_cell_20_bias_v:
identity_35ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Д
valueЊBЇ#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B а
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ђ
_output_shapes
:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp)assignvariableop_gru_9_gru_cell_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_1AssignVariableOp5assignvariableop_1_gru_9_gru_cell_18_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp)assignvariableop_2_gru_9_gru_cell_18_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp,assignvariableop_3_gru_10_gru_cell_19_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_4AssignVariableOp6assignvariableop_4_gru_10_gru_cell_19_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_10_gru_cell_19_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_gru_11_gru_cell_20_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_7AssignVariableOp6assignvariableop_7_gru_11_gru_cell_20_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_gru_11_gru_cell_20_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_16AssignVariableOp3assignvariableop_16_adam_gru_9_gru_cell_18_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_17AssignVariableOp=assignvariableop_17_adam_gru_9_gru_cell_18_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_gru_9_gru_cell_18_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_gru_10_gru_cell_19_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_gru_10_gru_cell_19_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_gru_10_gru_cell_19_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_gru_11_gru_cell_20_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_gru_11_gru_cell_20_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_gru_11_gru_cell_20_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_gru_9_gru_cell_18_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_gru_9_gru_cell_18_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_gru_9_gru_cell_18_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_gru_10_gru_cell_19_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_gru_10_gru_cell_19_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_gru_10_gru_cell_19_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_gru_11_gru_cell_20_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_gru_11_gru_cell_20_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_gru_11_gru_cell_20_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Л
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: Ј
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
=

while_body_1961483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_19_readvariableop_resource_0:	ЌF
2while_gru_cell_19_matmul_readvariableop_resource_0:
ЌЌG
4while_gru_cell_19_matmul_1_readvariableop_resource_0:	dЌ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_19_readvariableop_resource:	ЌD
0while_gru_cell_19_matmul_readvariableop_resource:
ЌЌE
2while_gru_cell_19_matmul_1_readvariableop_resource:	dЌЂ'while/gru_cell_19/MatMul/ReadVariableOpЂ)while/gru_cell_19/MatMul_1/ReadVariableOpЂ while/gru_cell_19/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype0
 while/gru_cell_19/ReadVariableOpReadVariableOp+while_gru_cell_19_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype0
while/gru_cell_19/unstackUnpack(while/gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
'while/gru_cell_19/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
ЌЌ*
dtype0И
while/gru_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
while/gru_cell_19/BiasAddBiasAdd"while/gru_cell_19/MatMul:product:0"while/gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌl
!while/gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџй
while/gru_cell_19/splitSplit*while/gru_cell_19/split/split_dim:output:0"while/gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
)while/gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dЌ*
dtype0
while/gru_cell_19/MatMul_1MatMulwhile_placeholder_21while/gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌЃ
while/gru_cell_19/BiasAdd_1BiasAdd$while/gru_cell_19/MatMul_1:product:0"while/gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌl
while/gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџn
#while/gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
while/gru_cell_19/split_1SplitV$while/gru_cell_19/BiasAdd_1:output:0 while/gru_cell_19/Const:output:0,while/gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
while/gru_cell_19/addAddV2 while/gru_cell_19/split:output:0"while/gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdq
while/gru_cell_19/SigmoidSigmoidwhile/gru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_1AddV2 while/gru_cell_19/split:output:1"while/gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_1Sigmoidwhile/gru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mulMulwhile/gru_cell_19/Sigmoid_1:y:0"while/gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_2AddV2 while/gru_cell_19/split:output:2while/gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdu
while/gru_cell_19/Sigmoid_2Sigmoidwhile/gru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_1Mulwhile/gru_cell_19/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџd\
while/gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_19/subSub while/gru_cell_19/sub/x:output:0while/gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/mul_2Mulwhile/gru_cell_19/sub:z:0while/gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/gru_cell_19/add_3AddV2while/gru_cell_19/mul_1:z:0while/gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_19/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_19/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdХ

while/NoOpNoOp(^while/gru_cell_19/MatMul/ReadVariableOp*^while/gru_cell_19/MatMul_1/ReadVariableOp!^while/gru_cell_19/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_19_matmul_1_readvariableop_resource4while_gru_cell_19_matmul_1_readvariableop_resource_0"f
0while_gru_cell_19_matmul_readvariableop_resource2while_gru_cell_19_matmul_readvariableop_resource_0"X
)while_gru_cell_19_readvariableop_resource+while_gru_cell_19_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџd: : : : : 2R
'while/gru_cell_19/MatMul/ReadVariableOp'while/gru_cell_19/MatMul/ReadVariableOp2V
)while/gru_cell_19/MatMul_1/ReadVariableOp)while/gru_cell_19/MatMul_1/ReadVariableOp2D
 while/gru_cell_19/ReadVariableOp while/gru_cell_19/ReadVariableOp: 
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
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
оM

C__inference_gru_10_layer_call_and_return_conditional_losses_1961572
inputs_06
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961483*
condR
while_cond_1961482*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
с
Џ
while_cond_1961132
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1961132___redundant_placeholder05
1while_while_cond_1961132___redundant_placeholder15
1while_while_cond_1961132___redundant_placeholder25
1while_while_cond_1961132___redundant_placeholder3
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
.: : : : :џџџџџџџџџЌ: ::::: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
:

з
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958437

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpf
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
:џџџџџџџџџh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџQ
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Ѕ
л
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957618

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
Ќ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџІ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌR
	Sigmoid_2Sigmoid	add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:џџџџџџџџџЌJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ:џџџџџџџџџЌ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_namestates

з
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1958294

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:d2
 matmul_1_readvariableop_resource:
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpf
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
:џџџџџџџџџh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџQ
SoftplusSoftplus	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_2Mulsub:z:0Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџd:џџџџџџџџџ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
ІM

C__inference_gru_10_layer_call_and_return_conditional_losses_1959412

inputs6
#gru_cell_19_readvariableop_resource:	Ќ>
*gru_cell_19_matmul_readvariableop_resource:
ЌЌ?
,gru_cell_19_matmul_1_readvariableop_resource:	dЌ
identityЂ!gru_cell_19/MatMul/ReadVariableOpЂ#gru_cell_19/MatMul_1/ReadVariableOpЂgru_cell_19/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:њџџџџџџџџџЌD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask
gru_cell_19/ReadVariableOpReadVariableOp#gru_cell_19_readvariableop_resource*
_output_shapes
:	Ќ*
dtype0y
gru_cell_19/unstackUnpack"gru_cell_19/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
num
!gru_cell_19/MatMul/ReadVariableOpReadVariableOp*gru_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0
gru_cell_19/MatMulMatMulstrided_slice_2:output:0)gru_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAddBiasAddgru_cell_19/MatMul:product:0gru_cell_19/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_19/splitSplit$gru_cell_19/split/split_dim:output:0gru_cell_19/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#gru_cell_19/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0
gru_cell_19/MatMul_1MatMulzeros:output:0+gru_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_19/BiasAdd_1BiasAddgru_cell_19/MatMul_1:product:0gru_cell_19/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_19/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџh
gru_cell_19/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_19/split_1SplitVgru_cell_19/BiasAdd_1:output:0gru_cell_19/Const:output:0&gru_cell_19/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
gru_cell_19/addAddV2gru_cell_19/split:output:0gru_cell_19/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
gru_cell_19/SigmoidSigmoidgru_cell_19/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/add_1AddV2gru_cell_19/split:output:1gru_cell_19/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_1Sigmoidgru_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
gru_cell_19/mulMulgru_cell_19/Sigmoid_1:y:0gru_cell_19/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd}
gru_cell_19/add_2AddV2gru_cell_19/split:output:2gru_cell_19/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdi
gru_cell_19/Sigmoid_2Sigmoidgru_cell_19/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџds
gru_cell_19/mul_1Mulgru_cell_19/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
gru_cell_19/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_19/subSubgru_cell_19/sub/x:output:0gru_cell_19/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/mul_2Mulgru_cell_19/sub:z:0gru_cell_19/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdz
gru_cell_19/add_3AddV2gru_cell_19/mul_1:z:0gru_cell_19/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_19_readvariableop_resource*gru_cell_19_matmul_readvariableop_resource,gru_cell_19_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1959323*
condR
while_cond_1959322*8
output_shapes'
%: : : : :џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:њџџџџџџџџџd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџњd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњdЕ
NoOpNoOp"^gru_cell_19/MatMul/ReadVariableOp$^gru_cell_19/MatMul_1/ReadVariableOp^gru_cell_19/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџњЌ: : : 2F
!gru_cell_19/MatMul/ReadVariableOp!gru_cell_19/MatMul/ReadVariableOp2J
#gru_cell_19/MatMul_1/ReadVariableOp#gru_cell_19/MatMul_1/ReadVariableOp28
gru_cell_19/ReadVariableOpgru_cell_19/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:џџџџџџџџџњЌ
 
_user_specified_nameinputs
 
О
while_body_1957631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_18_1957653_0:	.
while_gru_cell_18_1957655_0:	/
while_gru_cell_18_1957657_0:
Ќ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_18_1957653:	,
while_gru_cell_18_1957655:	-
while_gru_cell_18_1957657:
ЌЂ)while/gru_cell_18/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
)while/gru_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_18_1957653_0while_gru_cell_18_1957655_0while_gru_cell_18_1957657_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџЌ:џџџџџџџџџЌ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1957618л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/gru_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌx

while/NoOpNoOp*^while/gru_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_18_1957653while_gru_cell_18_1957653_0"8
while_gru_cell_18_1957655while_gru_cell_18_1957655_0"8
while_gru_cell_18_1957657while_gru_cell_18_1957657_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2V
)while/gru_cell_18/StatefulPartitionedCall)while/gru_cell_18/StatefulPartitionedCall: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 

н
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962860

inputs
states_0*
readvariableop_resource:	Ќ2
matmul_readvariableop_resource:
ЌЌ3
 matmul_1_readvariableop_resource:	dЌ
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	Ќ*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:Ќ:Ќ*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЌЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dЌ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџdQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
mul_2Mulsub:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0
жM

C__inference_gru_11_layer_call_and_return_conditional_losses_1962381
inputs_05
#gru_cell_20_readvariableop_resource:<
*gru_cell_20_matmul_readvariableop_resource:d>
,gru_cell_20_matmul_1_readvariableop_resource:
identityЂ!gru_cell_20/MatMul/ReadVariableOpЂ#gru_cell_20/MatMul_1/ReadVariableOpЂgru_cell_20/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџdD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask~
gru_cell_20/ReadVariableOpReadVariableOp#gru_cell_20_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_20/unstackUnpack"gru_cell_20/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_20/MatMul/ReadVariableOpReadVariableOp*gru_cell_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
gru_cell_20/MatMulMatMulstrided_slice_2:output:0)gru_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAddBiasAddgru_cell_20/MatMul:product:0gru_cell_20/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
gru_cell_20/splitSplit$gru_cell_20/split/split_dim:output:0gru_cell_20/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
#gru_cell_20/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_20/MatMul_1MatMulzeros:output:0+gru_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/BiasAdd_1BiasAddgru_cell_20/MatMul_1:product:0gru_cell_20/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
gru_cell_20/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      џџџџh
gru_cell_20/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
gru_cell_20/split_1SplitVgru_cell_20/BiasAdd_1:output:0gru_cell_20/Const:output:0&gru_cell_20/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
gru_cell_20/addAddV2gru_cell_20/split:output:0gru_cell_20/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
gru_cell_20/SigmoidSigmoidgru_cell_20/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/add_1AddV2gru_cell_20/split:output:1gru_cell_20/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/Sigmoid_1Sigmoidgru_cell_20/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mulMulgru_cell_20/Sigmoid_1:y:0gru_cell_20/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
gru_cell_20/add_2AddV2gru_cell_20/split:output:2gru_cell_20/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
gru_cell_20/SoftplusSoftplusgru_cell_20/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџs
gru_cell_20/mul_1Mulgru_cell_20/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
gru_cell_20/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_20/subSubgru_cell_20/sub/x:output:0gru_cell_20/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
gru_cell_20/mul_2Mulgru_cell_20/sub:z:0"gru_cell_20/Softplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџz
gru_cell_20/add_3AddV2gru_cell_20/mul_1:z:0gru_cell_20/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_20_readvariableop_resource*gru_cell_20_matmul_readvariableop_resource,gru_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1962292*
condR
while_cond_1962291*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЕ
NoOpNoOp"^gru_cell_20/MatMul/ReadVariableOp$^gru_cell_20/MatMul_1/ReadVariableOp^gru_cell_20/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџd: : : 2F
!gru_cell_20/MatMul/ReadVariableOp!gru_cell_20/MatMul/ReadVariableOp2J
#gru_cell_20/MatMul_1/ReadVariableOp#gru_cell_20/MatMul_1/ReadVariableOp28
gru_cell_20/ReadVariableOpgru_cell_20/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
"
_user_specified_name
inputs/0
щD
П	
gru_9_while_body_1959881(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0D
1gru_9_while_gru_cell_18_readvariableop_resource_0:	K
8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0:	N
:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0:
Ќ
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensorB
/gru_9_while_gru_cell_18_readvariableop_resource:	I
6gru_9_while_gru_cell_18_matmul_readvariableop_resource:	L
8gru_9_while_gru_cell_18_matmul_1_readvariableop_resource:
ЌЂ-gru_9/while/gru_cell_18/MatMul/ReadVariableOpЂ/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpЂ&gru_9/while/gru_cell_18/ReadVariableOp
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ф
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
&gru_9/while/gru_cell_18/ReadVariableOpReadVariableOp1gru_9_while_gru_cell_18_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_9/while/gru_cell_18/unstackUnpack.gru_9/while/gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numЇ
-gru_9/while/gru_cell_18/MatMul/ReadVariableOpReadVariableOp8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ъ
gru_9/while/gru_cell_18/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:05gru_9/while/gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџБ
gru_9/while/gru_cell_18/BiasAddBiasAdd(gru_9/while/gru_cell_18/MatMul:product:0(gru_9/while/gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџr
'gru_9/while/gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџю
gru_9/while/gru_cell_18/splitSplit0gru_9/while/gru_cell_18/split/split_dim:output:0(gru_9/while/gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitЌ
/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype0Б
 gru_9/while/gru_cell_18/MatMul_1MatMulgru_9_while_placeholder_27gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
!gru_9/while/gru_cell_18/BiasAdd_1BiasAdd*gru_9/while/gru_cell_18/MatMul_1:product:0(gru_9/while/gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџr
gru_9/while/gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџt
)gru_9/while/gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЉ
gru_9/while/gru_cell_18/split_1SplitV*gru_9/while/gru_cell_18/BiasAdd_1:output:0&gru_9/while/gru_cell_18/Const:output:02gru_9/while/gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_splitЉ
gru_9/while/gru_cell_18/addAddV2&gru_9/while/gru_cell_18/split:output:0(gru_9/while/gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_9/while/gru_cell_18/SigmoidSigmoidgru_9/while/gru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌЋ
gru_9/while/gru_cell_18/add_1AddV2&gru_9/while/gru_cell_18/split:output:1(gru_9/while/gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌ
!gru_9/while/gru_cell_18/Sigmoid_1Sigmoid!gru_9/while/gru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌІ
gru_9/while/gru_cell_18/mulMul%gru_9/while/gru_cell_18/Sigmoid_1:y:0(gru_9/while/gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌЂ
gru_9/while/gru_cell_18/add_2AddV2&gru_9/while/gru_cell_18/split:output:2gru_9/while/gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
!gru_9/while/gru_cell_18/Sigmoid_2Sigmoid!gru_9/while/gru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/mul_1Mul#gru_9/while/gru_cell_18/Sigmoid:y:0gru_9_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџЌb
gru_9/while/gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
gru_9/while/gru_cell_18/subSub&gru_9/while/gru_cell_18/sub/x:output:0#gru_9/while/gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/mul_2Mulgru_9/while/gru_cell_18/sub:z:0%gru_9/while/gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_9/while/gru_cell_18/add_3AddV2!gru_9/while/gru_cell_18/mul_1:z:0!gru_9/while/gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌм
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder!gru_9/while/gru_cell_18/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвS
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: U
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations^gru_9/while/NoOp*
T0*
_output_shapes
: k
gru_9/while/Identity_2Identitygru_9/while/add:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_4Identity!gru_9/while/gru_cell_18/add_3:z:0^gru_9/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌн
gru_9/while/NoOpNoOp.^gru_9/while/gru_cell_18/MatMul/ReadVariableOp0^gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp'^gru_9/while/gru_cell_18/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"v
8gru_9_while_gru_cell_18_matmul_1_readvariableop_resource:gru_9_while_gru_cell_18_matmul_1_readvariableop_resource_0"r
6gru_9_while_gru_cell_18_matmul_readvariableop_resource8gru_9_while_gru_cell_18_matmul_readvariableop_resource_0"d
/gru_9_while_gru_cell_18_readvariableop_resource1gru_9_while_gru_cell_18_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"Р
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџЌ: : : : : 2^
-gru_9/while/gru_cell_18/MatMul/ReadVariableOp-gru_9/while/gru_cell_18/MatMul/ReadVariableOp2b
/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp/gru_9/while/gru_cell_18/MatMul_1/ReadVariableOp2P
&gru_9/while/gru_cell_18/ReadVariableOp&gru_9/while/gru_cell_18/ReadVariableOp: 
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
:џџџџџџџџџЌ:

_output_shapes
: :

_output_shapes
: 
КM

B__inference_gru_9_layer_call_and_return_conditional_losses_1961222

inputs6
#gru_cell_18_readvariableop_resource:	=
*gru_cell_18_matmul_readvariableop_resource:	@
,gru_cell_18_matmul_1_readvariableop_resource:
Ќ
identityЂ!gru_cell_18/MatMul/ReadVariableOpЂ#gru_cell_18/MatMul_1/ReadVariableOpЂgru_cell_18/ReadVariableOpЂwhile;
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
valueB:б
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
B :Ќs
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
:џџџџџџџџџЌc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:њџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell_18/ReadVariableOpReadVariableOp#gru_cell_18_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_18/unstackUnpack"gru_cell_18/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_18/MatMul/ReadVariableOpReadVariableOp*gru_cell_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_18/MatMulMatMulstrided_slice_2:output:0)gru_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAddBiasAddgru_cell_18/MatMul:product:0gru_cell_18/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЪ
gru_cell_18/splitSplit$gru_cell_18/split/split_dim:output:0gru_cell_18/BiasAdd:output:0*
T0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
#gru_cell_18/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype0
gru_cell_18/MatMul_1MatMulzeros:output:0+gru_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
gru_cell_18/BiasAdd_1BiasAddgru_cell_18/MatMul_1:product:0gru_cell_18/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџf
gru_cell_18/ConstConst*
_output_shapes
:*
dtype0*!
valueB",  ,  џџџџh
gru_cell_18/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџљ
gru_cell_18/split_1SplitVgru_cell_18/BiasAdd_1:output:0gru_cell_18/Const:output:0&gru_cell_18/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:џџџџџџџџџЌ:џџџџџџџџџЌ:џџџџџџџџџЌ*
	num_split
gru_cell_18/addAddV2gru_cell_18/split:output:0gru_cell_18/split_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌf
gru_cell_18/SigmoidSigmoidgru_cell_18/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/add_1AddV2gru_cell_18/split:output:1gru_cell_18/split_1:output:1*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_1Sigmoidgru_cell_18/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
gru_cell_18/mulMulgru_cell_18/Sigmoid_1:y:0gru_cell_18/split_1:output:2*
T0*(
_output_shapes
:џџџџџџџџџЌ~
gru_cell_18/add_2AddV2gru_cell_18/split:output:2gru_cell_18/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌj
gru_cell_18/Sigmoid_2Sigmoidgru_cell_18/add_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌt
gru_cell_18/mul_1Mulgru_cell_18/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌV
gru_cell_18/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
gru_cell_18/subSubgru_cell_18/sub/x:output:0gru_cell_18/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/mul_2Mulgru_cell_18/sub:z:0gru_cell_18/Sigmoid_2:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ{
gru_cell_18/add_3AddV2gru_cell_18/mul_1:z:0gru_cell_18/mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_18_readvariableop_resource*gru_cell_18_matmul_readvariableop_resource,gru_cell_18_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1961133*
condR
while_cond_1961132*9
output_shapes(
&: : : : :џџџџџџџџџЌ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  Ф
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:њџџџџџџџџџЌ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџњЌ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџњЌЕ
NoOpNoOp"^gru_cell_18/MatMul/ReadVariableOp$^gru_cell_18/MatMul_1/ReadVariableOp^gru_cell_18/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : 2F
!gru_cell_18/MatMul/ReadVariableOp!gru_cell_18/MatMul/ReadVariableOp2J
#gru_cell_18/MatMul_1/ReadVariableOp#gru_cell_18/MatMul_1/ReadVariableOp28
gru_cell_18/ReadVariableOpgru_cell_18/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
М

н
-__inference_gru_cell_19_layer_call_fn_1962821

inputs
states_0
unknown:	Ќ
	unknown_0:
ЌЌ
	unknown_1:	dЌ
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1958099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
H
gru_9_input9
serving_default_gru_9_input:0џџџџџџџџџњ?
gru_115
StatefulPartitionedCall:0џџџџџџџџџњtensorflow/serving/predict:ЋЃ
л
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
к
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
к
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
к
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
Ъ
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
ю
6trace_0
7trace_1
8trace_2
9trace_32
.__inference_sequential_3_layer_call_fn_1959071
.__inference_sequential_3_layer_call_fn_1959794
.__inference_sequential_3_layer_call_fn_1959817
.__inference_sequential_3_layer_call_fn_1959690Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 z6trace_0z7trace_1z8trace_2z9trace_3
к
:trace_0
;trace_1
<trace_2
=trace_32я
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960268
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960719
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959715
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959740Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 z:trace_0z;trace_1z<trace_2z=trace_3
бBЮ
"__inference__wrapped_model_1957548gru_9_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

>iter

?beta_1

@beta_2
	Adecay
Blearning_rate(mЃ)mЄ*mЅ+mІ,mЇ-mЈ.mЉ/mЊ0mЋ(vЌ)v­*vЎ+vЏ,vА-vБ.vВ/vГ0vД"
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
Й

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
ч
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32ќ
'__inference_gru_9_layer_call_fn_1960730
'__inference_gru_9_layer_call_fn_1960741
'__inference_gru_9_layer_call_fn_1960752
'__inference_gru_9_layer_call_fn_1960763е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
г
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_32ш
B__inference_gru_9_layer_call_and_return_conditional_losses_1960916
B__inference_gru_9_layer_call_and_return_conditional_losses_1961069
B__inference_gru_9_layer_call_and_return_conditional_losses_1961222
B__inference_gru_9_layer_call_and_return_conditional_losses_1961375е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zNtrace_0zOtrace_1zPtrace_2zQtrace_3
"
_generic_user_object
ш
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
Й

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
ы
_trace_0
`trace_1
atrace_2
btrace_32
(__inference_gru_10_layer_call_fn_1961386
(__inference_gru_10_layer_call_fn_1961397
(__inference_gru_10_layer_call_fn_1961408
(__inference_gru_10_layer_call_fn_1961419е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 z_trace_0z`trace_1zatrace_2zbtrace_3
з
ctrace_0
dtrace_1
etrace_2
ftrace_32ь
C__inference_gru_10_layer_call_and_return_conditional_losses_1961572
C__inference_gru_10_layer_call_and_return_conditional_losses_1961725
C__inference_gru_10_layer_call_and_return_conditional_losses_1961878
C__inference_gru_10_layer_call_and_return_conditional_losses_1962031е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zctrace_0zdtrace_1zetrace_2zftrace_3
"
_generic_user_object
ш
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
Й

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
ы
ttrace_0
utrace_1
vtrace_2
wtrace_32
(__inference_gru_11_layer_call_fn_1962042
(__inference_gru_11_layer_call_fn_1962053
(__inference_gru_11_layer_call_fn_1962064
(__inference_gru_11_layer_call_fn_1962075е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
з
xtrace_0
ytrace_1
ztrace_2
{trace_32ь
C__inference_gru_11_layer_call_and_return_conditional_losses_1962228
C__inference_gru_11_layer_call_and_return_conditional_losses_1962381
C__inference_gru_11_layer_call_and_return_conditional_losses_1962534
C__inference_gru_11_layer_call_and_return_conditional_losses_1962687е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zxtrace_0zytrace_1zztrace_2z{trace_3
"
_generic_user_object
ы
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

.kernel
/recurrent_kernel
0bias"
_tf_keras_layer
 "
trackable_list_wrapper
+:)	2gru_9/gru_cell_18/kernel
6:4
Ќ2"gru_9/gru_cell_18/recurrent_kernel
):'	2gru_9/gru_cell_18/bias
-:+
ЌЌ2gru_10/gru_cell_19/kernel
6:4	dЌ2#gru_10/gru_cell_19/recurrent_kernel
*:(	Ќ2gru_10/gru_cell_19/bias
+:)d2gru_11/gru_cell_20/kernel
5:32#gru_11/gru_cell_20/recurrent_kernel
):'2gru_11/gru_cell_20/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_3_layer_call_fn_1959071gru_9_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_3_layer_call_fn_1959794inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B§
.__inference_sequential_3_layer_call_fn_1959817inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
.__inference_sequential_3_layer_call_fn_1959690gru_9_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960268inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960719inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
 B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959715gru_9_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
 B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959740gru_9_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
аBЭ
%__inference_signature_wrapper_1959771gru_9_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
B
'__inference_gru_9_layer_call_fn_1960730inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1960741inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1960752inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1960763inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЋBЈ
B__inference_gru_9_layer_call_and_return_conditional_losses_1960916inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЋBЈ
B__inference_gru_9_layer_call_and_return_conditional_losses_1961069inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЉBІ
B__inference_gru_9_layer_call_and_return_conditional_losses_1961222inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЉBІ
B__inference_gru_9_layer_call_and_return_conditional_losses_1961375inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
к
trace_0
trace_12
-__inference_gru_cell_18_layer_call_fn_1962701
-__inference_gru_cell_18_layer_call_fn_1962715О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12е
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962754
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962793О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
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
B
(__inference_gru_10_layer_call_fn_1961386inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_10_layer_call_fn_1961397inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_10_layer_call_fn_1961408inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_10_layer_call_fn_1961419inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЌBЉ
C__inference_gru_10_layer_call_and_return_conditional_losses_1961572inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЌBЉ
C__inference_gru_10_layer_call_and_return_conditional_losses_1961725inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЊBЇ
C__inference_gru_10_layer_call_and_return_conditional_losses_1961878inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЊBЇ
C__inference_gru_10_layer_call_and_return_conditional_losses_1962031inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
к
trace_0
trace_12
-__inference_gru_cell_19_layer_call_fn_1962807
-__inference_gru_cell_19_layer_call_fn_1962821О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12е
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962860
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962899О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
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
B
(__inference_gru_11_layer_call_fn_1962042inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_11_layer_call_fn_1962053inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_11_layer_call_fn_1962064inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_gru_11_layer_call_fn_1962075inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЌBЉ
C__inference_gru_11_layer_call_and_return_conditional_losses_1962228inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЌBЉ
C__inference_gru_11_layer_call_and_return_conditional_losses_1962381inputs/0"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЊBЇ
C__inference_gru_11_layer_call_and_return_conditional_losses_1962534inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЊBЇ
C__inference_gru_11_layer_call_and_return_conditional_losses_1962687inputs"е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
к
trace_0
trace_12
-__inference_gru_cell_20_layer_call_fn_1962913
-__inference_gru_cell_20_layer_call_fn_1962927О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12е
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1962966
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1963005О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
R
	variables
 	keras_api

Ёtotal

Ђcount"
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
B
-__inference_gru_cell_18_layer_call_fn_1962701inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
-__inference_gru_cell_18_layer_call_fn_1962715inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962754inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962793inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
B
-__inference_gru_cell_19_layer_call_fn_1962807inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
-__inference_gru_cell_19_layer_call_fn_1962821inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962860inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962899inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
B
-__inference_gru_cell_20_layer_call_fn_1962913inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
-__inference_gru_cell_20_layer_call_fn_1962927inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1962966inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЂB
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1963005inputsstates/0"О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
0
Ё0
Ђ1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0:.	2Adam/gru_9/gru_cell_18/kernel/m
;:9
Ќ2)Adam/gru_9/gru_cell_18/recurrent_kernel/m
.:,	2Adam/gru_9/gru_cell_18/bias/m
2:0
ЌЌ2 Adam/gru_10/gru_cell_19/kernel/m
;:9	dЌ2*Adam/gru_10/gru_cell_19/recurrent_kernel/m
/:-	Ќ2Adam/gru_10/gru_cell_19/bias/m
0:.d2 Adam/gru_11/gru_cell_20/kernel/m
::82*Adam/gru_11/gru_cell_20/recurrent_kernel/m
.:,2Adam/gru_11/gru_cell_20/bias/m
0:.	2Adam/gru_9/gru_cell_18/kernel/v
;:9
Ќ2)Adam/gru_9/gru_cell_18/recurrent_kernel/v
.:,	2Adam/gru_9/gru_cell_18/bias/v
2:0
ЌЌ2 Adam/gru_10/gru_cell_19/kernel/v
;:9	dЌ2*Adam/gru_10/gru_cell_19/recurrent_kernel/v
/:-	Ќ2Adam/gru_10/gru_cell_19/bias/v
0:.d2 Adam/gru_11/gru_cell_20/kernel/v
::82*Adam/gru_11/gru_cell_20/recurrent_kernel/v
.:,2Adam/gru_11/gru_cell_20/bias/vЂ
"__inference__wrapped_model_1957548|	*()-+,0./9Ђ6
/Ђ,
*'
gru_9_inputџџџџџџџџџњ
Њ "4Њ1
/
gru_11%"
gru_11џџџџџџџџџњг
C__inference_gru_10_layer_call_and_return_conditional_losses_1961572-+,PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџd
 г
C__inference_gru_10_layer_call_and_return_conditional_losses_1961725-+,PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџd
 Л
C__inference_gru_10_layer_call_and_return_conditional_losses_1961878t-+,AЂ>
7Ђ4
&#
inputsџџџџџџџџџњЌ

 
p 

 
Њ "*Ђ'
 
0џџџџџџџџџњd
 Л
C__inference_gru_10_layer_call_and_return_conditional_losses_1962031t-+,AЂ>
7Ђ4
&#
inputsџџџџџџџџџњЌ

 
p

 
Њ "*Ђ'
 
0џџџџџџџџџњd
 Њ
(__inference_gru_10_layer_call_fn_1961386~-+,PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџdЊ
(__inference_gru_10_layer_call_fn_1961397~-+,PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџd
(__inference_gru_10_layer_call_fn_1961408g-+,AЂ>
7Ђ4
&#
inputsџџџџџџџџџњЌ

 
p 

 
Њ "џџџџџџџџџњd
(__inference_gru_10_layer_call_fn_1961419g-+,AЂ>
7Ђ4
&#
inputsџџџџџџџџџњЌ

 
p

 
Њ "џџџџџџџџџњdв
C__inference_gru_11_layer_call_and_return_conditional_losses_19622280./OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџd

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 в
C__inference_gru_11_layer_call_and_return_conditional_losses_19623810./OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџd

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 К
C__inference_gru_11_layer_call_and_return_conditional_losses_1962534s0./@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњd

 
p 

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 К
C__inference_gru_11_layer_call_and_return_conditional_losses_1962687s0./@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњd

 
p

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 Љ
(__inference_gru_11_layer_call_fn_1962042}0./OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџd

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџЉ
(__inference_gru_11_layer_call_fn_1962053}0./OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџd

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ
(__inference_gru_11_layer_call_fn_1962064f0./@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњd

 
p 

 
Њ "џџџџџџџџџњ
(__inference_gru_11_layer_call_fn_1962075f0./@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњd

 
p

 
Њ "џџџџџџџџџњв
B__inference_gru_9_layer_call_and_return_conditional_losses_1960916*()OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџЌ
 в
B__inference_gru_9_layer_call_and_return_conditional_losses_1961069*()OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџЌ
 К
B__inference_gru_9_layer_call_and_return_conditional_losses_1961222t*()@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњ

 
p 

 
Њ "+Ђ(
!
0џџџџџџџџџњЌ
 К
B__inference_gru_9_layer_call_and_return_conditional_losses_1961375t*()@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњ

 
p

 
Њ "+Ђ(
!
0џџџџџџџџџњЌ
 Љ
'__inference_gru_9_layer_call_fn_1960730~*()OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "&#џџџџџџџџџџџџџџџџџџЌЉ
'__inference_gru_9_layer_call_fn_1960741~*()OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "&#џџџџџџџџџџџџџџџџџџЌ
'__inference_gru_9_layer_call_fn_1960752g*()@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњ

 
p 

 
Њ "џџџџџџџџџњЌ
'__inference_gru_9_layer_call_fn_1960763g*()@Ђ=
6Ђ3
%"
inputsџџџџџџџџџњ

 
p

 
Њ "џџџџџџџџџњЌ
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962754К*()]ЂZ
SЂP
 
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџЌ
p 
Њ "TЂQ
JЂG

0/0џџџџџџџџџЌ
%"
 
0/1/0џџџџџџџџџЌ
 
H__inference_gru_cell_18_layer_call_and_return_conditional_losses_1962793К*()]ЂZ
SЂP
 
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџЌ
p
Њ "TЂQ
JЂG

0/0џџџџџџџџџЌ
%"
 
0/1/0џџџџџџџџџЌ
 о
-__inference_gru_cell_18_layer_call_fn_1962701Ќ*()]ЂZ
SЂP
 
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџЌ
p 
Њ "FЂC

0џџџџџџџџџЌ
# 

1/0џџџџџџџџџЌо
-__inference_gru_cell_18_layer_call_fn_1962715Ќ*()]ЂZ
SЂP
 
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџЌ
p
Њ "FЂC

0џџџџџџџџџЌ
# 

1/0џџџџџџџџџЌ
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962860И-+,]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџd
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџd
$!

0/1/0џџџџџџџџџd
 
H__inference_gru_cell_19_layer_call_and_return_conditional_losses_1962899И-+,]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџd
p
Њ "RЂO
HЂE

0/0џџџџџџџџџd
$!

0/1/0џџџџџџџџџd
 м
-__inference_gru_cell_19_layer_call_fn_1962807Њ-+,]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџd
p 
Њ "DЂA

0џџџџџџџџџd
"

1/0џџџџџџџџџdм
-__inference_gru_cell_19_layer_call_fn_1962821Њ-+,]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџd
p
Њ "DЂA

0џџџџџџџџџd
"

1/0џџџџџџџџџd
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1962966З0./\ЂY
RЂO
 
inputsџџџџџџџџџd
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 
H__inference_gru_cell_20_layer_call_and_return_conditional_losses_1963005З0./\ЂY
RЂO
 
inputsџџџџџџџџџd
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 л
-__inference_gru_cell_20_layer_call_fn_1962913Љ0./\ЂY
RЂO
 
inputsџџџџџџџџџd
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџл
-__inference_gru_cell_20_layer_call_fn_1962927Љ0./\ЂY
RЂO
 
inputsџџџџџџџџџd
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџЧ
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959715z	*()-+,0./AЂ>
7Ђ4
*'
gru_9_inputџџџџџџџџџњ
p 

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 Ч
I__inference_sequential_3_layer_call_and_return_conditional_losses_1959740z	*()-+,0./AЂ>
7Ђ4
*'
gru_9_inputџџџџџџџџџњ
p

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 Т
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960268u	*()-+,0./<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p 

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 Т
I__inference_sequential_3_layer_call_and_return_conditional_losses_1960719u	*()-+,0./<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p

 
Њ "*Ђ'
 
0џџџџџџџџџњ
 
.__inference_sequential_3_layer_call_fn_1959071m	*()-+,0./AЂ>
7Ђ4
*'
gru_9_inputџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџњ
.__inference_sequential_3_layer_call_fn_1959690m	*()-+,0./AЂ>
7Ђ4
*'
gru_9_inputџџџџџџџџџњ
p

 
Њ "џџџџџџџџџњ
.__inference_sequential_3_layer_call_fn_1959794h	*()-+,0./<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџњ
.__inference_sequential_3_layer_call_fn_1959817h	*()-+,0./<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p

 
Њ "џџџџџџџџџњЕ
%__inference_signature_wrapper_1959771	*()-+,0./HЂE
Ђ 
>Њ;
9
gru_9_input*'
gru_9_inputџџџџџџџџџњ"4Њ1
/
gru_11%"
gru_11џџџџџџџџџњ