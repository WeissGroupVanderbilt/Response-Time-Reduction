»ш(
µК
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
delete_old_dirsbool(И
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018Г≤&
∞
,Adam/simple_rnn_20/simple_rnn_cell_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_20/simple_rnn_cell_38/bias/v
©
@Adam/simple_rnn_20/simple_rnn_cell_38/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_20/simple_rnn_cell_38/bias/v*
_output_shapes
:*
dtype0
ћ
8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v
≈
LAdam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v*
_output_shapes

:*
dtype0
Є
.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*?
shared_name0.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/v
±
BAdam/simple_rnn_20/simple_rnn_cell_38/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/v*
_output_shapes

:d*
dtype0
∞
,Adam/simple_rnn_19/simple_rnn_cell_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_19/simple_rnn_cell_37/bias/v
©
@Adam/simple_rnn_19/simple_rnn_cell_37/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_19/simple_rnn_cell_37/bias/v*
_output_shapes
:d*
dtype0
ћ
8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v
≈
LAdam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
є
.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*?
shared_name0.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/v
≤
BAdam/simple_rnn_19/simple_rnn_cell_37/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/v*
_output_shapes
:	ђd*
dtype0
±
,Adam/simple_rnn_18/simple_rnn_cell_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*=
shared_name.,Adam/simple_rnn_18/simple_rnn_cell_36/bias/v
™
@Adam/simple_rnn_18/simple_rnn_cell_36/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_18/simple_rnn_cell_36/bias/v*
_output_shapes	
:ђ*
dtype0
ќ
8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*I
shared_name:8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v
«
LAdam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v* 
_output_shapes
:
ђђ*
dtype0
є
.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*?
shared_name0.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/v
≤
BAdam/simple_rnn_18/simple_rnn_cell_36/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/v*
_output_shapes
:	ђ*
dtype0
∞
,Adam/simple_rnn_20/simple_rnn_cell_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_20/simple_rnn_cell_38/bias/m
©
@Adam/simple_rnn_20/simple_rnn_cell_38/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_20/simple_rnn_cell_38/bias/m*
_output_shapes
:*
dtype0
ћ
8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m
≈
LAdam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m*
_output_shapes

:*
dtype0
Є
.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*?
shared_name0.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/m
±
BAdam/simple_rnn_20/simple_rnn_cell_38/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/m*
_output_shapes

:d*
dtype0
∞
,Adam/simple_rnn_19/simple_rnn_cell_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_19/simple_rnn_cell_37/bias/m
©
@Adam/simple_rnn_19/simple_rnn_cell_37/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_19/simple_rnn_cell_37/bias/m*
_output_shapes
:d*
dtype0
ћ
8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m
≈
LAdam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
є
.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*?
shared_name0.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/m
≤
BAdam/simple_rnn_19/simple_rnn_cell_37/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/m*
_output_shapes
:	ђd*
dtype0
±
,Adam/simple_rnn_18/simple_rnn_cell_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*=
shared_name.,Adam/simple_rnn_18/simple_rnn_cell_36/bias/m
™
@Adam/simple_rnn_18/simple_rnn_cell_36/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_18/simple_rnn_cell_36/bias/m*
_output_shapes	
:ђ*
dtype0
ќ
8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*I
shared_name:8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m
«
LAdam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m* 
_output_shapes
:
ђђ*
dtype0
є
.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*?
shared_name0.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/m
≤
BAdam/simple_rnn_18/simple_rnn_cell_36/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/m*
_output_shapes
:	ђ*
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
Ґ
%simple_rnn_20/simple_rnn_cell_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%simple_rnn_20/simple_rnn_cell_38/bias
Ы
9simple_rnn_20/simple_rnn_cell_38/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_20/simple_rnn_cell_38/bias*
_output_shapes
:*
dtype0
Њ
1simple_rnn_20/simple_rnn_cell_38/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31simple_rnn_20/simple_rnn_cell_38/recurrent_kernel
Ј
Esimple_rnn_20/simple_rnn_cell_38/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_20/simple_rnn_cell_38/recurrent_kernel*
_output_shapes

:*
dtype0
™
'simple_rnn_20/simple_rnn_cell_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*8
shared_name)'simple_rnn_20/simple_rnn_cell_38/kernel
£
;simple_rnn_20/simple_rnn_cell_38/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_20/simple_rnn_cell_38/kernel*
_output_shapes

:d*
dtype0
Ґ
%simple_rnn_19/simple_rnn_cell_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%simple_rnn_19/simple_rnn_cell_37/bias
Ы
9simple_rnn_19/simple_rnn_cell_37/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_19/simple_rnn_cell_37/bias*
_output_shapes
:d*
dtype0
Њ
1simple_rnn_19/simple_rnn_cell_37/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*B
shared_name31simple_rnn_19/simple_rnn_cell_37/recurrent_kernel
Ј
Esimple_rnn_19/simple_rnn_cell_37/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_19/simple_rnn_cell_37/recurrent_kernel*
_output_shapes

:dd*
dtype0
Ђ
'simple_rnn_19/simple_rnn_cell_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*8
shared_name)'simple_rnn_19/simple_rnn_cell_37/kernel
§
;simple_rnn_19/simple_rnn_cell_37/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_19/simple_rnn_cell_37/kernel*
_output_shapes
:	ђd*
dtype0
£
%simple_rnn_18/simple_rnn_cell_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%simple_rnn_18/simple_rnn_cell_36/bias
Ь
9simple_rnn_18/simple_rnn_cell_36/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_18/simple_rnn_cell_36/bias*
_output_shapes	
:ђ*
dtype0
ј
1simple_rnn_18/simple_rnn_cell_36/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*B
shared_name31simple_rnn_18/simple_rnn_cell_36/recurrent_kernel
є
Esimple_rnn_18/simple_rnn_cell_36/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_18/simple_rnn_cell_36/recurrent_kernel* 
_output_shapes
:
ђђ*
dtype0
Ђ
'simple_rnn_18/simple_rnn_cell_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*8
shared_name)'simple_rnn_18/simple_rnn_cell_36/kernel
§
;simple_rnn_18/simple_rnn_cell_36/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_18/simple_rnn_cell_36/kernel*
_output_shapes
:	ђ*
dtype0

NoOpNoOp
ЁK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ШK
valueОKBЛK BДK
Ѕ
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
™
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
™
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
™
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#cell
$
state_spec*
C
%0
&1
'2
(3
)4
*5
+6
,7
-8*
C
%0
&1
'2
(3
)4
*5
+6
,7
-8*
* 
∞
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 
ш
;iter

<beta_1

=beta_2
	>decay
?learning_rate%m†&m°'mҐ(m£)m§*m•+m¶,mІ-m®%v©&v™'vЂ(vђ)v≠*vЃ+vѓ,v∞-v±*

@serving_default* 

%0
&1
'2*

%0
&1
'2*
* 
Я

Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
”
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator

%kernel
&recurrent_kernel
'bias*
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
Я

Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
”
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator

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
Я

kstates
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
6
qtrace_0
rtrace_1
strace_2
ttrace_3* 
6
utrace_0
vtrace_1
wtrace_2
xtrace_3* 
”
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator

+kernel
,recurrent_kernel
-bias*
* 
ga
VARIABLE_VALUE'simple_rnn_18/simple_rnn_cell_36/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_18/simple_rnn_cell_36/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_18/simple_rnn_cell_36/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_19/simple_rnn_cell_37/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_19/simple_rnn_cell_37/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_19/simple_rnn_cell_37/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_20/simple_rnn_cell_38/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_20/simple_rnn_cell_38/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_20/simple_rnn_cell_38/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

А0*
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

0*
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
%0
&1
'2*

%0
&1
'2*
* 
Ш
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Жtrace_0
Зtrace_1* 

Иtrace_0
Йtrace_1* 
* 
* 
* 

0*
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
Ш
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

Пtrace_0
Рtrace_1* 

Сtrace_0
Тtrace_1* 
* 
* 
* 

#0*
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
Ш
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
* 
<
Ь	variables
Э	keras_api

Юtotal

Яcount*
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
Ю0
Я1*

Ь	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_18/simple_rnn_cell_36/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_19/simple_rnn_cell_37/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_20/simple_rnn_cell_38/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_18/simple_rnn_cell_36/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_19/simple_rnn_cell_37/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_20/simple_rnn_cell_38/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Р
#serving_default_simple_rnn_18_inputPlaceholder*,
_output_shapes
:€€€€€€€€€ъ*
dtype0*!
shape:€€€€€€€€€ъ
в
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_18_input'simple_rnn_18/simple_rnn_cell_36/kernel%simple_rnn_18/simple_rnn_cell_36/bias1simple_rnn_18/simple_rnn_cell_36/recurrent_kernel'simple_rnn_19/simple_rnn_cell_37/kernel%simple_rnn_19/simple_rnn_cell_37/bias1simple_rnn_19/simple_rnn_cell_37/recurrent_kernel'simple_rnn_20/simple_rnn_cell_38/kernel%simple_rnn_20/simple_rnn_cell_38/bias1simple_rnn_20/simple_rnn_cell_38/recurrent_kernel*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_8477588
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
’
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;simple_rnn_18/simple_rnn_cell_36/kernel/Read/ReadVariableOpEsimple_rnn_18/simple_rnn_cell_36/recurrent_kernel/Read/ReadVariableOp9simple_rnn_18/simple_rnn_cell_36/bias/Read/ReadVariableOp;simple_rnn_19/simple_rnn_cell_37/kernel/Read/ReadVariableOpEsimple_rnn_19/simple_rnn_cell_37/recurrent_kernel/Read/ReadVariableOp9simple_rnn_19/simple_rnn_cell_37/bias/Read/ReadVariableOp;simple_rnn_20/simple_rnn_cell_38/kernel/Read/ReadVariableOpEsimple_rnn_20/simple_rnn_cell_38/recurrent_kernel/Read/ReadVariableOp9simple_rnn_20/simple_rnn_cell_38/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpBAdam/simple_rnn_18/simple_rnn_cell_36/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_18/simple_rnn_cell_36/bias/m/Read/ReadVariableOpBAdam/simple_rnn_19/simple_rnn_cell_37/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_19/simple_rnn_cell_37/bias/m/Read/ReadVariableOpBAdam/simple_rnn_20/simple_rnn_cell_38/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_20/simple_rnn_cell_38/bias/m/Read/ReadVariableOpBAdam/simple_rnn_18/simple_rnn_cell_36/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_18/simple_rnn_cell_36/bias/v/Read/ReadVariableOpBAdam/simple_rnn_19/simple_rnn_cell_37/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_19/simple_rnn_cell_37/bias/v/Read/ReadVariableOpBAdam/simple_rnn_20/simple_rnn_cell_38/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_20/simple_rnn_cell_38/bias/v/Read/ReadVariableOpConst*/
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_8480005
®
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'simple_rnn_18/simple_rnn_cell_36/kernel1simple_rnn_18/simple_rnn_cell_36/recurrent_kernel%simple_rnn_18/simple_rnn_cell_36/bias'simple_rnn_19/simple_rnn_cell_37/kernel1simple_rnn_19/simple_rnn_cell_37/recurrent_kernel%simple_rnn_19/simple_rnn_cell_37/bias'simple_rnn_20/simple_rnn_cell_38/kernel1simple_rnn_20/simple_rnn_cell_38/recurrent_kernel%simple_rnn_20/simple_rnn_cell_38/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/m8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m,Adam/simple_rnn_18/simple_rnn_cell_36/bias/m.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/m8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m,Adam/simple_rnn_19/simple_rnn_cell_37/bias/m.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/m8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m,Adam/simple_rnn_20/simple_rnn_cell_38/bias/m.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/v8Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v,Adam/simple_rnn_18/simple_rnn_cell_36/bias/v.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/v8Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v,Adam/simple_rnn_19/simple_rnn_cell_37/bias/v.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/v8Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v,Adam/simple_rnn_20/simple_rnn_cell_38/bias/v*.
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_8480117ше$
я
ѓ
while_cond_8476811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476811___redundant_placeholder05
1while_while_cond_8476811___redundant_placeholder15
1while_while_cond_8476811___redundant_placeholder25
1while_while_cond_8476811___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
Џ

є
 simple_rnn_19_while_cond_84780958
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_2:
6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8478095___redundant_placeholder0Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8478095___redundant_placeholder1Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8478095___redundant_placeholder2Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8478095___redundant_placeholder3 
simple_rnn_19_while_identity
Ъ
simple_rnn_19/while/LessLesssimple_rnn_19_while_placeholder6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
щ,
Џ
while_body_8478460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_8476926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476926___redundant_placeholder05
1while_while_cond_8476926___redundant_placeholder15
1while_while_cond_8476926___redundant_placeholder25
1while_while_cond_8476926___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_8479303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479303___redundant_placeholder05
1while_while_cond_8479303___redundant_placeholder15
1while_while_cond_8479303___redundant_placeholder25
1while_while_cond_8479303___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Х
н
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479818

inputs
states_01
matmul_readvariableop_resource:	ђd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€dZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d\

Identity_1IdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€d
"
_user_specified_name
states/0
я
ѓ
while_cond_8479043
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479043___redundant_placeholder05
1while_while_cond_8479043___redundant_placeholder15
1while_while_cond_8479043___redundant_placeholder25
1while_while_cond_8479043___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
и,
‘
while_body_8477208
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
’
√
.sequential_16_simple_rnn_19_while_cond_8475602T
Psequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_loop_counterZ
Vsequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_maximum_iterations1
-sequential_16_simple_rnn_19_while_placeholder3
/sequential_16_simple_rnn_19_while_placeholder_13
/sequential_16_simple_rnn_19_while_placeholder_2V
Rsequential_16_simple_rnn_19_while_less_sequential_16_simple_rnn_19_strided_slice_1m
isequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_cond_8475602___redundant_placeholder0m
isequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_cond_8475602___redundant_placeholder1m
isequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_cond_8475602___redundant_placeholder2m
isequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_cond_8475602___redundant_placeholder3.
*sequential_16_simple_rnn_19_while_identity
“
&sequential_16/simple_rnn_19/while/LessLess-sequential_16_simple_rnn_19_while_placeholderRsequential_16_simple_rnn_19_while_less_sequential_16_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: Г
*sequential_16/simple_rnn_19/while/IdentityIdentity*sequential_16/simple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_16_simple_rnn_19_while_identity3sequential_16/simple_rnn_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
РG
¶
.sequential_16_simple_rnn_18_while_body_8475499T
Psequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_loop_counterZ
Vsequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_maximum_iterations1
-sequential_16_simple_rnn_18_while_placeholder3
/sequential_16_simple_rnn_18_while_placeholder_13
/sequential_16_simple_rnn_18_while_placeholder_2S
Osequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_strided_slice_1_0Р
Лsequential_16_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0h
Usequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђe
Vsequential_16_simple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђk
Wsequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ.
*sequential_16_simple_rnn_18_while_identity0
,sequential_16_simple_rnn_18_while_identity_10
,sequential_16_simple_rnn_18_while_identity_20
,sequential_16_simple_rnn_18_while_identity_30
,sequential_16_simple_rnn_18_while_identity_4Q
Msequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_strided_slice_1О
Йsequential_16_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorf
Ssequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђc
Tsequential_16_simple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђi
Usequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐKsequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐJsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpҐLsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp§
Ssequential_16/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ≥
Esequential_16/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЛsequential_16_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0-sequential_16_simple_rnn_18_while_placeholder\sequential_16/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0б
Jsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOpUsequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0Ъ
;sequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMulMatMulLsequential_16/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђя
Ksequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOpVsequential_16_simple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0Ц
<sequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAddBiasAddEsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul:product:0Ssequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђж
Lsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpWsequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Б
=sequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1MatMul/sequential_16_simple_rnn_18_while_placeholder_2Tsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђД
8sequential_16/simple_rnn_18/while/simple_rnn_cell_36/addAddV2Esequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd:output:0Gsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђЄ
<sequential_16/simple_rnn_18/while/simple_rnn_cell_36/SigmoidSigmoid<sequential_16/simple_rnn_18/while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђљ
Fsequential_16/simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_16_simple_rnn_18_while_placeholder_1-sequential_16_simple_rnn_18_while_placeholder@sequential_16/simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“i
'sequential_16/simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :∞
%sequential_16/simple_rnn_18/while/addAddV2-sequential_16_simple_rnn_18_while_placeholder0sequential_16/simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_16/simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :„
'sequential_16/simple_rnn_18/while/add_1AddV2Psequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_loop_counter2sequential_16/simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: ≠
*sequential_16/simple_rnn_18/while/IdentityIdentity+sequential_16/simple_rnn_18/while/add_1:z:0'^sequential_16/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_18/while/Identity_1IdentityVsequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_maximum_iterations'^sequential_16/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ≠
,sequential_16/simple_rnn_18/while/Identity_2Identity)sequential_16/simple_rnn_18/while/add:z:0'^sequential_16/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_18/while/Identity_3IdentityVsequential_16/simple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_16/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ÷
,sequential_16/simple_rnn_18/while/Identity_4Identity@sequential_16/simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0'^sequential_16/simple_rnn_18/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ“
&sequential_16/simple_rnn_18/while/NoOpNoOpL^sequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpK^sequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpM^sequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_16_simple_rnn_18_while_identity3sequential_16/simple_rnn_18/while/Identity:output:0"e
,sequential_16_simple_rnn_18_while_identity_15sequential_16/simple_rnn_18/while/Identity_1:output:0"e
,sequential_16_simple_rnn_18_while_identity_25sequential_16/simple_rnn_18/while/Identity_2:output:0"e
,sequential_16_simple_rnn_18_while_identity_35sequential_16/simple_rnn_18/while/Identity_3:output:0"e
,sequential_16_simple_rnn_18_while_identity_45sequential_16/simple_rnn_18/while/Identity_4:output:0"†
Msequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_strided_slice_1Osequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_strided_slice_1_0"Ѓ
Tsequential_16_simple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resourceVsequential_16_simple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"∞
Usequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resourceWsequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"ђ
Ssequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resourceUsequential_16_simple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0"Ъ
Йsequential_16_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorЛsequential_16_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2Ъ
Ksequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpKsequential_16/simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2Ш
Jsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpJsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp2Ь
Lsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpLsequential_16/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
и,
‘
while_body_8476812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_8479411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479411___redundant_placeholder05
1while_while_cond_8479411___redundant_placeholder15
1while_while_cond_8479411___redundant_placeholder25
1while_while_cond_8479411___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
С
є
/__inference_simple_rnn_20_layer_call_fn_8479262

inputs
unknown:d
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8477144t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
Џ

є
 simple_rnn_20_while_cond_84778838
4simple_rnn_20_while_simple_rnn_20_while_loop_counter>
:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations#
simple_rnn_20_while_placeholder%
!simple_rnn_20_while_placeholder_1%
!simple_rnn_20_while_placeholder_2:
6simple_rnn_20_while_less_simple_rnn_20_strided_slice_1Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8477883___redundant_placeholder0Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8477883___redundant_placeholder1Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8477883___redundant_placeholder2Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8477883___redundant_placeholder3 
simple_rnn_20_while_identity
Ъ
simple_rnn_20/while/LessLesssimple_rnn_20_while_placeholder6simple_rnn_20_while_less_simple_rnn_20_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_20/while/IdentityIdentitysimple_rnn_20/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_20_while_identity%simple_rnn_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ґ
р
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479739

inputs
states_01
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ]

Identity_1IdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
states/0
†Џ
Ґ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8478266

inputsR
?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource:	ђO
@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђU
Asimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђR
?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdN
@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource:dS
Asimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddQ
?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource:dN
@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource:S
Asimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpҐ8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpҐsimple_rnn_18/whileҐ7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpҐ8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpҐsimple_rnn_19/whileҐ7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpҐ8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpҐsimple_rnn_20/whileI
simple_rnn_18/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_18/strided_sliceStridedSlicesimple_rnn_18/Shape:output:0*simple_rnn_18/strided_slice/stack:output:0,simple_rnn_18/strided_slice/stack_1:output:0,simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђЭ
simple_rnn_18/zeros/packedPack$simple_rnn_18/strided_slice:output:0%simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ч
simple_rnn_18/zerosFill#simple_rnn_18/zeros/packed:output:0"simple_rnn_18/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђq
simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          К
simple_rnn_18/transpose	Transposeinputs%simple_rnn_18/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€`
simple_rnn_18/Shape_1Shapesimple_rnn_18/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_18/strided_slice_1StridedSlicesimple_rnn_18/Shape_1:output:0,simple_rnn_18/strided_slice_1/stack:output:0.simple_rnn_18/strided_slice_1/stack_1:output:0.simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_18/TensorArrayV2TensorListReserve2simple_rnn_18/TensorArrayV2/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_18/transpose:y:0Lsimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_18/strided_slice_2StridedSlicesimple_rnn_18/transpose:y:0,simple_rnn_18/strided_slice_2/stack:output:0.simple_rnn_18/strided_slice_2/stack_1:output:0.simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЈ
6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ћ
'simple_rnn_18/simple_rnn_cell_36/MatMulMatMul&simple_rnn_18/strided_slice_2:output:0>simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђµ
7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
(simple_rnn_18/simple_rnn_cell_36/BiasAddBiasAdd1simple_rnn_18/simple_rnn_cell_36/MatMul:product:0?simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЉ
8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0∆
)simple_rnn_18/simple_rnn_cell_36/MatMul_1MatMulsimple_rnn_18/zeros:output:0@simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ»
$simple_rnn_18/simple_rnn_cell_36/addAddV21simple_rnn_18/simple_rnn_cell_36/BiasAdd:output:03simple_rnn_18/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђР
(simple_rnn_18/simple_rnn_cell_36/SigmoidSigmoid(simple_rnn_18/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ|
+simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  в
simple_rnn_18/TensorArrayV2_1TensorListReserve4simple_rnn_18/TensorArrayV2_1/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
simple_rnn_18/whileWhile)simple_rnn_18/while/loop_counter:output:0/simple_rnn_18/while/maximum_iterations:output:0simple_rnn_18/time:output:0&simple_rnn_18/TensorArrayV2_1:handle:0simple_rnn_18/zeros:output:0&simple_rnn_18/strided_slice_1:output:0Esimple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resourceAsimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_18_while_body_8477992*,
cond$R"
 simple_rnn_18_while_cond_8477991*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations П
>simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  о
0simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_18/while:output:3Gsimple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0v
#simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
simple_rnn_18/strided_slice_3StridedSlice9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_18/strided_slice_3/stack:output:0.simple_rnn_18/strided_slice_3/stack_1:output:0.simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_masks
simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
simple_rnn_18/transpose_1	Transpose9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_18/transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђ`
simple_rnn_19/ShapeShapesimple_rnn_18/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_19/strided_sliceStridedSlicesimple_rnn_19/Shape:output:0*simple_rnn_19/strided_slice/stack:output:0,simple_rnn_19/strided_slice/stack_1:output:0,simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЭ
simple_rnn_19/zeros/packedPack$simple_rnn_19/strided_slice:output:0%simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_19/zerosFill#simple_rnn_19/zeros/packed:output:0"simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€dq
simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
simple_rnn_19/transpose	Transposesimple_rnn_18/transpose_1:y:0%simple_rnn_19/transpose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђ`
simple_rnn_19/Shape_1Shapesimple_rnn_19/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_19/strided_slice_1StridedSlicesimple_rnn_19/Shape_1:output:0,simple_rnn_19/strided_slice_1/stack:output:0.simple_rnn_19/strided_slice_1/stack_1:output:0.simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_19/TensorArrayV2TensorListReserve2simple_rnn_19/TensorArrayV2/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  К
5simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_19/transpose:y:0Lsimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
simple_rnn_19/strided_slice_2StridedSlicesimple_rnn_19/transpose:y:0,simple_rnn_19/strided_slice_2/stack:output:0.simple_rnn_19/strided_slice_2/stack_1:output:0.simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЈ
6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0Ћ
'simple_rnn_19/simple_rnn_cell_37/MatMulMatMul&simple_rnn_19/strided_slice_2:output:0>simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dі
7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ў
(simple_rnn_19/simple_rnn_cell_37/BiasAddBiasAdd1simple_rnn_19/simple_rnn_cell_37/MatMul:product:0?simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЇ
8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0≈
)simple_rnn_19/simple_rnn_cell_37/MatMul_1MatMulsimple_rnn_19/zeros:output:0@simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d«
$simple_rnn_19/simple_rnn_cell_37/addAddV21simple_rnn_19/simple_rnn_cell_37/BiasAdd:output:03simple_rnn_19/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dП
(simple_rnn_19/simple_rnn_cell_37/SigmoidSigmoid(simple_rnn_19/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€d|
+simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   в
simple_rnn_19/TensorArrayV2_1TensorListReserve4simple_rnn_19/TensorArrayV2_1/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_19/whileWhile)simple_rnn_19/while/loop_counter:output:0/simple_rnn_19/while/maximum_iterations:output:0simple_rnn_19/time:output:0&simple_rnn_19/TensorArrayV2_1:handle:0simple_rnn_19/zeros:output:0&simple_rnn_19/strided_slice_1:output:0Esimple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resourceAsimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_19_while_body_8478096*,
cond$R"
 simple_rnn_19_while_cond_8478095*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations П
>simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   н
0simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_19/while:output:3Gsimple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0v
#simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_19/strided_slice_3StridedSlice9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_19/strided_slice_3/stack:output:0.simple_rnn_19/strided_slice_3/stack_1:output:0.simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_masks
simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
simple_rnn_19/transpose_1	Transpose9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъd`
simple_rnn_20/ShapeShapesimple_rnn_19/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_20/strided_sliceStridedSlicesimple_rnn_20/Shape:output:0*simple_rnn_20/strided_slice/stack:output:0,simple_rnn_20/strided_slice/stack_1:output:0,simple_rnn_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Э
simple_rnn_20/zeros/packedPack$simple_rnn_20/strided_slice:output:0%simple_rnn_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_20/zerosFill#simple_rnn_20/zeros/packed:output:0"simple_rnn_20/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
simple_rnn_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          °
simple_rnn_20/transpose	Transposesimple_rnn_19/transpose_1:y:0%simple_rnn_20/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€d`
simple_rnn_20/Shape_1Shapesimple_rnn_20/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_20/strided_slice_1StridedSlicesimple_rnn_20/Shape_1:output:0,simple_rnn_20/strided_slice_1/stack:output:0.simple_rnn_20/strided_slice_1/stack_1:output:0.simple_rnn_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_20/TensorArrayV2TensorListReserve2simple_rnn_20/TensorArrayV2/element_shape:output:0&simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   К
5simple_rnn_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_20/transpose:y:0Lsimple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_20/strided_slice_2StridedSlicesimple_rnn_20/transpose:y:0,simple_rnn_20/strided_slice_2/stack:output:0.simple_rnn_20/strided_slice_2/stack_1:output:0.simple_rnn_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskґ
6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ћ
'simple_rnn_20/simple_rnn_cell_38/MatMulMatMul&simple_rnn_20/strided_slice_2:output:0>simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€і
7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
(simple_rnn_20/simple_rnn_cell_38/BiasAddBiasAdd1simple_rnn_20/simple_rnn_cell_38/MatMul:product:0?simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0≈
)simple_rnn_20/simple_rnn_cell_38/MatMul_1MatMulsimple_rnn_20/zeros:output:0@simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€«
$simple_rnn_20/simple_rnn_cell_38/addAddV21simple_rnn_20/simple_rnn_cell_38/BiasAdd:output:03simple_rnn_20/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)simple_rnn_20/simple_rnn_cell_38/SoftplusSoftplus(simple_rnn_20/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
+simple_rnn_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   в
simple_rnn_20/TensorArrayV2_1TensorListReserve4simple_rnn_20/TensorArrayV2_1/element_shape:output:0&simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_20/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_20/whileWhile)simple_rnn_20/while/loop_counter:output:0/simple_rnn_20/while/maximum_iterations:output:0simple_rnn_20/time:output:0&simple_rnn_20/TensorArrayV2_1:handle:0simple_rnn_20/zeros:output:0&simple_rnn_20/strided_slice_1:output:0Esimple_rnn_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resourceAsimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_20_while_body_8478200*,
cond$R"
 simple_rnn_20_while_cond_8478199*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations П
>simple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   н
0simple_rnn_20/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_20/while:output:3Gsimple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0v
#simple_rnn_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_20/strided_slice_3StridedSlice9simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_20/strided_slice_3/stack:output:0.simple_rnn_20/strided_slice_3/stack_1:output:0.simple_rnn_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_masks
simple_rnn_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
simple_rnn_20/transpose_1	Transpose9simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъq
IdentityIdentitysimple_rnn_20/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъТ
NoOpNoOp8^simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp7^simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp9^simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp^simple_rnn_18/while8^simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp7^simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp9^simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp^simple_rnn_19/while8^simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp7^simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp9^simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp^simple_rnn_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2r
7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp2p
6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp2t
8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp2*
simple_rnn_18/whilesimple_rnn_18/while2r
7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp2p
6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp2t
8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp2*
simple_rnn_19/whilesimple_rnn_19/while2r
7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp2p
6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp2t
8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp2*
simple_rnn_20/whilesimple_rnn_20/while:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
я
ѓ
while_cond_8479519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479519___redundant_placeholder05
1while_while_cond_8479519___redundant_placeholder15
1while_while_cond_8479519___redundant_placeholder25
1while_while_cond_8479519___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
и,
‘
while_body_8479152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
©
м
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479863

inputs
states_00
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€O
SoftplusSoftplusadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g

Identity_1IdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
б
ѓ
while_cond_8478567
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478567___redundant_placeholder05
1while_while_cond_8478567___redundant_placeholder15
1while_while_cond_8478567___redundant_placeholder25
1while_while_cond_8478567___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Д:
ю
 simple_rnn_18_while_body_84779928
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_27
3simple_rnn_18_while_simple_rnn_18_strided_slice_1_0s
osimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђW
Hsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђ]
Isimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ 
simple_rnn_18_while_identity"
simple_rnn_18_while_identity_1"
simple_rnn_18_while_identity_2"
simple_rnn_18_while_identity_3"
simple_rnn_18_while_identity_45
1simple_rnn_18_while_simple_rnn_18_strided_slice_1q
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђU
Fsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђ[
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpЦ
Esimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_18_while_placeholderNsimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0≈
<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0р
-simple_rnn_18/while/simple_rnn_cell_36/MatMulMatMul>simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ√
=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0м
.simple_rnn_18/while/simple_rnn_cell_36/BiasAddBiasAdd7simple_rnn_18/while/simple_rnn_cell_36/MatMul:product:0Esimple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ 
>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0„
/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1MatMul!simple_rnn_18_while_placeholder_2Fsimple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЏ
*simple_rnn_18/while/simple_rnn_cell_36/addAddV27simple_rnn_18/while/simple_rnn_cell_36/BiasAdd:output:09simple_rnn_18/while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђЬ
.simple_rnn_18/while/simple_rnn_cell_36/SigmoidSigmoid.simple_rnn_18/while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЕ
8simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_18_while_placeholder_1simple_rnn_18_while_placeholder2simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_18/while/addAddV2simple_rnn_18_while_placeholder"simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_18/while/add_1AddV24simple_rnn_18_while_simple_rnn_18_while_loop_counter$simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/add_1:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_18/while/Identity_1Identity:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_18/while/Identity_2Identitysimple_rnn_18/while/add:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_18/while/Identity_3IdentityHsimple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ђ
simple_rnn_18/while/Identity_4Identity2simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0^simple_rnn_18/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђЪ
simple_rnn_18/while/NoOpNoOp>^simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp=^simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp?^simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0"I
simple_rnn_18_while_identity_1'simple_rnn_18/while/Identity_1:output:0"I
simple_rnn_18_while_identity_2'simple_rnn_18/while/Identity_2:output:0"I
simple_rnn_18_while_identity_3'simple_rnn_18/while/Identity_3:output:0"I
simple_rnn_18_while_identity_4'simple_rnn_18/while/Identity_4:output:0"h
1simple_rnn_18_while_simple_rnn_18_strided_slice_13simple_rnn_18_while_simple_rnn_18_strided_slice_1_0"Т
Fsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resourceHsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resourceIsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resourceGsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0"а
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2~
=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2|
<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp2А
>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
†Џ
Ґ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477950

inputsR
?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource:	ђO
@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђU
Asimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђR
?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdN
@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource:dS
Asimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddQ
?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource:dN
@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource:S
Asimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpҐ8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpҐsimple_rnn_18/whileҐ7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpҐ8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpҐsimple_rnn_19/whileҐ7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpҐ8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpҐsimple_rnn_20/whileI
simple_rnn_18/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_18/strided_sliceStridedSlicesimple_rnn_18/Shape:output:0*simple_rnn_18/strided_slice/stack:output:0,simple_rnn_18/strided_slice/stack_1:output:0,simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђЭ
simple_rnn_18/zeros/packedPack$simple_rnn_18/strided_slice:output:0%simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ч
simple_rnn_18/zerosFill#simple_rnn_18/zeros/packed:output:0"simple_rnn_18/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђq
simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          К
simple_rnn_18/transpose	Transposeinputs%simple_rnn_18/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€`
simple_rnn_18/Shape_1Shapesimple_rnn_18/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_18/strided_slice_1StridedSlicesimple_rnn_18/Shape_1:output:0,simple_rnn_18/strided_slice_1/stack:output:0.simple_rnn_18/strided_slice_1/stack_1:output:0.simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_18/TensorArrayV2TensorListReserve2simple_rnn_18/TensorArrayV2/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_18/transpose:y:0Lsimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_18/strided_slice_2StridedSlicesimple_rnn_18/transpose:y:0,simple_rnn_18/strided_slice_2/stack:output:0.simple_rnn_18/strided_slice_2/stack_1:output:0.simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЈ
6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ћ
'simple_rnn_18/simple_rnn_cell_36/MatMulMatMul&simple_rnn_18/strided_slice_2:output:0>simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђµ
7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
(simple_rnn_18/simple_rnn_cell_36/BiasAddBiasAdd1simple_rnn_18/simple_rnn_cell_36/MatMul:product:0?simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЉ
8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0∆
)simple_rnn_18/simple_rnn_cell_36/MatMul_1MatMulsimple_rnn_18/zeros:output:0@simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ»
$simple_rnn_18/simple_rnn_cell_36/addAddV21simple_rnn_18/simple_rnn_cell_36/BiasAdd:output:03simple_rnn_18/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђР
(simple_rnn_18/simple_rnn_cell_36/SigmoidSigmoid(simple_rnn_18/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ|
+simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  в
simple_rnn_18/TensorArrayV2_1TensorListReserve4simple_rnn_18/TensorArrayV2_1/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
simple_rnn_18/whileWhile)simple_rnn_18/while/loop_counter:output:0/simple_rnn_18/while/maximum_iterations:output:0simple_rnn_18/time:output:0&simple_rnn_18/TensorArrayV2_1:handle:0simple_rnn_18/zeros:output:0&simple_rnn_18/strided_slice_1:output:0Esimple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource@simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resourceAsimple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_18_while_body_8477676*,
cond$R"
 simple_rnn_18_while_cond_8477675*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations П
>simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  о
0simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_18/while:output:3Gsimple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0v
#simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
simple_rnn_18/strided_slice_3StridedSlice9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_18/strided_slice_3/stack:output:0.simple_rnn_18/strided_slice_3/stack_1:output:0.simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_masks
simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
simple_rnn_18/transpose_1	Transpose9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_18/transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђ`
simple_rnn_19/ShapeShapesimple_rnn_18/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_19/strided_sliceStridedSlicesimple_rnn_19/Shape:output:0*simple_rnn_19/strided_slice/stack:output:0,simple_rnn_19/strided_slice/stack_1:output:0,simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЭ
simple_rnn_19/zeros/packedPack$simple_rnn_19/strided_slice:output:0%simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_19/zerosFill#simple_rnn_19/zeros/packed:output:0"simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€dq
simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
simple_rnn_19/transpose	Transposesimple_rnn_18/transpose_1:y:0%simple_rnn_19/transpose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђ`
simple_rnn_19/Shape_1Shapesimple_rnn_19/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_19/strided_slice_1StridedSlicesimple_rnn_19/Shape_1:output:0,simple_rnn_19/strided_slice_1/stack:output:0.simple_rnn_19/strided_slice_1/stack_1:output:0.simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_19/TensorArrayV2TensorListReserve2simple_rnn_19/TensorArrayV2/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  К
5simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_19/transpose:y:0Lsimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
simple_rnn_19/strided_slice_2StridedSlicesimple_rnn_19/transpose:y:0,simple_rnn_19/strided_slice_2/stack:output:0.simple_rnn_19/strided_slice_2/stack_1:output:0.simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЈ
6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0Ћ
'simple_rnn_19/simple_rnn_cell_37/MatMulMatMul&simple_rnn_19/strided_slice_2:output:0>simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dі
7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ў
(simple_rnn_19/simple_rnn_cell_37/BiasAddBiasAdd1simple_rnn_19/simple_rnn_cell_37/MatMul:product:0?simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЇ
8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0≈
)simple_rnn_19/simple_rnn_cell_37/MatMul_1MatMulsimple_rnn_19/zeros:output:0@simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d«
$simple_rnn_19/simple_rnn_cell_37/addAddV21simple_rnn_19/simple_rnn_cell_37/BiasAdd:output:03simple_rnn_19/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dП
(simple_rnn_19/simple_rnn_cell_37/SigmoidSigmoid(simple_rnn_19/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€d|
+simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   в
simple_rnn_19/TensorArrayV2_1TensorListReserve4simple_rnn_19/TensorArrayV2_1/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_19/whileWhile)simple_rnn_19/while/loop_counter:output:0/simple_rnn_19/while/maximum_iterations:output:0simple_rnn_19/time:output:0&simple_rnn_19/TensorArrayV2_1:handle:0simple_rnn_19/zeros:output:0&simple_rnn_19/strided_slice_1:output:0Esimple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource@simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resourceAsimple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_19_while_body_8477780*,
cond$R"
 simple_rnn_19_while_cond_8477779*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations П
>simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   н
0simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_19/while:output:3Gsimple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0v
#simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_19/strided_slice_3StridedSlice9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_19/strided_slice_3/stack:output:0.simple_rnn_19/strided_slice_3/stack_1:output:0.simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_masks
simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
simple_rnn_19/transpose_1	Transpose9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъd`
simple_rnn_20/ShapeShapesimple_rnn_19/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_20/strided_sliceStridedSlicesimple_rnn_20/Shape:output:0*simple_rnn_20/strided_slice/stack:output:0,simple_rnn_20/strided_slice/stack_1:output:0,simple_rnn_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Э
simple_rnn_20/zeros/packedPack$simple_rnn_20/strided_slice:output:0%simple_rnn_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_20/zerosFill#simple_rnn_20/zeros/packed:output:0"simple_rnn_20/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
simple_rnn_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          °
simple_rnn_20/transpose	Transposesimple_rnn_19/transpose_1:y:0%simple_rnn_20/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€d`
simple_rnn_20/Shape_1Shapesimple_rnn_20/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_20/strided_slice_1StridedSlicesimple_rnn_20/Shape_1:output:0,simple_rnn_20/strided_slice_1/stack:output:0.simple_rnn_20/strided_slice_1/stack_1:output:0.simple_rnn_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_20/TensorArrayV2TensorListReserve2simple_rnn_20/TensorArrayV2/element_shape:output:0&simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   К
5simple_rnn_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_20/transpose:y:0Lsimple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_20/strided_slice_2StridedSlicesimple_rnn_20/transpose:y:0,simple_rnn_20/strided_slice_2/stack:output:0.simple_rnn_20/strided_slice_2/stack_1:output:0.simple_rnn_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskґ
6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ћ
'simple_rnn_20/simple_rnn_cell_38/MatMulMatMul&simple_rnn_20/strided_slice_2:output:0>simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€і
7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
(simple_rnn_20/simple_rnn_cell_38/BiasAddBiasAdd1simple_rnn_20/simple_rnn_cell_38/MatMul:product:0?simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0≈
)simple_rnn_20/simple_rnn_cell_38/MatMul_1MatMulsimple_rnn_20/zeros:output:0@simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€«
$simple_rnn_20/simple_rnn_cell_38/addAddV21simple_rnn_20/simple_rnn_cell_38/BiasAdd:output:03simple_rnn_20/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)simple_rnn_20/simple_rnn_cell_38/SoftplusSoftplus(simple_rnn_20/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
+simple_rnn_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   в
simple_rnn_20/TensorArrayV2_1TensorListReserve4simple_rnn_20/TensorArrayV2_1/element_shape:output:0&simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_20/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_20/whileWhile)simple_rnn_20/while/loop_counter:output:0/simple_rnn_20/while/maximum_iterations:output:0simple_rnn_20/time:output:0&simple_rnn_20/TensorArrayV2_1:handle:0simple_rnn_20/zeros:output:0&simple_rnn_20/strided_slice_1:output:0Esimple_rnn_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource@simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resourceAsimple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_20_while_body_8477884*,
cond$R"
 simple_rnn_20_while_cond_8477883*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations П
>simple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   н
0simple_rnn_20/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_20/while:output:3Gsimple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0v
#simple_rnn_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_20/strided_slice_3StridedSlice9simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_20/strided_slice_3/stack:output:0.simple_rnn_20/strided_slice_3/stack_1:output:0.simple_rnn_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_masks
simple_rnn_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
simple_rnn_20/transpose_1	Transpose9simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъq
IdentityIdentitysimple_rnn_20/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъТ
NoOpNoOp8^simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp7^simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp9^simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp^simple_rnn_18/while8^simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp7^simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp9^simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp^simple_rnn_19/while8^simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp7^simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp9^simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp^simple_rnn_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2r
7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp7simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp2p
6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp6simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp2t
8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp8simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp2*
simple_rnn_18/whilesimple_rnn_18/while2r
7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp7simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp2p
6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp6simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp2t
8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp8simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp2*
simple_rnn_19/whilesimple_rnn_19/while2r
7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp7simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp2p
6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp6simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp2t
8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp8simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp2*
simple_rnn_20/whilesimple_rnn_20/while:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
„
√
.sequential_16_simple_rnn_18_while_cond_8475498T
Psequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_loop_counterZ
Vsequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_maximum_iterations1
-sequential_16_simple_rnn_18_while_placeholder3
/sequential_16_simple_rnn_18_while_placeholder_13
/sequential_16_simple_rnn_18_while_placeholder_2V
Rsequential_16_simple_rnn_18_while_less_sequential_16_simple_rnn_18_strided_slice_1m
isequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_cond_8475498___redundant_placeholder0m
isequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_cond_8475498___redundant_placeholder1m
isequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_cond_8475498___redundant_placeholder2m
isequential_16_simple_rnn_18_while_sequential_16_simple_rnn_18_while_cond_8475498___redundant_placeholder3.
*sequential_16_simple_rnn_18_while_identity
“
&sequential_16/simple_rnn_18/while/LessLess-sequential_16_simple_rnn_18_while_placeholderRsequential_16_simple_rnn_18_while_less_sequential_16_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: Г
*sequential_16/simple_rnn_18/while/IdentityIdentity*sequential_16/simple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_16_simple_rnn_18_while_identity3sequential_16/simple_rnn_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Л

з
%__inference_signature_wrapper_8477588
simple_rnn_18_input
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_8475773t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
И:
ц
 simple_rnn_20_while_body_84778848
4simple_rnn_20_while_simple_rnn_20_while_loop_counter>
:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations#
simple_rnn_20_while_placeholder%
!simple_rnn_20_while_placeholder_1%
!simple_rnn_20_while_placeholder_27
3simple_rnn_20_while_simple_rnn_20_strided_slice_1_0s
osimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dV
Hsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:[
Isimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0: 
simple_rnn_20_while_identity"
simple_rnn_20_while_identity_1"
simple_rnn_20_while_identity_2"
simple_rnn_20_while_identity_3"
simple_rnn_20_while_identity_45
1simple_rnn_20_while_simple_rnn_20_strided_slice_1q
msimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource:dT
Fsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource:Y
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpЦ
Esimple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   м
7simple_rnn_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_20_while_placeholderNsimple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0ƒ
<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0п
-simple_rnn_20/while/simple_rnn_cell_38/MatMulMatMul>simple_rnn_20/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¬
=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0л
.simple_rnn_20/while/simple_rnn_cell_38/BiasAddBiasAdd7simple_rnn_20/while/simple_rnn_cell_38/MatMul:product:0Esimple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€»
>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0÷
/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1MatMul!simple_rnn_20_while_placeholder_2Fsimple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ў
*simple_rnn_20/while/simple_rnn_cell_38/addAddV27simple_rnn_20/while/simple_rnn_cell_38/BiasAdd:output:09simple_rnn_20/while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Э
/simple_rnn_20/while/simple_rnn_cell_38/SoftplusSoftplus.simple_rnn_20/while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Р
8simple_rnn_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_20_while_placeholder_1simple_rnn_20_while_placeholder=simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_20/while/addAddV2simple_rnn_20_while_placeholder"simple_rnn_20/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_20/while/add_1AddV24simple_rnn_20_while_simple_rnn_20_while_loop_counter$simple_rnn_20/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_20/while/IdentityIdentitysimple_rnn_20/while/add_1:z:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_20/while/Identity_1Identity:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_20/while/Identity_2Identitysimple_rnn_20/while/add:z:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_20/while/Identity_3IdentityHsimple_rnn_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: ґ
simple_rnn_20/while/Identity_4Identity=simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0^simple_rnn_20/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_20/while/NoOpNoOp>^simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp=^simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp?^simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_20_while_identity%simple_rnn_20/while/Identity:output:0"I
simple_rnn_20_while_identity_1'simple_rnn_20/while/Identity_1:output:0"I
simple_rnn_20_while_identity_2'simple_rnn_20/while/Identity_2:output:0"I
simple_rnn_20_while_identity_3'simple_rnn_20/while/Identity_3:output:0"I
simple_rnn_20_while_identity_4'simple_rnn_20/while/Identity_4:output:0"h
1simple_rnn_20_while_simple_rnn_20_strided_slice_13simple_rnn_20_while_simple_rnn_20_strided_slice_1_0"Т
Fsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resourceHsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resourceIsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resourceGsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0"а
msimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensorosimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2~
=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2|
<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp2А
>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_8477207
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8477207___redundant_placeholder05
1while_while_cond_8477207___redundant_placeholder15
1while_while_cond_8477207___redundant_placeholder25
1while_while_cond_8477207___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
љ
њ
/__inference_simple_rnn_18_layer_call_fn_8478288
inputs_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476056}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
э,
“
while_body_8479412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ь
о
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475821

inputs

states1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ]

Identity_1IdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_namestates
£>
…
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478418
inputs_0D
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478352*
condR
while_cond_8478351*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
б
ѓ
while_cond_8478351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478351___redundant_placeholder05
1while_while_cond_8478351___redundant_placeholder15
1while_while_cond_8478351___redundant_placeholder25
1while_while_cond_8478351___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
л=
«
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478742

inputsD
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478676*
condR
while_cond_8478675*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ƒ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ш
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
Р>
≈
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479370
inputs_0C
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479304*
condR
while_cond_8479303*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
"
_user_specified_name
inputs/0
э,
“
while_body_8477078
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
№

є
 simple_rnn_18_while_cond_84776758
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_2:
6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477675___redundant_placeholder0Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477675___redundant_placeholder1Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477675___redundant_placeholder2Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477675___redundant_placeholder3 
simple_rnn_18_while_identity
Ъ
simple_rnn_18/while/LessLesssimple_rnn_18_while_placeholder6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
б
ѓ
while_cond_8478459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478459___redundant_placeholder05
1while_while_cond_8478459___redundant_placeholder15
1while_while_cond_8478459___redundant_placeholder25
1while_while_cond_8478459___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
щ,
Џ
while_body_8478352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
л=
«
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478634

inputsD
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478568*
condR
while_cond_8478567*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ƒ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ш
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
я
ѓ
while_cond_8478827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478827___redundant_placeholder05
1while_while_cond_8478827___redundant_placeholder15
1while_while_cond_8478827___redundant_placeholder25
1while_while_cond_8478827___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
і
Ћ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477002

inputs(
simple_rnn_18_8476764:	ђ$
simple_rnn_18_8476766:	ђ)
simple_rnn_18_8476768:
ђђ(
simple_rnn_19_8476879:	ђd#
simple_rnn_19_8476881:d'
simple_rnn_19_8476883:dd'
simple_rnn_20_8476994:d#
simple_rnn_20_8476996:'
simple_rnn_20_8476998:
identityИҐ%simple_rnn_18/StatefulPartitionedCallҐ%simple_rnn_19/StatefulPartitionedCallҐ%simple_rnn_20/StatefulPartitionedCall©
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_18_8476764simple_rnn_18_8476766simple_rnn_18_8476768*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476763–
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0simple_rnn_19_8476879simple_rnn_19_8476881simple_rnn_19_8476883*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476878–
%simple_rnn_20/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0simple_rnn_20_8476994simple_rnn_20_8476996simple_rnn_20_8476998*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476993В
IdentityIdentity.simple_rnn_20/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъЊ
NoOpNoOp&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall&^simple_rnn_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall2N
%simple_rnn_20/StatefulPartitionedCall%simple_rnn_20/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
Ф>
∆
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479002
inputs_0D
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478936*
condR
while_cond_8478935*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
С
є
/__inference_simple_rnn_20_layer_call_fn_8479251

inputs
unknown:d
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476993t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
Р>
≈
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479478
inputs_0C
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479412*
condR
while_cond_8479411*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
"
_user_specified_name
inputs/0
и,
‘
while_body_8479044
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
№=
ƒ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479218

inputsD
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479152*
condR
while_cond_8479151*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъdc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
џ
Ў
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477532
simple_rnn_18_input(
simple_rnn_18_8477510:	ђ$
simple_rnn_18_8477512:	ђ)
simple_rnn_18_8477514:
ђђ(
simple_rnn_19_8477517:	ђd#
simple_rnn_19_8477519:d'
simple_rnn_19_8477521:dd'
simple_rnn_20_8477524:d#
simple_rnn_20_8477526:'
simple_rnn_20_8477528:
identityИҐ%simple_rnn_18/StatefulPartitionedCallҐ%simple_rnn_19/StatefulPartitionedCallҐ%simple_rnn_20/StatefulPartitionedCallґ
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputsimple_rnn_18_8477510simple_rnn_18_8477512simple_rnn_18_8477514*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476763–
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0simple_rnn_19_8477517simple_rnn_19_8477519simple_rnn_19_8477521*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476878–
%simple_rnn_20/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0simple_rnn_20_8477524simple_rnn_20_8477526simple_rnn_20_8477528*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476993В
IdentityIdentity.simple_rnn_20/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъЊ
NoOpNoOp&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall&^simple_rnn_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall2N
%simple_rnn_20/StatefulPartitionedCall%simple_rnn_20/StatefulPartitionedCall:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
ї4
®
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8475897

inputs-
simple_rnn_cell_36_8475822:	ђ)
simple_rnn_cell_36_8475824:	ђ.
simple_rnn_cell_36_8475826:
ђђ
identityИҐ*simple_rnn_cell_36/StatefulPartitionedCallҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskх
*simple_rnn_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_36_8475822simple_rnn_cell_36_8475824simple_rnn_cell_36_8475826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475821n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_36_8475822simple_rnn_cell_36_8475824simple_rnn_cell_36_8475826*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8475834*
condR
while_cond_8475833*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ{
NoOpNoOp+^simple_rnn_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_36/StatefulPartitionedCall*simple_rnn_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ф9
ш
 simple_rnn_19_while_body_84780968
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_27
3simple_rnn_19_while_simple_rnn_19_strided_slice_1_0s
osimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdV
Hsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:d[
Isimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd 
simple_rnn_19_while_identity"
simple_rnn_19_while_identity_1"
simple_rnn_19_while_identity_2"
simple_rnn_19_while_identity_3"
simple_rnn_19_while_identity_45
1simple_rnn_19_while_simple_rnn_19_strided_slice_1q
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdT
Fsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource:dY
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpЦ
Esimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  н
7simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_19_while_placeholderNsimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0≈
<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0п
-simple_rnn_19/while/simple_rnn_cell_37/MatMulMatMul>simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¬
=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0л
.simple_rnn_19/while/simple_rnn_cell_37/BiasAddBiasAdd7simple_rnn_19/while/simple_rnn_cell_37/MatMul:product:0Esimple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d»
>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0÷
/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1MatMul!simple_rnn_19_while_placeholder_2Fsimple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dў
*simple_rnn_19/while/simple_rnn_cell_37/addAddV27simple_rnn_19/while/simple_rnn_cell_37/BiasAdd:output:09simple_rnn_19/while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dЫ
.simple_rnn_19/while/simple_rnn_cell_37/SigmoidSigmoid.simple_rnn_19/while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЕ
8simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_19_while_placeholder_1simple_rnn_19_while_placeholder2simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_19/while/addAddV2simple_rnn_19_while_placeholder"simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_19/while/add_1AddV24simple_rnn_19_while_simple_rnn_19_while_loop_counter$simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/add_1:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_19/while/Identity_1Identity:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_19/while/Identity_2Identitysimple_rnn_19/while/add:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_19/while/Identity_3IdentityHsimple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_19/while/Identity_4Identity2simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0^simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dЪ
simple_rnn_19/while/NoOpNoOp>^simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp=^simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp?^simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0"I
simple_rnn_19_while_identity_1'simple_rnn_19/while/Identity_1:output:0"I
simple_rnn_19_while_identity_2'simple_rnn_19/while/Identity_2:output:0"I
simple_rnn_19_while_identity_3'simple_rnn_19/while/Identity_3:output:0"I
simple_rnn_19_while_identity_4'simple_rnn_19/while/Identity_4:output:0"h
1simple_rnn_19_while_simple_rnn_19_strided_slice_13simple_rnn_19_while_simple_rnn_19_strided_slice_1_0"Т
Fsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resourceHsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resourceIsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resourceGsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0"а
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2~
=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2|
<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp2А
>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
Х
н
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479801

inputs
states_01
matmul_readvariableop_resource:	ђd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€dZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d\

Identity_1IdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€d
"
_user_specified_name
states/0
№=
ƒ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476878

inputsD
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476812*
condR
while_cond_8476811*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъdc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
б
ѓ
while_cond_8476696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476696___redundant_placeholder05
1while_while_cond_8476696___redundant_placeholder15
1while_while_cond_8476696___redundant_placeholder25
1while_while_cond_8476696___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
–Q
ч
 __inference__traced_save_8480005
file_prefixF
Bsavev2_simple_rnn_18_simple_rnn_cell_36_kernel_read_readvariableopP
Lsavev2_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_18_simple_rnn_cell_36_bias_read_readvariableopF
Bsavev2_simple_rnn_19_simple_rnn_cell_37_kernel_read_readvariableopP
Lsavev2_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_19_simple_rnn_cell_37_bias_read_readvariableopF
Bsavev2_simple_rnn_20_simple_rnn_cell_38_kernel_read_readvariableopP
Lsavev2_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_20_simple_rnn_cell_38_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopM
Isavev2_adam_simple_rnn_18_simple_rnn_cell_36_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_18_simple_rnn_cell_36_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_19_simple_rnn_cell_37_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_19_simple_rnn_cell_37_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_20_simple_rnn_cell_38_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_20_simple_rnn_cell_38_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_18_simple_rnn_cell_36_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_18_simple_rnn_cell_36_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_19_simple_rnn_cell_37_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_19_simple_rnn_cell_37_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_20_simple_rnn_cell_38_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_20_simple_rnn_cell_38_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Л
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*і
value™BІ#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH≥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ‘
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_simple_rnn_18_simple_rnn_cell_36_kernel_read_readvariableopLsavev2_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_read_readvariableop@savev2_simple_rnn_18_simple_rnn_cell_36_bias_read_readvariableopBsavev2_simple_rnn_19_simple_rnn_cell_37_kernel_read_readvariableopLsavev2_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_read_readvariableop@savev2_simple_rnn_19_simple_rnn_cell_37_bias_read_readvariableopBsavev2_simple_rnn_20_simple_rnn_cell_38_kernel_read_readvariableopLsavev2_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_read_readvariableop@savev2_simple_rnn_20_simple_rnn_cell_38_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopIsavev2_adam_simple_rnn_18_simple_rnn_cell_36_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_18_simple_rnn_cell_36_bias_m_read_readvariableopIsavev2_adam_simple_rnn_19_simple_rnn_cell_37_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_19_simple_rnn_cell_37_bias_m_read_readvariableopIsavev2_adam_simple_rnn_20_simple_rnn_cell_38_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_20_simple_rnn_cell_38_bias_m_read_readvariableopIsavev2_adam_simple_rnn_18_simple_rnn_cell_36_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_18_simple_rnn_cell_36_bias_v_read_readvariableopIsavev2_adam_simple_rnn_19_simple_rnn_cell_37_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_19_simple_rnn_cell_37_bias_v_read_readvariableopIsavev2_adam_simple_rnn_20_simple_rnn_cell_38_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_20_simple_rnn_cell_38_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*†
_input_shapesО
Л: :	ђ:
ђђ:ђ:	ђd:dd:d:d::: : : : : : : :	ђ:
ђђ:ђ:	ђd:dd:d:d:::	ђ:
ђђ:ђ:	ђd:dd:d:d::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђd:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d:$ 

_output_shapes

:: 	

_output_shapes
::
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
:	ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђd:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђd:$ 

_output_shapes

:dd: 

_output_shapes
:d:$  

_output_shapes

:d:$! 

_output_shapes

:: "

_output_shapes
::#

_output_shapes
: 
Ј
ї
/__inference_simple_rnn_20_layer_call_fn_8479240
inputs_0
unknown:d
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476640|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
"
_user_specified_name
inputs/0
£
к
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476525

inputs

states0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€O
SoftplusSoftplusadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g

Identity_1IdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
щ,
Џ
while_body_8477338
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
џ
Ў
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477557
simple_rnn_18_input(
simple_rnn_18_8477535:	ђ$
simple_rnn_18_8477537:	ђ)
simple_rnn_18_8477539:
ђђ(
simple_rnn_19_8477542:	ђd#
simple_rnn_19_8477544:d'
simple_rnn_19_8477546:dd'
simple_rnn_20_8477549:d#
simple_rnn_20_8477551:'
simple_rnn_20_8477553:
identityИҐ%simple_rnn_18/StatefulPartitionedCallҐ%simple_rnn_19/StatefulPartitionedCallҐ%simple_rnn_20/StatefulPartitionedCallґ
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputsimple_rnn_18_8477535simple_rnn_18_8477537simple_rnn_18_8477539*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8477404–
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0simple_rnn_19_8477542simple_rnn_19_8477544simple_rnn_19_8477546*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8477274–
%simple_rnn_20/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0simple_rnn_20_8477549simple_rnn_20_8477551simple_rnn_20_8477553*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8477144В
IdentityIdentity.simple_rnn_20/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъЊ
NoOpNoOp&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall&^simple_rnn_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall2N
%simple_rnn_20/StatefulPartitionedCall%simple_rnn_20/StatefulPartitionedCall:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
≠4
§
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476481

inputs,
simple_rnn_cell_38_8476406:d(
simple_rnn_cell_38_8476408:,
simple_rnn_cell_38_8476410:
identityИҐ*simple_rnn_cell_38/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskу
*simple_rnn_cell_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_38_8476406simple_rnn_cell_38_8476408simple_rnn_cell_38_8476410*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476405n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_38_8476406simple_rnn_cell_38_8476408simple_rnn_cell_38_8476410*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476418*
condR
while_cond_8476417*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€{
NoOpNoOp+^simple_rnn_cell_38/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 2X
*simple_rnn_cell_38/StatefulPartitionedCall*simple_rnn_cell_38/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
 
_user_specified_nameinputs
э,
“
while_body_8479628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ј

№
4__inference_simple_rnn_cell_38_layer_call_fn_8479846

inputs
states_0
unknown:d
	unknown_0:
	unknown_1:
identity

identity_1ИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476525o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
№=
ƒ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8477274

inputsD
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8477208*
condR
while_cond_8477207*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъdc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
б
ѓ
while_cond_8475992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8475992___redundant_placeholder05
1while_while_cond_8475992___redundant_placeholder15
1while_while_cond_8475992___redundant_placeholder25
1while_while_cond_8475992___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Ф>
∆
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8478894
inputs_0D
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478828*
condR
while_cond_8478827*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
°!
я
while_body_8476577
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_38_8476599_0:d0
"while_simple_rnn_cell_38_8476601_0:4
"while_simple_rnn_cell_38_8476603_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_38_8476599:d.
 while_simple_rnn_cell_38_8476601:2
 while_simple_rnn_cell_38_8476603:ИҐ0while/simple_rnn_cell_38/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0Ѓ
0while/simple_rnn_cell_38/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_38_8476599_0"while_simple_rnn_cell_38_8476601_0"while_simple_rnn_cell_38_8476603_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476525в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_38/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ц
while/Identity_4Identity9while/simple_rnn_cell_38/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€

while/NoOpNoOp1^while/simple_rnn_cell_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_38_8476599"while_simple_rnn_cell_38_8476599_0"F
 while_simple_rnn_cell_38_8476601"while_simple_rnn_cell_38_8476601_0"F
 while_simple_rnn_cell_38_8476603"while_simple_rnn_cell_38_8476603_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2d
0while/simple_rnn_cell_38/StatefulPartitionedCall0while/simple_rnn_cell_38/StatefulPartitionedCall: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
≤4
•
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476189

inputs-
simple_rnn_cell_37_8476114:	ђd(
simple_rnn_cell_37_8476116:d,
simple_rnn_cell_37_8476118:dd
identityИҐ*simple_rnn_cell_37/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskу
*simple_rnn_cell_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_37_8476114simple_rnn_cell_37_8476116simple_rnn_cell_37_8476118*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476113n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_37_8476114simple_rnn_cell_37_8476116simple_rnn_cell_37_8476118*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476126*
condR
while_cond_8476125*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d{
NoOpNoOp+^simple_rnn_cell_37/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2X
*simple_rnn_cell_37/StatefulPartitionedCall*simple_rnn_cell_37/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
Ґ
р
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479756

inputs
states_01
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ]

Identity_1IdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
states/0
’
√
.sequential_16_simple_rnn_20_while_cond_8475706T
Psequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_loop_counterZ
Vsequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_maximum_iterations1
-sequential_16_simple_rnn_20_while_placeholder3
/sequential_16_simple_rnn_20_while_placeholder_13
/sequential_16_simple_rnn_20_while_placeholder_2V
Rsequential_16_simple_rnn_20_while_less_sequential_16_simple_rnn_20_strided_slice_1m
isequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_cond_8475706___redundant_placeholder0m
isequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_cond_8475706___redundant_placeholder1m
isequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_cond_8475706___redundant_placeholder2m
isequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_cond_8475706___redundant_placeholder3.
*sequential_16_simple_rnn_20_while_identity
“
&sequential_16/simple_rnn_20/while/LessLess-sequential_16_simple_rnn_20_while_placeholderRsequential_16_simple_rnn_20_while_less_sequential_16_simple_rnn_20_strided_slice_1*
T0*
_output_shapes
: Г
*sequential_16/simple_rnn_20/while/IdentityIdentity*sequential_16/simple_rnn_20/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_16_simple_rnn_20_while_identity3sequential_16/simple_rnn_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
№

є
 simple_rnn_18_while_cond_84779918
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_2:
6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477991___redundant_placeholder0Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477991___redundant_placeholder1Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477991___redundant_placeholder2Q
Msimple_rnn_18_while_simple_rnn_18_while_cond_8477991___redundant_placeholder3 
simple_rnn_18_while_identity
Ъ
simple_rnn_18/while/LessLesssimple_rnn_18_while_placeholder6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Ў=
√
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476993

inputsC
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476927*
condR
while_cond_8476926*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
і
Ћ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477463

inputs(
simple_rnn_18_8477441:	ђ$
simple_rnn_18_8477443:	ђ)
simple_rnn_18_8477445:
ђђ(
simple_rnn_19_8477448:	ђd#
simple_rnn_19_8477450:d'
simple_rnn_19_8477452:dd'
simple_rnn_20_8477455:d#
simple_rnn_20_8477457:'
simple_rnn_20_8477459:
identityИҐ%simple_rnn_18/StatefulPartitionedCallҐ%simple_rnn_19/StatefulPartitionedCallҐ%simple_rnn_20/StatefulPartitionedCall©
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_18_8477441simple_rnn_18_8477443simple_rnn_18_8477445*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8477404–
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0simple_rnn_19_8477448simple_rnn_19_8477450simple_rnn_19_8477452*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8477274–
%simple_rnn_20/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0simple_rnn_20_8477455simple_rnn_20_8477457simple_rnn_20_8477459*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8477144В
IdentityIdentity.simple_rnn_20/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъЊ
NoOpNoOp&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall&^simple_rnn_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall2N
%simple_rnn_20/StatefulPartitionedCall%simple_rnn_20/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
б
ѓ
while_cond_8477337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8477337___redundant_placeholder05
1while_while_cond_8477337___redundant_placeholder15
1while_while_cond_8477337___redundant_placeholder25
1while_while_cond_8477337___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Џ

є
 simple_rnn_19_while_cond_84777798
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_2:
6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8477779___redundant_placeholder0Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8477779___redundant_placeholder1Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8477779___redundant_placeholder2Q
Msimple_rnn_19_while_simple_rnn_19_while_cond_8477779___redundant_placeholder3 
simple_rnn_19_while_identity
Ъ
simple_rnn_19/while/LessLesssimple_rnn_19_while_placeholder6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
Ф
Ї
/__inference_simple_rnn_19_layer_call_fn_8478775

inputs
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476878t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
£>
…
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478526
inputs_0D
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8478460*
condR
while_cond_8478459*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
б
ѓ
while_cond_8475833
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8475833___redundant_placeholder05
1while_while_cond_8475833___redundant_placeholder15
1while_while_cond_8475833___redundant_placeholder25
1while_while_cond_8475833___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
Ў=
√
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479694

inputsC
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479628*
condR
while_cond_8479627*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
щ,
Џ
while_body_8476697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
Ѓ!
з
while_body_8475834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_simple_rnn_cell_36_8475856_0:	ђ1
"while_simple_rnn_cell_36_8475858_0:	ђ6
"while_simple_rnn_cell_36_8475860_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_simple_rnn_cell_36_8475856:	ђ/
 while_simple_rnn_cell_36_8475858:	ђ4
 while_simple_rnn_cell_36_8475860:
ђђИҐ0while/simple_rnn_cell_36/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0∞
0while/simple_rnn_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_36_8475856_0"while_simple_rnn_cell_36_8475858_0"while_simple_rnn_cell_36_8475860_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475821в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ч
while/Identity_4Identity9while/simple_rnn_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ

while/NoOpNoOp1^while/simple_rnn_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_36_8475856"while_simple_rnn_cell_36_8475856_0"F
 while_simple_rnn_cell_36_8475858"while_simple_rnn_cell_36_8475858_0"F
 while_simple_rnn_cell_36_8475860"while_simple_rnn_cell_36_8475860_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2d
0while/simple_rnn_cell_36/StatefulPartitionedCall0while/simple_rnn_cell_36/StatefulPartitionedCall: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
Ў=
√
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8477144

inputsC
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8477078*
condR
while_cond_8477077*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
Ј
ї
/__inference_simple_rnn_20_layer_call_fn_8479229
inputs_0
unknown:d
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476481|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
"
_user_specified_name
inputs/0
я
ѓ
while_cond_8476576
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476576___redundant_placeholder05
1while_while_cond_8476576___redundant_placeholder15
1while_while_cond_8476576___redundant_placeholder25
1while_while_cond_8476576___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
√С
…
#__inference__traced_restore_8480117
file_prefixK
8assignvariableop_simple_rnn_18_simple_rnn_cell_36_kernel:	ђX
Dassignvariableop_1_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel:
ђђG
8assignvariableop_2_simple_rnn_18_simple_rnn_cell_36_bias:	ђM
:assignvariableop_3_simple_rnn_19_simple_rnn_cell_37_kernel:	ђdV
Dassignvariableop_4_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel:ddF
8assignvariableop_5_simple_rnn_19_simple_rnn_cell_37_bias:dL
:assignvariableop_6_simple_rnn_20_simple_rnn_cell_38_kernel:dV
Dassignvariableop_7_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel:F
8assignvariableop_8_simple_rnn_20_simple_rnn_cell_38_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: U
Bassignvariableop_16_adam_simple_rnn_18_simple_rnn_cell_36_kernel_m:	ђ`
Lassignvariableop_17_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_m:
ђђO
@assignvariableop_18_adam_simple_rnn_18_simple_rnn_cell_36_bias_m:	ђU
Bassignvariableop_19_adam_simple_rnn_19_simple_rnn_cell_37_kernel_m:	ђd^
Lassignvariableop_20_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_m:ddN
@assignvariableop_21_adam_simple_rnn_19_simple_rnn_cell_37_bias_m:dT
Bassignvariableop_22_adam_simple_rnn_20_simple_rnn_cell_38_kernel_m:d^
Lassignvariableop_23_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_m:N
@assignvariableop_24_adam_simple_rnn_20_simple_rnn_cell_38_bias_m:U
Bassignvariableop_25_adam_simple_rnn_18_simple_rnn_cell_36_kernel_v:	ђ`
Lassignvariableop_26_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_v:
ђђO
@assignvariableop_27_adam_simple_rnn_18_simple_rnn_cell_36_bias_v:	ђU
Bassignvariableop_28_adam_simple_rnn_19_simple_rnn_cell_37_kernel_v:	ђd^
Lassignvariableop_29_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_v:ddN
@assignvariableop_30_adam_simple_rnn_19_simple_rnn_cell_37_bias_v:dT
Bassignvariableop_31_adam_simple_rnn_20_simple_rnn_cell_38_kernel_v:d^
Lassignvariableop_32_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_v:N
@assignvariableop_33_adam_simple_rnn_20_simple_rnn_cell_38_bias_v:
identity_35ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9О
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*і
value™BІ#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHґ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B –
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ґ
_output_shapesП
М:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOpAssignVariableOp8assignvariableop_simple_rnn_18_simple_rnn_cell_36_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_1AssignVariableOpDassignvariableop_1_simple_rnn_18_simple_rnn_cell_36_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_2AssignVariableOp8assignvariableop_2_simple_rnn_18_simple_rnn_cell_36_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_3AssignVariableOp:assignvariableop_3_simple_rnn_19_simple_rnn_cell_37_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_4AssignVariableOpDassignvariableop_4_simple_rnn_19_simple_rnn_cell_37_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp8assignvariableop_5_simple_rnn_19_simple_rnn_cell_37_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_6AssignVariableOp:assignvariableop_6_simple_rnn_20_simple_rnn_cell_38_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_7AssignVariableOpDassignvariableop_7_simple_rnn_20_simple_rnn_cell_38_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_8AssignVariableOp8assignvariableop_8_simple_rnn_20_simple_rnn_cell_38_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_16AssignVariableOpBassignvariableop_16_adam_simple_rnn_18_simple_rnn_cell_36_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_17AssignVariableOpLassignvariableop_17_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_simple_rnn_18_simple_rnn_cell_36_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_simple_rnn_19_simple_rnn_cell_37_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_20AssignVariableOpLassignvariableop_20_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_19_simple_rnn_cell_37_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_simple_rnn_20_simple_rnn_cell_38_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_23AssignVariableOpLassignvariableop_23_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_20_simple_rnn_cell_38_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_simple_rnn_18_simple_rnn_cell_36_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_26AssignVariableOpLassignvariableop_26_adam_simple_rnn_18_simple_rnn_cell_36_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_18_simple_rnn_cell_36_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_simple_rnn_19_simple_rnn_cell_37_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_29AssignVariableOpLassignvariableop_29_adam_simple_rnn_19_simple_rnn_cell_37_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_19_simple_rnn_cell_37_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_31AssignVariableOpBassignvariableop_31_adam_simple_rnn_20_simple_rnn_cell_38_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_32AssignVariableOpLassignvariableop_32_adam_simple_rnn_20_simple_rnn_cell_38_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_simple_rnn_20_simple_rnn_cell_38_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ї
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: ®
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
Ф
Ї
/__inference_simple_rnn_19_layer_call_fn_8478786

inputs
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8477274t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
ф9
ш
 simple_rnn_19_while_body_84777808
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_27
3simple_rnn_19_while_simple_rnn_19_strided_slice_1_0s
osimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdV
Hsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:d[
Isimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd 
simple_rnn_19_while_identity"
simple_rnn_19_while_identity_1"
simple_rnn_19_while_identity_2"
simple_rnn_19_while_identity_3"
simple_rnn_19_while_identity_45
1simple_rnn_19_while_simple_rnn_19_strided_slice_1q
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdT
Fsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource:dY
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpЦ
Esimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  н
7simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_19_while_placeholderNsimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0≈
<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0п
-simple_rnn_19/while/simple_rnn_cell_37/MatMulMatMul>simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¬
=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0л
.simple_rnn_19/while/simple_rnn_cell_37/BiasAddBiasAdd7simple_rnn_19/while/simple_rnn_cell_37/MatMul:product:0Esimple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d»
>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0÷
/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1MatMul!simple_rnn_19_while_placeholder_2Fsimple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dў
*simple_rnn_19/while/simple_rnn_cell_37/addAddV27simple_rnn_19/while/simple_rnn_cell_37/BiasAdd:output:09simple_rnn_19/while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dЫ
.simple_rnn_19/while/simple_rnn_cell_37/SigmoidSigmoid.simple_rnn_19/while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЕ
8simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_19_while_placeholder_1simple_rnn_19_while_placeholder2simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_19/while/addAddV2simple_rnn_19_while_placeholder"simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_19/while/add_1AddV24simple_rnn_19_while_simple_rnn_19_while_loop_counter$simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/add_1:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_19/while/Identity_1Identity:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_19/while/Identity_2Identitysimple_rnn_19/while/add:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_19/while/Identity_3IdentityHsimple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_19/while/Identity_4Identity2simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0^simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dЪ
simple_rnn_19/while/NoOpNoOp>^simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp=^simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp?^simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0"I
simple_rnn_19_while_identity_1'simple_rnn_19/while/Identity_1:output:0"I
simple_rnn_19_while_identity_2'simple_rnn_19/while/Identity_2:output:0"I
simple_rnn_19_while_identity_3'simple_rnn_19/while/Identity_3:output:0"I
simple_rnn_19_while_identity_4'simple_rnn_19/while/Identity_4:output:0"h
1simple_rnn_19_while_simple_rnn_19_strided_slice_13simple_rnn_19_while_simple_rnn_19_strided_slice_1_0"Т
Fsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resourceHsimple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resourceIsimple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resourceGsimple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0"а
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2~
=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp=simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2|
<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp<simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp2А
>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp>simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
и,
‘
while_body_8478828
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
э,
“
while_body_8479304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_8477077
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8477077___redundant_placeholder05
1while_while_cond_8477077___redundant_placeholder15
1while_while_cond_8477077___redundant_placeholder25
1while_while_cond_8477077___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
щ,
Џ
while_body_8478568
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
ј

№
4__inference_simple_rnn_cell_38_layer_call_fn_8479832

inputs
states_0
unknown:d
	unknown_0:
	unknown_1:
identity

identity_1ИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
АG
†
.sequential_16_simple_rnn_19_while_body_8475603T
Psequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_loop_counterZ
Vsequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_maximum_iterations1
-sequential_16_simple_rnn_19_while_placeholder3
/sequential_16_simple_rnn_19_while_placeholder_13
/sequential_16_simple_rnn_19_while_placeholder_2S
Osequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_strided_slice_1_0Р
Лsequential_16_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0h
Usequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdd
Vsequential_16_simple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:di
Wsequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd.
*sequential_16_simple_rnn_19_while_identity0
,sequential_16_simple_rnn_19_while_identity_10
,sequential_16_simple_rnn_19_while_identity_20
,sequential_16_simple_rnn_19_while_identity_30
,sequential_16_simple_rnn_19_while_identity_4Q
Msequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_strided_slice_1О
Йsequential_16_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorf
Ssequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdb
Tsequential_16_simple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource:dg
Usequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐKsequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐJsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpҐLsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp§
Ssequential_16/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  і
Esequential_16/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЛsequential_16_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0-sequential_16_simple_rnn_19_while_placeholder\sequential_16/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0б
Jsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOpUsequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0Щ
;sequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMulMatMulLsequential_16/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dё
Ksequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOpVsequential_16_simple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Х
<sequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAddBiasAddEsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul:product:0Ssequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dд
Lsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpWsequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0А
=sequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1MatMul/sequential_16_simple_rnn_19_while_placeholder_2Tsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dГ
8sequential_16/simple_rnn_19/while/simple_rnn_cell_37/addAddV2Esequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd:output:0Gsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dЈ
<sequential_16/simple_rnn_19/while/simple_rnn_cell_37/SigmoidSigmoid<sequential_16/simple_rnn_19/while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dљ
Fsequential_16/simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_16_simple_rnn_19_while_placeholder_1-sequential_16_simple_rnn_19_while_placeholder@sequential_16/simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“i
'sequential_16/simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :∞
%sequential_16/simple_rnn_19/while/addAddV2-sequential_16_simple_rnn_19_while_placeholder0sequential_16/simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_16/simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :„
'sequential_16/simple_rnn_19/while/add_1AddV2Psequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_loop_counter2sequential_16/simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: ≠
*sequential_16/simple_rnn_19/while/IdentityIdentity+sequential_16/simple_rnn_19/while/add_1:z:0'^sequential_16/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_19/while/Identity_1IdentityVsequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_while_maximum_iterations'^sequential_16/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ≠
,sequential_16/simple_rnn_19/while/Identity_2Identity)sequential_16/simple_rnn_19/while/add:z:0'^sequential_16/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_19/while/Identity_3IdentityVsequential_16/simple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_16/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ’
,sequential_16/simple_rnn_19/while/Identity_4Identity@sequential_16/simple_rnn_19/while/simple_rnn_cell_37/Sigmoid:y:0'^sequential_16/simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€d“
&sequential_16/simple_rnn_19/while/NoOpNoOpL^sequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpK^sequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpM^sequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_16_simple_rnn_19_while_identity3sequential_16/simple_rnn_19/while/Identity:output:0"e
,sequential_16_simple_rnn_19_while_identity_15sequential_16/simple_rnn_19/while/Identity_1:output:0"e
,sequential_16_simple_rnn_19_while_identity_25sequential_16/simple_rnn_19/while/Identity_2:output:0"e
,sequential_16_simple_rnn_19_while_identity_35sequential_16/simple_rnn_19/while/Identity_3:output:0"e
,sequential_16_simple_rnn_19_while_identity_45sequential_16/simple_rnn_19/while/Identity_4:output:0"†
Msequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_strided_slice_1Osequential_16_simple_rnn_19_while_sequential_16_simple_rnn_19_strided_slice_1_0"Ѓ
Tsequential_16_simple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resourceVsequential_16_simple_rnn_19_while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"∞
Usequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resourceWsequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"ђ
Ssequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resourceUsequential_16_simple_rnn_19_while_simple_rnn_cell_37_matmul_readvariableop_resource_0"Ъ
Йsequential_16_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorЛsequential_16_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2Ъ
Ksequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpKsequential_16/simple_rnn_19/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2Ш
Jsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOpJsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul/ReadVariableOp2Ь
Lsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOpLsequential_16/simple_rnn_19/while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
л=
«
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8477404

inputsD
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8477338*
condR
while_cond_8477337*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ƒ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ш
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
Ї
Љ
/__inference_simple_rnn_19_layer_call_fn_8478753
inputs_0
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476189|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
°!
я
while_body_8476418
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_38_8476440_0:d0
"while_simple_rnn_cell_38_8476442_0:4
"while_simple_rnn_cell_38_8476444_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_38_8476440:d.
 while_simple_rnn_cell_38_8476442:2
 while_simple_rnn_cell_38_8476444:ИҐ0while/simple_rnn_cell_38/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0Ѓ
0while/simple_rnn_cell_38/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_38_8476440_0"while_simple_rnn_cell_38_8476442_0"while_simple_rnn_cell_38_8476444_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476405в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_38/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ц
while/Identity_4Identity9while/simple_rnn_cell_38/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€

while/NoOpNoOp1^while/simple_rnn_cell_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_38_8476440"while_simple_rnn_cell_38_8476440_0"F
 while_simple_rnn_cell_38_8476442"while_simple_rnn_cell_38_8476442_0"F
 while_simple_rnn_cell_38_8476444"while_simple_rnn_cell_38_8476444_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2d
0while/simple_rnn_cell_38/StatefulPartitionedCall0while/simple_rnn_cell_38/StatefulPartitionedCall: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ї
Љ
/__inference_simple_rnn_19_layer_call_fn_8478764
inputs_0
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476348|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
я
ѓ
while_cond_8478935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478935___redundant_placeholder05
1while_while_cond_8478935___redundant_placeholder15
1while_while_cond_8478935___redundant_placeholder25
1while_while_cond_8478935___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
љ
њ
/__inference_simple_rnn_18_layer_call_fn_8478277
inputs_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8475897}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Ь
о
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475941

inputs

states1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ]

Identity_1IdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_namestates
И:
ц
 simple_rnn_20_while_body_84782008
4simple_rnn_20_while_simple_rnn_20_while_loop_counter>
:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations#
simple_rnn_20_while_placeholder%
!simple_rnn_20_while_placeholder_1%
!simple_rnn_20_while_placeholder_27
3simple_rnn_20_while_simple_rnn_20_strided_slice_1_0s
osimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dV
Hsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:[
Isimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0: 
simple_rnn_20_while_identity"
simple_rnn_20_while_identity_1"
simple_rnn_20_while_identity_2"
simple_rnn_20_while_identity_3"
simple_rnn_20_while_identity_45
1simple_rnn_20_while_simple_rnn_20_strided_slice_1q
msimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource:dT
Fsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource:Y
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpЦ
Esimple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   м
7simple_rnn_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_20_while_placeholderNsimple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0ƒ
<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0п
-simple_rnn_20/while/simple_rnn_cell_38/MatMulMatMul>simple_rnn_20/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¬
=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0л
.simple_rnn_20/while/simple_rnn_cell_38/BiasAddBiasAdd7simple_rnn_20/while/simple_rnn_cell_38/MatMul:product:0Esimple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€»
>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0÷
/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1MatMul!simple_rnn_20_while_placeholder_2Fsimple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ў
*simple_rnn_20/while/simple_rnn_cell_38/addAddV27simple_rnn_20/while/simple_rnn_cell_38/BiasAdd:output:09simple_rnn_20/while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Э
/simple_rnn_20/while/simple_rnn_cell_38/SoftplusSoftplus.simple_rnn_20/while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Р
8simple_rnn_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_20_while_placeholder_1simple_rnn_20_while_placeholder=simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_20/while/addAddV2simple_rnn_20_while_placeholder"simple_rnn_20/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_20/while/add_1AddV24simple_rnn_20_while_simple_rnn_20_while_loop_counter$simple_rnn_20/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_20/while/IdentityIdentitysimple_rnn_20/while/add_1:z:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_20/while/Identity_1Identity:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_20/while/Identity_2Identitysimple_rnn_20/while/add:z:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_20/while/Identity_3IdentityHsimple_rnn_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_20/while/NoOp*
T0*
_output_shapes
: ґ
simple_rnn_20/while/Identity_4Identity=simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0^simple_rnn_20/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_20/while/NoOpNoOp>^simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp=^simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp?^simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_20_while_identity%simple_rnn_20/while/Identity:output:0"I
simple_rnn_20_while_identity_1'simple_rnn_20/while/Identity_1:output:0"I
simple_rnn_20_while_identity_2'simple_rnn_20/while/Identity_2:output:0"I
simple_rnn_20_while_identity_3'simple_rnn_20/while/Identity_3:output:0"I
simple_rnn_20_while_identity_4'simple_rnn_20/while/Identity_4:output:0"h
1simple_rnn_20_while_simple_rnn_20_strided_slice_13simple_rnn_20_while_simple_rnn_20_strided_slice_1_0"Т
Fsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resourceHsimple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resourceIsimple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resourceGsimple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0"а
msimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensorosimple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2~
=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp=simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2|
<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp<simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp2А
>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp>simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
щ,
Џ
while_body_8478676
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђI
:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђO
;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђG
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђM
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0©
.while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0∆
while/simple_rnn_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђІ
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0¬
 while/simple_rnn_cell_36/BiasAddBiasAdd)while/simple_rnn_cell_36/MatMul:product:07while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЃ
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0≠
!while/simple_rnn_cell_36/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ∞
while/simple_rnn_cell_36/addAddV2)while/simple_rnn_cell_36/BiasAdd:output:0+while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђА
 while/simple_rnn_cell_36/SigmoidSigmoid while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: В
while/Identity_4Identity$while/simple_rnn_cell_36/Sigmoid:y:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђв

while/NoOpNoOp0^while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_36/MatMul/ReadVariableOp1^while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_36_biasadd_readvariableop_resource:while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_36_matmul_1_readvariableop_resource;while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_36_matmul_readvariableop_resource9while_simple_rnn_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2b
/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_36/MatMul/ReadVariableOp.while/simple_rnn_cell_36/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp0while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
 

а
4__inference_simple_rnn_cell_36_layer_call_fn_8479722

inputs
states_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475941p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
states/0
£
к
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476405

inputs

states0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€O
SoftplusSoftplusadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g

Identity_1IdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
§!
б
while_body_8476126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_simple_rnn_cell_37_8476148_0:	ђd0
"while_simple_rnn_cell_37_8476150_0:d4
"while_simple_rnn_cell_37_8476152_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_simple_rnn_cell_37_8476148:	ђd.
 while_simple_rnn_cell_37_8476150:d2
 while_simple_rnn_cell_37_8476152:ddИҐ0while/simple_rnn_cell_37/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0Ѓ
0while/simple_rnn_cell_37/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_37_8476148_0"while_simple_rnn_cell_37_8476150_0"while_simple_rnn_cell_37_8476152_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476113в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_37/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ц
while/Identity_4Identity9while/simple_rnn_cell_37/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€d

while/NoOpNoOp1^while/simple_rnn_cell_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_37_8476148"while_simple_rnn_cell_37_8476148_0"F
 while_simple_rnn_cell_37_8476150"while_simple_rnn_cell_37_8476150_0"F
 while_simple_rnn_cell_37_8476152"while_simple_rnn_cell_37_8476152_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2d
0while/simple_rnn_cell_37/StatefulPartitionedCall0while/simple_rnn_cell_37/StatefulPartitionedCall: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
Ц

д
/__inference_sequential_16_layer_call_fn_8477611

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477002t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
Д:
ю
 simple_rnn_18_while_body_84776768
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_27
3simple_rnn_18_while_simple_rnn_18_strided_slice_1_0s
osimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0:	ђW
Hsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0:	ђ]
Isimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0:
ђђ 
simple_rnn_18_while_identity"
simple_rnn_18_while_identity_1"
simple_rnn_18_while_identity_2"
simple_rnn_18_while_identity_3"
simple_rnn_18_while_identity_45
1simple_rnn_18_while_simple_rnn_18_strided_slice_1q
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource:	ђU
Fsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђ[
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђИҐ=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpҐ>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpЦ
Esimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_18_while_placeholderNsimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0≈
<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0р
-simple_rnn_18/while/simple_rnn_cell_36/MatMulMatMul>simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ√
=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0м
.simple_rnn_18/while/simple_rnn_cell_36/BiasAddBiasAdd7simple_rnn_18/while/simple_rnn_cell_36/MatMul:product:0Esimple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ 
>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0„
/simple_rnn_18/while/simple_rnn_cell_36/MatMul_1MatMul!simple_rnn_18_while_placeholder_2Fsimple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЏ
*simple_rnn_18/while/simple_rnn_cell_36/addAddV27simple_rnn_18/while/simple_rnn_cell_36/BiasAdd:output:09simple_rnn_18/while/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђЬ
.simple_rnn_18/while/simple_rnn_cell_36/SigmoidSigmoid.simple_rnn_18/while/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђЕ
8simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_18_while_placeholder_1simple_rnn_18_while_placeholder2simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_18/while/addAddV2simple_rnn_18_while_placeholder"simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_18/while/add_1AddV24simple_rnn_18_while_simple_rnn_18_while_loop_counter$simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/add_1:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_18/while/Identity_1Identity:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_18/while/Identity_2Identitysimple_rnn_18/while/add:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_18/while/Identity_3IdentityHsimple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ђ
simple_rnn_18/while/Identity_4Identity2simple_rnn_18/while/simple_rnn_cell_36/Sigmoid:y:0^simple_rnn_18/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђЪ
simple_rnn_18/while/NoOpNoOp>^simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp=^simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp?^simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0"I
simple_rnn_18_while_identity_1'simple_rnn_18/while/Identity_1:output:0"I
simple_rnn_18_while_identity_2'simple_rnn_18/while/Identity_2:output:0"I
simple_rnn_18_while_identity_3'simple_rnn_18/while/Identity_3:output:0"I
simple_rnn_18_while_identity_4'simple_rnn_18/while/Identity_4:output:0"h
1simple_rnn_18_while_simple_rnn_18_strided_slice_13simple_rnn_18_while_simple_rnn_18_strided_slice_1_0"Т
Fsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resourceHsimple_rnn_18_while_simple_rnn_cell_36_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resourceIsimple_rnn_18_while_simple_rnn_cell_36_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resourceGsimple_rnn_18_while_simple_rnn_cell_36_matmul_readvariableop_resource_0"а
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2~
=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp=simple_rnn_18/while/simple_rnn_cell_36/BiasAdd/ReadVariableOp2|
<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp<simple_rnn_18/while/simple_rnn_cell_36/MatMul/ReadVariableOp2А
>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp>simple_rnn_18/while/simple_rnn_cell_36/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
√

Ё
4__inference_simple_rnn_cell_37_layer_call_fn_8479784

inputs
states_0
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identity

identity_1ИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€d
"
_user_specified_name
states/0
бВ
≠
"__inference__wrapped_model_8475773
simple_rnn_18_input`
Msequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource:	ђ]
Nsequential_16_simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource:	ђc
Osequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ`
Msequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource:	ђd\
Nsequential_16_simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource:da
Osequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource:dd_
Msequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource:d\
Nsequential_16_simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource:a
Osequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐEsequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpҐDsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpҐFsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpҐ!sequential_16/simple_rnn_18/whileҐEsequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐDsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpҐFsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpҐ!sequential_16/simple_rnn_19/whileҐEsequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐDsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpҐFsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpҐ!sequential_16/simple_rnn_20/whiled
!sequential_16/simple_rnn_18/ShapeShapesimple_rnn_18_input*
T0*
_output_shapes
:y
/sequential_16/simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_16/simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_16/simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential_16/simple_rnn_18/strided_sliceStridedSlice*sequential_16/simple_rnn_18/Shape:output:08sequential_16/simple_rnn_18/strided_slice/stack:output:0:sequential_16/simple_rnn_18/strided_slice/stack_1:output:0:sequential_16/simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*sequential_16/simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђ«
(sequential_16/simple_rnn_18/zeros/packedPack2sequential_16/simple_rnn_18/strided_slice:output:03sequential_16/simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_16/simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ѕ
!sequential_16/simple_rnn_18/zerosFill1sequential_16/simple_rnn_18/zeros/packed:output:00sequential_16/simple_rnn_18/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ
*sequential_16/simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≥
%sequential_16/simple_rnn_18/transpose	Transposesimple_rnn_18_input3sequential_16/simple_rnn_18/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€|
#sequential_16/simple_rnn_18/Shape_1Shape)sequential_16/simple_rnn_18/transpose:y:0*
T0*
_output_shapes
:{
1sequential_16/simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential_16/simple_rnn_18/strided_slice_1StridedSlice,sequential_16/simple_rnn_18/Shape_1:output:0:sequential_16/simple_rnn_18/strided_slice_1/stack:output:0<sequential_16/simple_rnn_18/strided_slice_1/stack_1:output:0<sequential_16/simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7sequential_16/simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
)sequential_16/simple_rnn_18/TensorArrayV2TensorListReserve@sequential_16/simple_rnn_18/TensorArrayV2/element_shape:output:04sequential_16/simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ґ
Qsequential_16/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   і
Csequential_16/simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_16/simple_rnn_18/transpose:y:0Zsequential_16/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“{
1sequential_16/simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+sequential_16/simple_rnn_18/strided_slice_2StridedSlice)sequential_16/simple_rnn_18/transpose:y:0:sequential_16/simple_rnn_18/strided_slice_2/stack:output:0<sequential_16/simple_rnn_18/strided_slice_2/stack_1:output:0<sequential_16/simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask”
Dsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOpMsequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ц
5sequential_16/simple_rnn_18/simple_rnn_cell_36/MatMulMatMul4sequential_16/simple_rnn_18/strided_slice_2:output:0Lsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ—
Esequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOpNsequential_16_simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
6sequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAddBiasAdd?sequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul:product:0Msequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЎ
Fsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOpOsequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0р
7sequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1MatMul*sequential_16/simple_rnn_18/zeros:output:0Nsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђт
2sequential_16/simple_rnn_18/simple_rnn_cell_36/addAddV2?sequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd:output:0Asequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђђ
6sequential_16/simple_rnn_18/simple_rnn_cell_36/SigmoidSigmoid6sequential_16/simple_rnn_18/simple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђК
9sequential_16/simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  М
+sequential_16/simple_rnn_18/TensorArrayV2_1TensorListReserveBsequential_16/simple_rnn_18/TensorArrayV2_1/element_shape:output:04sequential_16/simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“b
 sequential_16/simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_16/simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€p
.sequential_16/simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ћ
!sequential_16/simple_rnn_18/whileWhile7sequential_16/simple_rnn_18/while/loop_counter:output:0=sequential_16/simple_rnn_18/while/maximum_iterations:output:0)sequential_16/simple_rnn_18/time:output:04sequential_16/simple_rnn_18/TensorArrayV2_1:handle:0*sequential_16/simple_rnn_18/zeros:output:04sequential_16/simple_rnn_18/strided_slice_1:output:0Ssequential_16/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_readvariableop_resourceNsequential_16_simple_rnn_18_simple_rnn_cell_36_biasadd_readvariableop_resourceOsequential_16_simple_rnn_18_simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.sequential_16_simple_rnn_18_while_body_8475499*:
cond2R0
.sequential_16_simple_rnn_18_while_cond_8475498*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Э
Lsequential_16/simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Ш
>sequential_16/simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_16/simple_rnn_18/while:output:3Usequential_16/simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0Д
1sequential_16/simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential_16/simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
+sequential_16/simple_rnn_18/strided_slice_3StridedSliceGsequential_16/simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_16/simple_rnn_18/strided_slice_3/stack:output:0<sequential_16/simple_rnn_18/strided_slice_3/stack_1:output:0<sequential_16/simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskБ
,sequential_16/simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
'sequential_16/simple_rnn_18/transpose_1	TransposeGsequential_16/simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:05sequential_16/simple_rnn_18/transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђ|
!sequential_16/simple_rnn_19/ShapeShape+sequential_16/simple_rnn_18/transpose_1:y:0*
T0*
_output_shapes
:y
/sequential_16/simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_16/simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_16/simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential_16/simple_rnn_19/strided_sliceStridedSlice*sequential_16/simple_rnn_19/Shape:output:08sequential_16/simple_rnn_19/strided_slice/stack:output:0:sequential_16/simple_rnn_19/strided_slice/stack_1:output:0:sequential_16/simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_16/simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d«
(sequential_16/simple_rnn_19/zeros/packedPack2sequential_16/simple_rnn_19/strided_slice:output:03sequential_16/simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_16/simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
!sequential_16/simple_rnn_19/zerosFill1sequential_16/simple_rnn_19/zeros/packed:output:00sequential_16/simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
*sequential_16/simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ћ
%sequential_16/simple_rnn_19/transpose	Transpose+sequential_16/simple_rnn_18/transpose_1:y:03sequential_16/simple_rnn_19/transpose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђ|
#sequential_16/simple_rnn_19/Shape_1Shape)sequential_16/simple_rnn_19/transpose:y:0*
T0*
_output_shapes
:{
1sequential_16/simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential_16/simple_rnn_19/strided_slice_1StridedSlice,sequential_16/simple_rnn_19/Shape_1:output:0:sequential_16/simple_rnn_19/strided_slice_1/stack:output:0<sequential_16/simple_rnn_19/strided_slice_1/stack_1:output:0<sequential_16/simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7sequential_16/simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
)sequential_16/simple_rnn_19/TensorArrayV2TensorListReserve@sequential_16/simple_rnn_19/TensorArrayV2/element_shape:output:04sequential_16/simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ґ
Qsequential_16/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  і
Csequential_16/simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_16/simple_rnn_19/transpose:y:0Zsequential_16/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“{
1sequential_16/simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
+sequential_16/simple_rnn_19/strided_slice_2StridedSlice)sequential_16/simple_rnn_19/transpose:y:0:sequential_16/simple_rnn_19/strided_slice_2/stack:output:0<sequential_16/simple_rnn_19/strided_slice_2/stack_1:output:0<sequential_16/simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask”
Dsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOpMsequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0х
5sequential_16/simple_rnn_19/simple_rnn_cell_37/MatMulMatMul4sequential_16/simple_rnn_19/strided_slice_2:output:0Lsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d–
Esequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOpNsequential_16_simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Г
6sequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAddBiasAdd?sequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul:product:0Msequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d÷
Fsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOpOsequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0п
7sequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1MatMul*sequential_16/simple_rnn_19/zeros:output:0Nsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dс
2sequential_16/simple_rnn_19/simple_rnn_cell_37/addAddV2?sequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd:output:0Asequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dЂ
6sequential_16/simple_rnn_19/simple_rnn_cell_37/SigmoidSigmoid6sequential_16/simple_rnn_19/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dК
9sequential_16/simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   М
+sequential_16/simple_rnn_19/TensorArrayV2_1TensorListReserveBsequential_16/simple_rnn_19/TensorArrayV2_1/element_shape:output:04sequential_16/simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“b
 sequential_16/simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_16/simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€p
.sequential_16/simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : …
!sequential_16/simple_rnn_19/whileWhile7sequential_16/simple_rnn_19/while/loop_counter:output:0=sequential_16/simple_rnn_19/while/maximum_iterations:output:0)sequential_16/simple_rnn_19/time:output:04sequential_16/simple_rnn_19/TensorArrayV2_1:handle:0*sequential_16/simple_rnn_19/zeros:output:04sequential_16/simple_rnn_19/strided_slice_1:output:0Ssequential_16/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_readvariableop_resourceNsequential_16_simple_rnn_19_simple_rnn_cell_37_biasadd_readvariableop_resourceOsequential_16_simple_rnn_19_simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.sequential_16_simple_rnn_19_while_body_8475603*:
cond2R0
.sequential_16_simple_rnn_19_while_cond_8475602*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Э
Lsequential_16/simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Ч
>sequential_16/simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_16/simple_rnn_19/while:output:3Usequential_16/simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0Д
1sequential_16/simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential_16/simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:У
+sequential_16/simple_rnn_19/strided_slice_3StridedSliceGsequential_16/simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_16/simple_rnn_19/strided_slice_3/stack:output:0<sequential_16/simple_rnn_19/strided_slice_3/stack_1:output:0<sequential_16/simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskБ
,sequential_16/simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          л
'sequential_16/simple_rnn_19/transpose_1	TransposeGsequential_16/simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:05sequential_16/simple_rnn_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъd|
!sequential_16/simple_rnn_20/ShapeShape+sequential_16/simple_rnn_19/transpose_1:y:0*
T0*
_output_shapes
:y
/sequential_16/simple_rnn_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_16/simple_rnn_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_16/simple_rnn_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential_16/simple_rnn_20/strided_sliceStridedSlice*sequential_16/simple_rnn_20/Shape:output:08sequential_16/simple_rnn_20/strided_slice/stack:output:0:sequential_16/simple_rnn_20/strided_slice/stack_1:output:0:sequential_16/simple_rnn_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_16/simple_rnn_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :«
(sequential_16/simple_rnn_20/zeros/packedPack2sequential_16/simple_rnn_20/strided_slice:output:03sequential_16/simple_rnn_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_16/simple_rnn_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
!sequential_16/simple_rnn_20/zerosFill1sequential_16/simple_rnn_20/zeros/packed:output:00sequential_16/simple_rnn_20/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
*sequential_16/simple_rnn_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
%sequential_16/simple_rnn_20/transpose	Transpose+sequential_16/simple_rnn_19/transpose_1:y:03sequential_16/simple_rnn_20/transpose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€d|
#sequential_16/simple_rnn_20/Shape_1Shape)sequential_16/simple_rnn_20/transpose:y:0*
T0*
_output_shapes
:{
1sequential_16/simple_rnn_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential_16/simple_rnn_20/strided_slice_1StridedSlice,sequential_16/simple_rnn_20/Shape_1:output:0:sequential_16/simple_rnn_20/strided_slice_1/stack:output:0<sequential_16/simple_rnn_20/strided_slice_1/stack_1:output:0<sequential_16/simple_rnn_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7sequential_16/simple_rnn_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
)sequential_16/simple_rnn_20/TensorArrayV2TensorListReserve@sequential_16/simple_rnn_20/TensorArrayV2/element_shape:output:04sequential_16/simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ґ
Qsequential_16/simple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   і
Csequential_16/simple_rnn_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_16/simple_rnn_20/transpose:y:0Zsequential_16/simple_rnn_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“{
1sequential_16/simple_rnn_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_16/simple_rnn_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+sequential_16/simple_rnn_20/strided_slice_2StridedSlice)sequential_16/simple_rnn_20/transpose:y:0:sequential_16/simple_rnn_20/strided_slice_2/stack:output:0<sequential_16/simple_rnn_20/strided_slice_2/stack_1:output:0<sequential_16/simple_rnn_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_mask“
Dsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOpMsequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0х
5sequential_16/simple_rnn_20/simple_rnn_cell_38/MatMulMatMul4sequential_16/simple_rnn_20/strided_slice_2:output:0Lsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€–
Esequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOpNsequential_16_simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
6sequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAddBiasAdd?sequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul:product:0Msequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€÷
Fsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpOsequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0п
7sequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1MatMul*sequential_16/simple_rnn_20/zeros:output:0Nsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€с
2sequential_16/simple_rnn_20/simple_rnn_cell_38/addAddV2?sequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd:output:0Asequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€≠
7sequential_16/simple_rnn_20/simple_rnn_cell_38/SoftplusSoftplus6sequential_16/simple_rnn_20/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
9sequential_16/simple_rnn_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   М
+sequential_16/simple_rnn_20/TensorArrayV2_1TensorListReserveBsequential_16/simple_rnn_20/TensorArrayV2_1/element_shape:output:04sequential_16/simple_rnn_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“b
 sequential_16/simple_rnn_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_16/simple_rnn_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€p
.sequential_16/simple_rnn_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : …
!sequential_16/simple_rnn_20/whileWhile7sequential_16/simple_rnn_20/while/loop_counter:output:0=sequential_16/simple_rnn_20/while/maximum_iterations:output:0)sequential_16/simple_rnn_20/time:output:04sequential_16/simple_rnn_20/TensorArrayV2_1:handle:0*sequential_16/simple_rnn_20/zeros:output:04sequential_16/simple_rnn_20/strided_slice_1:output:0Ssequential_16/simple_rnn_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_readvariableop_resourceNsequential_16_simple_rnn_20_simple_rnn_cell_38_biasadd_readvariableop_resourceOsequential_16_simple_rnn_20_simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.sequential_16_simple_rnn_20_while_body_8475707*:
cond2R0
.sequential_16_simple_rnn_20_while_cond_8475706*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Э
Lsequential_16/simple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ч
>sequential_16/simple_rnn_20/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_16/simple_rnn_20/while:output:3Usequential_16/simple_rnn_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0Д
1sequential_16/simple_rnn_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential_16/simple_rnn_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_16/simple_rnn_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:У
+sequential_16/simple_rnn_20/strided_slice_3StridedSliceGsequential_16/simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_16/simple_rnn_20/strided_slice_3/stack:output:0<sequential_16/simple_rnn_20/strided_slice_3/stack_1:output:0<sequential_16/simple_rnn_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskБ
,sequential_16/simple_rnn_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          л
'sequential_16/simple_rnn_20/transpose_1	TransposeGsequential_16/simple_rnn_20/TensorArrayV2Stack/TensorListStack:tensor:05sequential_16/simple_rnn_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ
IdentityIdentity+sequential_16/simple_rnn_20/transpose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъЇ
NoOpNoOpF^sequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpE^sequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpG^sequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp"^sequential_16/simple_rnn_18/whileF^sequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpE^sequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpG^sequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp"^sequential_16/simple_rnn_19/whileF^sequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpE^sequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpG^sequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp"^sequential_16/simple_rnn_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 2О
Esequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOpEsequential_16/simple_rnn_18/simple_rnn_cell_36/BiasAdd/ReadVariableOp2М
Dsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOpDsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul/ReadVariableOp2Р
Fsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOpFsequential_16/simple_rnn_18/simple_rnn_cell_36/MatMul_1/ReadVariableOp2F
!sequential_16/simple_rnn_18/while!sequential_16/simple_rnn_18/while2О
Esequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOpEsequential_16/simple_rnn_19/simple_rnn_cell_37/BiasAdd/ReadVariableOp2М
Dsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOpDsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul/ReadVariableOp2Р
Fsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOpFsequential_16/simple_rnn_19/simple_rnn_cell_37/MatMul_1/ReadVariableOp2F
!sequential_16/simple_rnn_19/while!sequential_16/simple_rnn_19/while2О
Esequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOpEsequential_16/simple_rnn_20/simple_rnn_cell_38/BiasAdd/ReadVariableOp2М
Dsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOpDsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul/ReadVariableOp2Р
Fsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOpFsequential_16/simple_rnn_20/simple_rnn_cell_38/MatMul_1/ReadVariableOp2F
!sequential_16/simple_rnn_20/while!sequential_16/simple_rnn_20/while:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
э,
“
while_body_8476927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
≤4
•
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8476348

inputs-
simple_rnn_cell_37_8476273:	ђd(
simple_rnn_cell_37_8476275:d,
simple_rnn_cell_37_8476277:dd
identityИҐ*simple_rnn_cell_37/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskу
*simple_rnn_cell_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_37_8476273simple_rnn_cell_37_8476275simple_rnn_cell_37_8476277*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476233n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_37_8476273simple_rnn_cell_37_8476275simple_rnn_cell_37_8476277*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476285*
condR
while_cond_8476284*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d{
NoOpNoOp+^simple_rnn_cell_37/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2X
*simple_rnn_cell_37/StatefulPartitionedCall*simple_rnn_cell_37/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
ФG
Ю
.sequential_16_simple_rnn_20_while_body_8475707T
Psequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_loop_counterZ
Vsequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_maximum_iterations1
-sequential_16_simple_rnn_20_while_placeholder3
/sequential_16_simple_rnn_20_while_placeholder_13
/sequential_16_simple_rnn_20_while_placeholder_2S
Osequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_strided_slice_1_0Р
Лsequential_16_simple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dd
Vsequential_16_simple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:i
Wsequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:.
*sequential_16_simple_rnn_20_while_identity0
,sequential_16_simple_rnn_20_while_identity_10
,sequential_16_simple_rnn_20_while_identity_20
,sequential_16_simple_rnn_20_while_identity_30
,sequential_16_simple_rnn_20_while_identity_4Q
Msequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_strided_slice_1О
Йsequential_16_simple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_20_tensorarrayunstack_tensorlistfromtensore
Ssequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource:db
Tsequential_16_simple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource:g
Usequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐKsequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐJsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpҐLsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp§
Ssequential_16/simple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ≥
Esequential_16/simple_rnn_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЛsequential_16_simple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0-sequential_16_simple_rnn_20_while_placeholder\sequential_16/simple_rnn_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0а
Jsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOpUsequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0Щ
;sequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMulMatMulLsequential_16/simple_rnn_20/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ё
Ksequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOpVsequential_16_simple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Х
<sequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAddBiasAddEsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul:product:0Ssequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€д
Lsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOpWsequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0А
=sequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1MatMul/sequential_16_simple_rnn_20_while_placeholder_2Tsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Г
8sequential_16/simple_rnn_20/while/simple_rnn_cell_38/addAddV2Esequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd:output:0Gsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€є
=sequential_16/simple_rnn_20/while/simple_rnn_cell_38/SoftplusSoftplus<sequential_16/simple_rnn_20/while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€»
Fsequential_16/simple_rnn_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_16_simple_rnn_20_while_placeholder_1-sequential_16_simple_rnn_20_while_placeholderKsequential_16/simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“i
'sequential_16/simple_rnn_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :∞
%sequential_16/simple_rnn_20/while/addAddV2-sequential_16_simple_rnn_20_while_placeholder0sequential_16/simple_rnn_20/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_16/simple_rnn_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :„
'sequential_16/simple_rnn_20/while/add_1AddV2Psequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_loop_counter2sequential_16/simple_rnn_20/while/add_1/y:output:0*
T0*
_output_shapes
: ≠
*sequential_16/simple_rnn_20/while/IdentityIdentity+sequential_16/simple_rnn_20/while/add_1:z:0'^sequential_16/simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_20/while/Identity_1IdentityVsequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_while_maximum_iterations'^sequential_16/simple_rnn_20/while/NoOp*
T0*
_output_shapes
: ≠
,sequential_16/simple_rnn_20/while/Identity_2Identity)sequential_16/simple_rnn_20/while/add:z:0'^sequential_16/simple_rnn_20/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_16/simple_rnn_20/while/Identity_3IdentityVsequential_16/simple_rnn_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_16/simple_rnn_20/while/NoOp*
T0*
_output_shapes
: а
,sequential_16/simple_rnn_20/while/Identity_4IdentityKsequential_16/simple_rnn_20/while/simple_rnn_cell_38/Softplus:activations:0'^sequential_16/simple_rnn_20/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
&sequential_16/simple_rnn_20/while/NoOpNoOpL^sequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpK^sequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpM^sequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_16_simple_rnn_20_while_identity3sequential_16/simple_rnn_20/while/Identity:output:0"e
,sequential_16_simple_rnn_20_while_identity_15sequential_16/simple_rnn_20/while/Identity_1:output:0"e
,sequential_16_simple_rnn_20_while_identity_25sequential_16/simple_rnn_20/while/Identity_2:output:0"e
,sequential_16_simple_rnn_20_while_identity_35sequential_16/simple_rnn_20/while/Identity_3:output:0"e
,sequential_16_simple_rnn_20_while_identity_45sequential_16/simple_rnn_20/while/Identity_4:output:0"†
Msequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_strided_slice_1Osequential_16_simple_rnn_20_while_sequential_16_simple_rnn_20_strided_slice_1_0"Ѓ
Tsequential_16_simple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resourceVsequential_16_simple_rnn_20_while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"∞
Usequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resourceWsequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"ђ
Ssequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resourceUsequential_16_simple_rnn_20_while_simple_rnn_cell_38_matmul_readvariableop_resource_0"Ъ
Йsequential_16_simple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_20_tensorarrayunstack_tensorlistfromtensorЛsequential_16_simple_rnn_20_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2Ъ
Ksequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpKsequential_16/simple_rnn_20/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2Ш
Jsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOpJsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul/ReadVariableOp2Ь
Lsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOpLsequential_16/simple_rnn_20/while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Џ

є
 simple_rnn_20_while_cond_84781998
4simple_rnn_20_while_simple_rnn_20_while_loop_counter>
:simple_rnn_20_while_simple_rnn_20_while_maximum_iterations#
simple_rnn_20_while_placeholder%
!simple_rnn_20_while_placeholder_1%
!simple_rnn_20_while_placeholder_2:
6simple_rnn_20_while_less_simple_rnn_20_strided_slice_1Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8478199___redundant_placeholder0Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8478199___redundant_placeholder1Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8478199___redundant_placeholder2Q
Msimple_rnn_20_while_simple_rnn_20_while_cond_8478199___redundant_placeholder3 
simple_rnn_20_while_identity
Ъ
simple_rnn_20/while/LessLesssimple_rnn_20_while_placeholder6simple_rnn_20_while_less_simple_rnn_20_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_20/while/IdentityIdentitysimple_rnn_20/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_20_while_identity%simple_rnn_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_8479151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479151___redundant_placeholder05
1while_while_cond_8479151___redundant_placeholder15
1while_while_cond_8479151___redundant_placeholder25
1while_while_cond_8479151___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
√

Ё
4__inference_simple_rnn_cell_37_layer_call_fn_8479770

inputs
states_0
unknown:	ђd
	unknown_0:d
	unknown_1:dd
identity

identity_1ИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476113o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€d
"
_user_specified_name
states/0
Ц

д
/__inference_sequential_16_layer_call_fn_8477634

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477463t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
я
ѓ
while_cond_8476125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476125___redundant_placeholder05
1while_while_cond_8476125___redundant_placeholder15
1while_while_cond_8476125___redundant_placeholder25
1while_while_cond_8476125___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
ї4
®
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476056

inputs-
simple_rnn_cell_36_8475981:	ђ)
simple_rnn_cell_36_8475983:	ђ.
simple_rnn_cell_36_8475985:
ђђ
identityИҐ*simple_rnn_cell_36/StatefulPartitionedCallҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskх
*simple_rnn_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_36_8475981simple_rnn_cell_36_8475983simple_rnn_cell_36_8475985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475941n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_36_8475981simple_rnn_cell_36_8475983simple_rnn_cell_36_8475985*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8475993*
condR
while_cond_8475992*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ{
NoOpNoOp+^simple_rnn_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_36/StatefulPartitionedCall*simple_rnn_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
э,
“
while_body_8479520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_38_matmul_readvariableop_resource_0:dH
:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_38_matmul_readvariableop_resource:dF
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_38/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€d*
element_dtype0®
.while/simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_38_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0≈
while/simple_rnn_cell_38/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_38/BiasAddBiasAdd)while/simple_rnn_cell_38/MatMul:product:07while/simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_38/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_38/addAddV2)while/simple_rnn_cell_38/BiasAdd:output:0+while/simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!while/simple_rnn_cell_38/SoftplusSoftplus while/simple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/simple_rnn_cell_38/Softplus:activations:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: М
while/Identity_4Identity/while/simple_rnn_cell_38/Softplus:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

while/NoOpNoOp0^while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_38/MatMul/ReadVariableOp1^while/simple_rnn_cell_38/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_38_biasadd_readvariableop_resource:while_simple_rnn_cell_38_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_38_matmul_1_readvariableop_resource;while_simple_rnn_cell_38_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_38_matmul_readvariableop_resource9while_simple_rnn_cell_38_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp/while/simple_rnn_cell_38/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_38/MatMul/ReadVariableOp.while/simple_rnn_cell_38/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp0while/simple_rnn_cell_38/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ѓ!
з
while_body_8475993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_simple_rnn_cell_36_8476015_0:	ђ1
"while_simple_rnn_cell_36_8476017_0:	ђ6
"while_simple_rnn_cell_36_8476019_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_simple_rnn_cell_36_8476015:	ђ/
 while_simple_rnn_cell_36_8476017:	ђ4
 while_simple_rnn_cell_36_8476019:
ђђИҐ0while/simple_rnn_cell_36/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0∞
0while/simple_rnn_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_36_8476015_0"while_simple_rnn_cell_36_8476017_0"while_simple_rnn_cell_36_8476019_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475941в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ч
while/Identity_4Identity9while/simple_rnn_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ

while/NoOpNoOp1^while/simple_rnn_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_36_8476015"while_simple_rnn_cell_36_8476015_0"F
 while_simple_rnn_cell_36_8476017"while_simple_rnn_cell_36_8476017_0"F
 while_simple_rnn_cell_36_8476019"while_simple_rnn_cell_36_8476019_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :€€€€€€€€€ђ: : : : : 2d
0while/simple_rnn_cell_36/StatefulPartitionedCall0while/simple_rnn_cell_36/StatefulPartitionedCall: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
: 
љ

с
/__inference_sequential_16_layer_call_fn_8477023
simple_rnn_18_input
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477002t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
№=
ƒ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479110

inputsD
1simple_rnn_cell_37_matmul_readvariableop_resource:	ђd@
2simple_rnn_cell_37_biasadd_readvariableop_resource:dE
3simple_rnn_cell_37_matmul_1_readvariableop_resource:dd
identityИҐ)simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_37/MatMul/ReadVariableOpҐ*simple_rnn_cell_37/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:ъ€€€€€€€€€ђD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maskЫ
(simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_37_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0°
simple_rnn_cell_37/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dШ
)simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_37_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ѓ
simple_rnn_cell_37/BiasAddBiasAdd#simple_rnn_cell_37/MatMul:product:01simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЮ
*simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_37_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0Ы
simple_rnn_cell_37/MatMul_1MatMulzeros:output:02simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЭ
simple_rnn_cell_37/addAddV2#simple_rnn_cell_37/BiasAdd:output:0%simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ds
simple_rnn_cell_37/SigmoidSigmoidsimple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_37_matmul_readvariableop_resource2simple_rnn_cell_37_biasadd_readvariableop_resource3simple_rnn_cell_37_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479044*
condR
while_cond_8479043*8
output_shapes'
%: : : : :€€€€€€€€€d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъdc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъd“
NoOpNoOp*^simple_rnn_cell_37/BiasAdd/ReadVariableOp)^simple_rnn_cell_37/MatMul/ReadVariableOp+^simple_rnn_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ъђ: : : 2V
)simple_rnn_cell_37/BiasAdd/ReadVariableOp)simple_rnn_cell_37/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_37/MatMul/ReadVariableOp(simple_rnn_cell_37/MatMul/ReadVariableOp2X
*simple_rnn_cell_37/MatMul_1/ReadVariableOp*simple_rnn_cell_37/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ъђ
 
_user_specified_nameinputs
П
л
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476113

inputs

states1
matmul_readvariableop_resource:	ђd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€dZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d\

Identity_1IdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_namestates
Ў=
√
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479586

inputsC
1simple_rnn_cell_38_matmul_readvariableop_resource:d@
2simple_rnn_cell_38_biasadd_readvariableop_resource:E
3simple_rnn_cell_38_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_38/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_38/MatMul/ReadVariableOpҐ*simple_rnn_cell_38/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskЪ
(simple_rnn_cell_38/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_38_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0°
simple_rnn_cell_38/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_38/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_38/BiasAddBiasAdd#simple_rnn_cell_38/MatMul:product:01simple_rnn_cell_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_38/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_38_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_38/MatMul_1MatMulzeros:output:02simple_rnn_cell_38/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_38/addAddV2#simple_rnn_cell_38/BiasAdd:output:0%simple_rnn_cell_38/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€u
simple_rnn_cell_38/SoftplusSoftplussimple_rnn_cell_38/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_38_matmul_readvariableop_resource2simple_rnn_cell_38_biasadd_readvariableop_resource3simple_rnn_cell_38_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8479520*
condR
while_cond_8479519*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ъ€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ“
NoOpNoOp*^simple_rnn_cell_38/BiasAdd/ReadVariableOp)^simple_rnn_cell_38/MatMul/ReadVariableOp+^simple_rnn_cell_38/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъd: : : 2V
)simple_rnn_cell_38/BiasAdd/ReadVariableOp)simple_rnn_cell_38/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_38/MatMul/ReadVariableOp(simple_rnn_cell_38/MatMul/ReadVariableOp2X
*simple_rnn_cell_38/MatMul_1/ReadVariableOp*simple_rnn_cell_38/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъd
 
_user_specified_nameinputs
 

а
4__inference_simple_rnn_cell_36_layer_call_fn_8479708

inputs
states_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8475821p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
states/0
л=
«
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476763

inputsD
1simple_rnn_cell_36_matmul_readvariableop_resource:	ђA
2simple_rnn_cell_36_biasadd_readvariableop_resource:	ђG
3simple_rnn_cell_36_matmul_1_readvariableop_resource:
ђђ
identityИҐ)simple_rnn_cell_36/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_36/MatMul/ReadVariableOpҐ*simple_rnn_cell_36/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
B :ђs
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
:€€€€€€€€€ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ъ€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЫ
(simple_rnn_cell_36/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_36_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ґ
simple_rnn_cell_36/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЩ
)simple_rnn_cell_36/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0∞
simple_rnn_cell_36/BiasAddBiasAdd#simple_rnn_cell_36/MatMul:product:01simple_rnn_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ†
*simple_rnn_cell_36/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_36_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ь
simple_rnn_cell_36/MatMul_1MatMulzeros:output:02simple_rnn_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЮ
simple_rnn_cell_36/addAddV2#simple_rnn_cell_36/BiasAdd:output:0%simple_rnn_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
simple_rnn_cell_36/SigmoidSigmoidsimple_rnn_cell_36/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_36_matmul_readvariableop_resource2simple_rnn_cell_36_biasadd_readvariableop_resource3simple_rnn_cell_36_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476697*
condR
while_cond_8476696*9
output_shapes(
&: : : : :€€€€€€€€€ђ: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  ƒ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:ъ€€€€€€€€€ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ш
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:€€€€€€€€€ъђd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ“
NoOpNoOp*^simple_rnn_cell_36/BiasAdd/ReadVariableOp)^simple_rnn_cell_36/MatMul/ReadVariableOp+^simple_rnn_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 2V
)simple_rnn_cell_36/BiasAdd/ReadVariableOp)simple_rnn_cell_36/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_36/MatMul/ReadVariableOp(simple_rnn_cell_36/MatMul/ReadVariableOp2X
*simple_rnn_cell_36/MatMul_1/ReadVariableOp*simple_rnn_cell_36/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
П
л
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476233

inputs

states1
matmul_readvariableop_resource:	ђd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€dZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d\

Identity_1IdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_namestates
и,
‘
while_body_8478936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_37_matmul_readvariableop_resource_0:	ђdH
:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0:dM
;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_37_matmul_readvariableop_resource:	ђdF
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:dK
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource:ddИҐ/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_37/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0©
.while/simple_rnn_cell_37/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	ђd*
dtype0≈
while/simple_rnn_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0Ѕ
 while/simple_rnn_cell_37/BiasAddBiasAdd)while/simple_rnn_cell_37/MatMul:product:07while/simple_rnn_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dђ
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0ђ
!while/simple_rnn_cell_37/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dѓ
while/simple_rnn_cell_37/addAddV2)while/simple_rnn_cell_37/BiasAdd:output:0+while/simple_rnn_cell_37/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€d
 while/simple_rnn_cell_37/SigmoidSigmoid while/simple_rnn_cell_37/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€dЌ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/simple_rnn_cell_37/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Б
while/Identity_4Identity$while/simple_rnn_cell_37/Sigmoid:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€dв

while/NoOpNoOp0^while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_37/MatMul/ReadVariableOp1^while/simple_rnn_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_37_biasadd_readvariableop_resource:while_simple_rnn_cell_37_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_37_matmul_1_readvariableop_resource;while_simple_rnn_cell_37_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_37_matmul_readvariableop_resource9while_simple_rnn_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2b
/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp/while/simple_rnn_cell_37/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_37/MatMul/ReadVariableOp.while/simple_rnn_cell_37/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp0while/simple_rnn_cell_37/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
©
м
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479880

inputs
states_00
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€O
SoftplusSoftplusadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g

Identity_1IdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€d:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
Ч
љ
/__inference_simple_rnn_18_layer_call_fn_8478310

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8477404u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs
≠4
§
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8476640

inputs,
simple_rnn_cell_38_8476565:d(
simple_rnn_cell_38_8476567:,
simple_rnn_cell_38_8476569:
identityИҐ*simple_rnn_cell_38/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€dD
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€d   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€d*
shrink_axis_maskу
*simple_rnn_cell_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_38_8476565simple_rnn_cell_38_8476567simple_rnn_cell_38_8476569*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8476525n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_38_8476565simple_rnn_cell_38_8476567simple_rnn_cell_38_8476569*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8476577*
condR
while_cond_8476576*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€{
NoOpNoOp+^simple_rnn_cell_38/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€d: : : 2X
*simple_rnn_cell_38/StatefulPartitionedCall*simple_rnn_cell_38/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€d
 
_user_specified_nameinputs
я
ѓ
while_cond_8476284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476284___redundant_placeholder05
1while_while_cond_8476284___redundant_placeholder15
1while_while_cond_8476284___redundant_placeholder25
1while_while_cond_8476284___redundant_placeholder3
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
-: : : : :€€€€€€€€€d: ::::: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
:
§!
б
while_body_8476285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_simple_rnn_cell_37_8476307_0:	ђd0
"while_simple_rnn_cell_37_8476309_0:d4
"while_simple_rnn_cell_37_8476311_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_simple_rnn_cell_37_8476307:	ђd.
 while_simple_rnn_cell_37_8476309:d2
 while_simple_rnn_cell_37_8476311:ddИҐ0while/simple_rnn_cell_37/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype0Ѓ
0while/simple_rnn_cell_37/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_37_8476307_0"while_simple_rnn_cell_37_8476309_0"while_simple_rnn_cell_37_8476311_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8476233в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_37/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ц
while/Identity_4Identity9while/simple_rnn_cell_37/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€d

while/NoOpNoOp1^while/simple_rnn_cell_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_37_8476307"while_simple_rnn_cell_37_8476307_0"F
 while_simple_rnn_cell_37_8476309"while_simple_rnn_cell_37_8476309_0"F
 while_simple_rnn_cell_37_8476311"while_simple_rnn_cell_37_8476311_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€d: : : : : 2d
0while/simple_rnn_cell_37/StatefulPartitionedCall0while/simple_rnn_cell_37/StatefulPartitionedCall: 
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
:€€€€€€€€€d:

_output_shapes
: :

_output_shapes
: 
љ

с
/__inference_sequential_16_layer_call_fn_8477507
simple_rnn_18_input
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477463t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€ъ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:€€€€€€€€€ъ
-
_user_specified_namesimple_rnn_18_input
б
ѓ
while_cond_8478675
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8478675___redundant_placeholder05
1while_while_cond_8478675___redundant_placeholder15
1while_while_cond_8478675___redundant_placeholder25
1while_while_cond_8478675___redundant_placeholder3
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
.: : : : :€€€€€€€€€ђ: ::::: 
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
:€€€€€€€€€ђ:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_8476417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8476417___redundant_placeholder05
1while_while_cond_8476417___redundant_placeholder15
1while_while_cond_8476417___redundant_placeholder25
1while_while_cond_8476417___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_8479627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_8479627___redundant_placeholder05
1while_while_cond_8479627___redundant_placeholder15
1while_while_cond_8479627___redundant_placeholder25
1while_while_cond_8479627___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ч
љ
/__inference_simple_rnn_18_layer_call_fn_8478299

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ъђ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8476763u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:€€€€€€€€€ъђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ъ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ
 
_user_specified_nameinputs"њL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*“
serving_defaultЊ
X
simple_rnn_18_inputA
%serving_default_simple_rnn_18_input:0€€€€€€€€€ъF
simple_rnn_205
StatefulPartitionedCall:0€€€€€€€€€ъtensorflow/serving/predict:Бђ
џ
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
√
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
√
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
√
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#cell
$
state_spec"
_tf_keras_rnn_layer
_
%0
&1
'2
(3
)4
*5
+6
,7
-8"
trackable_list_wrapper
_
%0
&1
'2
(3
)4
*5
+6
,7
-8"
trackable_list_wrapper
 "
trackable_list_wrapper
 
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
т
3trace_0
4trace_1
5trace_2
6trace_32З
/__inference_sequential_16_layer_call_fn_8477023
/__inference_sequential_16_layer_call_fn_8477611
/__inference_sequential_16_layer_call_fn_8477634
/__inference_sequential_16_layer_call_fn_8477507ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z3trace_0z4trace_1z5trace_2z6trace_3
ё
7trace_0
8trace_1
9trace_2
:trace_32у
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477950
J__inference_sequential_16_layer_call_and_return_conditional_losses_8478266
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477532
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477557ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z7trace_0z8trace_1z9trace_2z:trace_3
ўB÷
"__inference__wrapped_model_8475773simple_rnn_18_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
З
;iter

<beta_1

=beta_2
	>decay
?learning_rate%m†&m°'mҐ(m£)m§*m•+m¶,mІ-m®%v©&v™'vЂ(vђ)v≠*vЃ+vѓ,v∞-v±"
	optimizer
,
@serving_default"
signature_map
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
З
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32Ь
/__inference_simple_rnn_18_layer_call_fn_8478277
/__inference_simple_rnn_18_layer_call_fn_8478288
/__inference_simple_rnn_18_layer_call_fn_8478299
/__inference_simple_rnn_18_layer_call_fn_8478310’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
у
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32И
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478418
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478526
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478634
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478742’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
и
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
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
є

Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
З
\trace_0
]trace_1
^trace_2
_trace_32Ь
/__inference_simple_rnn_19_layer_call_fn_8478753
/__inference_simple_rnn_19_layer_call_fn_8478764
/__inference_simple_rnn_19_layer_call_fn_8478775
/__inference_simple_rnn_19_layer_call_fn_8478786’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z\trace_0z]trace_1z^trace_2z_trace_3
у
`trace_0
atrace_1
btrace_2
ctrace_32И
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8478894
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479002
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479110
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479218’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z`trace_0zatrace_1zbtrace_2zctrace_3
и
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator

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
є

kstates
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
З
qtrace_0
rtrace_1
strace_2
ttrace_32Ь
/__inference_simple_rnn_20_layer_call_fn_8479229
/__inference_simple_rnn_20_layer_call_fn_8479240
/__inference_simple_rnn_20_layer_call_fn_8479251
/__inference_simple_rnn_20_layer_call_fn_8479262’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zqtrace_0zrtrace_1zstrace_2zttrace_3
у
utrace_0
vtrace_1
wtrace_2
xtrace_32И
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479370
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479478
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479586
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479694’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zutrace_0zvtrace_1zwtrace_2zxtrace_3
и
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator

+kernel
,recurrent_kernel
-bias"
_tf_keras_layer
 "
trackable_list_wrapper
::8	ђ2'simple_rnn_18/simple_rnn_cell_36/kernel
E:C
ђђ21simple_rnn_18/simple_rnn_cell_36/recurrent_kernel
4:2ђ2%simple_rnn_18/simple_rnn_cell_36/bias
::8	ђd2'simple_rnn_19/simple_rnn_cell_37/kernel
C:Add21simple_rnn_19/simple_rnn_cell_37/recurrent_kernel
3:1d2%simple_rnn_19/simple_rnn_cell_37/bias
9:7d2'simple_rnn_20/simple_rnn_cell_38/kernel
C:A21simple_rnn_20/simple_rnn_cell_38/recurrent_kernel
3:12%simple_rnn_20/simple_rnn_cell_38/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
(
А0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ОBЛ
/__inference_sequential_16_layer_call_fn_8477023simple_rnn_18_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
БBю
/__inference_sequential_16_layer_call_fn_8477611inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
БBю
/__inference_sequential_16_layer_call_fn_8477634inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ОBЛ
/__inference_sequential_16_layer_call_fn_8477507simple_rnn_18_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЬBЩ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477950inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЬBЩ
J__inference_sequential_16_layer_call_and_return_conditional_losses_8478266inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477532simple_rnn_18_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477557simple_rnn_18_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЎB’
%__inference_signature_wrapper_8477588simple_rnn_18_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBХ
/__inference_simple_rnn_18_layer_call_fn_8478277inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ШBХ
/__inference_simple_rnn_18_layer_call_fn_8478288inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_18_layer_call_fn_8478299inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_18_layer_call_fn_8478310inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478418inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478526inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478634inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478742inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
и
Жtrace_0
Зtrace_12≠
4__inference_simple_rnn_cell_36_layer_call_fn_8479708
4__inference_simple_rnn_cell_36_layer_call_fn_8479722Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zЖtrace_0zЗtrace_1
Ю
Иtrace_0
Йtrace_12г
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479739
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479756Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zИtrace_0zЙtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBХ
/__inference_simple_rnn_19_layer_call_fn_8478753inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ШBХ
/__inference_simple_rnn_19_layer_call_fn_8478764inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_19_layer_call_fn_8478775inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_19_layer_call_fn_8478786inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8478894inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479002inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479110inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479218inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
≤
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
и
Пtrace_0
Рtrace_12≠
4__inference_simple_rnn_cell_37_layer_call_fn_8479770
4__inference_simple_rnn_cell_37_layer_call_fn_8479784Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zПtrace_0zРtrace_1
Ю
Сtrace_0
Тtrace_12г
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479801
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479818Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zСtrace_0zТtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBХ
/__inference_simple_rnn_20_layer_call_fn_8479229inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ШBХ
/__inference_simple_rnn_20_layer_call_fn_8479240inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_20_layer_call_fn_8479251inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
/__inference_simple_rnn_20_layer_call_fn_8479262inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479370inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥B∞
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479478inputs/0"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479586inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
±BЃ
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479694inputs"’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
≤
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
и
Шtrace_0
Щtrace_12≠
4__inference_simple_rnn_cell_38_layer_call_fn_8479832
4__inference_simple_rnn_cell_38_layer_call_fn_8479846Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zШtrace_0zЩtrace_1
Ю
Ъtrace_0
Ыtrace_12г
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479863
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479880Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zЪtrace_0zЫtrace_1
"
_generic_user_object
R
Ь	variables
Э	keras_api

Юtotal

Яcount"
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
ОBЛ
4__inference_simple_rnn_cell_36_layer_call_fn_8479708inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ОBЛ
4__inference_simple_rnn_cell_36_layer_call_fn_8479722inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479739inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479756inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
ОBЛ
4__inference_simple_rnn_cell_37_layer_call_fn_8479770inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ОBЛ
4__inference_simple_rnn_cell_37_layer_call_fn_8479784inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479801inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479818inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
ОBЛ
4__inference_simple_rnn_cell_38_layer_call_fn_8479832inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ОBЛ
4__inference_simple_rnn_cell_38_layer_call_fn_8479846inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479863inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©B¶
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479880inputsstates/0"Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
0
Ю0
Я1"
trackable_list_wrapper
.
Ь	variables"
_generic_user_object
:  (2total
:  (2count
?:=	ђ2.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/m
J:H
ђђ28Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/m
9:7ђ2,Adam/simple_rnn_18/simple_rnn_cell_36/bias/m
?:=	ђd2.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/m
H:Fdd28Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/m
8:6d2,Adam/simple_rnn_19/simple_rnn_cell_37/bias/m
>:<d2.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/m
H:F28Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/m
8:62,Adam/simple_rnn_20/simple_rnn_cell_38/bias/m
?:=	ђ2.Adam/simple_rnn_18/simple_rnn_cell_36/kernel/v
J:H
ђђ28Adam/simple_rnn_18/simple_rnn_cell_36/recurrent_kernel/v
9:7ђ2,Adam/simple_rnn_18/simple_rnn_cell_36/bias/v
?:=	ђd2.Adam/simple_rnn_19/simple_rnn_cell_37/kernel/v
H:Fdd28Adam/simple_rnn_19/simple_rnn_cell_37/recurrent_kernel/v
8:6d2,Adam/simple_rnn_19/simple_rnn_cell_37/bias/v
>:<d2.Adam/simple_rnn_20/simple_rnn_cell_38/kernel/v
H:F28Adam/simple_rnn_20/simple_rnn_cell_38/recurrent_kernel/v
8:62,Adam/simple_rnn_20/simple_rnn_cell_38/bias/vє
"__inference__wrapped_model_8475773Т	%'&(*)+-,AҐ>
7Ґ4
2К/
simple_rnn_18_input€€€€€€€€€ъ
™ "B™?
=
simple_rnn_20,К)
simple_rnn_20€€€€€€€€€ъ—
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477532В	%'&(*)+-,IҐF
?Ґ<
2К/
simple_rnn_18_input€€€€€€€€€ъ
p 

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ —
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477557В	%'&(*)+-,IҐF
?Ґ<
2К/
simple_rnn_18_input€€€€€€€€€ъ
p

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ √
J__inference_sequential_16_layer_call_and_return_conditional_losses_8477950u	%'&(*)+-,<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ъ
p 

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ √
J__inference_sequential_16_layer_call_and_return_conditional_losses_8478266u	%'&(*)+-,<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ъ
p

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ ®
/__inference_sequential_16_layer_call_fn_8477023u	%'&(*)+-,IҐF
?Ґ<
2К/
simple_rnn_18_input€€€€€€€€€ъ
p 

 
™ "К€€€€€€€€€ъ®
/__inference_sequential_16_layer_call_fn_8477507u	%'&(*)+-,IҐF
?Ґ<
2К/
simple_rnn_18_input€€€€€€€€€ъ
p

 
™ "К€€€€€€€€€ъЫ
/__inference_sequential_16_layer_call_fn_8477611h	%'&(*)+-,<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ъ
p 

 
™ "К€€€€€€€€€ъЫ
/__inference_sequential_16_layer_call_fn_8477634h	%'&(*)+-,<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ъ
p

 
™ "К€€€€€€€€€ъ”
%__inference_signature_wrapper_8477588©	%'&(*)+-,XҐU
Ґ 
N™K
I
simple_rnn_18_input2К/
simple_rnn_18_input€€€€€€€€€ъ"B™?
=
simple_rnn_20,К)
simple_rnn_20€€€€€€€€€ъЏ
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478418Л%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€ђ
Ъ Џ
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478526Л%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€ђ
Ъ ¬
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478634t%'&@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъ

 
p 

 
™ "+Ґ(
!К
0€€€€€€€€€ъђ
Ъ ¬
J__inference_simple_rnn_18_layer_call_and_return_conditional_losses_8478742t%'&@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъ

 
p

 
™ "+Ґ(
!К
0€€€€€€€€€ъђ
Ъ ±
/__inference_simple_rnn_18_layer_call_fn_8478277~%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "&К#€€€€€€€€€€€€€€€€€€ђ±
/__inference_simple_rnn_18_layer_call_fn_8478288~%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "&К#€€€€€€€€€€€€€€€€€€ђЪ
/__inference_simple_rnn_18_layer_call_fn_8478299g%'&@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъ

 
p 

 
™ "К€€€€€€€€€ъђЪ
/__inference_simple_rnn_18_layer_call_fn_8478310g%'&@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъ

 
p

 
™ "К€€€€€€€€€ъђЏ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8478894Л(*)PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€d
Ъ Џ
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479002Л(*)PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€d
Ъ ¬
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479110t(*)AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ъђ

 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€ъd
Ъ ¬
J__inference_simple_rnn_19_layer_call_and_return_conditional_losses_8479218t(*)AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ъђ

 
p

 
™ "*Ґ'
 К
0€€€€€€€€€ъd
Ъ ±
/__inference_simple_rnn_19_layer_call_fn_8478753~(*)PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€d±
/__inference_simple_rnn_19_layer_call_fn_8478764~(*)PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€dЪ
/__inference_simple_rnn_19_layer_call_fn_8478775g(*)AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ъђ

 
p 

 
™ "К€€€€€€€€€ъdЪ
/__inference_simple_rnn_19_layer_call_fn_8478786g(*)AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ъђ

 
p

 
™ "К€€€€€€€€€ъdў
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479370К+-,OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€d

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ў
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479478К+-,OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€d

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ѕ
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479586s+-,@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъd

 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ Ѕ
J__inference_simple_rnn_20_layer_call_and_return_conditional_losses_8479694s+-,@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъd

 
p

 
™ "*Ґ'
 К
0€€€€€€€€€ъ
Ъ ∞
/__inference_simple_rnn_20_layer_call_fn_8479229}+-,OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€d

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€∞
/__inference_simple_rnn_20_layer_call_fn_8479240}+-,OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€d

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€Щ
/__inference_simple_rnn_20_layer_call_fn_8479251f+-,@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъd

 
p 

 
™ "К€€€€€€€€€ъЩ
/__inference_simple_rnn_20_layer_call_fn_8479262f+-,@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€ъd

 
p

 
™ "К€€€€€€€€€ъО
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479739Ї%'&]ҐZ
SҐP
 К
inputs€€€€€€€€€
(Ґ%
#К 
states/0€€€€€€€€€ђ
p 
™ "TҐQ
JҐG
К
0/0€€€€€€€€€ђ
%Ъ"
 К
0/1/0€€€€€€€€€ђ
Ъ О
O__inference_simple_rnn_cell_36_layer_call_and_return_conditional_losses_8479756Ї%'&]ҐZ
SҐP
 К
inputs€€€€€€€€€
(Ґ%
#К 
states/0€€€€€€€€€ђ
p
™ "TҐQ
JҐG
К
0/0€€€€€€€€€ђ
%Ъ"
 К
0/1/0€€€€€€€€€ђ
Ъ е
4__inference_simple_rnn_cell_36_layer_call_fn_8479708ђ%'&]ҐZ
SҐP
 К
inputs€€€€€€€€€
(Ґ%
#К 
states/0€€€€€€€€€ђ
p 
™ "FҐC
К
0€€€€€€€€€ђ
#Ъ 
К
1/0€€€€€€€€€ђе
4__inference_simple_rnn_cell_36_layer_call_fn_8479722ђ%'&]ҐZ
SҐP
 К
inputs€€€€€€€€€
(Ґ%
#К 
states/0€€€€€€€€€ђ
p
™ "FҐC
К
0€€€€€€€€€ђ
#Ъ 
К
1/0€€€€€€€€€ђМ
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479801Є(*)]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€d
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€d
$Ъ!
К
0/1/0€€€€€€€€€d
Ъ М
O__inference_simple_rnn_cell_37_layer_call_and_return_conditional_losses_8479818Є(*)]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€d
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€d
$Ъ!
К
0/1/0€€€€€€€€€d
Ъ г
4__inference_simple_rnn_cell_37_layer_call_fn_8479770™(*)]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€d
p 
™ "DҐA
К
0€€€€€€€€€d
"Ъ
К
1/0€€€€€€€€€dг
4__inference_simple_rnn_cell_37_layer_call_fn_8479784™(*)]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€d
p
™ "DҐA
К
0€€€€€€€€€d
"Ъ
К
1/0€€€€€€€€€dЛ
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479863Ј+-,\ҐY
RҐO
 К
inputs€€€€€€€€€d
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ Л
O__inference_simple_rnn_cell_38_layer_call_and_return_conditional_losses_8479880Ј+-,\ҐY
RҐO
 К
inputs€€€€€€€€€d
'Ґ$
"К
states/0€€€€€€€€€
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ в
4__inference_simple_rnn_cell_38_layer_call_fn_8479832©+-,\ҐY
RҐO
 К
inputs€€€€€€€€€d
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€в
4__inference_simple_rnn_cell_38_layer_call_fn_8479846©+-,\ҐY
RҐO
 К
inputs€€€€€€€€€d
'Ґ$
"К
states/0€€€€€€€€€
p
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€