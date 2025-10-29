# Prepare data

include("nn-classification.jl")
include("plots.jl")

fig_train_nelder_mead = fig_nelder_mead_fit()

Ps_log = []
loss_log = []
Ps = train_NN_2_3_2(; loss_log, Pval_log=Ps_log)

fig_loss_2_3_2_SG = fig_NN_2_3_2_loss_lines(loss_log)

fig_2_3_2_training = fig_NN_2_3_2_decisions_train_interactive(Ps_log, loss_log, :)

model_2_3_2_sigmoid = dense_NN_model([2, 2, 3, 2], NNlib.sigmoid)
model_2_3_2_relu = dense_NN_model([2, 2, 3, 2], NNlib.relu)
#
model_deep_sigmoid = train_NN_lux(model_2_3_2_sigmoid)
model_deep_relu = train_NN_lux(model_2_3_2_relu)
