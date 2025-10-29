using Optim

include("nn-classification.jl")

# TODO: Find out how Matlab solves `lsqnonlin` and imitate <16-03-25> 
function fit_NN_2_3_2(P0=0.5randn(23), x=data_x, y=float.(data_y); activation=NNlib.sigmoid, optimizer=NelderMead())
    opt = Optim.optimize(P -> NN_2_3_2_loss(P, x, y; activation), P0, optimizer, Optim.Options(iterations=1_000_000))
    if !opt.ls_success # FIX: How to check whether the optimization succeeded? <16-03-25> 
        throw(error("Optimization did not convege"))
    end
    return opt.minimizer, opt.minimum
end
