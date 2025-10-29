using NNlib, Lux, ProgressLogging
include("nn-classification.jl")

"""
    train_NN_3_2_3([P0=0.5randn(23)], [x=data_x], [y=data_y]; <kwargs>)

Train a neural network with architecture input-2-3-2-output and a sigmoid activation using hand written back propagation
stochastic gradient descent.

# Arguments
- `eta=0.05`: Learning rate for gradient descent.
- `iterations=1e6`: Number of training iterations.
- `loss_log::Union{Vector,Nothing}=nothing`: Optional vector to log loss values during training.
- `Pval_log::Union{Vector,Nothing}=nothing`: Optional vector to log parameter values during training.

```jlrepl
julia> Ps = train_NN_3_2_3()
23-element Vector{Float64}:
   6.271708093417881
   9.763763573507966
 -10.795445301414254
   8.707378422806054
  -3.99535740055764
  -4.881155404213543
   ⋮
   6.133278582400668
   7.336842412236986
  -6.559704435057771
  -1.5701866463355216
   2.0555246030403533

julia> loss = [];

julia> Ps_log = [];

julia> train_NN_3_2_3(;loss_log=loss, Pval_log=Ps_log)
```
"""
function train_NN_3_2_3(P0=0.5randn(23), x=data_x, y=data_y; eta=0.05, iterations=Int(1e6), loss_log::Union{Vector,Nothing}=nothing, Pval_log::Union{Vector,Nothing}=nothing)
    W2, b2, W3, b3, W4, b4 = _NN_2_3_2_split_params(P0)

    @progress for _ in Base.OneTo(iterations)
        ind = rand(Base.OneTo(size(x, 2)))
        x_train = x[:, ind]
        y_train = y[:, ind]

        # Forward pass
        a2 = NNlib.σ.(W2 * x_train .+ b2)
        a3 = NNlib.σ.(W3 * a2 .+ b3)
        a4 = NNlib.σ.(W4 * a3 .+ b4) # a4

        # Backward pass
        delta4 = @. a4 * (1 - a4) * (a4 - y_train)
        delta3 = a3 .* (1 .- a3) .* (W4' * delta4)
        delta2 = a2 .* (1 .- a2) .* (W3' * delta3)

        # Step gradient
        W2 .-= eta * delta2 * x_train'
        W3 .-= eta * delta3 * a2'
        W4 .-= eta * delta4 * a3'
        b2 .-= eta * delta2
        b3 .-= eta * delta3
        b4 .-= eta * delta4

        if !isnothing(loss_log)
            Pval = _NN_2_3_2_join_params(W2, b2, W3, b3, W4, b4)
            push!(loss_log, NN_2_3_2_loss(Pval, x, y))
        end
        if !isnothing(Pval_log)
            Pval = _NN_2_3_2_join_params(W2, b2, W3, b3, W4, b4)
            push!(Pval_log, Pval)
        end
    end

    return _NN_2_3_2_join_params(W2, b2, W3, b3, W4, b4)
end
