"""
    _NN_2_3_2_split_params(Pval)

Split the parameter vector `Pval` into weight matrices and bias vectors for a 2-3-2 neural network. The order of the
returned parameters is `W2` (2×2), `b2`, `W3` (2×3), `b3`, `W4` (3×2), `b4`.
"""
function _NN_2_3_2_split_params(Pval)
    W2 = reshape(Pval[1:4], (2, 2))
    b2 = Pval[17:18]
    W3 = reshape(Pval[5:10], (3, 2))
    b3 = Pval[19:21]
    W4 = reshape(Pval[11:16], (2, 3))
    b4 = Pval[22:23]
    return W2, b2, W3, b3, W4, b4
end


"""
    _NN_2_3_2_join_params(W2, b2, W3, b3, W4, b4)

Concatenate the parameters of a 2-3-2 neural network into a single vector. For internal use in functions.
Returns 1-4, Flattened W2, 17-18, b2, 5-10, Flattened W3, 19-21, b3, 11-16, Flattened W4, 22-23, b4
"""
function _NN_2_3_2_join_params(W2, b2, W3, b3, W4, b4)
    Pval = Vector{eltype(W2)}(undef, 23)
    Pval[1:4] .= W2[:]
    Pval[17:18] .= b2[:]
    Pval[5:10] .= W3[:]
    Pval[19:21] .= b3[:]
    Pval[11:16] .= W4[:]
    Pval[22:23] .= b4[:]
    return Pval
end

"""
    NN_2_3_2_forward(x, Pval, activation=NNlib.sigmoid)


Manual implementation of a 2-3-2 densely connected neural net forward pass. `Pval` are the 23 parameters in the of the network concatenated into a vector (see [`_NN_2_3_2_split_params`](@ref).

See also [`NN_2_3_2_loss`](@ref)
"""
function NN_2_3_2_forward(x; Pval, activation=NNlib.sigmoid)
    W2, b2, W3, b3, W4, b4 = _NN_2_3_2_split_params(Pval)
    a2 = activation.(W2 * x .+ b2)
    a3 = activation.(W3 * a2 .+ b3)
    y = activation.(W4 * a3 .+ b4) # a4
    return y
end
NN_2_3_2_forward(x1, x2; kwargs...) = NN_2_3_2_forward([x1, x2]; kwargs...)


"""
    NN_2_3_2_loss(Pval, x=data_x, y=data_y; activation=NNlib.sigmoid)

Calculates the L2 loss for a 2-3-2 neural network. Takes network parameters `Pval` (see [`_NN_2_3_2_join_params`](@ref)), 
input data `x`, target data `y`, and `activation` function. Returns the L2 norm of 
individual sample losses.
"""
function NN_2_3_2_loss(Pval, x::AbstractMatrix=data_x, y::AbstractMatrix=data_y; activation=NNlib.sigmoid)
    cost = map(Base.OneTo(10)) do ind
        input = x[:, ind]
        label = y[:, ind]
        output = NN_2_3_2_forward(input; Pval, activation)

        return √(sum((output .- label) .^ 2)) # L2 norm (norm(y - label))
    end
    return norm(cost, 2)
end

"""
    train_NN_2_3_2([P0=0.5randn(23)], [x=data_x], [y=data_y]; <kwargs>)

Train a neural network with architecture input-2-3-2-output and a sigmoid activation using hand written back propagation
stochastic gradient descent.

# Arguments
- `eta=0.05`: Learning rate for gradient descent.
- `iterations=1e6`: Number of training iterations.
- `loss_log::Union{Vector,Nothing}=nothing`: Optional vector to log loss values during training.
- `Pval_log::Union{Vector,Nothing}=nothing`: Optional vector to log parameter values during training.

```jlrepl
julia> Ps = train_NN_2_3_2()
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

julia> train_NN_2_3_2(;loss_log=loss, Pval_log=Ps_log)
```
"""
function train_NN_2_3_2(P0=0.5randn(23), x=data_x, y=data_y; eta=0.05, iterations=Int(1e6), loss_log::Union{Vector,Nothing}=nothing, Pval_log::Union{Vector,Nothing}=nothing)
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
