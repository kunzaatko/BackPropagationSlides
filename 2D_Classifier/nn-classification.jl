using AMDGPU: functional
using NNlib, LinearAlgebra, Lux, Random, Optimisers, Printf, Zygote, AMDGPU, IterTools, ProgressLogging
data_x = [
    0.1 0.3 0.1 0.6 0.4 0.6 0.5 0.9 0.4 0.7;
    0.1 0.4 0.5 0.9 0.2 0.3 0.6 0.2 0.4 0.6
]

data_y = [trues(1, 5) falses(1, 5); falses(1, 5) trues(1, 5)]

dense_NN_model(layer_sizes, activation=NNlib.sigmoid) = Chain(
    (Dense(sz[1] => sz[2], activation) for (i, sz) in enumerate(partition(layer_sizes, 2, 1)))...
)

NN_2_3_2_model(activation=NNlib.sigmoid) = dense_NN_model([2, 2, 3, 2], activation)

function train_NN_lux(model, x=data_x, y=data_y; epochs=Int(1e4), device=cpu_device())
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model) |> device
    lossfn = MSELoss()
    opt = Optimisers.Adam(0.03f0)
    tstate = Training.TrainState(model, ps, st, opt)
    x_dev = x |> device
    y_dev = y |> device
    @info "Training $model with $epochs epochs"
    @progress for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), lossfn, (x_dev, y_dev), tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate, tstate.model, tstate.parameters, tstate.states
end

forward_NN_lux(x, tstate; device=cpu_device()) = Lux.apply(tstate.model, device(x), tstate.parameters, Lux.testmode(tstate.states))

function train_NN_2_3_2_lux(x=data_x, y=data_y; activation=NNlib.sigmoid, eta=0.05, iterations=Int(1e6), device=cpu_device(), kwargs...)
    model = NN_2_3_2_model(activation)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model) |> device
    lossfn = MSELoss()
    tstate = Training.TrainState(model, ps, st, Descent(; eta))
    @info "Training $model with $iterations iterations"
    @progress for i in Base.OneTo(iterations)
        ind = rand(Base.OneTo(size(x, 2)))
        x_train = x[:, ind] |> device
        y_train = y[:, ind] |> device

        _, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), lossfn, (x_train, y_train), tstate
        )
        if i % 1000 == 1 || i == iterations
            @printf "Loss Value after %6d iterations: %.8f\n" i loss
        end
    end
    return tstate.model, tstate.parameters, tstate.states
end

include("nn-2-3-2-manual_implementation.jl")
