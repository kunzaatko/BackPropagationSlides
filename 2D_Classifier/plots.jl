using MakieMaestro, LazyGrids, ProgressLogging
using Makie: @lift

include("nn-classification.jl")
include("nn-lsqfit.jl")

labels(ys) =
    map(ys) do y
        y[1] > y[2] ? 0 : 1
    end

label_values(ys) = begin
    A = map(ys) do y
        y[1] > y[2] ? y[1] : missing
    end
    B = map(ys) do y
        y[1] < y[2] ? y[2] : missing
    end
    return Array{Union{eltype(ys).parameters[1],Missing}}(A), Array{Union{eltype(ys).parameters[1],Missing}}(B)
end

function data_scatters!(ax, x=data_x, y=data_y; scatter_kwargs...)
    labels = map(y -> y == [true; false] ? 1 : 0, eachcol(y))
    scatter!(ax, x[:, labels.==1];
        markersize=15,
        marker=:circle,
        color=:red,
        label=L"y=1"
    )
    scatter!(ax, x[:, labels.==0];
        markersize=15,
        marker=:xcross,
        color=:blue, label=L"y=0")
end


function fig_scatter_data(x=data_x, y=data_y; scatter_kwargs...)
    f = Figure()
    a = Axis(f[1, 1])
    a.xlabel = L"x_1"
    a.ylabel = L"x_2"
    data_scatters!(a)
    axislegend(a)
    return f
end

# TODO: Add the points and interaction with finding the best optimum when a point is added <16-03-25> 

function fig_NN_decisions!(a, func)
    x1_range = range(0, 1, 1000)
    x2_range = range(0, 1, 1000)
    x1s, x2s = ndgrid(x1_range, x2_range)
    l_values_A, l_values_B = label_values(func.(x1s, x2s))
    if !(all(ismissing.(l_values_A)))
        heatmap!(a, x1_range, x2_range, l_values_A, colormap=:blues, alpha=0.5)
    end
    if !(all(ismissing.(l_values_B)))
        heatmap!(a, x1_range, x2_range, l_values_B, colormap=:reds, alpha=0.5)
    end
end

function fig_NN_2_3_2_decisions(Pval)
    f = Figure()
    a = Axis(f[1, 1])
    fig_NN_decisions!(a, (x1, x2) -> NN_2_3_2_forward(x1, x2; Pval))
    data_scatters!(a)
    return f
end

function _fig_NN_2_3_2_decisions_train!(a, Ps_log, iteration::Observable)
    x1_range = range(0, 1, 200)
    x2_range = range(0, 1, 200)
    x1s, x2s = ndgrid(x1_range, x2_range)

    l_values = @lift label_values(map((x1, x2) -> NN_2_3_2_forward(x1, x2; Pval=Ps_log[$iteration]), x1s, x2s))
    l_values_A = @lift ($l_values)[1]
    l_values_B = @lift ($l_values)[2]

    heatmap!(a, x1_range, x2_range, l_values_A, colormap=:blues, alpha=0.5)
    heatmap!(a, x1_range, x2_range, l_values_B, colormap=:reds, alpha=0.5)

    data_scatters!(a)
    return l_values_A, l_values_B
end

function fig_NN_2_3_2_decisions_train_interactive(Ps_log, loss_log, iterations=Colon(); kwargs...)
    f = Figure()
    a_boundary = Axis(f[1, 1])

    iteration_indices = eachindex(Ps_log)[iterations]
    slg = SliderGrid(f[2, 1:2], (label=L("iteration"), range=iteration_indices, startvalue=first(iteration_indices)))
    iteration = slg.sliders[1].value

    # TODO: Create a macro for this Observable deference functionality in MakieMaestro.jl <24-03-25> 
    iteration_lazy = Observable{Int64}(iteration[])
    Timer(0.0, interval=0.2) do _
        if iteration_lazy[] != iteration[]
            iteration_lazy[] = iteration[]
            notify(iteration_lazy)
        end
    end

    _, _ = _fig_NN_2_3_2_decisions_train!(a_boundary, Ps_log, iteration_lazy)

    a_loss = Axis(f[1, 2])

    lines!(a_loss, iteration_indices, loss_log[iteration_indices];)
    scatter!(a_loss, iteration, @lift(loss_log[$iteration]); color=:red, markersize=20)

    return f
end

function fig_NN_2_3_2_decisions_train_animation(Ps_log, loss_log, iterations=Colon(); filename="training_animation.mkv", kwargs...)
    f = Figure()
    a = Axis(f[1, 1])

    iteration_indices = eachindex(Ps_log)[iterations]
    iteration = Observable{Int64}(first(iteration_indices))

    _, _ = _fig_NN_2_3_2_decisions_train!(a, Ps_log, iteration)

    @withprogress name = "recording" begin
        loss_pad_size = maximum(length.(string.(round.(loss_log[iteration_indices]; sigdigits=3))))
        record(f, filename, enumerate(iteration_indices); framerate=round(Int, length(iteration_indices) / 10), backend=CairoMakie, kwargs...) do (ind, i)
            iteration[] = i
            a.title = L("Iteration: " * lpad(string(i), length(string(last(iteration_indices)))))
            a.subtitle = L(" Loss: " * lpad(string(round(loss_log[i]; sigdigits=3)), loss_pad_size))
            @logprogress ind / length(iteration_indices)
        end
    end
end

function fig_NN_2_3_2_loss_lines(loss_log)
    f, a, _ = lines(loss_log; axis=(; yscale=log10))
    a.xlabel = L"i"
    a.ylabel = L"\ell_2(N)"
    a.title = L("Loss during the training")
    return f
end

function fig_NN_2_3_2_lux_contourf(model, params, state)
    f = Figure()
    a = Axis(f[1, 1])
    fig_NN_decisions!(a, (x1, x2) -> first(model([x1, x2], params, state)))
    data_scatters!(a)
    return f
end
fig_NN_2_3_2_lux_contourf(trainstate) = fig_NN_2_3_2_lux_contourf(trainstate.model, trainstate.parameters, trainstate.states)

function fig_ReLU_variants(params=[(w=1, b=0), (w=1.5, b=0), (w=1, b=0.5), (w=2.5, b=-1)], activation=relu)
    f = Figure()
    a = Axis(f[1, 1]; title=L"\mathrm{ReLU}(wx+b)")
    for ps in params
        lines!(a, -1 .. 1, x -> activation(ps.w * x + ps.b);
            label=latexstring("w=$(ps.w), b=$(ps.b)")
        )
    end
    axislegend(a; position=:lt)
    return f
end

const gate_activations = [(relu, name=L"\mathrm{ReLU}"), (gelu, name=L"\mathrm{GeLU}"), (mish, name=L"\mathrm{Mish}")]
const math_activations = [(sigmoid, name=L"\sigma"), (tanh, name=L"\mathrm{tanh}"),]

function fig_activations(activations=math_activations, range=-5 .. 5)
    f = Figure()
    a = Axis(f[1, 1]; title=L("Activation functions"))
    for act in activations
        lines!(a, range, x -> act[1](x);
            label=act.name
        )
    end
    axislegend(a; position=:lt)
    return f
end

function fig_nelder_mead_fit()
    f = Figure()
    axs = [Axis(f[i, j]) for i in 1:3, j in 1:3]
    @progress for a in axs
        best_fit, loss = fit_NN_2_3_2()
        a.title = L("Loss: " * string(round(loss; digits=2)))
        fig_NN_decisions!(a, (x1, x2) -> NN_2_3_2_forward(x1, x2; Pval=best_fit))
        data_scatters!(a)
    end
    linkaxes!(axs...)
    return f
end
