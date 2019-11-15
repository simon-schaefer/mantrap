#= The DiscreteTimeVelocity Ado is a general definition of a dynamic ado that is moving
over time given some defined probability density function in the velocity space (vpdf). It
implements methods to sample trajectories, build the next pdf in the position space and stores
the ados history (position space). =#
export DTVAdo


struct DTVAdo <:AdoAgent

    history::Trajectory
    num_modes::Int8
    dt::Float32
end


function DTVAdo(history::Array{Float64, 2}, num_modes::Int64, dt::Float64=1.0)
    @assert num_modes > 0 "number of modes must be larger than 0"
    @assert length(history) % 2 == 0 "history array must consist of two-dimensional points"
    @assert length(history) > 0 "history array must not be empty"
    @assert dt > 0 "time-delta between discrete time-steps must be larger 0"

    new(history, num_modes, dt)
end


position() = history[-1, :]