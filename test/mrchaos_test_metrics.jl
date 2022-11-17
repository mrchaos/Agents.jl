using Test, Agents, Random
using Agents.Graphs, Agents.DataFrames
using Agents.Pathfinding
using StatsBase: mean
using StableRNGs
using PrettyPrint:pprintln

mutable struct Agent0 <: AbstractAgent
    id::Int
end

mutable struct Agent1 <: AbstractAgent
    id::Int
    pos::Dims{2}
end

mutable struct Agent2 <: AbstractAgent
    id::Int
    weight::Float64
end

mutable struct Agent3 <: AbstractAgent
    id::Int
    pos::Dims{2}
    weight::Float64
end

mutable struct Agent4 <: AbstractAgent
    id::Int
    pos::Dims{2}
    p::Int
end

mutable struct Agent5 <: AbstractAgent
    id::Int
    pos::Int
    weight::Float64
end

mutable struct Agent6 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    weight::Float64
end

mutable struct Agent7 <: AbstractAgent
    id::Int
    pos::Int
    f1::Bool
    f2::Int
end

Agent7(id, pos; f1, f2) = Agent7(id, pos, f1, f2)

mutable struct Agent8 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    f1::Bool
    f2::Int
end

moore = Pathfinding.moore_neighborhood(2)
vonneumann = Pathfinding.vonneumann_neighborhood(2)
gspace = GridSpace((5, 5))
cspace = ContinuousSpace((5., 5.))
atol = 0.0001

# struct AStar{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
# function AStar{D,P,M,T,C}(
#    agent_paths::Dict,
#    dims::NTuple{D,T},
#    neighborhood::Vector{CartesianIndex{D}},
#    admissibility::Float64,
#    walkmap::BitArray{D},
#    cost_metric::C,
# ) where {D,P,M,C,T}
#
# D : dimension, P : periodic, M : diagonal_movement, T : type
# C :  cost metric type

pfinder_2d_np_m = AStar{2,false,true,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(moore),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_np_nm = AStar{2,false,false,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(vonneumann),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_p_m = AStar{2,true,true,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(moore),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_p_nm = AStar{2,true,false,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(vonneumann),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)